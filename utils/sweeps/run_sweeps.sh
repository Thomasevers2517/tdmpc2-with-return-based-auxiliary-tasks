#!/usr/bin/env bash
set -euo pipefail

# Powerful W&B sweep orchestrator for TD-MPC2 experiments
# --------------------------------------------------------
# Features
# - Processes leaf sweeps lexicographically; eager rollover: as soon as ANY agent
#   reports the sweep is finished (no runs left), remaining agents are cancelled and
#   we immediately move to the next sweep.
# - Local and Slurm modes, with robust log capture and completion detection.
# - Verbose, timestamped logging; optional dry-run; optional watch/daemon mode.
# - Estimates expected grid size from sweep.yaml (best effort) and records metadata
#   per sweep to sweep_state.json and progress.txt.
# - Optional W&B progress probing (best-effort) if Python W&B public API is available; 
#   emits per-sweep runs_index.json and runs_configs.csv.
# - Respects a completion marker (completed.txt) but can resume/force as requested.
#
# Usage (local):
#   utils/sweeps/run_sweeps.sh [--dry-run] [--resume] [--force] [--max-sweeps N] [--watch] [--interval SEC] <root_or_chapter_folder> <COUNT>
# Example:
#   utils/sweeps/run_sweeps.sh sweep_list/midterm_sweep 4
#
# Usage (slurm):
#   utils/sweeps/run_sweeps.sh --slurm --partition <P> --gres <GRES> --cpus <N> --mem <GB> --time <HH:MM:SS> --env <conda_env> \
#                              [--dry-run] [--resume] [--force] [--max-sweeps N] [--watch] [--interval SEC] <root> <COUNT>

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "$(ts) | INFO  | $*"; }
warn(){ echo "$(ts) | WARN  | $*" >&2; }
err() { echo "$(ts) | ERROR | $*" >&2; }

# Return 0 if file appears to be a leaf sweep (has sweep.yaml and no nested sweep.yaml under it)
is_leaf_sweep_dir() {
  local dir="$1"
  [[ -f "$dir/sweep.yaml" ]] || return 1
  if find "$dir" -mindepth 2 -type f -name sweep.yaml | grep -q .; then
    return 1
  fi
  return 0
}

# Best-effort estimator of expected grid size from sweep.yaml.
# Sets EXPECTED_RUNS, METHOD, and PARAM_COUNTS (JSON-like string).
estimate_sweep_size() {
  local yaml="$1"
  METHOD="unknown"
  EXPECTED_RUNS="?"
  PARAM_COUNTS="{}"

  # Try Python + PyYAML first for robust parsing
  local pyout
  pyout=$(python - << 'PY' "$yaml" 2>/dev/null || true)
import sys, json
try:
    import yaml
except Exception:
    print("NOYAML")
    sys.exit(0)
from pathlib import Path
data = yaml.safe_load(Path(sys.argv[1]).read_text())
method = data.get('method', 'unknown')
params = data.get('parameters', {}) or {}
counts = {}
total = 1
def count_vals(v):
    if isinstance(v, dict):
        if 'values' in v:
            vv = v['values']
            if isinstance(vv, list):
                return len(vv)
            else:
                return 1
        elif 'value' in v:
            return 1
        else:
            return 1
    else:
        return 1
for k, v in params.items():
    c = count_vals(v)
    counts[k] = c
    total *= c
print(json.dumps({"method": method, "total": total, "counts": counts}))
PY
  )

  if [[ -n "$pyout" && "$pyout" != "NOYAML" ]]; then
    METHOD=$(echo "$pyout" | python - << 'PY'
import sys, json
obj=json.loads(sys.stdin.read())
print(obj.get('method','unknown'))
PY
    )
    EXPECTED_RUNS=$(echo "$pyout" | python - << 'PY'
import sys, json
obj=json.loads(sys.stdin.read())
print(obj.get('total','?'))
PY
    )
    PARAM_COUNTS=$(echo "$pyout" | python - << 'PY'
import sys, json
obj=json.loads(sys.stdin.read())
print(json.dumps(obj.get('counts',{})))
PY
    )
    log "Estimated sweep size via PyYAML: method=$METHOD expected_runs=$EXPECTED_RUNS counts=$PARAM_COUNTS"
    return 0
  fi

  warn "PyYAML not available; falling back to naive estimator. Install PyYAML for robust parsing."
  # Naive parser: count occurrences of list items under 'values:' or inline [a,b]
  METHOD=$(grep -E '^method:' "$yaml" | awk '{print $2}' || echo unknown)
  local counts_json="{"
  local total=1
  # Parse parameters section roughly
  awk '/^parameters:/{flag=1;next}/^[^[:space:]]/{if(flag){exit}}flag{print}' "$yaml" | \
  awk 'BEGIN{param=""}
  /^[[:space:]]+[A-Za-z0-9_.-]+:$/ {gsub(":","",$1); param=$1; next}
  /values:[[:space:]]*\[/ {
     n=gsub(",","&"); cnt=n+1; if (cnt<1) cnt=1; print param"="cnt; next
  }
  /values:/ {mode=1; cnt=0; next}
  /^[[:space:]]*-[[:space:]]/ && mode==1 {cnt++}
  /^[[:space:]]*[^-].*$/ && mode==1 {if(cnt>0){print param"="cnt}; mode=0}
  END{if(mode==1 && cnt>0){print param"="cnt}}' | while IFS='=' read -r k v; do
    if [[ -n "$k" && -n "$v" ]]; then
      counts_json+="\"$k\":$v,"
      total=$(( total * v ))
    fi
  done
  if [[ "$counts_json" == "{" ]]; then counts_json+="}"; else counts_json=${counts_json%,}"}"; fi
  PARAM_COUNTS="$counts_json"
  EXPECTED_RUNS="$total"
  log "Estimated sweep size (naive): method=$METHOD expected_runs=$EXPECTED_RUNS counts=$PARAM_COUNTS"
}

# Best-effort probe of W&B completed runs for a sweep path entity/proj/id.
# Sets WANDB_DONE (int) and WANDB_TOTAL (int or '?') if possible.
probe_wandb_progress() {
  local sweep_path="$1"
  WANDB_DONE="?"; WANDB_TOTAL="?"
  local py
  py=$(python - << 'PY' "$sweep_path" 2>/dev/null || true)
import sys
sp=sys.argv[1]
try:
    import wandb
    from wandb.apis import public
except Exception:
    print("NOWANDB")
    sys.exit(0)
api=public.Api()
try:
    ent, proj, sid = sp.split('/')
    sw = api.sweep(f"{ent}/{proj}/{sid}")
    runs = list(sw.runs)
    done = sum(1 for r in runs if r.state in ("finished","killed","crashed","failed"))
    total = len(runs)
    print(f"{done} {total}")
except Exception:
    print("ERR")
PY
  )
  if [[ "$py" == "NOWANDB" ]] || [[ "$py" == "ERR" ]]; then
    warn "W&B public API not available for progress probe (this is optional)."
    return 1
  fi
  WANDB_DONE=$(echo "$py" | awk '{print $1}')
  WANDB_TOTAL=$(echo "$py" | awk '{print $2}')
  log "W&B progress: done=$WANDB_DONE total=$WANDB_TOTAL"
}

# Write a global status snapshot for all leaf sweeps under a root
write_global_status() {
  local root="$1"
  local status_file="$root/sweeps_status.txt"
  local tmp_file="${status_file}.tmp"
  echo "$(ts) | status snapshot for root: $root" > "$tmp_file"
  local ymls
  mapfile -t ymls < <(find "$root" -type f -name sweep.yaml | sort)
  for y in "${ymls[@]}"; do
    local dir
    dir=$(dirname "$y")
    if ! is_leaf_sweep_dir "$dir"; then
      continue
    fi
    local idf="$dir/id.txt"
    local sp="(missing)"
    [[ -s "$idf" ]] && sp=$(cat "$idf")
    local completed="no"
    [[ -f "$dir/completed.txt" ]] && completed="yes"
    local method="?" exp="?" wc_done="?" wc_total="?"
    if [[ -f "$dir/sweep_state.json" ]]; then
      method=$(grep -o '"method"\s*:\s*"[^"]*"' "$dir/sweep_state.json" | sed -E 's/.*"([^"]+)"/\1/')
      exp=$(grep -o '"expected_runs"\s*:\s*"?[0-9?]+"?' "$dir/sweep_state.json" | grep -o '[0-9?]\+')
      wc_done=$(grep -o '"wandb_done"\s*:\s*"?[0-9?]+"?' "$dir/sweep_state.json" | grep -o '[0-9?]\+' | head -n1)
      wc_total=$(grep -o '"wandb_total"\s*:\s*"?[0-9?]+"?' "$dir/sweep_state.json" | grep -o '[0-9?]\+' | head -n1)
    fi
    local npids=$(ls "$dir"/.agent_*.pid 2>/dev/null | wc -l | awk '{print $1}')
    local njobs=$(ls "$dir"/.job_*.id 2>/dev/null | wc -l | awk '{print $1}')
    echo "sweep: $sp | dir: $dir | completed: $completed | local_agents: $npids | slurm_jobs: $njobs | method: $method | expected_runs: $exp | wandb: $wc_done/$wc_total" >> "$tmp_file"
  done
  mv "$tmp_file" "$status_file"
}

# Export W&B run index and configs to files and write a per-sweep progress.txt
update_per_sweep_progress_files() {
  local dir="$1"; local sweep_path="$2"; local expected="$3"; local method="$4"
  local progress_file="$dir/progress.txt"
  local runs_json="$dir/runs_index.json"
  local runs_csv="$dir/runs_configs.csv"
  local py
  py=$(python - << 'PY' "$sweep_path" "$runs_json" "$runs_csv" 2>/dev/null || true)
import sys,json
sp, runs_json, runs_csv = sys.argv[1:4]
try:
    import wandb
    from wandb.apis import public
except Exception:
    print("NOWANDB")
    sys.exit(0)
api=public.Api()
try:
    ent, proj, sid = sp.split('/')
    sw = api.sweep(f"{ent}/{proj}/{sid}")
    runs = list(sw.runs)
    # Write JSON index
    with open(runs_json,'w') as f:
        json.dump([
            {
                'id': r.id,
                'name': r.name,
                'state': r.state,
                'created_at': str(r.created_at),
                'config': getattr(r,'config',{})
            } for r in runs
        ], f)
    # Write CSV of selected config keys (flattened, best-effort)
    import csv
    keys=set()
    for r in runs:
        cfg=getattr(r,'config',{}) or {}
        for k in cfg.keys():
            keys.add(k)
    keys=sorted(list(keys))
    with open(runs_csv,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['id','name','state']+keys)
        for r in runs:
            cfg=getattr(r,'config',{}) or {}
            row=[r.id,r.name,r.state] + [cfg.get(k,'') for k in keys]
            w.writerow(row)
    print(len(runs))
except Exception:
    print("ERR")
PY
  )
  local done="?" total="?"
  if [[ "$py" != "NOWANDB" && "$py" != "ERR" && -n "$py" ]]; then
    total="$py"
    if [[ -f "$runs_csv" ]]; then
      done=$(awk -F',' 'NR>1 && ($3=="finished" || $3=="failed" || $3=="crashed" || $3=="killed"){c++} END{print c+0}' "$runs_csv")
    fi
  fi
  echo "$(ts) | method=$method expected_runs=$expected wandb=$done/$total" > "$progress_file"
}

MODE="local"
PARTITION=""; GRES=""; CPUS=""; MEM=""; TIME=""; CONDA_ENV=""
DRY_RUN=0; RESUME=0; FORCE=0; MAX_SWEEPS=""
WATCH=0; INTERVAL=60
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slurm) MODE="slurm"; shift ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume) RESUME=1; shift ;;
    --force) FORCE=1; shift ;;
    --max-sweeps) MAX_SWEEPS="$2"; shift 2 ;;
    --watch) WATCH=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

ROOT_DIR=${ARGS[0]:-sweep_list}
COUNT=${ARGS[1]:-1}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

if ! command -v wandb >/dev/null 2>&1; then
  err "wandb CLI not found in PATH. Activate the tdmpc2 conda env first."
  exit 1
fi

if [[ "$MODE" == "slurm" ]]; then
  # Require explicit slurm resources to avoid accidental defaults.
  if [[ -z "$PARTITION" || -z "$GRES" || -z "$CPUS" || -z "$MEM" || -z "$TIME" || -z "$CONDA_ENV" ]]; then
    echo "Error: --slurm requires --partition, --gres, --cpus, --mem, --time, and --env <conda_env>" >&2
    exit 1
  fi
  mkdir -p slurm_logs
fi

# Optional watcher that periodically writes global and per-sweep status files.
start_watcher() {
  local root="$1"; local interval="$2"
  local pidfile="$root/.watcher.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    warn "Watcher already running (PID $(cat "$pidfile")); leaving it active."
    return
  fi
  (
    while true; do
      write_global_status "$root" || true
      # per-sweep progress files
      mapfile -t ymls < <(find "$root" -type f -name sweep.yaml | sort)
      for y in "${ymls[@]}"; do
        local dir
        dir=$(dirname "$y")
        if ! is_leaf_sweep_dir "$dir"; then continue; fi
        local idf="$dir/id.txt"
        [[ -s "$idf" ]] || continue
        local sp
        sp=$(cat "$idf")
        local method="?" expected="?"
        if [[ -f "$dir/sweep_state.json" ]]; then
          method=$(grep -o '"method"\s*:\s*"[^"]*"' "$dir/sweep_state.json" | sed -E 's/.*"([^"]+)"/\1/')
          expected=$(grep -o '"expected_runs"\s*:\s*"?[0-9?]+"?' "$dir/sweep_state.json" | grep -o '[0-9?]\+')
        else
          # fallback to quick estimation
          estimate_sweep_size "$y" || true
          method="$METHOD"; expected="$EXPECTED_RUNS"
        fi
        update_per_sweep_progress_files "$dir" "$sp" "$expected" "$method" || true
      done
      sleep "$interval"
    done
  ) &
  echo $! > "$pidfile"
  log "Started watcher (PID $(cat "$pidfile"), interval=${interval}s)"
}

stop_watcher() {
  local root="$1"; local pidfile="$root/.watcher.pid"
  if [[ -f "$pidfile" ]]; then
    local wpid
    wpid=$(cat "$pidfile")
    if kill -0 "$wpid" 2>/dev/null; then
      kill "$wpid" 2>/dev/null || true
      rm -f "$pidfile"
      log "Stopped watcher (PID $wpid)"
    else
      rm -f "$pidfile"
    fi
  fi
}

mapfile -t YAML_FILES < <(find "$ROOT_DIR" -type f -name sweep.yaml | sort)
if [[ ${#YAML_FILES[@]} -eq 0 ]]; then
  warn "No sweep.yaml files found under $ROOT_DIR"
  exit 0
fi

COUNT_SWEEPS=0

# Start watcher if requested
if [[ $WATCH -eq 1 ]]; then
  start_watcher "$ROOT_DIR" "$INTERVAL"
fi

for YAML in "${YAML_FILES[@]}"; do
  DIR=$(dirname "$YAML")
  # Only leaf sweep folders
  if ! is_leaf_sweep_dir "$DIR"; then
    log "Skipping non-leaf sweep folder: $DIR"
    continue
  fi

  # Skip if already completed
  if [[ -f "$DIR/completed.txt" && $FORCE -eq 0 && $RESUME -eq 0 ]]; then
    log "Sweep already marked completed: $DIR (use --resume or --force to override)"
    continue
  fi

  ID_FILE="$DIR/id.txt"
  if [[ ! -s "$ID_FILE" ]]; then
    log "id.txt missing or empty in $DIR; creating sweep..."
    "$SCRIPT_DIR/create_or_update_sweeps.sh" "$DIR"
  fi
  SWEEP_PATH=$(cat "$ID_FILE")
  if [[ -z "$SWEEP_PATH" ]]; then
    err "empty sweep id in $ID_FILE"
    exit 1
  fi

  echo "----------------------------------------"
  echo "Running sweep: $SWEEP_PATH"
  echo "Folder: $DIR"
  echo "Agents: $COUNT | Mode: $MODE | Dry-run: $DRY_RUN | Resume: $RESUME | Force: $FORCE"
  echo "----------------------------------------"

  # Estimate grid size (best effort) and write/refresh sweep_state.json
  estimate_sweep_size "$YAML" || true
  probe_wandb_progress "$SWEEP_PATH" || true
  STATE_FILE="$DIR/sweep_state.json"
  (
    echo '{'
    echo "  \"sweep_path\": \"$SWEEP_PATH\","
    echo "  \"method\": \"$METHOD\","
    echo "  \"expected_runs\": \"$EXPECTED_RUNS\","
    echo "  \"param_counts\": $PARAM_COUNTS,"
    echo "  \"wandb_done\": \"${WANDB_DONE:-?}\","
    echo "  \"wandb_total\": \"${WANDB_TOTAL:-?}\","
    echo "  \"timestamp\": \"$(ts)\""
    echo '}'
  ) > "$STATE_FILE"
  log "Wrote/updated $STATE_FILE"

  # Update per-sweep progress snapshot immediately
  update_per_sweep_progress_files "$DIR" "$SWEEP_PATH" "$EXPECTED_RUNS" "$METHOD" || true

  if [[ $DRY_RUN -eq 1 ]]; then
    log "Dry-run: not launching agents for $SWEEP_PATH"
    COUNT_SWEEPS=$((COUNT_SWEEPS+1))
    if [[ -n "$MAX_SWEEPS" && $COUNT_SWEEPS -ge $MAX_SWEEPS ]]; then
      log "Reached --max-sweeps=$MAX_SWEEPS, stopping."
      break
    fi
    continue
  fi

  # Common completion regex for wandb agent messages
  COMPLETE_RE="(sweep finished|no runs left|no more runs|sweep is finished)"

  if [[ "$MODE" == "local" ]]; then
    # Launch COUNT local agents with very large --count so they keep pulling
    # runs until the sweep is completed; then wait for all to finish.
    PIDS=()
    LOGS=()
    for ((i=1; i<=COUNT; i++)); do
      log "[local] starting agent $i/$COUNT for $SWEEP_PATH"
      LOG="$DIR/.agent_$i.log"
      # capture both stdout+stderr for completion detection
      ( wandb agent --count 99999999 "$SWEEP_PATH" |& tee "$LOG" ) &
      PID=$!
      echo "$PID" >"$DIR/.agent_$i.pid"
      PIDS+=("$PID")
      LOGS+=("$LOG")
      sleep 1
    done
    # Monitor agents: as soon as ANY agent log shows completion, stop others
    COMPLETED=0
    while :; do
      # Check logs for completion signal
      for LOG in "${LOGS[@]}"; do
        if [[ -f "$LOG" ]] && grep -qiE "$COMPLETE_RE" "$LOG"; then
          COMPLETED=1
          break
        fi
      done
      if [[ $COMPLETED -eq 1 ]]; then
  log "Completion detected in one agent; stopping remaining agents..."
        # Terminate any still-running agents
        for pid in "${PIDS[@]}"; do
          if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
          fi
        done
        # Wait a short grace period for clean exits
        sleep 5
        # Best-effort wait for remaining pids
        for pid in "${PIDS[@]}"; do
          if kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
          fi
        done
        rm -f "$DIR"/.agent_*.pid || true
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$DIR/completed.txt"
        echo "Marked completed: $DIR"
        update_per_sweep_progress_files "$DIR" "$SWEEP_PATH" "$EXPECTED_RUNS" "$METHOD" || true
        break
      fi
      # If all agents have exited without completion signal, break
      ANY_RUNNING=0
      for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then ANY_RUNNING=1; break; fi
      done
      if [[ $ANY_RUNNING -eq 0 ]]; then
  log "All agents exited; checking logs for completion..."
        # Final check
        for LOG in "${LOGS[@]}"; do
          if [[ -f "$LOG" ]] && grep -qiE "$COMPLETE_RE" "$LOG"; then
            COMPLETED=1; break
          fi
        done
        if [[ $COMPLETED -eq 1 ]]; then
          date -u +"%Y-%m-%dT%H:%M:%SZ" > "$DIR/completed.txt"
          log "Marked completed: $DIR"
          update_per_sweep_progress_files "$DIR" "$SWEEP_PATH" "$EXPECTED_RUNS" "$METHOD" || true
        else
          warn "Not marking completed (no W&B completion signal found)."
        fi
        rm -f "$DIR"/.agent_*.pid || true
        break
      fi
      sleep 5
    done
  else
    # Submit COUNT Slurm jobs and wait until all job IDs are gone from the queue.
    JOBS=()
    for ((i=1; i<=COUNT; i++)); do
  log "[slurm] submitting agent $i/$COUNT for $SWEEP_PATH"
      JID=$(sbatch -p "$PARTITION" --gres="$GRES" -c "$CPUS" --mem="$MEM" -t "$TIME" \
            --job-name "wb_sweep_${i}" --output "slurm_logs/%j.out" --error "slurm_logs/%j.err" \
            --wrap "source ~/.bashrc && conda activate $CONDA_ENV && wandb agent --count 99999999 $SWEEP_PATH" \
            | awk '{print $4}')
      echo "$JID" > "$DIR/.job_${i}.id"
      JOBS+=("$JID")
      sleep 1
    done
    # Poll until either completion detected in logs or all jobs finished
    COMPLETED=0
    while :; do
      # Check logs for completion signal early
      for J in "${JOBS[@]}"; do
        OUT="slurm_logs/${J}.out"; ERR="slurm_logs/${J}.err"
        if [[ -f "$OUT" ]] && grep -qiE "$COMPLETE_RE" "$OUT"; then COMPLETED=1; break; fi
        if [[ -f "$ERR" ]] && grep -qiE "$COMPLETE_RE" "$ERR"; then COMPLETED=1; break; fi
      done
      if [[ $COMPLETED -eq 1 ]]; then
  log "[slurm] completion detected; cancelling remaining jobs..."
        for J in "${JOBS[@]}"; do
          if squeue -j "$J" 2>/dev/null | grep -q "$J"; then
            scancel "$J" 2>/dev/null || true
          fi
        done
        # wait until none remain
        while :; do
          REM=0
          for J in "${JOBS[@]}"; do
            if squeue -j "$J" 2>/dev/null | grep -q "$J"; then REM=$((REM+1)); fi
          done
          [[ $REM -eq 0 ]] && break
          sleep 5
        done
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$DIR/completed.txt"
        log "Marked completed: $DIR"
        update_per_sweep_progress_files "$DIR" "$SWEEP_PATH" "$EXPECTED_RUNS" "$METHOD" || true
        break
      fi
      # Otherwise, continue polling active jobs
      REM=0
      for J in "${JOBS[@]}"; do
        if squeue -j "$J" 2>/dev/null | grep -q "$J"; then REM=$((REM+1)); fi
      done
      if [[ $REM -eq 0 ]]; then
        echo "[slurm] all jobs finished; checking logs for completion..."
        # Final check
        for J in "${JOBS[@]}"; do
          OUT="slurm_logs/${J}.out"; ERR="slurm_logs/${J}.err"
          if [[ -f "$OUT" ]] && grep -qiE "$COMPLETE_RE" "$OUT"; then COMPLETED=1; break; fi
          if [[ -f "$ERR" ]] && grep -qiE "$COMPLETE_RE" "$ERR"; then COMPLETED=1; break; fi
        done
        if [[ $COMPLETED -eq 1 ]]; then
          date -u +"%Y-%m-%dT%H:%M:%SZ" > "$DIR/completed.txt"
          echo "Marked completed: $DIR"
          update_per_sweep_progress_files "$DIR" "$SWEEP_PATH" "$EXPECTED_RUNS" "$METHOD" || true
        else
          echo "Not marking completed (no W&B completion signal found)."
        fi
        break
      fi
      log "[slurm] waiting... $REM agents still running"
      sleep 10
    done
    rm -f "$DIR"/.job_*.id || true
  fi

  COUNT_SWEEPS=$((COUNT_SWEEPS+1))
  if [[ -n "$MAX_SWEEPS" && $COUNT_SWEEPS -ge $MAX_SWEEPS ]]; then
    log "Reached --max-sweeps=$MAX_SWEEPS, stopping."
    break
  fi
done

log "All sweeps processed."

# Stop watcher if running
if [[ $WATCH -eq 1 ]]; then
  stop_watcher "$ROOT_DIR"
fi
