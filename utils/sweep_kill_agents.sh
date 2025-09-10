#!/usr/bin/env bash
# Kill running W&B sweep agents and their spawned runs by GPU id(s).
# Usage: bash utils/sweep_kill_agents.sh <gpu_list>
# Example: bash utils/sweep_kill_agents.sh 2,3

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <gpu_list (comma-separated)>" >&2
  exit 1
fi

GPU_LIST="$1"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# Read an env var from a PID's environment
get_env_var() {
  local PID="$1"; local VAR="$2"
  if [[ -r "/proc/$PID/environ" ]]; then
    tr '\0' '\n' < "/proc/$PID/environ" | awk -F= -v v="$VAR" '$1==v{print $2; exit}'
  else
    echo ""
  fi
}

get_env_cuda() { get_env_var "$1" "CUDA_VISIBLE_DEVICES"; }

# Return all descendants (children, grandchildren, ...) of a PID
list_descendants() {
  local ROOT="$1"
  local queue=($ROOT)
  local out=()
  local seen=" $ROOT "
  while ((${#queue[@]})); do
    local p="${queue[0]}"; queue=(${queue[@]:1})
    mapfile -t kids < <(pgrep -P "$p" 2>/dev/null || true)
    for k in "${kids[@]:-}"; do
      [[ -n "$k" ]] || continue
      if [[ "$seen" != *" $k "* ]]; then
        out+=("$k"); queue+=("$k"); seen+=" $k "
      fi
    done
  done
  printf '%s\n' "${out[@]}" | sort -u
}

# Kill a PID tree (children first), with TERM then KILL
kill_tree() {
  local ROOT="$1"; local WHY="$2"
  if ! kill -0 "$ROOT" 2>/dev/null; then
    echo "PID $ROOT not running ($WHY)"
    return 0
  fi
  local DESC=($(list_descendants "$ROOT"))
  # Kill descendants first
  for pid in "${DESC[@]:-}"; do
    local cmd="$(ps -o cmd= -p "$pid" 2>/dev/null || true)"
    echo "TERM child PID $pid of $ROOT ($WHY): $cmd"
    kill -TERM "$pid" 2>/dev/null || true
  done
  # Then kill root
  local rcmd="$(ps -o cmd= -p "$ROOT" 2>/dev/null || true)"
  echo "TERM root PID $ROOT ($WHY): $rcmd"
  kill -TERM "$ROOT" 2>/dev/null || true

  # Wait a bit
  for i in {1..20}; do
    sleep 0.5
    local alive=0
    for pid in "${DESC[@]:-}" "$ROOT"; do
      if kill -0 "$pid" 2>/dev/null; then alive=1; fi
    done
    ((alive==0)) && break
  done

  # Force kill survivors
  for pid in "${DESC[@]:-}" "$ROOT"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "KILL PID $pid ($WHY)"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

# Collect agent PIDs using pidfiles and process scans
collect_agent_pids_for_gpu() {
  local GPU="$1"
  local -a results=()
  # From pidfiles
  mapfile -t PID_FILES < <(find "$REPO_ROOT" -maxdepth 5 -type f -name "agent_gpu${GPU}.pid" 2>/dev/null || true)
  for f in "${PID_FILES[@]:-}"; do
    [[ -f "$f" ]] || continue
    local p; p="$(cat "$f" 2>/dev/null || true)"
    [[ -n "$p" ]] || continue
    if kill -0 "$p" 2>/dev/null; then
      local envgpu; envgpu="$(get_env_cuda "$p")"
      if [[ "$envgpu" == "$GPU" ]]; then results+=("$p"); fi
    fi
  done
  # Process scan for wandb agents
  mapfile -t SCAN1 < <(pgrep -f "python -m wandb agent" 2>/dev/null || true)
  mapfile -t SCAN2 < <(pgrep -f "wandb agent" 2>/dev/null || true)
  local uniq=("${SCAN1[@]}" "${SCAN2[@]}")
  for p in "${uniq[@]:-}"; do
    [[ -n "$p" ]] || continue
    if kill -0 "$p" 2>/dev/null; then
      local envgpu; envgpu="$(get_env_cuda "$p")"
      if [[ "$envgpu" == "$GPU" ]]; then results+=("$p"); fi
    fi
  done
  printf '%s\n' "${results[@]}" | sort -u
}

# Collect any processes on GPU with WANDB_AGENT set (likely agent or its run)
collect_wandb_proc_for_gpu() {
  local GPU="$1"
  local -a results=()
  mapfile -t PSALL < <(pgrep -a . 2>/dev/null || true)
  for line in "${PSALL[@]:-}"; do
    local p="${line%% *}"  # pid is first token
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    if ! kill -0 "$p" 2>/dev/null; then continue; fi
    local envgpu; envgpu="$(get_env_cuda "$p")"
    [[ "$envgpu" == "$GPU" ]] || continue
    local wanagent; wanagent="$(get_env_var "$p" "WANDB_AGENT" )"
    if [[ -n "$wanagent" ]]; then results+=("$p"); fi
  done
  printf '%s\n' "${results[@]}" | sort -u
}

# Collect training script roots (tdmpc2/train.py) on given GPU
collect_train_roots_for_gpu() {
  local GPU="$1"
  local -a results=()
  mapfile -t PIDS < <(pgrep -f "tdmpc2/train.py" 2>/dev/null || true)
  for p in "${PIDS[@]:-}"; do
    [[ -n "$p" ]] || continue
    if ! kill -0 "$p" 2>/dev/null; then continue; fi
    local envgpu; envgpu="$(get_env_cuda "$p")"
    [[ "$envgpu" == "$GPU" ]] || continue
    results+=("$p")
  done
  printf '%s\n' "${results[@]}" | sort -u
}

# Collect wandb-service helper processes on given GPU
collect_wandb_service_for_gpu() {
  local GPU="$1"
  local -a results=()
  mapfile -t PIDS < <(pgrep -f "wandb-service\(" 2>/dev/null || true)
  for p in "${PIDS[@]:-}"; do
    [[ -n "$p" ]] || continue
    if ! kill -0 "$p" 2>/dev/null; then continue; fi
    local envgpu; envgpu="$(get_env_cuda "$p")"
    [[ "$envgpu" == "$GPU" ]] || continue
    results+=("$p")
  done
  printf '%s\n' "${results[@]}" | sort -u
}

IFS=',' read -ra GPUS <<< "$GPU_LIST"
TOTAL_KILLED_AGENTS=0
TOTAL_KILLED_RUNS=0

for GPU in "${GPUS[@]}"; do
  GPU="${GPU// /}"
  if [[ -z "$GPU" ]]; then continue; fi
  echo "--- Processing GPU $GPU ---"

  # Phase 1: kill agents first
  AGENTS=()
  mapfile -t AGENTS < <(collect_agent_pids_for_gpu "$GPU")
  if ((${#AGENTS[@]}==0)); then
    echo "No wandb agents found on GPU $GPU"
  else
    for ROOT in "${AGENTS[@]}"; do
      [[ -n "$ROOT" ]] || continue
      ENV_GPU="$(get_env_cuda "$ROOT")"
      [[ "$ENV_GPU" == "$GPU" ]] || { echo "Skip agent PID $ROOT (CUDA_VISIBLE_DEVICES=$ENV_GPU)"; continue; }
      kill_tree "$ROOT" "GPU $GPU wandb agent"
    done
  fi
  # Count agent survivors
  SURV_AG=0
  for p in "${AGENTS[@]}"; do
    if kill -0 "$p" 2>/dev/null; then ((SURV_AG++)); fi
  done
  AG_COUNT=${#AGENTS[@]}
  KILLED_AG=$(( AG_COUNT - SURV_AG ))
  echo "GPU $GPU: agents killed $KILLED_AG"
  ((TOTAL_KILLED_AGENTS+=KILLED_AG))

  # Brief pause to let children exit
  sleep 1

  # Phase 2: kill any remaining runs spawned by agents (WANDB_AGENT present)
  RUNS=()
  mapfile -t RUNS < <(collect_wandb_proc_for_gpu "$GPU")
  if ((${#RUNS[@]}==0)); then
    echo "No wandb runs found on GPU $GPU"
  else
    for ROOT in "${RUNS[@]}"; do
      [[ -n "$ROOT" ]] || continue
      ENV_GPU="$(get_env_cuda "$ROOT")"
      [[ "$ENV_GPU" == "$GPU" ]] || { echo "Skip run PID $ROOT (CUDA_VISIBLE_DEVICES=$ENV_GPU)"; continue; }
      kill_tree "$ROOT" "GPU $GPU wandb run"
    done
  fi
  # Count run survivors
  SURV_RN=0
  for p in "${RUNS[@]}"; do
    if kill -0 "$p" 2>/dev/null; then ((SURV_RN++)); fi
  done
  RN_COUNT=${#RUNS[@]}
  KILLED_RN=$(( RN_COUNT - SURV_RN ))
  echo "GPU $GPU: runs killed $KILLED_RN"
  ((TOTAL_KILLED_RUNS+=KILLED_RN))

  # Phase 3: kill any visible training roots (tdmpc2/train.py)
  TRAIN=(); mapfile -t TRAIN < <(collect_train_roots_for_gpu "$GPU")
  if ((${#TRAIN[@]})); then
    for ROOT in "${TRAIN[@]}"; do
      [[ -n "$ROOT" ]] || continue
      ENV_GPU="$(get_env_cuda "$ROOT")"; [[ "$ENV_GPU" == "$GPU" ]] || continue
      kill_tree "$ROOT" "GPU $GPU train root"
    done
  fi

  # Phase 4: kill any lingering wandb-service helpers
  WS=(); mapfile -t WS < <(collect_wandb_service_for_gpu "$GPU")
  if ((${#WS[@]})); then
    for ROOT in "${WS[@]}"; do
      [[ -n "$ROOT" ]] || continue
      ENV_GPU="$(get_env_cuda "$ROOT")"; [[ "$ENV_GPU" == "$GPU" ]] || continue
      kill_tree "$ROOT" "GPU $GPU wandb-service"
    done
  fi
done

echo "Total killed agents: $TOTAL_KILLED_AGENTS"
echo "Total killed runs: $TOTAL_KILLED_RUNS"
exit 0
