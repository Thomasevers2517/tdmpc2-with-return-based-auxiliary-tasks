#!/bin/bash
###############################################################################
# Inspect SLURM jobs for the current user and summarize GPU activity.
#
# - Runs squeue to list active jobs (optionally filter by job name prefix)
# - Prints job metadata (JOBID, ARRAY_TASK, NAME, STATE, TIME, PARTITION, NODELIST)
while IFS='|' read -r JOBID STATE NAME TIME PARTITION NODELIST ARRAY_TASK; do || true
#
# Usage:
#   utils/slurm/inspect_jobs.sh [--name-prefix PREFIX] [--max-jobs N] [--hosts-per-job M]
#
# Notes:
# - Requires passwordless SSH access to the compute nodes allocated to the job.
# - GPU stats are fetched with: nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory
# - When no GPU data can be collected, the script reports the failure reason and continues.
###############################################################################
set -euo pipefail

NAME_PREFIX=""
MAX_JOBS=3
HOSTS_PER_JOB=1
ACTIVE_STATES=(RUNNING COMPLETING)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name-prefix) NAME_PREFIX="$2"; shift 2;;
    --max-jobs) MAX_JOBS="$2"; shift 2;;
    --hosts-per-job) HOSTS_PER_JOB="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--name-prefix PREFIX] [--max-jobs N] [--hosts-per-job M]"; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

if ! [[ "$MAX_JOBS" =~ ^[0-9]+$ ]] || [[ "$MAX_JOBS" -le 0 ]]; then
  echo "ERROR: --max-jobs must be a positive integer" >&2
  exit 2
fi

if ! [[ "$HOSTS_PER_JOB" =~ ^[0-9]+$ ]] || [[ "$HOSTS_PER_JOB" -le 0 ]]; then
  echo "ERROR: --hosts-per-job must be a positive integer" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if ! command -v squeue >/dev/null 2>&1; then
  echo "ERROR: squeue not found in PATH" >&2
  exit 2
fi

if ! command -v scontrol >/dev/null 2>&1; then
  echo "ERROR: scontrol not found in PATH" >&2
  exit 2
fi

SQUEUE_CMD=(squeue -u "$USER" -o "%i|%T|%j|%M|%P|%R|%a")
SQUEUE_OUT="$(${SQUEUE_CMD[@]} | tail -n +2 || true)"

if [[ -z "$SQUEUE_OUT" ]]; then
  echo "No jobs found for user $USER"
  exit 0
fi

echo "Scanning active jobs for $USER (name prefix filter: '${NAME_PREFIX}')"
echo

JOB_COUNTER=0
ACTIVE_FOUND=0

while IFS='|' read -r JOBID STATE NAME TIME PARTITION NODELIST ARRAY_TASK; do
  # Optional filter by job name prefix
  if [[ -n "$NAME_PREFIX" ]] && [[ "$NAME" != ${NAME_PREFIX}* ]]; then
    continue
  fi

  IS_ACTIVE=false
  for ACTIVE_STATE in "${ACTIVE_STATES[@]}"; do
    if [[ "$STATE" == "$ACTIVE_STATE" ]]; then
      IS_ACTIVE=true
      break
    fi
  done
  if [[ "$IS_ACTIVE" == false ]]; then
    continue
  fi

  ((ACTIVE_FOUND++))
  ((JOB_COUNTER++))
  if (( JOB_COUNTER > MAX_JOBS )); then
    break
  fi

  # Derive array index from JOBID suffix if present (e.g., 12345_7)
  ARRAY_IDX=""
  JOBID_BASE="$JOBID"
  if [[ "$JOBID" == *_* ]]; then
    JOBID_BASE="${JOBID%%_*}"
    MAYBE_IDX="${JOBID##*_}"
    if [[ "$MAYBE_IDX" =~ ^[0-9]+$ ]]; then
      ARRAY_IDX="$MAYBE_IDX"
    fi
  fi

  echo "JOB: ${JOBID_BASE}${ARRAY_IDX:+[$ARRAY_IDX]}  NAME: $NAME  STATE: $STATE  TIME: $TIME  PART: $PARTITION  NODE: $NODELIST"

  if [[ -z "$NODELIST" || "$NODELIST" == "(null)" ]]; then
    echo "  GPU: no nodes assigned yet"
    echo
    continue
  fi

  mapfile -t HOST_CANDIDATES < <(scontrol show hostnames "$NODELIST" 2>/dev/null)

  if (( ${#HOST_CANDIDATES[@]} == 0 )); then
    echo "  GPU: unable to resolve hostnames for ${NODELIST}"
    echo
    continue
  fi

  HOST_COUNT=0
  for HOST in "${HOST_CANDIDATES[@]}"; do
    ((HOST_COUNT++))
    if (( HOST_COUNT > HOSTS_PER_JOB )); then
      break
    fi

    echo "  GPU (${HOST}):"
    GPU_QUERY=(nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory --format=csv,noheader)
    if GPU_OUTPUT=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" "${GPU_QUERY[@]}" 2>&1); then
      if [[ -z "$GPU_OUTPUT" ]]; then
        echo "    No GPU data returned (empty response)"
      else
        while IFS= read -r LINE; do
          echo "    $LINE"
        done <<< "$GPU_OUTPUT"
      fi
    else
      echo "    Failed to query nvidia-smi: $GPU_OUTPUT"
    fi
  done

  if (( ${#HOST_CANDIDATES[@]} > HOSTS_PER_JOB )); then
    echo "  GPU: remaining hosts skipped (total=${#HOST_CANDIDATES[@]}, shown=${HOSTS_PER_JOB})"
  fi

  echo

done <<< "$SQUEUE_OUT" || true

if (( ACTIVE_FOUND == 0 )); then
  echo "No active jobs (states: ${ACTIVE_STATES[*]}) matched the given filters"
fi
