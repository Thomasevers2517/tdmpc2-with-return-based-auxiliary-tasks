#!/bin/bash
###############################################################################
# Inspect SLURM jobs for the current user and show corresponding log files.
#
# - Runs squeue to list jobs (optionally filter by job name prefix)
# - Prints job metadata (JOBID, ARRAY_TASK, NAME, STATE, TIME, PARTITION, NODELIST)
# - Attempts to locate matching log files under slurm_logs/YYYYMMDD/HHMMSS
#   (and falls back to searching all subfolders under slurm_logs)
# - Shows the last N lines (tail) of each located log (.out and .err)
#
# Usage:
#   utils/slurm/inspect_jobs.sh [--name-prefix PREFIX] [--tail-lines N]
#
# Notes:
# - This script is read-only: it does not modify or delete logs.
# - It searches recursively within slurm_logs to handle dated/timestamped folders.
###############################################################################
set -euo pipefail

NAME_PREFIX=""
TAIL_LINES="50"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name-prefix) NAME_PREFIX="$2"; shift 2;;
    --tail-lines) TAIL_LINES="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--name-prefix PREFIX] [--tail-lines N]"; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_ROOT="${REPO_ROOT}/slurm_logs"

if ! command -v squeue >/dev/null 2>&1; then
  echo "ERROR: squeue not found in PATH" >&2
  exit 2
fi

SQUEUE_CMD=(squeue -u "$USER" -o "%i|%T|%j|%M|%P|%R|%a")
SQUEUE_OUT="$(${SQUEUE_CMD[@]} | tail -n +2 || true)"

if [[ -z "$SQUEUE_OUT" ]]; then
  echo "No jobs found for user $USER"
  exit 0
fi

echo "Found jobs for $USER (filtered by name prefix: '${NAME_PREFIX}')"
echo

while IFS='|' read -r JOBID STATE NAME TIME PARTITION NODELIST ARRAY_TASK; do
  # Optional filter by job name prefix
  if [[ -n "$NAME_PREFIX" ]] && [[ "$NAME" != ${NAME_PREFIX}* ]]; then
    continue
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

  # Build patterns to find logs. We search recursively under slurm_logs.
  # Filenames from run_sweep.sh: %x_%A_%a.out/.err (jobname_jobid_arrayidx)
  if [[ -n "$ARRAY_IDX" ]]; then
    OUT_GLOB="${LOG_ROOT}/**/${NAME}_${JOBID_BASE}_${ARRAY_IDX}.out"
    ERR_GLOB="${LOG_ROOT}/**/${NAME}_${JOBID_BASE}_${ARRAY_IDX}.err"
  else
    OUT_GLOB="${LOG_ROOT}/**/${NAME}_${JOBID_BASE}.out"
    ERR_GLOB="${LOG_ROOT}/**/${NAME}_${JOBID_BASE}.err"
  fi

  shopt -s nullglob globstar
  OUT_MATCH=( $OUT_GLOB )
  ERR_MATCH=( $ERR_GLOB )
  shopt -u globstar

  if (( ${#OUT_MATCH[@]} == 0 && ${#ERR_MATCH[@]} == 0 )); then
    echo "  Logs: not found under ${LOG_ROOT}"
    echo
    continue
  fi

  for f in "${OUT_MATCH[@]}" "${ERR_MATCH[@]}"; do
    [[ -e "$f" ]] || continue
    echo "  Log: $f"
    echo "  --- tail -n ${TAIL_LINES} ${f} ---"
    tail -n "$TAIL_LINES" "$f" || true
    echo
  done
done <<< "$SQUEUE_OUT"
