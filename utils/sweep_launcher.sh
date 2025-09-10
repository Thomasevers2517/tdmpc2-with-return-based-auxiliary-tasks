#!/bin/bash
# Usage: bash utils/sweep_launcher.sh <sweep_yaml> <gpu_list>
# Example: bash utils/sweep_launcher.sh wandb_sweeps/one_million_task_sweep.yaml 2,3

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <sweep_yaml> <gpu_list (comma-separated)>"
  exit 1
fi

SWEEP_YAML_INPUT="$1"
GPU_LIST="$2"

# Resolve repo root (script_dir/..), ensure we run from repo root
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# Resolve absolute path to YAML (assume relative to repo root if not absolute)
if [[ "$SWEEP_YAML_INPUT" = /* ]]; then
  SWEEP_YAML="$SWEEP_YAML_INPUT"
else
  SWEEP_YAML="$REPO_ROOT/$SWEEP_YAML_INPUT"
fi
if [ ! -f "$SWEEP_YAML" ]; then
  echo "Sweep file not found: $SWEEP_YAML" >&2
  exit 1
fi

# Get sweep name (basename without extension)
SWEEP_NAME=$(basename "$SWEEP_YAML" .yaml)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_ROOT/outputs/${SWEEP_NAME}/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Activate conda env (disable nounset during activation to avoid env script errors)
set +u
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate tdmpc2
set -u
echo "Python: $(python -V)" | tee "$LOG_DIR/env.txt"
echo "W&B: $(python -m wandb --version || echo not-found)" | tee -a "$LOG_DIR/env.txt"

# Create sweep and extract sweep id (use python -m wandb for reliability)
echo "Creating sweep from $SWEEP_YAML..." | tee "$LOG_DIR/sweep_create.log"
# Ensure W&B entity/project defaults (override with env if set)
export WANDB_ENTITY=${WANDB_ENTITY:-thomasevers9}
export WANDB_PROJECT=${WANDB_PROJECT:-tdmpc2-tdmpc2}
python -m wandb sweep "$SWEEP_YAML" 2>&1 | tee -a "$LOG_DIR/sweep_create.log"

# Parse sweep id from output (look for agent hint or sweeps/<id>)
SWEEP_LINE=$(grep -Eo "wandb agent [^ ]+/[^ ]+/[a-z0-9]+" "$LOG_DIR/sweep_create.log" | tail -n1 || true)
if [ -z "$SWEEP_LINE" ]; then
  SWEEP_LINE=$(grep -Eo "sweeps/[a-z0-9]+" "$LOG_DIR/sweep_create.log" | tail -n1 | sed -E "s#sweeps/#wandb agent thomasevers9/tdmpc2-tdmpc2/#" || true)
fi
if [ -z "$SWEEP_LINE" ]; then
  echo "Failed to extract sweep id from create output." | tee -a "$LOG_DIR/sweep_create.log"
  exit 2
fi
echo "Found: $SWEEP_LINE" | tee -a "$LOG_DIR/sweep_create.log"
SWEEP_ID=$(echo "$SWEEP_LINE" | awk -F'/' '{print $3}')
echo "$SWEEP_ID" > "$LOG_DIR/sweep_id.txt"

# Launch agents for each GPU (nohup, ensure env sourced in subshell)
IFS=',' read -ra GPUS <<< "$GPU_LIST"
for GPU in "${GPUS[@]}"; do
  LOG_FILE="$LOG_DIR/output_gpu${GPU}.log"
  PID_FILE="$LOG_DIR/agent_gpu${GPU}.pid"
  echo "Launching agent on GPU $GPU, logging to $LOG_FILE" | tee -a "$LOG_DIR/sweep_create.log"
  nohup bash -lc "set +u; source \"$(conda info --base)\"/etc/profile.d/conda.sh; conda activate tdmpc2; set -u; cd \"$REPO_ROOT\"; export CUDA_VISIBLE_DEVICES=$GPU WANDB_ENTITY=\"$WANDB_ENTITY\" WANDB_PROJECT=\"$WANDB_PROJECT\"; python -m wandb agent $WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID" > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "PID $(cat \"$PID_FILE\")" | tee -a "$LOG_DIR/sweep_create.log"
done

echo "All agents launched. Logs in $LOG_DIR."
