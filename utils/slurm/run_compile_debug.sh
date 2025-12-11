#!/bin/bash
# Submit a TD-MPC2 training run with verbose torch.compile logging for debugging.
# Logs go to slurm_logs/compile_debug_YYYYMMDD.

set -euo pipefail

# Fixed basic resources
ACCOUNT="tdsei8531"
PARTITION="gpu_a100"
TIME_LIMIT="00:30:00"
GPUS="1"
CPUS="9"
MEM="8G"
JOB_NAME="compile-debug"

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Prepare log directory
DATE_DIR="$(date +%Y%m%d)"
TIME_DIR="$(date +%H%M%S)"
OUT_DIR="${REPO_ROOT}/slurm_logs/compile_debug_${DATE_DIR}/${TIME_DIR}"
mkdir -p "${OUT_DIR}"

# Create a temporary batch script
BATCH_SCRIPT="$(mktemp)"
cat > "${BATCH_SCRIPT}" <<'BATCH'
#!/bin/bash
set -eo pipefail

echo "== Compile Debug Job ${SLURM_JOB_ID:-?} on ${SLURM_NODELIST:-?} =="
echo "Partition: ${SLURM_JOB_PARTITION:-?} | GPUs: ${SLURM_GPUS:-?} | CPUs: ${SLURM_CPUS_PER_TASK:-?}"

module load 2024 || true
module load CUDA/12.6.0 || true

# Activate conda env
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "conda command not found in PATH" >&2
  exit 2
fi
ENV_PATH="/projects/0/prjs0951/conda_envs/tdmpc2"
conda activate "${ENV_PATH}"

echo "Python: $(python -V)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# Ensure clean environment
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl

# ====== VERBOSE TORCH COMPILE LOGGING ======
# TORCH_COMPILE_DEBUG creates a debug directory with full traces
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="${SLURM_SUBMIT_DIR}/compile_debug_artifacts_${SLURM_JOB_ID}"
# Maximum verbosity for dynamo/inductor logging including cudagraphs
export TORCH_LOGS="+dynamo,+inductor,+aot,recompiles,graph_breaks,guards,cudagraphs,cudagraph_static_inputs,output_code"
export TORCHDYNAMO_VERBOSE=1
export TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL=1
export TORCHDYNAMO_EXTENDED_DEBUG_CPP=1
# Show guard failures and symbolic shape issues
# export TORCH_LOGS_FORMAT="short"
export TORCHINDUCTOR_TRACE=1
# Disable cudagraphs to simplify debugging
# export TORCHINDUCTOR_CUDAGRAPHS=0

# Triton debugging
export TRITON_DEBUG=1
export TL_LOG_LEVEL=DEBUG
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR="${SLURM_SUBMIT_DIR}/.triton_cache_${SLURM_JOB_ID}"

echo "========================================"
echo "TORCH_COMPILE_DEBUG=1"
echo "TORCH_COMPILE_DEBUG_DIR=${TORCH_COMPILE_DEBUG_DIR}"
echo "TORCH_LOGS=${TORCH_LOGS}"
echo "========================================"

# Run with unbuffered output (-u) and redirect both streams
set -x
srun stdbuf -oL -eL python -u tdmpc2/train.py \
  task=quadruped-walk \
  steps=10000 \
  compile=true \
  compile_type=reduce-overhead \
  planner_head_reduce=max \
  planner_head_reduce_eval=min \
  eval_freq=10000 \
  save_video=false \
  enable_wandb=false \
  "$@"
BATCH

chmod +x "${BATCH_SCRIPT}"

# Submit job
CMD=( sbatch
  -A "${ACCOUNT}"
  -p "${PARTITION}"
  -t "${TIME_LIMIT}"
  --gpus="${GPUS}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  -J "${JOB_NAME}"
  -o "${OUT_DIR}/%x_%j.out"
  -e "${OUT_DIR}/%x_%j.err"
  "${BATCH_SCRIPT}"
  -- "$@"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Submitted compile debug job. Logs: ${OUT_DIR}"
echo "Watch with: tail -f ${OUT_DIR}/*.err"
