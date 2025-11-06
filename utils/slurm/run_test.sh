#!/bin/bash
# Submit a single TD-MPC2 training run to SLURM with basic A100 resources.
# Logs go to slurm_logs/test_YYYYMMDD. The tdmpc2 conda env must exist at the cluster path.

set -euo pipefail

# Fixed basic resources (fail if the cluster rejects these)
ACCOUNT="tdsei8531"
PARTITION="gpu_a100"
TIME_LIMIT="00:15:00"
GPUS="1"
CPUS="9"
MEM="8G"
JOB_NAME="tdmpc2-test"

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Prepare log directory: slurm_logs/test_YYYYMMDD
DATE_DIR="$(date +%Y%m%d)"
TIME_DIR="$(date +%H%M%S)"
OUT_DIR="${REPO_ROOT}/slurm_logs/test_${DATE_DIR}/${TIME_DIR}"
mkdir -p "${OUT_DIR}"

# Create a temporary batch script
BATCH_SCRIPT="$(mktemp)"
cat > "${BATCH_SCRIPT}" <<'BATCH'
#!/bin/bash
set -eo pipefail

echo "== Job ${SLURM_JOB_ID:-?} on ${SLURM_NODELIST:-?} =="
echo "Partition: ${SLURM_JOB_PARTITION:-?} | GPUs: ${SLURM_GPUS:-?} | CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-?}"

# module purge || true
# module list
module load 2024 || true
module load CUDA/12.6.0 || true
module list

# Activate conda env (repo requires tdmpc2 env at fixed path)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "conda command not found in PATH; ensure the appropriate conda module is loaded" >&2
  exit 2
fi
ENV_PATH="/projects/0/prjs0951/conda_envs/tdmpc2"
if [[ ! -d "${ENV_PATH}" ]]; then
  echo "Conda environment path not found: ${ENV_PATH}" >&2
  exit 2
fi
conda activate "${ENV_PATH}"

echo "Conda environment activated."
python -V
which python

# Ensure user site-packages do not shadow the conda env
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl

set -x
srun python tdmpc2/train.py compile=true "$@"
BATCH

chmod +x "${BATCH_SCRIPT}"

# Submit single job
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

echo "Submitted. Logs: ${OUT_DIR}"
