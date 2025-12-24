#!/bin/bash
###############################################################################
# Submit a SLURM job array that runs Weights & Biases sweep agents.
#
# Each array task launches a wandb agent for the provided sweep and runs
# `--count <runs-per-job>` runs before exiting.
#
# Requirements:
# - Sweep folder contains an id.txt with the full sweep id (entity/project/sweepid)
# - Repo root has wandbapikey/key.txt (gitignored) containing the W&B API key
# - The tdmpc2 conda env must be available on the cluster
#
# Example:

  # utils/slurm/run_sweep.sh \
  #   --sweep-dir sweep_list/midterm_sweep/1aux_value/rgb \
  #   --jobs 8 \
  #   --runs-per-job 4 \
  #   --time 02:00:00 \

###############################################################################
set -euo pipefail

usage() {
  cat <<'END_USAGE'
Usage: run_sweep.sh \
  --sweep-dir PATH           # path to sweep folder containing id.txt (required)\
  --jobs N                   # number of parallel SLURM jobs (default: 1)\
  --runs-per-job N           # number of runs executed by each agent (default: 1)\
  --time HH:MM:SS            # walltime per job (default: 01:00:00)\
  --partition NAME           # SLURM partition, e.g. gpu_a100 (default: gpu_a100)\
  --gpus N                   # GPUs per job (default: 1)\
  --cpus N                   # CPUs per task (default: 8)\
  --mem MEM                  # memory per node, e.g. 8G (default: 8G)\
  --account NAME             # SLURM account/project (default: tdsei8531)\
  --job-name NAME            # job name prefix (default: tdmpc2-sweep)\
  [--max-parallel N]         # optional throttle for array concurrency\
  [--mail-user EMAIL]        # optional email for notifications
  [--wandb-entity NAME]      # W&B entity (default: thomasevers9); project inferred from project.txt when needed
END_USAGE
}

SWEEP_DIR=""
JOBS="1"
RUNS_PER_JOB="1"
TIME_LIMIT="01:00:00"
PARTITION="gpu_a100"
GPUS="1"
CPUS="9"
MEM="80G"
ACCOUNT="tdsei8531"
JOB_NAME="tdmpc2-sweep-weather"
CONDA_ENV="/projects/0/prjs0951/conda_envs/tdmpc2-new"
MAX_PARALLEL=""
MAIL_USER="t.evers-2@student.tudelft.nl"
WANDB_ENTITY="thomasevers9"
WANDB_PROJECT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep-dir) SWEEP_DIR="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --runs-per-job) RUNS_PER_JOB="$2"; shift 2;;
    --time) TIME_LIMIT="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --account) ACCOUNT="$2"; shift 2;;
    --job-name) JOB_NAME="$2"; shift 2;;
    --conda-env) CONDA_ENV="$2"; shift 2;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2;;
    --mail-user) MAIL_USER="$2"; shift 2;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done




# Validate required args
[[ -n "$SWEEP_DIR" ]] || { echo "--sweep-dir is required" >&2; usage; exit 2; }


# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SWEEP_DIR_ABS="$(cd "$SWEEP_DIR" && pwd)"
ID_FILE="${SWEEP_DIR_ABS}/id.txt"
PROJECT_FILE="${SWEEP_DIR_ABS}/project.txt"
KEY_FILE="${REPO_ROOT}/wandbapikey/key.txt"
DATE_DIR="$(date +%Y%m%d)"
TIME_DIR="$(date +%H%M%S)"
OUT_DIR="${REPO_ROOT}/slurm_logs/${DATE_DIR}/${TIME_DIR}"
mkdir -p "$OUT_DIR"

# # Make a job-local scratch dir on a fast FS
# export TMPDIR=/scratch-local/$USER/$DATE_DIR/$TIME_DIR  # or $SNIC_TMP, $WRKDIR, etc. depending on cluster
# mkdir -p $TMPDIR/{inductor,triton,cuda}

# # Point all the JIT/compile caches there
# export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
# export TRITON_CACHE_DIR=$TMPDIR/triton
# export CUDA_CACHE_PATH=$TMPDIR/cuda

# Validate inputs
if [[ ! -s "$ID_FILE" ]]; then
  echo "Missing or empty sweep id file: $ID_FILE" >&2
  exit 2
fi
if [[ ! -s "$KEY_FILE" ]]; then
  echo "Missing or empty W&B API key file: $KEY_FILE" >&2
  exit 2
fi

SWEEP_ID="$(<"$ID_FILE")"
SWEEP_ID="${SWEEP_ID%%[$'\r\n']*}"

if [[ -z "$SWEEP_ID" ]]; then
  echo "Sweep id read from $ID_FILE is empty" >&2
  exit 2
fi

# Ensure sweep id is fully-qualified (entity/project/sweepid). If not, read project from project.txt.
if [[ "$SWEEP_ID" != */*/* ]]; then
  if [[ ! -s "$PROJECT_FILE" ]]; then
    echo "Sweep id '$SWEEP_ID' is not fully-qualified (expected entity/project/sweepid), and project.txt is missing: $PROJECT_FILE" >&2
    echo "Run utils/create_sweep.py for this sweep folder or create project.txt with the correct W&B project name." >&2
    exit 2
  fi
  WANDB_PROJECT="$(<"$PROJECT_FILE")"
  WANDB_PROJECT="${WANDB_PROJECT%%[$'\r\n']*}"
  if [[ -z "$WANDB_PROJECT" ]]; then
    echo "project.txt is empty: $PROJECT_FILE" >&2
    exit 2
  fi
  SWEEP_ID="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
fi

ARRAY_RANGE="0-$((JOBS-1))"
if [[ -n "$MAX_PARALLEL" ]]; then
  ARRAY_RANGE="${ARRAY_RANGE}%${MAX_PARALLEL}"
fi

echo "Submitting sweep agents:"
echo "  Sweep dir     : $SWEEP_DIR_ABS"
echo "  Sweep id      : $SWEEP_ID"
echo "  Jobs (array)  : $JOBS (range $ARRAY_RANGE)"
echo "  Runs per job  : $RUNS_PER_JOB"
echo "  Partition     : $PARTITION"
echo "  GPUs / CPUs   : $GPUS / $CPUS"
echo "  Mem           : $MEM"
echo "  Time          : $TIME_LIMIT"
echo "  Account       : $ACCOUNT"
echo "  Job name      : $JOB_NAME"
echo "  Conda env     : $CONDA_ENV"

# Build optional mail args
MAIL_ARGS=()
if [[ -n "$MAIL_USER" ]]; then
  MAIL_ARGS+=("--mail-user" "$MAIL_USER" "--mail-type" "END,FAIL,BEGIN")
fi

# Create a temporary batch script
BATCH_SCRIPT="$(mktemp)"
cat > "$BATCH_SCRIPT" <<'BATCH'
#!/bin/bash
set -eo pipefail

echo "== Job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?} on ${SLURM_NODELIST:-?} =="
echo "Partition: ${SLURM_JOB_PARTITION:-?} | GPUs: ${SLURM_GPUS:-?}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-?}"
echo "TMPDIR: ${TMPDIR:-/tmp}"

# module purge
module load 2024
module load CUDA/12.6.0

echo "Modules loaded"

# Activate conda env (repo requires tdmpc2 env)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "conda command not found in PATH; ensure the appropriate conda module is loaded" >&2
  exit 2
fi
conda activate "${CONDA_ENV}"

echo "Conda environment activated."
python -V
which python

# Ensure user site-packages do not shadow the conda env (prevents ~/.local installs from overriding)
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl


# Helpful diagnostics: print TorchRL / TensorDict versions actually in use
python - <<'PYVERS'
try:
  import torchrl, tensordict
  print(f"TorchRL version: {getattr(torchrl, '__version__', 'unknown')}")
  print(f"TensorDict version: {getattr(tensordict, '__version__', 'unknown')}")
except Exception as e:
  print(f"Version check failed: {e}")
PYVERS


mkdir -p "${TMPDIR:-/tmp}/wandb"
export WANDB_DIR="${TMPDIR:-/tmp}/wandb"

# Read and login to W&B
if [[ ! -s "${KEY_FILE}" ]]; then
  echo "Missing or empty W&B API key file inside job: ${KEY_FILE}" >&2
  exit 2
fi
export WANDB_API_KEY="$(<"${KEY_FILE}")"
export WANDB_MODE="online"
export WANDB_ENTITY="${WANDB_ENTITY:-?}"
if [[ -n "${WANDB_PROJECT:-}" ]]; then export WANDB_PROJECT; fi
wandb login --relogin "${WANDB_API_KEY}" >/dev/null

echo "Starting wandb agent: ${SWEEP_ID} (count=${RUNS_PER_JOB})"
srun wandb agent --count "${RUNS_PER_JOB}" "${SWEEP_ID}"

echo "Agent finished."
BATCH

chmod +x "$BATCH_SCRIPT"

# Submit the array job
CMD=( sbatch
  -A "$ACCOUNT"
  -p "$PARTITION"
  -t "$TIME_LIMIT"
  --gpus="$GPUS"
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  -J "$JOB_NAME"
  -o "$OUT_DIR/%x_%A_%a.out"
  -e "$OUT_DIR/%x_%A_%a.err"
  --array "$ARRAY_RANGE"
  --export "ALL,SWEEP_ID=$SWEEP_ID,RUNS_PER_JOB=$RUNS_PER_JOB,KEY_FILE=$KEY_FILE,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,CONDA_ENV=$CONDA_ENV"
  "${MAIL_ARGS[@]:-}"
  "$BATCH_SCRIPT"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Submitted. Batch script: $BATCH_SCRIPT"
