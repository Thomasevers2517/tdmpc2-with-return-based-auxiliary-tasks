#!/bin/bash
###############################################################################
# Snellius MIG job: run python tdmpc2/train.py on an A100 MIG slice
# Submit: sbatch slurm/train_tdmpc2_mig.sbatch
###############################################################################

########################## SLURM DIRECTIVES (edit) ############################
#SBATCH -A tdsei8531                # project
#SBATCH -J weather_tdmpc2           # job name
#SBATCH -p gpu_a100                # MIG-enabled partition
#SBATCH -N 1                        # nodes
#SBATCH --ntasks=1                  # tasks
#SBATCH --cpus-per-task=5           # CPU cores per MIG slice (typical)
#SBATCH --gpus=1                    # number of MIG slices
#SBATCH --mem=4G                   # RAM per node
#SBATCH -t 00:09:15                 # walltime
#SBATCH -o slurm_logs/%x_%j.out     # stdout
#SBATCH -e slurm_logs/%x_%j.err     # stderr
# Optional mail:
#SBATCH --mail-user=T.Evers-2@student.tudelft.nl
#SBATCH --mail-type=END,FAIL,BEGIN
###############################################################################

set -eo pipefail

export WANDB_API_KEY=3cfa8ad4071af79aa5f7a00bb091ba6b46ac71e1      # from https://wandb.ai/authorize
export WANDB_ENTITY=thomasevers9              # your W&B username or team
export WANDB_PROJECT=snellius-test            # exact project name you own
# optional:
export WANDB_MODE=online                      # or "offline" to disable network
wandb login $WANDB_API_KEY
############################### USER KNOBS ####################################
# Repo paths
PROJECT_DIR="$PWD"                  # repo root
PY_ENTRY="tdmpc2/train.py"          # training entry
PY_ARGS="compile=true num_rollouts=16"                          # e.g. "--env taskX --seed 0"

# Staging (optional)
STAGE_INPUT_FROM=""                 # e.g. "$PROJECT_DIR/datasets/mydata"
STAGE_INPUT_TO_SUBDIR="data"        # becomes $TMPDIR/data
COPY_BACK_SUBDIRS=("outputs" "checkpoints" "wandb")

# Output folders in repo
OUT_DIR="$PROJECT_DIR/slurm_logs"
RUN_DIR="$PROJECT_DIR/results"
mkdir -p "$OUT_DIR" "$RUN_DIR"

# Runtime env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="$TMPDIR/wandb"    # keep W&B I/O on local scratch
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=INFO
###############################################################################

echo "== Job ${SLURM_JOB_ID} on ${SLURM_NODELIST} =="
echo "Partition: ${SLURM_JOB_PARTITION} | GPUs: ${SLURM_GPUS:-?}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "TMPDIR: ${TMPDIR}"

############################## ENV / MODULES ##################################
module purge
module load 2025
# module load CUDA/12.8.0        # match torch.version.cuda
conda init

echo "Modules loaded:"
source activate tdmpc2

conda activate /projects/0/prjs0951/conda_envs/tdmpc2
echo "Conda environment activated."

# Activate your conda env
python -V
which python

############################### STAGING #######################################
mkdir -p "$TMPDIR" "$WANDB_DIR"
if [[ -n "${STAGE_INPUT_FROM}" ]]; then
  echo "Staging input ${STAGE_INPUT_FROM} -> ${TMPDIR}/${STAGE_INPUT_TO_SUBDIR}"
  rsync -a --info=progress2 "${STAGE_INPUT_FROM}/" "${TMPDIR}/${STAGE_INPUT_TO_SUBDIR}/"
fi
for d in "${COPY_BACK_SUBDIRS[@]}"; do mkdir -p "${TMPDIR}/${d}"; done

################################# RUN #########################################
cd "${PROJECT_DIR}"
echo "Launching: python ${PY_ENTRY} ${PY_ARGS}"
srun python "${PY_ENTRY}" ${PY_ARGS}

############################### COPY BACK #####################################
echo "Copying results to ${RUN_DIR}"
for d in "${COPY_BACK_SUBDIRS[@]}"; do
  [[ -d "${TMPDIR}/${d}" ]] && rsync -a "${TMPDIR}/${d}/" "${RUN_DIR}/${d}/"
done

echo "Done. Job ${SLURM_JOB_ID}"
