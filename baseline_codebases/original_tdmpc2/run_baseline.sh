#!/bin/bash
#SBATCH --account=tdsei8531
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --job-name=orig-tdmpc2-baseline
#SBATCH --array=0-5
#SBATCH --output=/projects/prjs0951/Thomas/Thesis/RL_weather/tdmpc2-with-return-based-auxiliary-tasks/slurm_logs/original_baseline/%x_%A_%a.out
#SBATCH --error=/projects/prjs0951/Thomas/Thesis/RL_weather/tdmpc2-with-return-based-auxiliary-tasks/slurm_logs/original_baseline/%x_%A_%a.err

# Seeds 1-6 mapped from array index 0-5
SEED=$((SLURM_ARRAY_TASK_ID + 1))

REPO_ROOT="/projects/prjs0951/Thomas/Thesis/RL_weather/tdmpc2-with-return-based-auxiliary-tasks"
ORIG_DIR="${REPO_ROOT}/original_tdmpc2/tdmpc2/tdmpc2"
WANDB_KEY_FILE="${REPO_ROOT}/wandbapikey/key.txt"

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bmpc

# Set W&B API key
if [[ -f "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE")
fi

cd "$ORIG_DIR"

echo "Running original TD-MPC2 baseline: task=quadruped-walk seed=${SEED}"
echo "Working directory: $(pwd)"

python train.py \
    task=quadruped-walk \
    model_size=5 \
    steps=100000 \
    eval_freq=10000 \
    seed=${SEED} \
    wandb_project=tdmpc2-tdmpc2 \
    wandb_entity=thomasevers9 \
    exp_name=original_baseline \
    save_video=false \
    compile=true
