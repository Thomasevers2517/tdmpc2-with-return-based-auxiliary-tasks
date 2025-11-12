# How to Run TD-MPC2

Quick reference for launching training runs with Hydra overrides.

Environment setup (Conda)

```bash
# If the environment already exists
conda activate tdmpc2

# If you need to create it (one-time)
conda env create -f docker/environment.yaml -n tdmpc2
conda activate tdmpc2
```

Recommended GPU selection (4 GPUs available on this machine):

```bash
# Usually use GPU 1
export CUDA_VISIBLE_DEVICES=1
```

Basic training run (single-task):

```bash
CUDA_VISIBLE_DEVICES=1 python tdmpc2/train.py \
  task=quadruped-walk \
  steps=20000 \
  compile=true
```

Debug-friendly run (no compile):

```bash
CUDA_VISIBLE_DEVICES=1 python tdmpc2/train.py \
  task=quadruped-walk \
  steps=20000 \
  compile=false
```

Notes
- Hydra overrides come after the script (e.g., `task=... steps=... compile=...`).
- Use `eval_freq` to control evaluation cadence and `save_video=false` to avoid overhead.
- To run without MPC during training for ablations: `train_mpc=false` (evaluation still uses MPC if `eval_mpc=true`).
- Logs are under `logs/YYYYMMDD_HHMMSS/`. Set `wandb_*` keys in `config.yaml` to enable/disable Weights & Biases.
 - Always activate the `tdmpc2` conda environment before running scripts.
