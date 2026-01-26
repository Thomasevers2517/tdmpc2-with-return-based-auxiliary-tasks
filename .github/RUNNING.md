# How to Run TD-MPC2

Quick reference for launching training runs with Hydra overrides.

Environment setup (Conda)

```bash
# If the environment already exists
conda activate tdmpc2-new

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

---

## Running Sweeps on SLURM

### 1. Create the W&B Sweep

```bash
conda activate tdmpc2-new
python utils/create_sweep.py sweep_list/test/YOUR_SWEEP_NAME --wandb-project tdmpc2-tdmpc2
```

This reads `sweep.yaml` from the sweep folder, creates a W&B sweep, and writes the sweep ID to `id.txt` and project to `project.txt` in that folder.

**Note:** Always use `--wandb-project tdmpc2-tdmpc2` for consistency.

### 2. Submit to SLURM

```bash
bash utils/slurm/run_sweep.sh \
  --sweep-dir sweep_list/test/YOUR_SWEEP_NAME \
  --jobs N \
  --runs-per-job M \
  --time HH:MM:SS
```

**Parameters:**
- `--jobs N`: Number of SLURM jobs to submit. Each job launches **2 agents in parallel** (hardcoded in SLURM config), so `N` jobs = `2*N` parallel agents.
- `--runs-per-job M`: How many runs each agent executes **sequentially** before exiting. Default is 1.
- `--time HH:MM:SS`: Maximum walltime per job. **4 hours (`04:00:00`) is a good default** for state observations with typical UTD ratios.

**Example:** For a sweep with 54 total runs:
```bash
# 27 jobs × 2 agents/job = 54 parallel agents, each doing 1 run
bash utils/slurm/run_sweep.sh \
  --sweep-dir sweep_list/test/V_value6 \
  --jobs 27 \
  --time 04:00:00
```

**Calculating jobs needed:**
- Total runs = product of all `values` lists in sweep.yaml
- Jobs needed = `ceil(total_runs / 2)` (since each job spawns 2 agents)

---

## Notes
- Hydra overrides come after the script (e.g., `task=... steps=... compile=...`).
- Use `eval_freq` to control evaluation cadence and `save_video=false` to avoid overhead.
- To run without MPC during training for ablations: `train_mpc=false` (evaluation still uses MPC if `eval_mpc=true`).
- Logs are under `logs/YYYYMMDD_HHMMSS/`. Set `wandb_*` keys in `config.yaml` to enable/disable Weights & Biases.
- Always activate the `bmpc` conda environment before running scripts.
