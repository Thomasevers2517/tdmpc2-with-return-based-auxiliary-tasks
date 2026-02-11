# Copilot Instructions (Concise)
- add any files or temporary tests to /"copilot files"
- Descriptive naming: Explicit, self-explanatory names for variables, tensors, functions, classes, and files. Prefer `verb_object_detail` (e.g., `compute_episode_returns`). Avoid unclear abbreviations.
- Minimal & modular: Implement only what’s needed. Inline is fine for true single-use; on second use, refactor into a shared utility and replace call sites.
- Tensor shapes everywhere: Annotate shapes at introduction or transformation.
	- Inline: `# obs: float32[B, T, C, H, W]`
	- Docstrings: `Tensor[B, T, ...]` in Args/Returns.
- Stable shapes: Minimize `reshape/view/permute`; keep dimension order consistent. If reshaping is necessary, add a one-line rationale.
- Docstrings for all callables: Summary + Args (name, brief description, dtype, shape) + Returns (brief description + shape) + Raises (if any). Keep crisp. Shapes are mandatory immediately; docstrings may be deferred during exploration but are required before considering code complete.
- Comments explain why: Clarify intent, math, and non-obvious design choices—not what the code literally does.
- Ask before assuming: For any non-trivial design choice, pause and ask. Propose a minimal API plan (classes, functions, signatures, interactions) before coding and wait for confirmation.
- Retroactive modularization: When logic appears a second time, extract a minimal shared utility/module and replace both occurrences.
- Types: Add Python type hints throughout. Keep dimension names consistent within a module; track current conventions rather than hard-coding global standards.
- Validation: Add cheap `assert`/shape checks at boundaries and critical transforms; especially at the start of larger methods.
- Logging: Use the project logger. Prefer passing structured info dictionaries to specialized logger methods that can aggregate/analyze values. Avoid duplicating the same data in multiple variables—consume directly from dictionaries when used once. Avoid logging in high-frequency/hot code paths unless configured by logging frequency; ask before adding temporary logs for debugging and remove them after.
- Data structures: Prefer lightweight classes (e.g., dataclasses) for complex dict-like structures to make interactions explicit and type-safe. Store tensors directly; avoid duplicating values under new names; consume from the source dict/struct when used once.
- Performance & batching: Prioritize vectorized, batch-first ops. Pack values that are computed together into a single tensor; avoid many small tensors and Python loops. Minimize host↔device transfers and needless allocations; keep ops fuse-friendly for kernels/compilers.
- Compilability: Keep critical compute paths compatible with Torch compilation (e.g., `torch.compile`/`torch.jit`)—avoid unsupported ops, excessive Python-side effects, or dynamic control flow that breaks compilation.
- Contiguity: After any `permute`, call `.contiguous()` if needed; prefer avoiding permutations altogether when practical.
- Consistency checks: If you see missing tensor-shape comments or obvious duplication, flag it and propose a small fix or modularization.
- Constants: Use UPPERCASE for static configuration; avoid magic numbers inline.
- Tests/profiling: Skip by default unless requested. Shape-based micro-examples are optional for clarity, not required.

Workflow
- Propose-first: Share a short plan with class/function list, signatures, and interactions; ask clarifying questions.
- Large edits (>200 LOC): Write a plan describing the main goal, key changes, affected modules/files, expected ripple effects, and the sequence of edits. Apply changes one file at a time.
- Implement minimal pieces; refactor on reuse; iterate quickly while keeping shapes and names stable.

Defaults (confirm/override)
- Library: PyTorch.
- Docstring style: Google.
- Shape notation: Inline `float32[B, T, ...]`; docstrings `Tensor[B, T, ...]`.

Fail-Fast Expectations
- No silent defaults: Do not use `dict.get`/`getattr` with fallback values for required config or keys; access directly and let errors surface or raise a clear exception.
- No hidden backups: Avoid implicit backup code paths that change behavior when inputs/config are missing. Prefer explicit checks and fail fast.
- Exceptions policy: Do not swallow exceptions in `try/except`; only catch to add context and re-raise. Prefer context managers over `try/finally` for cleanup; if `finally` is required, always re-raise the original error.
- Required parameters: Avoid default argument values for parameters that are required for correctness; make them explicit and validate at boundaries.
 - Config access: Access required config flags/attrs directly (e.g., `cfg.fixed_value`). Do not use `getattr`/`dict.get` with fallbacks. Let missing keys raise immediately. Avoid hidden defaults.

Example style
```python
def compute_td_lambda_returns(
		rewards: torch.Tensor,          # float32[B, T]
		discounts: torch.Tensor,        # float32[B, T]
		values_bootstrap: torch.Tensor  # float32[B]
) -> torch.Tensor:                  # float32[B, T]
		"""Compute TD(λ) returns.

		Args:
			rewards (Tensor[B, T]): Per-timestep rewards to accumulate.
			discounts (Tensor[B, T]): Per-timestep discount factors in [0,1].
			values_bootstrap (Tensor[B]): Value estimate used to bootstrap the final step.
		Returns:
			Tensor[B, T]: Discounted returns per timestep.
		"""
		# Minimal, shape-stable implementation goes here.
		raise NotImplementedError  # Implement after API confirmation

Debugging / Understanding
- Temporary logs: Allowed when investigating behavior; prefer logging stats (shape/mean/std) over full tensors. Remove after.
- Shape focus: Ensure every tensor has an inline shape comment; verify shapes at boundaries with cheap asserts.
- Run notes: Prefer GPU selection via environment and run the trainer directly.
	- Activate environment first: `conda activate tdmpc2` (create with `conda env create -f docker/environment.yaml -n tdmpc2` if missing)
	- Quick commands:
		- Typical run (compile on): `CUDA_VISIBLE_DEVICES=1 python tdmpc2/train.py task=quadruped-walk steps=20000 compile=true`
		- Debug run (no compile): `CUDA_VISIBLE_DEVICES=1 python tdmpc2/train.py task=quadruped-walk steps=20000 compile=false`
		- Override examples: add Hydra overrides after the script, e.g. `eval_freq=5000 train_mpc=true save_video=false`
- Inspection loop: Run, inspect logs/metrics, adjust, and repeat. Keep hot paths clean of logging unless configured by frequency.
```

When uncertain: ask, propose, then implement.

Run experiments by creating a sweep and running it on slurm like so:

## Running Sweeps on SLURM

### 1. Create the W&B Sweep

```bash
conda activate bmpc
python utils/create_sweep.py sweep_list/test/YOUR_SWEEP_NAME --wandb-project tdmpc2-tdmpc2
```

This reads `sweep.yaml` from the sweep folder, creates a W&B sweep, and writes the sweep ID to `id.txt` and project to `project.txt` in that folder.

**Note:** Always use `--wandb-project bmpc` for consistency.

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
