# TD-MPC2 Analysis System

This document describes the architecture and conventions for analyzing experiment results and generating publication-quality figures and tables.

## Publication Target

**ICML 2026** — Double-column format on A4 paper.

Figures will be approximately half the width of an A4 page when printed, so:
- Use large fonts (title: 24pt, axis labels: 22pt, ticks: 20pt, legend: 18pt)
- Use thick lines (3.5pt width)
- Ensure text remains readable at reduced size
- Test figures by viewing them at ~50% zoom

## Overview

The analysis system is designed for:
1. **Fetching experiment data** from Weights & Biases (W&B) sweeps
2. **Processing and aggregating** results across seeds and configurations
3. **Generating publication-quality figures** for scientific papers
4. **Comparing against baselines** (TD-MPC2, DreamerV3, SAC, etc.)

## Folder Structure

```
analysis/
├── ANALYSIS_SYSTEM.md       # This documentation
├── EXPERIMENT_PLANS.md      # Notes on planned experiments
├── Understanding_WANDB.md   # W&B-specific documentation
├── notebooks/               # Analysis notebooks (numbered)
│   ├── 01_encoder_detach_consistency.ipynb
│   ├── 02_final_imagination_sweep.ipynb
│   ├── ...
│   └── 06_seminar/          # Subfolder with own numbering
│       ├── 01_multitask_comparison.ipynb
│       └── ...
├── results/                 # Generated outputs (numbered to match notebooks)
│   └── 06_seminar/
│       ├── 01_multitask_comparison/
│       └── ...
├── run_cache/               # Cached W&B data (auto-generated)
└── tools/                   # Reusable Python modules
    ├── __init__.py
    ├── aggregations.py      # DataFrame transformations
    ├── baselines.py         # Load baseline CSV results
    ├── cache.py             # On-disk JSON caching
    ├── config.py            # W&B entity/project defaults
    ├── consistency_analysis.py  # Consistency loss analysis
    ├── encodings.py         # Visual encoding utilities
    ├── filters.py           # DataFrame filtering
    ├── naming.py            # Task name conversions
    ├── paths.py             # Centralized path management
    ├── plotting.py          # Plotly-based visualization
    ├── selection.py         # Hyperparameter analysis
    ├── serialization.py     # JSON-safe conversions
    ├── style.py             # Publication styling constants
    └── wandb_io.py          # W&B API access with caching
```

## Notebook Conventions

### Numbering Scheme
- Main notebooks: `01_`, `02_`, `03_`, ...
- Subfolders: Folder gets a number (e.g., `06_seminar/`), inner notebooks restart at `01_`
- New notebooks continue the sequence (next main notebook would be `07_...`)

### Standard Notebook Structure

```python
# Cell 1: Setup and Imports
from analysis.tools import (
    wandb_io, aggregations, baselines, plotting, selection, paths, naming, config
)
from analysis.tools.style import STYLE  # Publication styling

NOTEBOOK_STEM = "07_my_analysis"  # Used for results directory
RESULTS_DIR = paths.notebook_results_dir(NOTEBOOK_STEM)

# Cell 2: Configuration
SWEEP_ID = "abc123xyz"
HISTORY_KEYS = ["eval/episode_reward", "_step"]
TASKS = ["quadruped-walk", "hopper-hop", ...]

# Cell 3: Fetch W&B Data
runs, manifest, source = wandb_io.fetch_sweep_runs(
    entity=config.WANDB_ENTITY,
    project=config.WANDB_PROJECT,
    sweep_id=SWEEP_ID,
    history_keys=HISTORY_KEYS,
    use_cache=True,
    force_refresh=False,  # Set True to bypass cache
)

# Cell 4: Process Data
runs_df = aggregations.runs_history_to_frame(runs, step_key="_step")
runs_df["task_baseline"] = runs_df["task"].map(naming.wandb_task_to_baseline)

# Cell 5: Load Baselines
baseline_df = baselines.load_many(TASKS, root=paths.BASELINE_TDMPC2)

# Cell 6: Generate Figures
fig = plotting.sample_efficiency_figure(...)
plotting.write_png(fig, output_path=RESULTS_DIR / "figure_name.png")

# Cell 7: Generate Tables
best = selection.best_configuration_at_step(...)
table = selection.comparison_table(...)
```

### Results Output
- All outputs go to `results/{notebook_stem}/`
- Use descriptive filenames: `quadruped_walk_ablation.png`, `comparison_table.png`
- Include task name and analysis type in filename

## Tools Reference

### `wandb_io.py` - W&B Data Fetching

```python
runs, manifest, source = wandb_io.fetch_sweep_runs(
    entity="thomasevers9",
    project="tdmpc2-tdmpc2",
    sweep_id="abc123",
    history_keys=["eval/episode_reward", "_step"],
    use_cache=True,
    force_refresh=False,  # Set True to re-download
)
# Returns: (list of run dicts, manifest metadata, "cache"|"remote")
```

### `aggregations.py` - Data Processing

```python
# Convert cached runs to DataFrame
df = aggregations.runs_history_to_frame(runs, step_key="_step")

# Aggregate at specific step across seeds
summary = aggregations.aggregate_at_step(df, step=100000, metric_key="eval/episode_reward")
```

### `baselines.py` - Baseline Results

```python
from analysis.tools import paths

# Single task
df = baselines.load_one("quadruped-walk", root=paths.BASELINE_TDMPC2)

# Multiple tasks
df = baselines.load_many(["quadruped-walk", "hopper-hop"], root=paths.BASELINE_DREAMERV3)
```

Baseline roots:
- `paths.BASELINE_TDMPC2` - TD-MPC2 state-based
- `paths.BASELINE_TDMPC2_PIXELS` - TD-MPC2 pixel-based
- `paths.BASELINE_DREAMERV3` - DreamerV3
- `paths.BASELINE_SAC` - SAC

### `plotting.py` - Visualization

```python
# Simple line plot with one variant column
fig = plotting.sample_efficiency_figure(
    frame=df,
    metric_key="eval/episode_reward",
    variant_column="entropy_coef",
    task_name="Quadruped Walk",
    baseline_frame=baseline_df,
    baseline_label="TD-MPC2",
)

# Multi-encoding plot (color, dash, width, marker)
fig = plotting.sample_efficiency_encoded_figure(
    frame=df,
    metric_key="eval/episode_reward",
    task_name="Quadruped Walk",
    baseline_frame=baseline_df,
    baseline_label="TD-MPC2",
    encodings={
        "color": {"column": "num_rollouts", "label": "Rollouts"},
        "dash": {"column": "entropy_coef", "label": "Entropy"},
    },
)

# Export
plotting.write_png(fig, output_path=Path("figure.png"))
```

### `selection.py` - Hyperparameter Analysis

```python
# Find best configuration
best = selection.best_configuration_at_step(
    frame=df,
    step=100000,
    metric_key="eval/episode_reward",
    hyperparameter_columns=["num_rollouts", "entropy_coef"],
)
# Returns BestConfigResult with .best_config, .task_summary, .all_configs

# Build comparison table
table = selection.comparison_table(
    model_task_summary=best.task_summary,
    model_label="Our Method",
    baselines=[
        {"frame": baseline_df, "label": "TD-MPC2"},
        {"frame": dreamer_df, "label": "DreamerV3"},
    ],
    step=100000,
)
```

### `style.py` - Publication Styling

```python
from analysis.tools.style import STYLE

# Access styling constants
STYLE.COLORS           # TU Delft color palette
STYLE.FONT_FAMILY      # "Helvetica"
STYLE.FONT_SIZE_TITLE  # Title font size
STYLE.FONT_SIZE_AXIS   # Axis label font size
STYLE.LINE_WIDTH       # Default line width
STYLE.FIGURE_WIDTH     # Default figure width (pixels)
STYLE.FIGURE_HEIGHT    # Default figure height (pixels)
STYLE.DPI              # Export DPI

# Apply to Plotly figure
fig.update_layout(**STYLE.plotly_layout())
```

### `config.py` - W&B Defaults

```python
from analysis.tools import config

config.WANDB_ENTITY   # "thomasevers9"
config.WANDB_PROJECT  # "tdmpc2-tdmpc2"
```

## Data Flow

```
W&B Sweep
    │
    ▼
wandb_io.fetch_sweep_runs()  ──►  Local Cache (run_cache/)
    │
    ▼
aggregations.runs_history_to_frame()
    │
    ▼
pd.DataFrame (tidy format)
    │
    ├──► plotting.sample_efficiency_*()  ──►  Plotly Figure  ──►  PNG
    │
    ├──► selection.best_configuration_at_step()  ──►  BestConfigResult
    │         │
    │         ▼
    │    selection.comparison_table()  ──►  Table DataFrame
    │         │
    │         ▼
    │    plotting.comparison_table_figure()  ──►  Plotly Figure  ──►  PNG
    │
    └──► aggregations.aggregate_at_step()  ──►  Summary DataFrame  ──►  CSV
```

## Baseline Results Structure

Located in `/results/` (project root, not analysis):

```
results/
├── tdmpc2/           # State-based (102 tasks)
├── tdmpc2-pixels/    # Pixel-based (12 tasks)
├── dreamerv3/        # DreamerV3 (102 tasks)
└── sac/              # SAC (102 tasks)
```

Each task has a CSV: `{task_name}.csv` with columns: `step`, `reward`, `seed`

## Sweep Configuration

Sweeps are configured in `/sweep_list/`:

```
sweep_list/
├── test/
│   ├── 01_ensemble_loss_weighting/
│   │   ├── sweep.yaml    # W&B sweep config
│   │   ├── id.txt        # Sweep ID
│   │   ├── project.txt   # W&B project name
│   │   └── description.txt
│   └── ...
└── ...
```

## Style Guidelines for Scientific Figures

### Color Palette (TU Delft)
1. **Cyan** `#00A6D6` - Primary/default
2. **Pink** `#EF60A3`
3. **Red** `#E03C31`
4. **Yellow** `#FFB81C`
5. **Green** `#009B77`
6. **Dark Blue** `#0C2340`
7. **Purple** `#6D1E70`

### Line Conventions
- **Experiment variants**: Solid lines with colored std-deviation shading
- **Baselines**: Dashed lines in gray (`#444444`)

### Typography
- Font family: Helvetica (or Arial fallback)
- Title: 16-20pt
- Axis labels: 12-14pt
- Legend: 11-13pt

### Figure Dimensions
- Single column: 600×400 pixels
- Double column: 1200×400 pixels
- Export DPI: 300 for print

## Common Patterns

### Ablation Study
Compare multiple values of a single hyperparameter:
```python
fig = plotting.sample_efficiency_figure(
    frame=df,
    metric_key="eval/episode_reward",
    variant_column="hyperparameter_name",
    ...
)
```

### Multi-Task Comparison
Aggregate across tasks with average row:
```python
table = selection.comparison_table(...)
fig = plotting.comparison_table_figure(
    table,
    title="Multi-Task Performance at 100k Steps",
)
```

### Runtime Analysis
Plot training time alongside reward:
```python
# Use "elapsed_time" or similar metric from W&B
fig = plotting.sample_efficiency_figure(
    frame=df,
    metric_key="elapsed_time",
    ...
)
```

## Cache Management

- Cache is stored in `analysis/run_cache/`
- To force refresh: `force_refresh=True` in `fetch_sweep_runs()`
- To clear all cache: delete `run_cache/` directory
- Cache does not auto-invalidate when sweep gets new runs

## Troubleshooting

### Missing W&B API Key
Place your key in `analysis/wandb-key.txt` (one line, no trailing newline).

### Step Key Not Found
Common step keys: `_step`, `step`, `total_env_steps`, `eval/step`. 
Add the correct one to `history_keys` in your fetch call.

### Baseline Task Name Mismatch
Use `naming.wandb_task_to_baseline()` to convert W&B task names (snake_case) 
to baseline format (kebab-case).
