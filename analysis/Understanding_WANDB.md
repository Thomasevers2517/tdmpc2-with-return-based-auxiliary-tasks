# Understanding W&B Analysis Workflow

This document serves as a knowledge base for interacting with the Weights & Biases (W&B) analysis tools in this repository. It documents common pitfalls, best practices, and structural details discovered during development.

## 1. Data Fetching (`analysis.tools.wandb_io`)

### Fetching Sweep Runs
Use `wandb_io.fetch_sweep_runs` to download data.
*   **`history_keys`**: You must explicitly list **ALL** metrics and step counters you want to retrieve from the run history.
*   **Caching**: The tools use a local cache (`analysis/run_cache`). Set `force_refresh=True` if you suspect the cache is stale or missing new keys.

### Critical: Step Keys
W&B runs often have multiple "step" counters depending on how data was logged (e.g., training steps vs. evaluation steps).
*   **Common Keys**: `total_env_steps`, `global_step`, `step`, `_step`.
*   **Namespaced Keys**: If a metric is logged as `eval/episode_reward`, the corresponding step might be `eval/step` or `eval/total_env_steps`.
*   **Best Practice**: Always request a broad list of step keys to ensure `runs_history_to_frame` can find a valid x-axis alignment.
    ```python
    STEP_KEYS = ["total_env_steps", "global_step", "step", "_step", "eval/step"]
    ```

### Critical: Metric Names
*   Verify the exact metric name in W&B.
*   TD-MPC2 often logs evaluation metrics with an `eval/` prefix (e.g., `eval/episode_reward`).
*   Training metrics might be `train/episode_reward` or `episode_reward`.

## 2. Data Processing (`analysis.tools.aggregations`)

### Converting to DataFrame
Use `aggregations.runs_history_to_frame` (NOT `wandb_io`) to convert raw run data into a Pandas DataFrame.

*   **Config vs. History**:
    *   **History**: Time-series data (rewards, losses). Defined in `history_keys`.
    *   **Config**: Static hyperparameters (seeds, UTD ratios).
*   **`config_to_columns`**: You must explicitly map config keys to DataFrame columns.
    ```python
    config_to_columns={"utd_ratio": "utd_ratio", "seed": "seed"}
    ```
    *   *Note*: You do **not** need to add config keys to `history_keys` during fetching. They are fetched automatically as part of the run object.

## 3. Baselines (`analysis.tools.baselines`)

*   Baselines (DreamerV3, SAC, TD-MPC2) are stored as CSVs in `results/`.
*   Use `baselines.load_task_baseline(task_name, root=...)` to load them.
*   Constants for roots: `baselines.DREAMERV3_BASELINE_ROOT`, `baselines.SAC_BASELINE_ROOT`, etc.

## 4. Plotting (`analysis.tools.plotting`)

*   The plotting tools have been updated to use the **TU Delft** color palette by default.
*   **`sample_efficiency_figure`**: Standard line plot with std deviation shading.
    *   `variant_column`: The column used to group lines (e.g., "Algorithm" or "UTD Ratio").
    *   `baseline_frame`: Optional frame to plot a dashed baseline comparison.

## 5. Common Errors & Fixes

### `StepMissingError`
*   **Cause**: The rows containing your requested metric (e.g., `eval/episode_reward`) do not contain any of the keys listed in `step_keys`.
*   **Fix**: Add `eval/step`, `_step`, or the specific step counter used for that metric to your `STEP_KEYS` list and re-fetch with `force_refresh=True`.

### `AttributeError: module 'analysis.tools.wandb_io' has no attribute 'runs_history_to_frame'`
*   **Cause**: The function was moved to `analysis.tools.aggregations`.
*   **Fix**: Import `aggregations` and call `aggregations.runs_history_to_frame`.

## 6. Authentication

The tools automatically look for a `wandb-key.txt` file in the `analysis/` root directory.
*   If found, the key is read and set as the `WANDB_API_KEY` environment variable before any W&B API calls are made.
*   This prevents the need to manually log in or paste keys into notebooks.
