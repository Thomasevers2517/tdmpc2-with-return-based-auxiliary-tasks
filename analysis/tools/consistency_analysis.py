from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from . import wandb_io


DEFAULT_STEP_KEYS = ["total_env_steps", "global_step", "step", "_step"]


def load_consistency_history(
    *,
    entity: str,
    project: str,
    sweep_ids: Dict[str, str],
    history_keys: Iterable[str],
) -> pd.DataFrame:
    """Fetch consistency-loss histories for one or more sweeps.

    The returned frame has one row per history entry and includes:

    - task
    - seed
    - detach_encoder_ratio
    - step (resolved from DEFAULT_STEP_KEYS)
    - one column per consistency metric present in *history_keys*.
    """

    all_records: List[dict] = []
    step_keys = list(DEFAULT_STEP_KEYS)

    for _, sweep_id in sweep_ids.items():
        runs, manifest, source = wandb_io.fetch_sweep_runs(
            entity=entity,
            project=project,
            sweep_id=sweep_id,
            history_keys=list(history_keys),
            use_cache=True,
            force_refresh=False,
        )

        for run in runs:
            config = run["config"]
            history = run["history"]
            rows = history["rows"]
            task = config.get("task")
            seed = config.get("seed")
            detach_ratio = config.get("detach_encoder_ratio")

            for row in rows:
                step_value = None
                for key in step_keys:
                    if key in row and row[key] is not None:
                        step_value = int(row[key])
                        break
                if step_value is None:
                    continue

                record: dict = {
                    "task": task,
                    "seed": seed,
                    "detach_encoder_ratio": detach_ratio,
                    "step": step_value,
                }

                for key in history_keys:
                    if "consistency_loss" in key and key in row and row[key] is not None:
                        record[key] = float(row[key])

                all_records.append(record)

    if not all_records:
        raise ValueError("No consistency-loss history rows found in fetched sweeps")

    frame = (
        pd.DataFrame.from_records(all_records)
        .sort_values(["task", "seed", "detach_encoder_ratio", "step"])
        .reset_index(drop=True)
    )
    return frame


def melt_consistency_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide consistency-loss metrics into long format.

    Any column containing "consistency_loss/step" is treated as a metric column.
    """

    metric_cols = [
        col for col in raw_df.columns if "consistency_loss/step" in col
    ]
    if not metric_cols:
        raise ValueError("No consistency loss columns found in dataframe")

    long_df = raw_df.melt(
        id_vars=["task", "seed", "detach_encoder_ratio", "step"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"])
    return long_df


def aggregate_consistency(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/CI across runs for each (task, detach, metric, step)."""

    grouped = (
        long_df.groupby(["task", "detach_encoder_ratio", "metric", "step"], as_index=False)
        .agg(
            mean_value=("value", "mean"),
            std_value=("value", "std"),
            count=("value", "count"),
        )
    )
    grouped["sem"] = grouped["std_value"] / grouped["count"].clip(lower=1).pow(0.5)
    grouped["ci_low"] = grouped["mean_value"] - 1.96 * grouped["sem"]
    grouped["ci_high"] = grouped["mean_value"] + 1.96 * grouped["sem"]
    return grouped


__all__ = [
    "load_consistency_history",
    "melt_consistency_frame",
    "aggregate_consistency",
]
