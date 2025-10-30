"""Helpers for selecting and summarising hyper-parameter configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class BestConfigResult:
    """Container for the best configuration discovered at a target training step."""

    config: Dict[str, Any]
    task_summary: pd.DataFrame
    config_summary: pd.DataFrame


def best_configuration_at_step(
    frame: pd.DataFrame,
    *,
    metric_key: str,
    step_value: int,
    hyperparam_columns: Sequence[str],
    task_column: str = "task",
) -> BestConfigResult:
    """Identify the hyper-parameter configuration with the highest mean reward.

    The *frame* must contain columns for `step`, *metric_key*, *task_column*, and every
    entry in *hyperparam_columns*. Only rows matching *step_value* are considered. The
    function averages seeds within a task and configuration first, then averages those
    per-task means to obtain the global score used for ranking configurations.
    """

    required_columns = {"step", metric_key, task_column, *hyperparam_columns}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise KeyError(f"Frame missing required columns: {sorted(missing_columns)}")

    step_frame = frame[frame["step"] == step_value]
    if step_frame.empty:
        raise ValueError(f"No rows found at step {step_value}")

    group_columns = list(hyperparam_columns) + [task_column]
    per_task = (
        step_frame.groupby(group_columns)[metric_key]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_reward", "std": "std_reward", "count": "num_runs"})
        .reset_index()
    )
    if per_task.empty:
        raise ValueError("No per-task aggregates could be computed")

    config_summary = (
        per_task.groupby(list(hyperparam_columns))
        .agg(
            mean_reward=("mean_reward", "mean"),
            std_across_tasks=("mean_reward", "std"),
            task_count=(task_column, "nunique"),
        )
        .reset_index()
    )
    if config_summary.empty:
        raise ValueError("No configuration aggregates could be computed")

    best_idx = config_summary["mean_reward"].idxmax()
    best_row = config_summary.loc[best_idx]
    best_config = {column: best_row[column] for column in hyperparam_columns}

    best_task_summary = per_task.copy()
    for column, value in best_config.items():
        best_task_summary = best_task_summary[best_task_summary[column] == value]
    if best_task_summary.empty:
        raise RuntimeError("Best configuration unexpectedly missing per-task summary")

    best_task_summary = best_task_summary[[task_column, "mean_reward", "std_reward", "num_runs"]].reset_index(drop=True)

    return BestConfigResult(
        config=best_config,
        task_summary=best_task_summary,
        config_summary=config_summary,
    )


def baseline_mean_at_step(
    baseline_frame: pd.DataFrame,
    *,
    step_value: int,
    task_column: str = "task",
    reward_column: str = "reward",
) -> pd.DataFrame:
    """Average baseline reward per task at *step_value*.

    Raises if any task present in *baseline_frame* lacks measurements at the requested
    step, preventing silent fallbacks.
    """

    required_columns = {task_column, "step", reward_column}
    missing_columns = required_columns.difference(baseline_frame.columns)
    if missing_columns:
        raise KeyError(f"Baseline frame missing columns: {sorted(missing_columns)}")

    subset = baseline_frame[baseline_frame["step"] == step_value]
    if subset.empty:
        raise ValueError(f"Baseline data contains no rows at step {step_value}")

    grouped = subset.groupby(task_column)[reward_column].mean().rename("mean_reward").reset_index()

    expected_tasks = baseline_frame[task_column].unique()
    missing_tasks = set(expected_tasks) - set(grouped[task_column])
    if missing_tasks:
        raise ValueError(
            f"Baseline data missing step {step_value} for tasks: {sorted(missing_tasks)}"
        )

    return grouped


def comparison_table(
    *,
    model_task_summary: pd.DataFrame,
    baselines: Mapping[str, pd.DataFrame],
    step_value: int,
    task_column: str = "task",
    model_label: str = "tdmpc2-best",
) -> pd.DataFrame:
    """Construct a per-task comparison table against multiple baselines.

    *model_task_summary* must contain the columns ``task``, ``mean_reward``, and
    ``std_reward`` representing the best hyper-parameter configuration. Each entry in
    *baselines* is loaded through :func:`baseline_mean_at_step` and aligned on
    *task_column*. The function returns a DataFrame ordered by *task_column* with an
    additional ``<metric>_avg`` row appended that reports unweighted means across tasks.
    """

    required_columns = {task_column, "mean_reward", "std_reward"}
    missing_columns = required_columns.difference(model_task_summary.columns)
    if missing_columns:
        raise KeyError(
            f"Model task summary missing columns: {sorted(missing_columns)}"
        )

    table = model_task_summary[[task_column, "mean_reward", "std_reward"]].rename(
        columns={"mean_reward": f"{model_label}_reward", "std_reward": f"{model_label}_std"}
    )

    for label, baseline_frame in baselines.items():
        summary = baseline_mean_at_step(baseline_frame, step_value=step_value, task_column=task_column)
        table = table.merge(
            summary.rename(columns={"mean_reward": f"{label}_reward"}),
            on=task_column,
            how="left",
            validate="one_to_one",
        )
        if table[f"{label}_reward"].isna().any():
            raise RuntimeError(
                f"Comparison table has missing values after merging baseline '{label}'"
            )

    metric_columns = [column for column in table.columns if column.endswith("_reward")]
    if not metric_columns:
        raise ValueError("No metric columns present in comparison table")

    mean_row = {task_column: "<avg>"}
    for column in table.columns:
        if column == task_column:
            continue
        mean_row[column] = table[column].mean()

    table_with_avg = pd.concat([table, pd.DataFrame([mean_row])], ignore_index=True)
    return table_with_avg


__all__ = [
    "BestConfigResult",
    "best_configuration_at_step",
    "baseline_mean_at_step",
    "comparison_table",
]
