"""Transform cached WANDB payloads into analysis-ready data frames."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd


class StepMissingError(RuntimeError):
    """Raised when a history row lacks any recognised step key."""


def runs_history_to_frame(
    runs: Iterable[Mapping[str, Any]],
    *,
    metric_key: str,
    step_keys: Iterable[str],
    config_to_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Convert cached sweep payload into a tidy DataFrame."""

    records: List[Dict[str, Any]] = []
    ordered_step_keys = list(step_keys)
    if not ordered_step_keys:
        raise ValueError("step_keys must not be empty")

    for run in runs:
        config = run["config"]
        history = run["history"]
        rows = history["rows"]
        config_columns = {
            column: _extract_config_value(config, config_key)
            for config_key, column in config_to_columns.items()
        }
        config_columns["run_id"] = run["run_id"]
        for row in rows:
            if metric_key not in row:
                continue
            metric_value = row[metric_key]
            if metric_value is None:
                continue
            step_value = _select_step(row, ordered_step_keys)
            record = {
                **config_columns,
                "step": step_value,
                metric_key: _coerce_float(metric_value, metric_key),
            }
            records.append(record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(
            "No history rows containing the requested metric were found in cache"
        )
    frame.sort_values(["run_id", "step"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def aggregate_at_step(
    frame: pd.DataFrame,
    *,
    step_value: int,
    metric_key: str,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    """Aggregate the metric at a specific *step_value* across seeds."""

    if metric_key not in frame.columns:
        raise KeyError(f"Frame missing metric column '{metric_key}'")
    subset = frame[frame["step"] == step_value]
    if subset.empty:
        raise ValueError(f"No rows found at step {step_value}")
    agg = (
        subset.groupby(list(group_cols))[metric_key]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_reward", "std": "std_reward", "count": "num_runs"})
        .reset_index()
    )
    return agg


def _extract_config_value(config: Mapping[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if part not in cursor:
            raise KeyError(f"Config missing key '{part}' within '{dotted_key}'")
        cursor = cursor[part]
    return _normalise_config_value(cursor)


def _normalise_config_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalise_config_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalise_config_value(item) for key, item in value.items()}
    return value


def _select_step(row: Mapping[str, Any], step_keys: List[str]) -> int:
    for key in step_keys:
        if key in row:
            value = row[key]
            if value is None:
                continue
            return _coerce_int(value, key)
    raise StepMissingError(
        "None of the provided step keys were present in the history row"
    )


def _coerce_int(value: Any, key: str) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    raise TypeError(f"Step key '{key}' must be numeric, received {type(value)!r}")


def _coerce_float(value: Any, key: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Metric '{key}' must be numeric, received {type(value)!r}")