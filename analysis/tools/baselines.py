"""Helpers for loading baseline CSV evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


from .paths import ANALYSIS_ROOT


BASELINE_ROOT = (ANALYSIS_ROOT.parent / "results" / "tdmpc2-pixels").resolve()


def load_task_baseline(task: str) -> pd.DataFrame:
    """Load the baseline CSV for *task* (hyphenated naming)."""

    csv_path = BASELINE_ROOT / f"{task}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline file missing: {csv_path}")
    frame = pd.read_csv(csv_path)
    required_columns = {"step", "reward", "seed"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Baseline '{task}' missing columns: {sorted(missing)}")
    frame["task"] = task
    return frame


def load_many(tasks: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for task in tasks:
        frames.append(load_task_baseline(task))
    combined = pd.concat(frames, ignore_index=True)
    return combined