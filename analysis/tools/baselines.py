"""Helpers for loading baseline CSV evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


from .paths import ANALYSIS_ROOT


PIXEL_BASELINE_ROOT = (ANALYSIS_ROOT.parent / "results" / "tdmpc2-pixels").resolve()
STATE_BASELINE_ROOT = (ANALYSIS_ROOT.parent / "results" / "tdmpc2").resolve()
DREAMERV3_BASELINE_ROOT = (ANALYSIS_ROOT.parent / "results" / "dreamerv3").resolve()
SAC_BASELINE_ROOT = (ANALYSIS_ROOT.parent / "results" / "sac").resolve()


def resolve_root(root: Optional[Path] = None) -> Path:
    if root is None:
        return PIXEL_BASELINE_ROOT
    return root


def baseline_path(task: str, *, root: Optional[Path] = None) -> Path:
    base_root = resolve_root(root)
    return base_root / f"{task}.csv"


def has_task(task: str, *, root: Optional[Path] = None) -> bool:
    return baseline_path(task, root=root).exists()


def load_task_baseline(task: str, *, root: Optional[Path] = None) -> pd.DataFrame:
    """Load the baseline CSV for *task* (hyphenated naming)."""

    csv_path = baseline_path(task, root=root)
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline file missing: {csv_path}")
    frame = pd.read_csv(csv_path)
    required_columns = {"step", "reward", "seed"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Baseline '{task}' missing columns: {sorted(missing)}")
    frame["task"] = task
    return frame


def load_many(tasks: Iterable[str], *, root: Optional[Path] = None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for task in tasks:
        frames.append(load_task_baseline(task, root=root))
    combined = pd.concat(frames, ignore_index=True)
    return combined


__all__ = [
    "PIXEL_BASELINE_ROOT",
    "STATE_BASELINE_ROOT",
    "DREAMERV3_BASELINE_ROOT",
    "SAC_BASELINE_ROOT",
    "baseline_path",
    "has_task",
    "load_task_baseline",
    "load_many",
]