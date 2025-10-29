"""Reusable DataFrame filters."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def by_column_values(frame: pd.DataFrame, column: str, values: Iterable) -> pd.DataFrame:
    if column not in frame.columns:
        raise KeyError(f"Column '{column}' missing from frame")
    filter_values = list(values)
    if not filter_values:
        raise ValueError("values must contain at least one selection")
    return frame[frame[column].isin(filter_values)].copy()


def exclude_column_values(frame: pd.DataFrame, column: str, values: Iterable) -> pd.DataFrame:
    if column not in frame.columns:
        raise KeyError(f"Column '{column}' missing from frame")
    filter_values = list(values)
    if not filter_values:
        raise ValueError("values must contain at least one selection")
    return frame[~frame[column].isin(filter_values)].copy()