"""Helpers to coerce data structures into JSON-compatible forms."""

from __future__ import annotations

import datetime as _dt
from typing import Any


try:  # Optional dependency; present in most experiment environments.
    import numpy as _np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled dynamically at runtime.
    _np = None  # type: ignore


def make_json_safe(value: Any) -> Any:
    """Return a JSON-serialisable representation of *value*.

    Raises:
        TypeError: when *value* cannot be converted without guesswork.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()

    if _np is not None:
        if isinstance(value, (_np.generic,)):  # type: ignore[arg-type]
            return value.item()
        if isinstance(value, _np.ndarray):  # type: ignore[arg-type]
            return [make_json_safe(item) for item in value.tolist()]

    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, set):
        items = [make_json_safe(item) for item in value]
        try:
            return sorted(items)
        except TypeError as exc:  # heterogeneous elements
            raise TypeError("Cannot deterministically serialise set") from exc

    raise TypeError(f"Unsupported type for JSON serialisation: {type(value)!r}")