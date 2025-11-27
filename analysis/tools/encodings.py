"""Visual encoding utilities for multi-parameter plots."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple, TypeVar

T = TypeVar("T")


class EncodingError(RuntimeError):
    """Raised when encodings cannot be constructed for the provided values."""


def _unique(values: Iterable[T]) -> List[T]:
    seen = []
    seen_set = set()
    for value in values:
        if value in seen_set:
            continue
        seen.append(value)
        seen_set.add(value)
    return seen


def _ensure_all_values_present(
    *,
    unique_values: Sequence[T],
    mapping: Mapping[T, T],
    label: str,
) -> None:
    missing = [value for value in unique_values if value not in mapping]
    if missing:
        raise EncodingError(
            f"Encoding for '{label}' missing values: {missing}"
        )


def _assign_from_sequence(
    *,
    unique_values: Sequence[T],
    sequence: Sequence[T],
    label: str,
) -> Mapping[T, T]:
    if len(unique_values) > len(sequence):
        raise EncodingError(
            f"Encoding for '{label}' requires {len(unique_values)} styles but only {len(sequence)} are available"
        )
    return {value: sequence[index] for index, value in enumerate(unique_values)}


def color_sequence() -> List[str]:
    """Return the TU Delft qualitative color sequence."""
    return [
        "#00A6D6",  # Cyan (Primary)
        "#EF60A3",  # Pink
        "#E03C31",  # Red
        "#FFB81C",  # Yellow
        "#009B77",  # Green
        "#0C2340",  # Dark Blue
        "#6D1E70",  # Purple
    ]


def dash_sequence() -> List[str]:
    return [
        "solid",
        "dash",
        "dot",
        "dashdot",
        "longdash",
        "longdashdot",
        "dashdotdot",
    ]


def marker_sequence() -> List[str]:
    return [
        "circle",
        "square",
        "diamond",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "star",
    ]


def width_sequence() -> List[float]:
    return [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def build_encoding(
    *,
    values: Iterable[T],
    label: str,
    override: Mapping[T, T] | None,
    sequence: Sequence[T],
) -> Mapping[T, T]:
    unique_values = _unique(values)
    if override is not None:
        _ensure_all_values_present(
            unique_values=unique_values,
            mapping=override,
            label=label,
        )
        return override
    return _assign_from_sequence(
        unique_values=unique_values,
        sequence=sequence,
        label=label,
    )


__all__ = [
    "EncodingError",
    "build_encoding",
    "color_sequence",
    "dash_sequence",
    "marker_sequence",
    "width_sequence",
]
