"""Plotting utilities for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import pandas as pd


def sample_efficiency_figure(
    frame: pd.DataFrame,
    *,
    metric_key: str,
    variant_column: str,
    task_name: str,
    baseline_frame: pd.DataFrame,
    baseline_label: str,
) -> Any:
    """Return a line plot comparing variants against the baseline."""

    go = _plotly_go()
    palette = _qualitative_colors()
    required_columns = {variant_column, "step", metric_key}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise KeyError(f"Frame missing required columns: {sorted(missing)}")

    summary = (
        frame.groupby([variant_column, "step"], as_index=False)
        .agg(
            mean_reward=(metric_key, "mean"),
            std_reward=(metric_key, "std"),
            num_seeds=(metric_key, "count"),
        )
    )
    summary["variant_label"] = summary[variant_column].apply(_format_variant)

    if "seed" in frame.columns:
        total_seeds = frame["seed"].nunique()
    else:
        total_seeds = int(summary["num_seeds"].max())

    fig = go.Figure()
    variant_labels = list(summary["variant_label"].unique())
    for variant_index, variant_label in enumerate(variant_labels):
        variant_rows = (
            summary[summary["variant_label"] == variant_label]
            .sort_values("step")
            .reset_index(drop=True)
        )
        std_series = variant_rows["std_reward"].fillna(0.0)
        upper = variant_rows["mean_reward"] + std_series
        lower = variant_rows["mean_reward"] - std_series
        color = palette[variant_index % len(palette)]
        fill_x = list(variant_rows["step"]) + list(variant_rows["step"][::-1])
        fill_y = list(upper) + list(lower[::-1])

        fig.add_trace(
            go.Scatter(
                x=fill_x,
                y=fill_y,
                mode="lines",
                line=dict(color=_rgba_with_alpha(color, 0.0), width=0),
                fill="toself",
                fillcolor=_rgba_with_alpha(color, 0.2),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=str(variant_label),
                name=f"{variant_label} std",
            )
        )

        custom_data = (
            variant_rows[["std_reward", "num_seeds"]]
            .fillna(0.0)
            .to_numpy()
        )
        fig.add_trace(
            go.Scatter(
                x=variant_rows["step"],
                y=variant_rows["mean_reward"],
                mode="lines",
                name=str(variant_label),
                hovertemplate=(
                    "Step=%{x}<br>Reward=%{y:.2f}<br>Std=%{customdata[0]:.2f}<br>Seeds=%{customdata[1]}<extra>%{fullData.name}</extra>"
                ),
                customdata=custom_data,
                line=dict(color=color),
                legendgroup=str(variant_label),
            )
        )

    if not baseline_frame.empty:
        baseline_summary = (
            baseline_frame.groupby("step", as_index=False)
            .agg(
                mean_reward=("reward", "mean"),
                std_reward=("reward", "std"),
                num_seeds=("reward", "count"),
            )
        )
    else:
        baseline_summary = pd.DataFrame(columns=["step", "mean_reward", "std_reward", "num_seeds"])

    if not baseline_summary.empty:
        baseline_rows = baseline_summary.sort_values("step").reset_index(drop=True)
        baseline_std = baseline_rows["std_reward"].fillna(0.0)
        baseline_upper = baseline_rows["mean_reward"] + baseline_std
        baseline_lower = baseline_rows["mean_reward"] - baseline_std
        baseline_color = "#444444"
        baseline_fill_x = list(baseline_rows["step"]) + list(baseline_rows["step"][::-1])
        baseline_fill_y = list(baseline_upper) + list(baseline_lower[::-1])

        fig.add_trace(
            go.Scatter(
                x=baseline_fill_x,
                y=baseline_fill_y,
                mode="lines",
                line=dict(color=_rgba_with_alpha(baseline_color, 0.0), width=0),
                fill="toself",
                fillcolor=_rgba_with_alpha(baseline_color, 0.15),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=baseline_label,
                name=f"{baseline_label} std",
            )
        )

        baseline_custom = (
            baseline_rows[["std_reward", "num_seeds"]]
            .fillna(0.0)
            .to_numpy()
        )
        fig.add_trace(
            go.Scatter(
                x=baseline_rows["step"],
                y=baseline_rows["mean_reward"],
                mode="lines",
                name=baseline_label,
                line=dict(color=baseline_color, dash="dash"),
                legendgroup=baseline_label,
                hovertemplate=(
                    "Step=%{x}<br>Reward=%{y:.2f}<br>Std=%{customdata[0]:.2f}<br>Seeds=%{customdata[1]}<extra>%{fullData.name}</extra>"
                ),
                customdata=baseline_custom,
            )
        )

    fig.update_layout(
        title=f"Sample Efficiency — {task_name} (n={total_seeds} seeds)",
        xaxis_title="Environment Steps",
        yaxis_title="Eval Episode Reward",
        legend_title="Variant",
    )
    return fig


def bar_chart_at_step(
    aggregated_frame: pd.DataFrame,
    *,
    metric_column: str,
    variant_column: str,
    baseline_rows: pd.DataFrame,
    baseline_label: str,
    task_name: str,
) -> Any:
    """Return a bar chart comparing variants at a fixed step."""

    data = aggregated_frame.copy()
    data["variant_label"] = data[variant_column].apply(_format_variant)
    data.rename(columns={metric_column: "mean_reward"}, inplace=True)

    plot_frame = data[["variant_label", "mean_reward"]].copy()
    if not baseline_rows.empty:
        plot_frame = pd.concat(
            [
                plot_frame,
                pd.DataFrame(
                    {
                        "variant_label": [baseline_label],
                        "mean_reward": [baseline_rows["reward"].mean()],
                    }
                ),
            ],
            ignore_index=True,
        )

    px = _plotly_px()
    fig = px.bar(
        plot_frame,
        x="variant_label",
        y="mean_reward",
        title=f"500k Step Evaluation Reward — {task_name}",
        labels={"variant_label": "Variant", "mean_reward": "Eval Episode Reward"},
    )
    return fig


def write_png(figure: Any, *, output_path: Path) -> None:
    """Render *figure* to *output_path* as a PNG image using kaleido."""

    if not hasattr(figure, "write_image"):
        raise TypeError("Figure object does not support write_image export")

    try:
        figure.write_image(str(output_path), format="png", engine="kaleido")
    except ValueError as exc:
        raise RuntimeError(
            "Static image export requires the kaleido engine. Install plotly[kaleido] in the tdmpc2 env."
        ) from exc


def _format_variant(value: object) -> str:
    if isinstance(value, tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


def _plotly_go():
    try:
        import plotly.graph_objects as go  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific.
        raise RuntimeError(
            "plotly is required for plotting. Install it in the tdmpc2 env."
        ) from exc
    return go


def _plotly_px():
    try:
        import plotly.express as px  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific.
        raise RuntimeError(
            "plotly is required for plotting. Install it in the tdmpc2 env."
        ) from exc
    return px


def _qualitative_colors():
    try:
        from plotly.colors import qualitative  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific.
        raise RuntimeError(
            "plotly is required for plotting. Install it in the tdmpc2 env."
        ) from exc
    return list(qualitative.Plotly)


def _rgba_with_alpha(color: str, alpha: float) -> str:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0.0, 1.0]")
    if color.startswith("#"):
        try:
            from plotly.colors import hex_to_rgb  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - environment specific.
            raise RuntimeError(
                "plotly is required for plotting. Install it in the tdmpc2 env."
            ) from exc
        r, g, b = hex_to_rgb(color)
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("rgb("):
        parts = color[color.find("(") + 1 : color.find(")")].split(",")
        if len(parts) != 3:
            raise ValueError(f"Unsupported rgb color format: {color}")
        r, g, b = (value.strip() for value in parts)
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("rgba("):
        parts = color[color.find("(") + 1 : color.find(")")].split(",")
        if len(parts) != 4:
            raise ValueError(f"Unsupported rgba color format: {color}")
        r, g, b = (value.strip() for value in parts[:3])
        return f"rgba({r},{g},{b},{alpha})"
    raise ValueError(f"Unsupported color format: {color}")