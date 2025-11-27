"""Plotting utilities for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any, Dict, Mapping, Optional

import pandas as pd

from . import encodings as encoding_utils


def sample_efficiency_figure(
    frame: pd.DataFrame,
    *,
    metric_key: str,
    variant_column: str,
    task_name: str,
    baseline_frame: pd.DataFrame,
    baseline_label: str,
    baseline_step_cap: Optional[int] = None,
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

    max_step_value = frame["step"].max()
    if pd.isna(max_step_value):
        raise ValueError("Frame does not contain any step values to determine plot range")
    max_step_value = int(max_step_value)
    default_cap = max(1, 2 * max_step_value)
    if baseline_step_cap is None:
        baseline_step_cap = default_cap
    else:
        baseline_step_cap = max(1, int(baseline_step_cap))

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
        baseline_rows = baseline_summary[baseline_summary["step"] <= baseline_step_cap]
        if baseline_rows.empty:
            baseline_rows = baseline_summary[baseline_summary["step"] == baseline_summary["step"].min()]
        baseline_rows = baseline_rows.sort_values("step").reset_index(drop=True)
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


def sample_efficiency_encoded_figure(
    frame: pd.DataFrame,
    *,
    metric_key: str,
    task_name: str,
    baseline_frame: pd.DataFrame,
    baseline_label: str,
    encodings: Mapping[str, Mapping[str, object]],
    baseline_step_cap: Optional[int] = None,
) -> Any:
    """Line plot where multiple hyperparameters control visual encodings."""

    if not encodings:
        raise ValueError("encodings must not be empty")
    if "color" not in encodings:
        raise ValueError("encodings must include a 'color' specification for shading")

    go = _plotly_go()
    required_columns = {"step", metric_key, "seed"}
    for spec in encodings.values():
        column = spec.get("column")
        if not isinstance(column, str):
            raise TypeError("Each encoding spec must define a 'column' string")
        required_columns.add(column)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise KeyError(f"Frame missing required columns: {sorted(missing)}")

    encoding_state: Dict[str, Dict[str, object]] = {}
    for encoding_type, spec in encodings.items():
        column = spec["column"]  # type: ignore[index]
        label = spec.get("label", column)  # type: ignore[assignment]
        override = spec.get("values")  # type: ignore[assignment]

        if encoding_type == "color":
            sequence = encoding_utils.color_sequence()
        elif encoding_type == "dash":
            sequence = encoding_utils.dash_sequence()
        elif encoding_type == "marker":
            sequence = encoding_utils.marker_sequence()
        elif encoding_type == "width":
            sequence = encoding_utils.width_sequence()
        else:
            raise ValueError(f"Unsupported encoding type '{encoding_type}'")

        if override is not None:
            if not isinstance(override, Mapping):
                raise TypeError("Encoding override must be a mapping of value -> style")
            mapping = encoding_utils.build_encoding(
                values=frame[column],
                label=str(label),
                override=override,  # type: ignore[arg-type]
                sequence=sequence,
            )
        else:
            mapping = encoding_utils.build_encoding(
                values=frame[column],
                label=str(label),
                override=None,
                sequence=sequence,
            )

        encoding_state[encoding_type] = {
            "column": column,
            "label": label,
            "mapping": dict(mapping),
        }

    encoding_columns = [state["column"] for state in encoding_state.values()]
    if len(set(encoding_columns)) != len(encoding_columns):
        raise ValueError("Each encoding must target a distinct column")

    summary = (
        frame.groupby(encoding_columns + ["step"], as_index=False)
        .agg(
            mean_reward=(metric_key, "mean"),
            std_reward=(metric_key, "std"),
            num_seeds=(metric_key, "count"),
        )
    )

    max_step_value = frame["step"].max()
    if pd.isna(max_step_value):
        raise ValueError("Frame does not contain any step values to determine plot range")
    max_step_value = int(max_step_value)
    default_cap = max(1, 2 * max_step_value)
    if baseline_step_cap is None:
        baseline_step_cap = default_cap
    else:
        baseline_step_cap = max(1, int(baseline_step_cap))

    total_seeds = frame["seed"].nunique()

    fig = go.Figure()
    combination_frame = summary[encoding_columns].drop_duplicates().reset_index(drop=True)
    color_state = encoding_state["color"]

    for _, combo_row in combination_frame.iterrows():
        mask = pd.Series(True, index=summary.index)
        for column in encoding_columns:
            mask &= summary[column] == combo_row[column]
        variant_rows = summary[mask].sort_values("step").reset_index(drop=True)
        if variant_rows.empty:
            continue

        fill_series = variant_rows["std_reward"].fillna(0.0)
        upper = variant_rows["mean_reward"] + fill_series
        lower = variant_rows["mean_reward"] - fill_series

        combo_color_value = combo_row[color_state["column"]]
        color_mapping = color_state["mapping"]  # type: ignore[assignment]
        if combo_color_value not in color_mapping:
            raise encoding_utils.EncodingError(
                f"Missing color mapping for value {combo_color_value!r}"
            )
        color = color_mapping[combo_color_value]

        fill_x = list(variant_rows["step"]) + list(variant_rows["step"][::-1])
        fill_y = list(upper) + list(lower[::-1])

        combo_label_parts = [
            f"{encoding_state[etype]['label']}={combo_row[encoding_state[etype]['column']]}"
            for etype in encoding_state
        ]
        combo_label = ", ".join(combo_label_parts)

        fig.add_trace(
            go.Scatter(
                x=fill_x,
                y=fill_y,
                mode="lines",
                line=dict(color=_rgba_with_alpha(str(color), 0.0), width=0),
                fill="toself",
                fillcolor=_rgba_with_alpha(str(color), 0.18),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=combo_label,
                name=f"{combo_label} std",
            )
        )

        line_kwargs: Dict[str, object] = {"color": color}
        marker_kwargs: Dict[str, object] | None = None

        if "dash" in encoding_state:
            dash_state = encoding_state["dash"]
            dash_value = combo_row[dash_state["column"]]
            dash_mapping = dash_state["mapping"]  # type: ignore[assignment]
            if dash_value not in dash_mapping:
                raise encoding_utils.EncodingError(
                    f"Missing dash mapping for value {dash_value!r}"
                )
            line_kwargs["dash"] = dash_mapping[dash_value]

        if "width" in encoding_state:
            width_state = encoding_state["width"]
            width_value = combo_row[width_state["column"]]
            width_mapping = width_state["mapping"]  # type: ignore[assignment]
            if width_value not in width_mapping:
                raise encoding_utils.EncodingError(
                    f"Missing width mapping for value {width_value!r}"
                )
            line_kwargs["width"] = width_mapping[width_value]

        trace_mode = "lines"
        if "marker" in encoding_state:
            marker_state = encoding_state["marker"]
            marker_value = combo_row[marker_state["column"]]
            marker_mapping = marker_state["mapping"]  # type: ignore[assignment]
            if marker_value not in marker_mapping:
                raise encoding_utils.EncodingError(
                    f"Missing marker mapping for value {marker_value!r}"
                )
            marker_kwargs = {
                "symbol": marker_mapping[marker_value],
                "size": 8,
                "line": dict(color=color, width=0),
            }
            trace_mode = "lines+markers"

        hover_columns = ["std_reward", "num_seeds"] + encoding_columns
        hover_data = variant_rows[hover_columns].copy()
        hover_data["std_reward"] = hover_data["std_reward"].fillna(0.0)
        custom_array = hover_data.to_numpy()

        hover_template_parts = [
            "Step=%{x}",
            "Reward=%{y:.2f}",
            "Std=%{customdata[0]:.2f}",
            "Seeds=%{customdata[1]}",
        ]
        for index, etype in enumerate(encoding_state):
            column_index = index + 2
            label = encoding_state[etype]["label"]
            hover_template_parts.append(
                f"{label}=%{{customdata[{column_index}]}}"
            )
        hover_template = "<br>".join(hover_template_parts) + "<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=variant_rows["step"],
                y=variant_rows["mean_reward"],
                mode=trace_mode,
                name=combo_label,
                hovertemplate=hover_template,
                customdata=custom_array,
                line=line_kwargs,
                marker=marker_kwargs,
                showlegend=False,
                legendgroup=combo_label,
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
        baseline_rows = baseline_summary[baseline_summary["step"] <= baseline_step_cap]
        if baseline_rows.empty:
            baseline_rows = baseline_summary[baseline_summary["step"] == baseline_summary["step"].min()]
        baseline_rows = baseline_rows.sort_values("step").reset_index(drop=True)
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

    for encoding_type, state in encoding_state.items():
        label = state["label"]
        mapping = state["mapping"]  # type: ignore[assignment]
        for value, style in mapping.items():
            legend_name = f"{label} = {value}"
            if encoding_type == "color":
                line = dict(color=style, width=3)
                mode = "lines"
                marker = None
            elif encoding_type == "dash":
                line = dict(color="#666666", dash=style, width=3)
                mode = "lines"
                marker = None
            elif encoding_type == "width":
                line = dict(color="#666666", width=style)
                mode = "lines"
                marker = None
            elif encoding_type == "marker":
                line = dict(color="#666666", width=2)
                mode = "markers"
                marker = dict(symbol=style, size=8, color="#666666")
            else:  # pragma: no cover - guarded above.
                continue
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode=mode,
                    line=line,
                    marker=marker,
                    name=legend_name,
                    hoverinfo="skip",
                    showlegend=True,
                    legendgroup=f"encoding-{encoding_type}",
                )
            )

    fig.update_layout(
        title=f"Sample Efficiency — {task_name} (n={total_seeds} seeds)",
        xaxis_title="Environment Steps",
        yaxis_title="Eval Episode Reward",
        legend_title="Encoding",
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


def comparison_table_figure(
    table: pd.DataFrame,
    *,
    title: str,
    footer_text: str | None = None,
) -> Any:
    """Render *table* as a styled Plotly figure suitable for reports."""

    if table.empty:
        raise ValueError("Comparison table must contain at least one row")

    go = _plotly_go()
    display = table.copy(deep=True)
    column_order = list(display.columns)
    if not column_order:
        raise ValueError("Comparison table missing column definitions")

    numeric_columns = display.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        display[column] = display[column].map(lambda value: f"{value:.2f}")

    headers: list[str] = []
    for column in column_order:
        if column == "task":
            headers.append("Task")
            continue
        label = column.replace("_reward", " Reward").replace("_std", " Std")
        label = label.replace("-", " ").replace("_", " ").title()
        headers.append(label)

    task_values = display[column_order[0]].tolist()
    row_colors = []
    for index, task_value in enumerate(task_values):
        if task_value == "<avg>":
            row_colors.append("#d8e9f8")
        elif index % 2 == 0:
            row_colors.append("#ffffff")
        else:
            row_colors.append("#f4f6fb")

    cell_values = [display[column].tolist() for column in column_order]
    fill_colors = [row_colors for _ in column_order]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="#1f2630",
                    font=dict(color="#ffffff", size=14, family="Helvetica"),
                    align="left",
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=fill_colors,
                    font=dict(color="#1a1a1a", size=13, family="Helvetica"),
                    align="left",
                ),
            )
        ]
    )

    bottom_margin = 24
    if footer_text is not None:
        bottom_margin = 96

    fig.update_layout(
        title=title,
        title_font=dict(size=20, family="Helvetica", color="#0b253a"),
        title_x=0.0,
        margin=dict(l=24, r=24, t=72, b=bottom_margin),
        paper_bgcolor="#ffffff",
    )

    if footer_text is not None:
        fig.add_annotation(
            text=footer_text,
            xref="paper",
            yref="paper",
            x=0.0,
            y=-0.18,
            showarrow=False,
            align="left",
            font=dict(size=12, family="Helvetica", color="#3a3a3a"),
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
    return encoding_utils.color_sequence()


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