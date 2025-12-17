"""Publication styling constants for scientific figures.

Centralizes TU Delft color palette, font sizes, line widths, figure dimensions,
and export settings for consistent, publication-quality figures across all notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class StyleConfig:
    """Immutable configuration for publication-quality figure styling."""

    # -------------------------------------------------------------------------
    # TU Delft Color Palette (qualitative)
    # -------------------------------------------------------------------------
    COLORS: tuple = (
        "#00A6D6",  # Cyan (Primary)
        "#EF60A3",  # Pink
        "#E03C31",  # Red
        "#FFB81C",  # Yellow
        "#009B77",  # Green
        "#0C2340",  # Dark Blue
        "#6D1E70",  # Purple
    )

    # Baseline/reference colors
    BASELINE_COLOR: str = "#444444"
    BASELINE_COLOR_LIGHT: str = "#888888"

    # Background colors for tables
    TABLE_HEADER_BG: str = "#1f2630"
    TABLE_HEADER_TEXT: str = "#ffffff"
    TABLE_ROW_EVEN: str = "#ffffff"
    TABLE_ROW_ODD: str = "#f4f6fb"
    TABLE_HIGHLIGHT_ROW: str = "#d8e9f8"  # For <avg> row

    # -------------------------------------------------------------------------
    # Typography
    # -------------------------------------------------------------------------
    FONT_FAMILY: str = "Helvetica"
    FONT_FAMILY_FALLBACK: str = "Arial, sans-serif"

    # Font sizes (in points for Plotly)
    FONT_SIZE_TITLE: int = 18
    FONT_SIZE_AXIS_TITLE: int = 14
    FONT_SIZE_AXIS_TICK: int = 12
    FONT_SIZE_LEGEND_TITLE: int = 13
    FONT_SIZE_LEGEND: int = 12
    FONT_SIZE_ANNOTATION: int = 11
    FONT_SIZE_TABLE_HEADER: int = 14
    FONT_SIZE_TABLE_CELL: int = 13

    # -------------------------------------------------------------------------
    # Line and Marker Styles
    # -------------------------------------------------------------------------
    LINE_WIDTH: float = 2.5
    LINE_WIDTH_BASELINE: float = 2.0
    MARKER_SIZE: int = 8
    STD_BAND_ALPHA: float = 0.18
    BASELINE_BAND_ALPHA: float = 0.12

    # -------------------------------------------------------------------------
    # Figure Dimensions
    # -------------------------------------------------------------------------
    # Single column figure (typical for ablations)
    FIGURE_WIDTH_SINGLE: int = 600
    FIGURE_HEIGHT_SINGLE: int = 400

    # Double column / wide figure
    FIGURE_WIDTH_DOUBLE: int = 1200
    FIGURE_HEIGHT_DOUBLE: int = 400

    # Square figure (for grid layouts)
    FIGURE_WIDTH_SQUARE: int = 500
    FIGURE_HEIGHT_SQUARE: int = 500

    # Default (single column)
    FIGURE_WIDTH: int = 600
    FIGURE_HEIGHT: int = 400

    # -------------------------------------------------------------------------
    # Export Settings
    # -------------------------------------------------------------------------
    DPI: int = 300
    EXPORT_FORMAT: str = "png"

    # -------------------------------------------------------------------------
    # Margins
    # -------------------------------------------------------------------------
    MARGIN_LEFT: int = 60
    MARGIN_RIGHT: int = 24
    MARGIN_TOP: int = 72
    MARGIN_BOTTOM: int = 60

    # -------------------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------------------

    def plotly_layout(
        self,
        *,
        title: str | None = None,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None,
        legend_title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        show_legend: bool = True,
    ) -> Dict[str, Any]:
        """Return a dict of Plotly layout settings using this style.

        Args:
            title: Figure title.
            xaxis_title: X-axis label.
            yaxis_title: Y-axis label.
            legend_title: Legend title.
            width: Figure width in pixels (defaults to FIGURE_WIDTH).
            height: Figure height in pixels (defaults to FIGURE_HEIGHT).
            show_legend: Whether to display the legend.

        Returns:
            Dict suitable for `fig.update_layout(**style.plotly_layout())`.
        """
        layout: Dict[str, Any] = {
            "font": {
                "family": f"{self.FONT_FAMILY}, {self.FONT_FAMILY_FALLBACK}",
                "size": self.FONT_SIZE_AXIS_TICK,
            },
            "title": {
                "font": {
                    "size": self.FONT_SIZE_TITLE,
                    "family": f"{self.FONT_FAMILY}, {self.FONT_FAMILY_FALLBACK}",
                },
            },
            "xaxis": {
                "title": {
                    "font": {"size": self.FONT_SIZE_AXIS_TITLE},
                },
                "tickfont": {"size": self.FONT_SIZE_AXIS_TICK},
            },
            "yaxis": {
                "title": {
                    "font": {"size": self.FONT_SIZE_AXIS_TITLE},
                },
                "tickfont": {"size": self.FONT_SIZE_AXIS_TICK},
            },
            "legend": {
                "font": {"size": self.FONT_SIZE_LEGEND},
                "title": {"font": {"size": self.FONT_SIZE_LEGEND_TITLE}},
            },
            "margin": {
                "l": self.MARGIN_LEFT,
                "r": self.MARGIN_RIGHT,
                "t": self.MARGIN_TOP,
                "b": self.MARGIN_BOTTOM,
            },
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "showlegend": show_legend,
            "width": width or self.FIGURE_WIDTH,
            "height": height or self.FIGURE_HEIGHT,
        }

        if title is not None:
            layout["title"]["text"] = title
        if xaxis_title is not None:
            layout["xaxis"]["title"]["text"] = xaxis_title
        if yaxis_title is not None:
            layout["yaxis"]["title"]["text"] = yaxis_title
        if legend_title is not None:
            layout["legend"]["title"]["text"] = legend_title

        return layout

    def color(self, index: int) -> str:
        """Return the color at the given index (wraps around)."""
        return self.COLORS[index % len(self.COLORS)]

    def colors_list(self) -> List[str]:
        """Return the color palette as a list."""
        return list(self.COLORS)


# Singleton instance for convenient import
STYLE = StyleConfig()


__all__ = ["STYLE", "StyleConfig"]
