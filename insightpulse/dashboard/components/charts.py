"""Plotly chart builders with a validated, consistent visual system.

Every chart on the dashboard goes through these builders so color usage,
mark weight, and chrome stay coherent across pages.

Color rules (validated with the dataviz palette checker on a white surface):
- Categorical hues live in CATEGORICAL in a fixed order and are assigned by
  entity, never by rank — "Synthetic" is always blue on every page.
- Likert/ordinal scales use ORDINAL_BLUES (one hue, light→dark).
- Status colors are reserved for pass/fail meaning and always ship with an
  icon + label, never color alone.
- Aqua/yellow/magenta sit below 3:1 contrast on white, so every chart is
  paired with a table view (expander) and tooltips as the relief channel.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

# --- Validated categorical palette (fixed order — the order IS the CVD-safety
# mechanism; never reorder or cycle past 8) ---
CATEGORICAL: list[str] = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]

# Color follows the entity: these assignments hold on every page.
SERIES_COLORS: dict[str, str] = {
    "Synthetic (raw)": CATEGORICAL[0],
    "Calibrated": CATEGORICAL[1],
    "Empirical": CATEGORICAL[2],
}

MODEL_COLORS: dict[str, str] = {
    "claude-sonnet-4-6": CATEGORICAL[0],
    "gpt-4o": CATEGORICAL[1],
    "ollama/llama3.1": CATEGORICAL[2],
    "claude-haiku-4-5": CATEGORICAL[3],
}

# One-hue ordinal ramp for Likert scales (validated with --ordinal).
ORDINAL_BLUES: list[str] = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Status palette — reserved meaning, always paired with icon + label.
STATUS: dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Chart chrome (light surface).
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"
FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def apply_base_layout(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    """Apply the shared chart chrome: recessive grid, thin marks, quiet axes.

    Args:
        fig: The figure to style.
        title: Optional chart title.
        height: Chart height in pixels (includes the x-axis band).

    Returns:
        The styled figure (same object, for chaining).
    """
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "color": SECONDARY_INK, "size": 12},
        title_font={"color": "#0b0b0b", "size": 15},
        margin={"l": 8, "r": 8, "t": 48 if title else 16, "b": 8},
        legend={
            "orientation": "h",
            "yanchor": "bottom", "y": 1.0,
            "xanchor": "left", "x": 0.0,
            "font": {"color": SECONDARY_INK},
        },
        bargap=0.35,
        bargroupgap=0.12,
        hoverlabel={"font": {"family": FONT_FAMILY}},
    )
    fig.update_xaxes(
        showgrid=False, linecolor=AXIS_LINE, linewidth=1,
        tickfont={"color": MUTED_INK, "size": 11},
        title_font={"color": MUTED_INK, "size": 12},
    )
    fig.update_yaxes(
        gridcolor=GRIDLINE, gridwidth=1, zeroline=False,
        showline=False, tickfont={"color": MUTED_INK, "size": 11},
        title_font={"color": MUTED_INK, "size": 12},
    )
    # Rounded data-ends (supported by plotly >= 5.19; skip silently on older).
    with contextlib.suppress(ValueError):
        fig.update_layout(barcornerradius=4)
    return fig


def distribution_chart(
    options: Sequence[str],
    series: dict[str, Sequence[float]],
    title: str | None = None,
    y_title: str = "% of respondents",
) -> go.Figure:
    """Grouped bar chart of response distributions.

    Series colors follow the fixed entity mapping (SERIES_COLORS), falling
    back to the next categorical slot for unknown series names.

    Args:
        options: Response options in scale order (x axis).
        series: Mapping of series name -> percentages aligned with options.
        title: Optional chart title.
        y_title: Y-axis title.

    Returns:
        A styled plotly figure.
    """
    fig = go.Figure()
    fallback = iter(CATEGORICAL)
    for name, values in series.items():
        color = SERIES_COLORS.get(name) or MODEL_COLORS.get(name) or next(fallback)
        fig.add_bar(
            x=list(options),
            y=list(values),
            name=name,
            marker={"color": color, "line": {"width": 0}},
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        )
    fig = apply_base_layout(fig, title)
    fig.update_yaxes(title=y_title)
    if len(series) == 1:
        fig.update_layout(showlegend=False)
    return fig


def metric_comparison_chart(
    df: pd.DataFrame,
    entity_col: str,
    value_col: str,
    color_map: dict[str, str],
    title: str,
    value_format: str = ".3f",
) -> go.Figure:
    """Single-metric bar comparison across entities (one chart per metric).

    Two measures of different scale never share a plot (no dual axes) —
    callers render one of these per metric as small multiples.

    Args:
        df: One row per entity.
        entity_col: Column holding entity names (x axis).
        value_col: Column holding the metric value.
        color_map: Fixed entity -> hex mapping (color follows the entity).
        title: Chart title (names the metric).
        value_format: d3 format string for the direct labels.

    Returns:
        A styled plotly figure.
    """
    entities = df[entity_col].tolist()
    values = df[value_col].tolist()
    fig = go.Figure(go.Bar(
        x=entities,
        y=values,
        marker={"color": [color_map.get(e, CATEGORICAL[0]) for e in entities],
                "line": {"width": 0}},
        text=[f"{v:{value_format}}" for v in values],
        textposition="outside",
        textfont={"color": SECONDARY_INK, "size": 11},
        hovertemplate="%{x}: %{y:" + value_format + "}<extra></extra>",
    ))
    fig = apply_base_layout(fig, title, height=320)
    fig.update_layout(showlegend=False)
    return fig


def convergence_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    series_col: str,
    title: str,
    y_title: str,
    log_y: bool = False,
) -> go.Figure:
    """Line chart of a metric converging over iterations.

    Args:
        df: Long-format data with one row per (series, iteration).
        x_col: Iteration column.
        y_col: Metric value column.
        series_col: Series identity column (e.g., epsilon value).
        title: Chart title.
        y_title: Y-axis title.
        log_y: Use a log scale on y (convergence curves span decades).

    Returns:
        A styled plotly figure.
    """
    fig = go.Figure()
    for i, (name, group) in enumerate(df.groupby(series_col, sort=False)):
        fig.add_scatter(
            x=group[x_col], y=group[y_col],
            mode="lines",
            name=str(name),
            line={"color": CATEGORICAL[i % len(CATEGORICAL)], "width": 2},
            hovertemplate=f"{name}<br>iter %{{x}}: %{{y:.4g}}<extra></extra>",
        )
    fig = apply_base_layout(fig, title)
    fig.update_yaxes(title=y_title)
    fig.update_xaxes(title="Sinkhorn iteration")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def drift_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    threshold: float,
    title: str,
    y_title: str,
) -> go.Figure:
    """Time series of drift with a retraining-trigger threshold line.

    Points above the threshold wear the reserved critical status color and
    are labeled — the trigger meaning is carried by marker + annotation,
    not color alone.

    Args:
        df: One row per period, sorted by time.
        x_col: Period column (e.g., month).
        y_col: Drift metric column.
        threshold: Retraining trigger level.
        title: Chart title.
        y_title: Y-axis title.

    Returns:
        A styled plotly figure.
    """
    fig = go.Figure()
    fig.add_scatter(
        x=df[x_col], y=df[y_col],
        mode="lines+markers",
        name="Drift (JS divergence)",
        line={"color": CATEGORICAL[0], "width": 2},
        marker={"size": 8, "color": CATEGORICAL[0],
                "line": {"width": 2, "color": "#ffffff"}},
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    )
    breaches = df[df[y_col] > threshold]
    if not breaches.empty:
        fig.add_scatter(
            x=breaches[x_col], y=breaches[y_col],
            mode="markers+text",
            name="⚠ Retrain trigger",
            marker={"size": 11, "color": STATUS["critical"],
                    "line": {"width": 2, "color": "#ffffff"}},
            text=["⚠ retrain"] * len(breaches),
            textposition="top center",
            textfont={"color": STATUS["critical"], "size": 11},
            hovertemplate="%{x}: %{y:.4f} — exceeds trigger<extra></extra>",
        )
    fig.add_hline(
        y=threshold, line={"color": AXIS_LINE, "width": 1},
        annotation_text=f"trigger = {threshold}",
        annotation_font={"color": MUTED_INK, "size": 11},
        annotation_position="top left",
    )
    fig = apply_base_layout(fig, title)
    fig.update_yaxes(title=y_title)
    return fig


def agent_timeline_chart(trace: list[dict], title: str = "Agent execution timeline") -> go.Figure:
    """Horizontal bar chart of per-agent execution durations.

    One series (pipeline duration), so every bar takes slot 1 and there is
    no legend — the title names the measure.

    Args:
        trace: Agent trace entries with 'agent_name' and 'duration_ms'.
        title: Chart title.

    Returns:
        A styled plotly figure.
    """
    names = [t["agent_name"] for t in trace]
    durations = [t["duration_ms"] for t in trace]
    fig = go.Figure(go.Bar(
        x=durations,
        y=names,
        orientation="h",
        marker={"color": CATEGORICAL[0], "line": {"width": 0}},
        text=[f"{d:,.0f} ms" for d in durations],
        textposition="outside",
        textfont={"color": SECONDARY_INK, "size": 11},
        hovertemplate="%{y}: %{x:,.0f} ms<extra></extra>",
    ))
    fig = apply_base_layout(fig, title, height=max(300, 40 * len(names) + 80))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="duration (ms)", showgrid=True, gridcolor=GRIDLINE)
    return fig
