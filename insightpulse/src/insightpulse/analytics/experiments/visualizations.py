"""Plotly figure builders for the experiment views.

All charts share the NielsenIQ palette (navy #003865, blue #00A4E4) and a
common base layout (titled axes, legend, gridlines). Selected/winning
configurations are highlighted in green; alternatives render in the
default blue/grey ramp.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

NAVY = "#003865"
BLUE = "#00A4E4"
GREEN = "#6CC24A"
AMBER = "#F2A900"
RED = "#E03C31"
GREY = "#9AA7B4"
GRID = "#E3E8EE"

# Ramp used when a chart needs several distinct series.
SERIES = [NAVY, BLUE, AMBER, GREEN, RED, "#7D5BA6", "#00877C", GREY]


def apply_base_layout(fig: go.Figure, title: str, x_title: str = "",
                      y_title: str = "", height: int = 420) -> go.Figure:
    """Apply the shared NIQ chart layout to a figure."""
    fig.update_layout(
        title={"text": title, "font": {"color": NAVY, "size": 16}},
        height=height,
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#22313F", "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0,
                "xanchor": "right", "x": 1.0},
        xaxis={"title": x_title, "gridcolor": GRID, "zeroline": False},
        yaxis={"title": y_title, "gridcolor": GRID, "zeroline": False},
    )
    return fig


def _bar_colors(rows: list[dict[str, Any]], selected_key: str = "selected") -> list[str]:
    """Green for the selected row, blue otherwise."""
    return [GREEN if row.get(selected_key) else BLUE for row in rows]


def comparison_bar(rows: list[dict[str, Any]], x_key: str, y_key: str,
                   title: str, y_title: str, text_fmt: str = "{:.3f}") -> go.Figure:
    """Single-metric bar comparison with the selected option in green."""
    xs = [str(row[x_key]) for row in rows]
    ys = [row.get(y_key) for row in rows]
    fig = go.Figure(go.Bar(
        x=xs, y=ys, marker_color=_bar_colors(rows),
        text=[text_fmt.format(y) if y is not None else "n/a" for y in ys],
        textposition="outside",
    ))
    return apply_base_layout(fig, title, "", y_title)


def grouped_bar(rows: list[dict[str, Any]], x_key: str,
                metrics: list[tuple[str, str]], title: str,
                y_title: str = "") -> go.Figure:
    """Grouped bars: one group per row, one bar per metric."""
    xs = [str(row[x_key]) for row in rows]
    fig = go.Figure()
    for color, (key, label) in zip(SERIES, metrics, strict=False):
        fig.add_bar(name=label, x=xs, y=[row.get(key) for row in rows],
                    marker_color=color)
    fig.update_layout(barmode="group")
    return apply_base_layout(fig, title, "", y_title)


def multi_metric_lines(rows: list[dict[str, Any]], x_key: str,
                       metrics: list[tuple[str, str]], title: str,
                       x_title: str, selected_x: Any = None) -> go.Figure:
    """Several metrics as lines over one x-axis (e.g. temperature sweep)."""
    xs = [row[x_key] for row in rows]
    fig = go.Figure()
    for color, (key, label) in zip(SERIES, metrics, strict=False):
        fig.add_scatter(x=xs, y=[row.get(key) for row in rows], name=label,
                        mode="lines+markers", line={"color": color, "width": 2.5})
    if selected_x is not None:
        fig.add_vline(x=selected_x, line_dash="dash", line_color=GREEN,
                      annotation_text="selected", annotation_font_color=GREEN)
    return apply_base_layout(fig, title, x_title)


def before_after_bars(before: dict[str, float], after: dict[str, float],
                      metrics: list[tuple[str, str]], title: str) -> go.Figure:
    """Paired before/after bars per metric (BDCL calibration impact)."""
    labels = [label for _, label in metrics]
    fig = go.Figure()
    fig.add_bar(name="Before BDCL", x=labels,
                y=[before.get(key) for key, _ in metrics], marker_color=GREY,
                text=[f"{before.get(key)}" for key, _ in metrics], textposition="outside")
    fig.add_bar(name="After BDCL", x=labels,
                y=[after.get(key) for key, _ in metrics], marker_color=GREEN,
                text=[f"{after.get(key)}" for key, _ in metrics], textposition="outside")
    fig.update_layout(barmode="group")
    return apply_base_layout(fig, title)


def convergence_chart(curves: dict[str, dict[str, Any]], selected: str,
                      title: str) -> go.Figure:
    """Sinkhorn marginal-violation curves on a log-scale y-axis."""
    fig = go.Figure()
    for color, (name, curve) in zip(SERIES, curves.items(), strict=False):
        label = name.replace("eps_", "ε=")
        width = 3.5 if name == selected else 1.8
        fig.add_scatter(
            x=curve["iterations"], y=curve["marginal_violation"],
            name=label + (" (selected)" if name == selected else ""),
            mode="lines+markers",
            line={"color": GREEN if name == selected else color, "width": width},
        )
    fig.add_hline(y=1e-6, line_dash="dot", line_color=RED,
                  annotation_text="tolerance 1e-6", annotation_font_color=RED)
    fig = apply_base_layout(fig, title, "Sinkhorn iteration", "Marginal violation")
    fig.update_yaxes(type="log", exponentformat="power")
    return fig


def radar_chart(entries: list[dict[str, Any]], axes: list[tuple[str, str]],
                title: str) -> go.Figure:
    """Normalised radar comparison (each axis scaled to best=1.0).

    Axis keys prefixed with ``-`` are lower-is-better and get inverted.
    """
    fig = go.Figure()
    theta = [label for _, label in axes]
    normalised: dict[str, list[float]] = {}
    for key, _ in axes:
        raw_key = key.lstrip("-")
        values = [float(e.get(raw_key) or 0.0) for e in entries]
        if key.startswith("-"):
            best = min(v for v in values if v > 0) if any(v > 0 for v in values) else 1.0
            normalised[key] = [best / v if v > 0 else 1.0 for v in values]
        else:
            best = max(values) or 1.0
            normalised[key] = [v / best for v in values]
    for i, entry in enumerate(entries):
        fig.add_scatterpolar(
            r=[round(normalised[key][i], 3) for key, _ in axes],
            theta=theta, fill="toself", name=str(entry.get("model", entry.get("name"))),
            line={"color": SERIES[i % len(SERIES)]}, opacity=0.75,
        )
    fig.update_layout(
        title={"text": title, "font": {"color": NAVY, "size": 16}},
        polar={"radialaxis": {"range": [0, 1.05], "gridcolor": GRID}},
        height=460, paper_bgcolor="white",
        legend={"orientation": "h", "y": -0.12},
    )
    return fig


def cost_quality_scatter(rows: list[dict[str, Any]], x_key: str, y_key: str,
                         label_key: str, title: str, x_title: str,
                         y_title: str) -> go.Figure:
    """Cost-vs-quality frontier scatter with model labels."""
    fig = go.Figure(go.Scatter(
        x=[row[x_key] for row in rows], y=[row[y_key] for row in rows],
        mode="markers+text", text=[str(row[label_key]) for row in rows],
        textposition="top center",
        marker={"size": 14, "color": SERIES[: len(rows)]},
    ))
    return apply_base_layout(fig, title, x_title, y_title)


def ablation_waterfall(rows: list[dict[str, Any]], baseline_key: str,
                       value_key: str, label_key: str, title: str,
                       y_title: str) -> go.Figure:
    """Deviation-from-baseline bars for the ablation study.

    The first row is the full-pipeline baseline; every other row shows how
    far the metric moves when that component is removed.
    """
    baseline = float(rows[0][value_key])
    labels = [str(row[label_key]).replace("Without ", "− ") for row in rows[1:]]
    deltas = [round(float(row[value_key]) - baseline, 4) for row in rows[1:]]
    fig = go.Figure(go.Waterfall(
        x=labels, y=deltas, base=baseline,
        measure=["relative"] * len(deltas),
        increasing={"marker": {"color": RED}},
        decreasing={"marker": {"color": GREEN}},
        connector={"line": {"color": GRID}},
    ))
    fig.add_hline(y=baseline, line_dash="dash", line_color=NAVY,
                  annotation_text=f"full pipeline {baseline}",
                  annotation_font_color=NAVY)
    return apply_base_layout(fig, title, "", y_title, height=460)


def drift_timeline(points: list[dict[str, Any]], threshold: float,
                   title: str) -> go.Figure:
    """Drift time series with the alert-threshold reference line."""
    months = [point["month"] for point in points]
    values = [point["js_drift"] for point in points]
    fig = go.Figure(go.Scatter(
        x=months, y=values, mode="lines+markers", name="JS drift",
        line={"color": BLUE, "width": 2.5}, marker={"size": 9},
        text=[point.get("annotation") or "" for point in points],
        hovertemplate="%{x}: %{y}<br>%{text}<extra></extra>",
    ))
    for point in points:
        if point.get("annotation"):
            fig.add_annotation(x=point["month"], y=point["js_drift"],
                               text=point["annotation"], showarrow=True,
                               arrowhead=0, ay=-28, font={"size": 10, "color": NAVY})
    fig.add_hline(y=threshold, line_dash="dash", line_color=RED,
                  annotation_text=f"alert threshold {threshold}",
                  annotation_font_color=RED)
    fig = apply_base_layout(fig, title, "", "JS divergence")
    fig.update_yaxes(range=[0, threshold * 1.4])
    return fig


def failure_mode_bar(modes: list[dict[str, Any]], title: str) -> go.Figure:
    """Horizontal bars for the residual failure-mode distribution."""
    ordered = sorted(modes, key=lambda mode: mode["rate_pct"])
    fig = go.Figure(go.Bar(
        y=[mode["type"] for mode in ordered],
        x=[mode["rate_pct"] for mode in ordered],
        orientation="h", marker_color=BLUE,
        text=[f"{mode['rate_pct']}%" for mode in ordered], textposition="outside",
    ))
    return apply_base_layout(fig, title, "Residual rate (%)", "", height=360)


def timing_bar(rows: list[dict[str, Any]], title: str) -> go.Figure:
    """Log-scale horizontal bars over heterogeneous timing magnitudes."""
    ordered = sorted(rows, key=lambda row: row["time_seconds"])
    fig = go.Figure(go.Bar(
        y=[row["component"] for row in ordered],
        x=[row["time_seconds"] for row in ordered],
        orientation="h",
        marker_color=[
            {"training": NAVY, "inference": BLUE, "e2e": GREEN}[row["category"]]
            for row in ordered
        ],
        text=[row["time_display"] for row in ordered], textposition="outside",
    ))
    fig = apply_base_layout(fig, title, "Seconds (log scale)", "", height=480)
    fig.update_xaxes(type="log", exponentformat="power")
    return fig
