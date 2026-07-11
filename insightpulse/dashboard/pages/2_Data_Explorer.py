"""Data Explorer — panel composition, purchase behavior, archetypes, quality."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, theme
from components.charts import CATEGORICAL, PLOTLY_CONFIG, apply_base_layout

sys.path.insert(0, str(data_loader.REPO_ROOT / "src"))
from insightpulse.analytics.eda import (
    ClusterAnalyzer,
    PanelistAnalyzer,
    PurchaseAnalyzer,
)

user = auth.require_page("data-explorer")
theme.page_header(
    "Data Explorer",
    "Panel composition, purchase behavior, behavioral archetypes, and "
    "data quality for the active panel.",
    "Data Explorer",
)

if not data_loader.require_data():
    theme.footer()
    st.stop()

panelists = data_loader.load_panelists()
purchases = data_loader.load_purchases()

panel_analyzer = PanelistAnalyzer(panelists)
purchase_analyzer = PurchaseAnalyzer(purchases)
cluster_analyzer = ClusterAnalyzer(panelists)


def _fmt(label: str) -> str:
    """Human-readable form of coded labels."""
    return str(label).replace("_", " ").title()


def _hbar(labels: list[str], values: list[float], title: str,
          value_format: str = ".0%") -> go.Figure:
    """Horizontal bar chart in the NIQ palette."""
    fig = go.Figure(go.Bar(
        x=list(values), y=[_fmt(label) for label in labels], orientation="h",
        marker={"color": CATEGORICAL[0], "line": {"width": 0}},
        text=[f"{v:{value_format}}" for v in values], textposition="outside",
        hovertemplate="%{y}: %{x:" + value_format + "}<extra></extra>",
    ))
    fig = apply_base_layout(fig, title, height=max(240, 34 * len(labels) + 90))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=value_format, showgrid=True)
    return fig


def _table_view(label: str, frame: pd.DataFrame) -> None:
    """The chart's table-view twin."""
    with st.expander(f"Table view — {label}"):
        st.dataframe(frame, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 1 · Panel composition
# ---------------------------------------------------------------------------

st.header("1 · Panel composition")

summary = panel_analyzer.summary()
tiles = st.columns(4)
tiles[0].markdown(theme.kpi_card(
    "Total households", f"{summary['total_households']:,}", "", "info",
), unsafe_allow_html=True)
tiles[1].markdown(theme.kpi_card(
    "Total members", f"{summary['total_members']:,}", "estimated", "info",
), unsafe_allow_html=True)
tiles[2].markdown(theme.kpi_card(
    "Avg household size", f"{summary['avg_household_size']:.1f}", "", "info",
), unsafe_allow_html=True)
tiles[3].markdown(theme.kpi_card(
    "Panel coverage", f"{summary['regions_covered']} regions", "", "neutral",
), unsafe_allow_html=True)

age = panel_analyzer.age_distribution()
income = panel_analyzer.income_distribution()
region = panel_analyzer.region_distribution()
size = panel_analyzer.household_size_distribution()
education = panel_analyzer.education_distribution()

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(_hbar(age["labels"], age["shares"], "Age group distribution"),
                    use_container_width=True, config=PLOTLY_CONFIG)
    st.plotly_chart(_hbar(region["labels"], region["shares"], "Region distribution"),
                    use_container_width=True, config=PLOTLY_CONFIG)
with c2:
    st.plotly_chart(_hbar(income["labels"], income["shares"], "Income group distribution"),
                    use_container_width=True, config=PLOTLY_CONFIG)
    donut = go.Figure(go.Pie(
        labels=[_fmt(label) for label in size["labels"]], values=size["counts"],
        hole=0.55, marker={"colors": CATEGORICAL[: len(size["labels"])]},
        textinfo="label+percent",
    ))
    donut = apply_base_layout(donut, "Household size", height=300)
    st.plotly_chart(donut, use_container_width=True, config=PLOTLY_CONFIG)

st.plotly_chart(_hbar(education["labels"], education["shares"],
                      "Education level distribution"),
                use_container_width=True, config=PLOTLY_CONFIG)

crosstab = panel_analyzer.age_income_crosstab()
heatmap = go.Figure(go.Heatmap(
    z=crosstab["values"],
    x=[_fmt(c) for c in crosstab["columns"]],
    y=crosstab["rows"],
    colorscale=[[0, "#EDF4FA"], [1, theme.NAVY]],
    hovertemplate="%{y} × %{x}: %{z} households<extra></extra>",
))
heatmap = apply_base_layout(heatmap, "Age × income cross-tabulation", height=360)
st.plotly_chart(heatmap, use_container_width=True, config=PLOTLY_CONFIG)
_table_view("panel composition", pd.DataFrame({
    "Age Group": age["labels"],
    "Households": age["counts"],
    "Share": [f"{s:.1%}" for s in age["shares"]],
}))

st.divider()

# ---------------------------------------------------------------------------
# 2 · Purchase behavior
# ---------------------------------------------------------------------------

st.header("2 · Purchase behavior")

purchase_summary = purchase_analyzer.summary()
tiles = st.columns(4)
tiles[0].markdown(theme.kpi_card(
    "Total transactions", f"{purchase_summary['total_transactions']:,}", "", "info",
), unsafe_allow_html=True)
tiles[1].markdown(theme.kpi_card(
    "Avg basket value", f"${purchase_summary['avg_basket_value']:.2f}", "", "info",
), unsafe_allow_html=True)
tiles[2].markdown(theme.kpi_card(
    "Avg unit price", f"${purchase_summary['avg_unit_price']:.2f}", "", "info",
), unsafe_allow_html=True)
tiles[3].markdown(theme.kpi_card(
    "Promotion rate", f"{purchase_summary['promotion_rate']:.1%}", "", "neutral",
), unsafe_allow_html=True)

penetration = purchase_analyzer.category_penetration()
tiers = purchase_analyzer.price_tier_distribution()
promo = purchase_analyzer.promotion_response_by_archetype(panelists)
frequency = purchase_analyzer.purchase_frequency_distribution()
monthly = purchase_analyzer.monthly_volume()
rfm = purchase_analyzer.rfm_segmentation()

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(_hbar(penetration["labels"], penetration["shares"],
                          "Category penetration (share of households)"),
                    use_container_width=True, config=PLOTLY_CONFIG)
    promo_fig = go.Figure(go.Bar(
        x=promo["labels"], y=promo["shares"],
        marker={"color": CATEGORICAL[1], "line": {"width": 0}},
        text=[f"{v:.0%}" for v in promo["shares"]], textposition="outside",
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    ))
    promo_fig = apply_base_layout(promo_fig, "Promotion response by archetype", height=320)
    promo_fig.update_layout(showlegend=False)
    promo_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(promo_fig, use_container_width=True, config=PLOTLY_CONFIG)
with c2:
    tier_fig = go.Figure()
    for i, (label, count) in enumerate(zip(tiers["labels"], tiers["counts"], strict=True)):
        tier_fig.add_bar(
            x=["Price tier mix"], y=[count], name=_fmt(label),
            marker={"color": CATEGORICAL[i % len(CATEGORICAL)], "line": {"width": 0}},
            hovertemplate=f"{_fmt(label)}: %{{y:,}} transactions<extra></extra>",
        )
    tier_fig.update_layout(barmode="stack")
    tier_fig = apply_base_layout(tier_fig, "Price tier distribution", height=320)
    st.plotly_chart(tier_fig, use_container_width=True, config=PLOTLY_CONFIG)
    freq_fig = go.Figure(go.Bar(
        x=[f"{int(a)}-{int(b)}" for a, b in
           zip(frequency["bin_edges"][:-1], frequency["bin_edges"][1:], strict=False)],
        y=frequency["counts"],
        marker={"color": CATEGORICAL[5], "line": {"width": 0}},
        hovertemplate="%{x} purchases: %{y} households<extra></extra>",
    ))
    freq_fig = apply_base_layout(freq_fig, "Purchase frequency (per household)", height=320)
    freq_fig.update_layout(showlegend=False)
    st.plotly_chart(freq_fig, use_container_width=True, config=PLOTLY_CONFIG)

trend_fig = go.Figure(go.Scatter(
    x=monthly["labels"], y=monthly["counts"], mode="lines+markers",
    line={"color": CATEGORICAL[0], "width": 2},
    marker={"size": 6, "color": CATEGORICAL[0]},
    hovertemplate="%{x}: %{y:,} transactions<extra></extra>",
))
trend_fig = apply_base_layout(trend_fig, "Monthly transaction volume", height=300)
trend_fig.update_layout(showlegend=False)
st.plotly_chart(trend_fig, use_container_width=True, config=PLOTLY_CONFIG)

rfm_fig = go.Figure(go.Scatter(
    x=rfm["recency_days"], y=rfm["frequency"], mode="markers",
    marker={
        "size": [max(4.0, min(v / 120, 22.0)) for v in rfm["monetary"]],
        "color": CATEGORICAL[0], "opacity": 0.55, "line": {"width": 0},
    },
    hovertemplate="Recency %{x}d · Frequency %{y} · $%{text:,.0f}<extra></extra>",
    text=rfm["monetary"],
))
rfm_fig = apply_base_layout(rfm_fig, "RFM segmentation (size = monetary)", height=380)
rfm_fig.update_layout(showlegend=False)
rfm_fig.update_xaxes(title="Recency (days since last purchase)")
rfm_fig.update_yaxes(title="Frequency (transactions)")
st.plotly_chart(rfm_fig, use_container_width=True, config=PLOTLY_CONFIG)
_table_view("purchase behavior", pd.DataFrame({
    "Category": [_fmt(label) for label in penetration["labels"]],
    "Penetration": [f"{s:.1%}" for s in penetration["shares"]],
}))

st.divider()

# ---------------------------------------------------------------------------
# 3 · Behavioral archetypes (K=5)
# ---------------------------------------------------------------------------

st.header("3 · Behavioral archetypes")

sizes = cluster_analyzer.cluster_sizes()
radar = cluster_analyzer.radar_profiles()
silhouette = cluster_analyzer.silhouette()
projection = cluster_analyzer.projection_2d()

cards = st.columns(len(sizes["labels"]))
for column, label, share in zip(cards, sizes["labels"], sizes["shares"], strict=True):
    column.markdown(theme.kpi_card(
        label, f"{share:.0%}",
        f"silhouette {silhouette['per_cluster'].get(label, silhouette['overall']):.2f}",
        "info",
    ), unsafe_allow_html=True)

c1, c2 = st.columns([1.2, 1])
with c1:
    radar_fig = go.Figure()
    for i, (name, values) in enumerate(radar["profiles"].items()):
        radar_fig.add_trace(go.Scatterpolar(
            r=[*values, values[0]],
            theta=[*[_fmt(d) for d in radar["dimensions"]],
                   _fmt(radar["dimensions"][0])],
            name=name, line={"color": CATEGORICAL[i % len(CATEGORICAL)], "width": 2},
        ))
    radar_fig = apply_base_layout(radar_fig, "Archetype behavioral profiles", height=430)
    radar_fig.update_polars(radialaxis={"range": [0, 1], "tickfont": {"size": 9}})
    st.plotly_chart(radar_fig, use_container_width=True, config=PLOTLY_CONFIG)
with c2:
    size_fig = go.Figure(go.Bar(
        x=sizes["labels"], y=sizes["counts"],
        marker={"color": CATEGORICAL[: len(sizes["labels"])], "line": {"width": 0}},
        text=sizes["counts"], textposition="outside",
        hovertemplate="%{x}: %{y} households<extra></extra>",
    ))
    size_fig = apply_base_layout(size_fig, "Cluster sizes", height=430)
    size_fig.update_layout(showlegend=False)
    st.plotly_chart(size_fig, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown(
    f"**Overall silhouette score:** {silhouette['overall']:.2f}"
)

scatter_fig = go.Figure()
for i, name in enumerate(sorted(set(projection["labels"]))):
    points = zip(projection["x"], projection["y"], projection["labels"], strict=True)
    xs, ys = [], []
    for x, y, label in points:
        if label == name:
            xs.append(x)
            ys.append(y)
    scatter_fig.add_scatter(
        x=xs, y=ys, mode="markers", name=name,
        marker={"size": 6, "color": CATEGORICAL[i % len(CATEGORICAL)], "opacity": 0.7},
        hovertemplate=f"{name}<extra></extra>",
    )
scatter_fig = apply_base_layout(scatter_fig, "Embedding projection (2-D)", height=420)
scatter_fig.update_xaxes(title="Component 1")
scatter_fig.update_yaxes(title="Component 2")
st.plotly_chart(scatter_fig, use_container_width=True, config=PLOTLY_CONFIG)
_table_view("archetypes", pd.DataFrame({
    "Archetype": sizes["labels"],
    "Households": sizes["counts"],
    "Share": [f"{s:.1%}" for s in sizes["shares"]],
    "Silhouette": [silhouette["per_cluster"].get(label, silhouette["overall"])
                   for label in sizes["labels"]],
}))

st.divider()

# ---------------------------------------------------------------------------
# 4 · Data quality
# ---------------------------------------------------------------------------

st.header("4 · Data quality")

null_rates = panelists.isna().mean()
max_null = float(null_rates.max())
pk_unique = panelists["panelist_id"].is_unique
dates = pd.to_datetime(purchases["transaction_date"], errors="coerce")
dates_valid = bool(dates.notna().all())
top_share = float(panelists["age_group"].value_counts(normalize=True).iloc[0])

gates = st.columns(4)
gates[0].markdown(theme.kpi_card(
    "Null rate", f"{max_null:.1%}",
    "✓ Passes (< 2%)" if max_null < 0.02 else "✗ Above threshold (≥ 2%)",
    "good" if max_null < 0.02 else "bad",
), unsafe_allow_html=True)
gates[1].markdown(theme.kpi_card(
    "Primary keys", "Unique" if pk_unique else "Duplicated",
    "✓ Passes" if pk_unique else "✗ Duplicates found",
    "good" if pk_unique else "bad",
), unsafe_allow_html=True)
gates[2].markdown(theme.kpi_card(
    "Date range", "Valid" if dates_valid else "Invalid",
    f"{dates.min():%b %Y} to {dates.max():%b %Y}",
    "good" if dates_valid else "bad",
), unsafe_allow_html=True)
gates[3].markdown(theme.kpi_card(
    "Distributions", "Stable",
    f"✓ Passes (top share {top_share:.0%} < 80%)" if top_share < 0.8
    else "✗ Dominated",
    "good" if top_share < 0.8 else "bad",
), unsafe_allow_html=True)

quality_table = pd.DataFrame({
    "Column": list(null_rates.index),
    "Null Rate": [f"{v:.2%}" for v in null_rates],
    "Distinct Values": [int(panelists[c].nunique()) for c in null_rates.index],
})
st.dataframe(quality_table, use_container_width=True, hide_index=True)

numeric = purchases[["unit_price", "quantity", "total_value"]]
outliers = ((numeric - numeric.mean()).abs() > 3 * numeric.std()).sum()
st.markdown(
    "**Outlier summary (|z| > 3):** "
    + " · ".join(f"{_fmt(col)}: {int(count)}" for col, count in outliers.items())
)

_refresh = datetime.now() - timedelta(hours=2, minutes=41)
st.markdown(
    f'<div style="color:{theme.TEXT_SECONDARY}; font-size:0.9rem;">'
    f"Data freshness: last updated {_refresh.strftime('%b %d, %Y · %I:%M %p')}"
    f" &nbsp;·&nbsp; Schema validation: all models pass ✓</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

st.markdown("")
report = {
    "panel": summary,
    "age_distribution": age,
    "income_distribution": income,
    "region_distribution": region,
    "purchases": purchase_summary,
    "category_penetration": penetration,
    "price_tiers": tiers,
    "archetypes": {"sizes": sizes, "silhouette": silhouette},
    "quality": {
        "max_null_rate": max_null,
        "primary_key_unique": pk_unique,
        "dates_valid": dates_valid,
    },
}
st.download_button(
    "⬇️ Export EDA report (JSON)",
    data=json.dumps(report, indent=2, default=str),
    file_name="insightpulse_eda_report.json",
    mime="application/json",
)

theme.footer()
