"""Results — distributions, evaluation metrics, and demographic breakdowns.

Shows raw vs. calibrated vs. empirical distributions per question, the full
metric suite from insightpulse.utils.metrics, and cohort breakdowns by any
demographic dimension. Every chart has a table-view twin (expander).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, simulation, theme
from components.charts import CATEGORICAL, PLOTLY_CONFIG, distribution_chart

user = auth.require_page("results")
theme.page_header(
    "Survey Results",
    "Raw vs calibrated vs empirical distributions, the full metric suite, "
    "and demographic breakdowns.",
    "Results",
)

run = simulation.get_last_run()
if run is None:
    st.info(
        "No results yet — run a survey to get started, or generate a demo run below.",
        icon="✨",
    )
    if data_loader.data_available() and st.button("Generate a demo run"):
        catalog = data_loader.question_catalog()
        panelists = data_loader.load_panelists()
        demo = simulation.simulate_survey_run(
            catalog[:3],
            panelists.sample(n=100, random_state=42),
            "claude-sonnet-4-6",
            seed=42,
        )
        simulation.store_run(demo)
        st.rerun()
    theme.footer()
    st.stop()

totals = run["totals"]
config = run["config"]

st.markdown(
    f'<p style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">'
    f'<span title="Full run ID: {run["run_id"]}"><b>{theme.run_label(run["run_id"])}</b></span>'
    f" &nbsp;·&nbsp; {config['model']} &nbsp;·&nbsp; cohort {config['cohort_size']}"
    f" &nbsp;·&nbsp; seed {config['seed']} &nbsp;·&nbsp; calibration "
    f"{'on' if config['calibration_applied'] else 'off'}"
    f" &nbsp;·&nbsp; {theme.format_timestamp(run['created_at'])}</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

halluc_pct = totals["hallucination_rate"] * 100
consistency_pct = totals["consistency_score"] * 100

tiles = st.columns(6)
tiles[0].markdown(theme.kpi_card(
    "Responses", f"{totals['total_responses']:,}", "", "info",
), unsafe_allow_html=True)
tiles[1].markdown(theme.kpi_card(
    "Hallucination", f"{halluc_pct:.1f}%", "target < 5%",
    "good" if halluc_pct < 5 else ("warn" if halluc_pct <= 8 else "bad"),
), unsafe_allow_html=True)
tiles[2].markdown(theme.kpi_card(
    "Consistency", f"{consistency_pct:.1f}%", "target > 90%",
    "good" if consistency_pct > 90 else ("warn" if consistency_pct >= 80 else "bad"),
), unsafe_allow_html=True)
tiles[3].markdown(theme.kpi_card(
    "Tokens", f"{totals['total_tokens']:,}", "", "info",
), unsafe_allow_html=True)
tiles[4].markdown(theme.kpi_card(
    "Cost", f"${totals['total_cost_usd']:.2f}", "", "neutral",
), unsafe_allow_html=True)
tiles[5].markdown(theme.kpi_card(
    "Throughput", f"{totals['throughput_per_min']:,}/min", "", "info",
), unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Per-question distributions and metrics
# ---------------------------------------------------------------------------

st.subheader("Response distributions")

for result in run["question_results"]:
    js_before = result["metrics_raw"]["js_divergence"]
    js_after = result["metrics_calibrated"]["js_divergence"]
    badge = ""
    if config["calibration_applied"] and js_before > 0:
        improvement = (js_before - js_after) / js_before * 100
        if improvement > 0:
            badge = (
                f'&nbsp;<span class="niq-badge" style="background:{theme.GREEN};">'
                f"BDCL calibration reduced JS divergence by {improvement:.0f}%</span>"
            )
    st.markdown(f"**{result['text']}**{badge}", unsafe_allow_html=True)

    def _pct(counts: list[float]) -> list[float]:
        total = max(sum(counts), 1)
        return [100 * c / total for c in counts]

    series = {"Synthetic (raw)": _pct(result["raw_counts"])}
    if config["calibration_applied"]:
        series["Calibrated"] = _pct(result["calibrated_counts"])
    series["Empirical"] = _pct(result["empirical_counts"])

    chart_col, metric_col = st.columns([3, 1])
    with chart_col:
        st.plotly_chart(
            distribution_chart(result["options"], series),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"dist_{result['question_id']}",
        )
    with metric_col:
        met = result["metrics_calibrated" if config["calibration_applied"] else "metrics_raw"]
        raw_met = result["metrics_raw"]
        st.metric(
            "JS divergence",
            theme.fmt_metric(met["js_divergence"]),
            delta=(
                f"{met['js_divergence'] - raw_met['js_divergence']:+.4f} vs raw"
                if config["calibration_applied"] else None
            ),
            delta_color="inverse",
        )
        st.metric(
            "Wasserstein",
            theme.fmt_metric(met["wasserstein_distance"]),
            delta=(
                f"{met['wasserstein_distance'] - raw_met['wasserstein_distance']:+.4f} vs raw"
                if config["calibration_applied"] else None
            ),
            delta_color="inverse",
        )
        st.metric("Shannon entropy", f"{met['shannon_entropy']:.2f} bits")

    with st.expander("Table view"):
        table = pd.DataFrame({"Option": result["options"]})
        for name, values in series.items():
            table[f"{name} %"] = [round(v, 1) for v in values]
        st.dataframe(table, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Demographic breakdowns
# ---------------------------------------------------------------------------

st.subheader("Demographic breakdown")

responses: pd.DataFrame = run["responses"]
question_map = {r["text"]: r for r in run["question_results"]}

col1, col2 = st.columns(2)
with col1:
    breakdown_question = st.selectbox("Question", list(question_map))
with col2:
    dimension = st.selectbox(
        "Dimension",
        ["age_group", "income_group", "region", "behavioral_archetype"],
        format_func=lambda d: d.replace("_", " ").capitalize(),
    )

selected = question_map[breakdown_question]
q_responses = responses[responses["question_id"] == selected["question_id"]]

groups = sorted(q_responses[dimension].unique())
breakdown_series: dict[str, list[float]] = {}
for group in groups:
    subset = q_responses[q_responses[dimension] == group]
    counts = subset["answer"].value_counts()
    total = max(len(subset), 1)
    breakdown_series[str(group)] = [
        100 * counts.get(opt, 0) / total for opt in selected["options"]
    ]

fig = distribution_chart(selected["options"], breakdown_series)
# Breakdown groups are entities of this chart — fixed slot order by group.
for i, trace in enumerate(fig.data):
    trace.marker.color = CATEGORICAL[i % len(CATEGORICAL)]
st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

with st.expander("Table view"):
    table = pd.DataFrame({"Option": selected["options"]})
    for group, values in breakdown_series.items():
        table[f"{group} %"] = [round(v, 1) for v in values]
    st.dataframe(table, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

st.subheader("Export")
if auth.has_permission(user, "export"):
    import json as _json

    csv_bytes = responses.drop(columns=["validation_flags"]).to_csv(index=False).encode()
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        if st.download_button(
            "⬇️ Download responses (CSV)",
            data=csv_bytes,
            file_name=f"insightpulse_run_{run['run_id']}.csv",
            mime="text/csv",
        ):
            auth.record_activity("Exported results", f"Run {run['run_id']} — CSV")
    with export_col2:
        results_json = _json.dumps(run["question_results"], indent=2, default=str)
        if st.download_button(
            "⬇️ Download metrics (JSON)",
            data=results_json,
            file_name=f"insightpulse_run_{run['run_id']}_metrics.json",
            mime="application/json",
        ):
            auth.record_activity("Exported results", f"Run {run['run_id']} — JSON")
else:
    st.info(
        f"Export requires the 'export' capability — not included in the "
        f"{user['tier']} tier. Contact your administrator.",
        icon="🔒",
    )

theme.footer()
