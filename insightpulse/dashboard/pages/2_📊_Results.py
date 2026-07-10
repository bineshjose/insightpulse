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

from components import data_loader, simulation
from components.charts import CATEGORICAL, distribution_chart

st.set_page_config(page_title="Results — InsightPulse", page_icon="📊", layout="wide")

st.title("📊 Survey Results")

run = simulation.get_last_run()
if run is None:
    st.info("No survey run in this session yet.")
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
    st.stop()

totals = run["totals"]
config = run["config"]

st.caption(
    f"Run `{run['run_id']}` · {config['model']} · cohort {config['cohort_size']} · "
    f"seed {config['seed']} · calibration "
    f"{'on' if config['calibration_applied'] else 'off'} · {run['created_at']}"
)

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

tiles = st.columns(6)
tiles[0].metric("Responses", f"{totals['total_responses']:,}")
tiles[1].metric("Hallucination", f"{totals['hallucination_rate']:.1%}")
tiles[2].metric("Consistency", f"{totals['consistency_score']:.1%}")
tiles[3].metric("Tokens", f"{totals['total_tokens']:,}")
tiles[4].metric("Cost", f"${totals['total_cost_usd']:.2f}")
tiles[5].metric("Throughput", f"{totals['throughput_per_min']:,}/min")

st.divider()

# ---------------------------------------------------------------------------
# Per-question distributions and metrics
# ---------------------------------------------------------------------------

st.subheader("Response distributions")

for result in run["question_results"]:
    st.markdown(f"**{result['text']}**")

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
            key=f"dist_{result['question_id']}",
        )
    with metric_col:
        met = result["metrics_calibrated" if config["calibration_applied"] else "metrics_raw"]
        raw_met = result["metrics_raw"]
        st.metric(
            "JS divergence",
            f"{met['js_divergence']:.4f}",
            delta=(
                f"{met['js_divergence'] - raw_met['js_divergence']:+.4f} vs raw"
                if config["calibration_applied"] else None
            ),
            delta_color="inverse",
        )
        st.metric(
            "Wasserstein",
            f"{met['wasserstein_distance']:.4f}",
            delta=(
                f"{met['wasserstein_distance'] - raw_met['wasserstein_distance']:+.4f} vs raw"
                if config["calibration_applied"] else None
            ),
            delta_color="inverse",
        )
        st.metric("Shannon entropy", f"{met['shannon_entropy']:.2f} bits")

    with st.expander("Table view"):
        table = pd.DataFrame({"option": result["options"]})
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
        "Dimension", ["age_group", "income_group", "region", "behavioral_archetype"]
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
st.plotly_chart(fig, use_container_width=True)

with st.expander("Table view"):
    table = pd.DataFrame({"option": selected["options"]})
    for group, values in breakdown_series.items():
        table[f"{group} %"] = [round(v, 1) for v in values]
    st.dataframe(table, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

st.subheader("Export")
csv_bytes = responses.drop(columns=["validation_flags"]).to_csv(index=False).encode()
st.download_button(
    "⬇️ Download responses (CSV)",
    data=csv_bytes,
    file_name=f"insightpulse_run_{run['run_id']}.csv",
    mime="text/csv",
)
