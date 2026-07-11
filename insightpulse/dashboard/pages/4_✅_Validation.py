"""Validation — cross-validation against empirical ground truth.

Addresses evaluator feedback:
- #5 Real-world validation: synthetic distributions vs. held-out empirical
  survey responses, per question, with the full metric suite.
- #4 Metric justification: why cosine / JS / Wasserstein / entropy fit this
  data, stated next to the numbers they justify.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, simulation, theme
from components.charts import STATUS, distribution_chart

sys.path.insert(0, str(data_loader.REPO_ROOT / "src"))
from insightpulse.utils import metrics as m

st.set_page_config(page_title="Validation | InsightPulse", page_icon="✅", layout="wide")

user = auth.require_auth("analyze")
theme.apply()
auth.render_sidebar(user)
theme.page_header(
    "✅ Validation",
    "Cross-validation of the synthetic panel against empirical ground truth. "
    "In demo mode the ground truth is the historical response bank; in "
    "production this slot is filled by Pew ATP and ESS benchmark waves.",
    "Validation",
)

if not data_loader.require_data():
    theme.footer()
    st.stop()

# Thesis acceptance targets per metric.
TARGETS = {
    "js_divergence": ("≤", 0.05),
    "wasserstein_distance": ("≤", 0.15),
    "normalized_entropy": ("≥", 0.60),
}

panelists = data_loader.load_panelists()
catalog = data_loader.question_catalog()

col1, col2, col3 = st.columns(3)
with col1:
    model = st.selectbox("Model to validate", list(simulation.MODEL_PROFILES))
with col2:
    cohort_size = st.slider("Validation cohort", 100, 500, 300, 50)
with col3:
    seed = st.number_input("Seed", value=42, min_value=0)

if st.button("▶ Run cross-validation"):
    cohort = panelists.sample(n=cohort_size, random_state=int(seed))
    with st.spinner("Generating synthetic panel and computing metrics..."):
        run = simulation.simulate_survey_run(
            catalog, cohort, model, seed=int(seed), calibrate=True
        )
    st.session_state["validation_run"] = run

run = st.session_state.get("validation_run")
if run is None:
    st.info(
        "No validation results yet — press **Run cross-validation** to compare "
        "synthetic vs. empirical.",
        icon="✨",
    )
    theme.footer()
    st.stop()

# ---------------------------------------------------------------------------
# Aggregate verdict
# ---------------------------------------------------------------------------

results = run["question_results"]
agg = pd.DataFrame([r["metrics_calibrated"] for r in results])

# Behavioral fidelity across the whole survey: cosine between concatenated
# calibrated and empirical distribution vectors.
syn_vec = np.concatenate([m.normalize_distribution(r["calibrated_counts"]) for r in results])
emp_vec = np.concatenate([m.normalize_distribution(r["empirical_counts"]) for r in results])
overall_cosine = m.cosine_similarity(syn_vec, emp_vec)

tiles = st.columns(4)
tiles[0].metric("Cosine similarity", f"{overall_cosine:.3f}", "target ≥ 0.80", delta_color="off")
tiles[1].metric("Mean JS divergence", f"{agg['js_divergence'].mean():.4f}",
                "target ≤ 0.05", delta_color="off")
tiles[2].metric("Mean Wasserstein", f"{agg['wasserstein_distance'].mean():.4f}",
                "target ≤ 0.15", delta_color="off")
tiles[3].metric("Mean entropy", f"{agg['shannon_entropy'].mean():.2f} bits",
                "diversity check", delta_color="off")

st.divider()

# ---------------------------------------------------------------------------
# Per-question validation
# ---------------------------------------------------------------------------

st.subheader("Per-question validation")

verdict_rows = []
for result in results:
    met = result["metrics_calibrated"]
    passed = all(
        met[name] <= target if op == "≤" else met[name] >= target
        for name, (op, target) in TARGETS.items()
    )
    verdict_rows.append({
        "question": result["text"],
        "js_divergence": met["js_divergence"],
        "wasserstein": met["wasserstein_distance"],
        "entropy_bits": met["shannon_entropy"],
        "normalized_entropy": met["normalized_entropy"],
        "verdict": "✅ pass" if passed else "❌ fail",
    })

verdicts = pd.DataFrame(verdict_rows)
st.dataframe(
    verdicts.style.format({
        "js_divergence": "{:.4f}", "wasserstein": "{:.4f}",
        "entropy_bits": "{:.2f}", "normalized_entropy": "{:.2f}",
    }).map(
        lambda v: (
            f"color: {STATUS['good']}" if v == "✅ pass"
            else f"color: {STATUS['critical']}" if v == "❌ fail" else ""
        ),
        subset=["verdict"],
    ),
    use_container_width=True, hide_index=True,
)

question_texts = [r["text"] for r in results]
chosen = st.selectbox("Inspect a question", question_texts)
result = results[question_texts.index(chosen)]

def _pct(counts: list[float]) -> list[float]:
    total = max(sum(counts), 1)
    return [100 * c / total for c in counts]

st.plotly_chart(distribution_chart(
    result["options"],
    {"Calibrated": _pct(result["calibrated_counts"]),
     "Empirical": _pct(result["empirical_counts"])},
    title="Calibrated synthetic vs. empirical ground truth",
), use_container_width=True)

with st.expander("Table view"):
    st.dataframe(pd.DataFrame({
        "option": result["options"],
        "Calibrated %": [round(v, 1) for v in _pct(result["calibrated_counts"])],
        "Empirical %": [round(v, 1) for v in _pct(result["empirical_counts"])],
    }), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Metric justification (evaluator feedback #4)
# ---------------------------------------------------------------------------

st.subheader("Why these metrics?")
JUSTIFICATIONS = [
    ("Cosine similarity",
     "Behavioral embeddings B_i ∈ ℝ¹²⁸ are unit-normalized, so angular distance "
     "is the meaningful comparison — magnitude carries no signal after normalization."),
    ("JS divergence",
     "Symmetric and bounded [0, 1]; stays finite when a response option has zero "
     "probability in one distribution (common for unpopular options), where KL "
     "divergence explodes."),
    ("Wasserstein-1",
     "Respects the *ordinal* structure of Likert/NPS scales: moving mass from "
     "'agree' to 'neutral' costs less than to 'strongly disagree'. JS treats all "
     "misplacements equally; Wasserstein does not."),
    ("Shannon entropy",
     "Detects mode collapse. A panel that always picks the modal option can still "
     "match aggregates; entropy checks that synthetic responses show a real "
     "population's variability."),
    ("Hallucination rate",
     "The trust metric: fraction of responses referencing non-existent products, "
     "studies, or events, flagged by the Validator agent."),
]

with st.expander("Metric justification for survey-response data", expanded=False):
    table = "| Metric | Why it fits this data |\n|---|---|\n"
    table += "\n".join(f"| **{name}** | {why} |" for name, why in JUSTIFICATIONS)
    st.markdown(table)

# ---------------------------------------------------------------------------
# External benchmarks
# ---------------------------------------------------------------------------

st.subheader("External benchmark integration")
st.dataframe(pd.DataFrame({
    "Benchmark": ["Pew American Trends Panel", "European Social Survey", "Twin-2K-500"],
    "Role": [
        "US attitudinal ground truth (wave-matched questions)",
        "Cross-country attitudinal validation",
        "Published digital-twin benchmark for direct comparison",
    ],
    "Status": ["🟡 mapping questions", "🟡 mapping questions", "🟢 format compatible"],
}), use_container_width=True, hide_index=True)
st.caption(
    "Benchmark CSVs drop into `data/benchmarks/` with the same schema as "
    "`survey_responses.csv`; the validation above runs unchanged against them. "
    "Sources: Pew Research Center American Trends Panel (waves 2023-2025); "
    "European Social Survey (ESS round 11); Toubia et al., Twin-2K-500 (2024)."
)

theme.footer()
