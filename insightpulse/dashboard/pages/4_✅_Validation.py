"""Validation — cross-validation against empirical ground truth.

Synthetic distributions vs. held-out empirical survey responses, per
question, with the full metric suite and the reasoning for why each metric
fits survey-response data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, simulation, theme
from components.charts import PLOTLY_CONFIG, STATUS, distribution_chart

sys.path.insert(0, str(data_loader.REPO_ROOT / "src"))
from insightpulse.utils import metrics as m

user = auth.require_page("validation")
theme.page_header(
    "Validation",
    "Cross-validation of the synthetic panel against empirical ground truth. "
    "Ground truth: synthetic response bank (demo) · Pew ATP & ESS benchmark "
    "waves (production).",
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
COSINE_TARGET = 0.80

panelists = data_loader.load_panelists()
catalog = data_loader.question_catalog()

col1, col2 = st.columns(2)
with col1:
    model = st.selectbox("Model to validate", list(simulation.MODEL_PROFILES))
with col2:
    cohort_size = st.slider("Validation cohort", 100, 500, 300, 50)
with st.expander("⚙ Advanced settings"):
    seed = st.number_input(
        "Random seed", value=42, min_value=0,
        help="Fixes the sampling so the validation can be reproduced exactly.",
    )

if st.button("▶ Run cross-validation"):
    cohort = panelists.sample(n=cohort_size, random_state=int(seed))
    with st.spinner("Generating synthetic panel and computing metrics..."):
        run = simulation.simulate_survey_run(
            catalog, cohort, model, seed=int(seed), calibrate=True
        )
    st.session_state["validation_run"] = run
    auth.record_activity("Viewed validation", f"Cross-validation ({model}, seed {seed})")

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


def _verdict_line(passed: bool, target_note: str) -> str:
    """Standardized pass/fail delta line for the metric tiles."""
    return f"✓ Passes ({target_note})" if passed else f"✗ Below threshold ({target_note})"


mean_js = agg["js_divergence"].mean()
mean_wass = agg["wasserstein_distance"].mean()
mean_norm_entropy = agg["normalized_entropy"].mean()

cosine_ok = overall_cosine >= COSINE_TARGET
js_ok = mean_js <= TARGETS["js_divergence"][1]
wass_ok = mean_wass <= TARGETS["wasserstein_distance"][1]
entropy_ok = mean_norm_entropy >= TARGETS["normalized_entropy"][1]

tiles = st.columns(4)
tiles[0].markdown(theme.kpi_card(
    "Cosine similarity", f"{overall_cosine:.3f}",
    _verdict_line(cosine_ok, f"≥ {COSINE_TARGET:.2f}"),
    "good" if cosine_ok else "bad",
), unsafe_allow_html=True)
tiles[1].markdown(theme.kpi_card(
    "Mean JS divergence", theme.fmt_metric(mean_js),
    _verdict_line(js_ok, f"≤ {TARGETS['js_divergence'][1]}"),
    "good" if js_ok else "bad",
), unsafe_allow_html=True)
tiles[2].markdown(theme.kpi_card(
    "Mean Wasserstein", theme.fmt_metric(mean_wass),
    _verdict_line(wass_ok, f"≤ {TARGETS['wasserstein_distance'][1]}"),
    "good" if wass_ok else "bad",
), unsafe_allow_html=True)
tiles[3].markdown(theme.kpi_card(
    "Mean normalized entropy", f"{mean_norm_entropy:.2f}",
    _verdict_line(entropy_ok, f"≥ {TARGETS['normalized_entropy'][1]:.2f}"),
    "good" if entropy_ok else "bad",
), unsafe_allow_html=True)

st.caption(
    "In demo mode the synthetic and empirical distributions share a common "
    "source, so cosine similarity is near-perfect. Production mode measures "
    "real behavioral-to-synthetic alignment (thesis result: 0.84)."
)

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

verdicts = theme.humanize_columns(pd.DataFrame(verdict_rows))
st.dataframe(
    verdicts.style.format({
        "JS Divergence": theme.fmt_metric,
        "Wasserstein": theme.fmt_metric,
        "Entropy (bits)": "{:.2f}",
        "Normalized Entropy": "{:.2f}",
    }).map(
        lambda v: (
            f"color: {STATUS['good']}" if v == "✅ pass"
            else f"color: {STATUS['critical']}" if v == "❌ fail" else ""
        ),
        subset=["Verdict"],
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
), use_container_width=True, config=PLOTLY_CONFIG)

with st.expander("Table view"):
    st.dataframe(pd.DataFrame({
        "Option": result["options"],
        "Calibrated %": [round(v, 1) for v in _pct(result["calibrated_counts"])],
        "Empirical %": [round(v, 1) for v in _pct(result["empirical_counts"])],
    }), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Metric justification
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
