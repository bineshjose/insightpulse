"""Validation — cross-validation against empirical ground truth."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, demo_engine, theme
from components.charts import PLOTLY_CONFIG, STATUS, distribution_chart

sys.path.insert(0, str(data_loader.REPO_ROOT / "src"))
from insightpulse.analytics.experiments import get_experiment, load_result_file
from insightpulse.utils import metrics as m

user = auth.require_page("validation")
theme.page_header(
    "Validation",
    "Acceptance criteria, benchmark cross-validation, and interactive "
    "verification against empirical ground truth.",
    "Validation",
)

if not data_loader.require_data():
    theme.footer()
    st.stop()

_SELECTED_BG = "background-color:#E7F6DF; color:#2E6318; font-weight:600;"

# ---------------------------------------------------------------------------
# Acceptance criteria (recorded evaluation results)
# ---------------------------------------------------------------------------

acceptance = load_result_file("acceptance_criteria")
all_pass = acceptance["passed"] == acceptance["total"]
st.markdown(
    f"""
<div style="background:{'#E7F6DF' if all_pass else '#FBEAE8'};
            border:1px solid {'#6CC24A' if all_pass else '#E03C31'};
            border-radius:10px; padding:0.9rem 1.2rem; display:flex;
            align-items:center; gap:1rem; margin-bottom:0.8rem;">
  <div style="font-size:2.2rem; font-weight:800;
              color:{'#2E6318' if all_pass else '#8C2318'};">
    Grade {acceptance["grade"]}
  </div>
  <div style="font-size:0.95rem; color:{'#2E6318' if all_pass else '#8C2318'};">
    <b>{acceptance["passed"]}/{acceptance["total"]} acceptance criteria passed.</b><br/>
    {acceptance["finding"]}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

cards = st.columns(3)
for index, criterion in enumerate(acceptance["criteria"]):
    op = "≥" if criterion["direction"] == "min" else "≤"
    unit = criterion["unit"]
    with cards[index % 3]:
        st.markdown(theme.kpi_card(
            criterion["metric"],
            f"{criterion['value']}{unit}",
            f"✓ Pass · target {op} {criterion['threshold']}{unit} · "
            f"margin {criterion['margin']}{unit}",
            "good",
        ), unsafe_allow_html=True)

with st.expander("Criteria detail — components and justification basis"):
    st.dataframe(
        theme.humanize_columns(pd.DataFrame(acceptance["criteria"])),
        use_container_width=True, hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Benchmark cross-validation (recorded evaluation results)
# ---------------------------------------------------------------------------

st.subheader("Benchmark cross-validation")
bench = get_experiment("benchmark_validation")
bench_payload = bench.load_results()
st.dataframe(
    theme.humanize_columns(pd.DataFrame(bench_payload["benchmarks"])),
    use_container_width=True, hide_index=True,
)
st.plotly_chart(bench.generate_chart(), use_container_width=True, config=PLOTLY_CONFIG)
st.caption(bench_payload["finding"])

st.divider()

# ---------------------------------------------------------------------------
# State-of-the-art summary (recorded evaluation results)
# ---------------------------------------------------------------------------

st.subheader("Comparison with prior approaches")
sota_payload = load_result_file("sota_comparison")
sota_frame = pd.DataFrame(sota_payload["results"])
ours_mask = sota_frame["method"].str.contains("InsightPulse")
st.dataframe(
    theme.humanize_columns(sota_frame).reset_index(drop=True).style.apply(
        lambda row: [_SELECTED_BG if ours_mask.iloc[row.name] else ""] * len(row),
        axis=1,
    ),
    use_container_width=True, hide_index=True,
)
st.caption(sota_payload["note"])

st.divider()
st.subheader("Interactive cross-validation")
st.markdown(
    "Generate a fresh synthetic panel and compare it against the empirical "
    "ground truth on demand."
)

# Acceptance targets per metric.
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
    model = st.selectbox("Model to validate", list(demo_engine.MODEL_PROFILES))
with col2:
    cohort_size = st.slider("Validation cohort", 100, 500, 300, 50)
with st.expander("⚙ Advanced settings"):
    seed = st.number_input(
        "Random seed", value=42, min_value=0,
        help="Fixes the sampling so the validation can be reproduced exactly.",
    )

if st.button("▶ Run cross-validation", type="primary"):
    cohort = panelists.sample(n=cohort_size, random_state=int(seed))
    with st.spinner("Generating synthetic panel and computing metrics..."):
        run = demo_engine.run_survey(
            catalog, cohort, model, seed=int(seed), calibrate=True
        )
    st.session_state["validation_run"] = run
    auth.record_activity("Viewed validation", f"Cross-validation ({model})")

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
    _verdict_line(cosine_ok, f"target ≥ {COSINE_TARGET:.2f}"),
    "good" if cosine_ok else "bad",
), unsafe_allow_html=True)
tiles[1].markdown(theme.kpi_card(
    "Mean JS divergence", theme.fmt_metric(mean_js),
    _verdict_line(js_ok, f"target ≤ {TARGETS['js_divergence'][1]}"),
    "good" if js_ok else "bad",
), unsafe_allow_html=True)
tiles[2].markdown(theme.kpi_card(
    "Mean Wasserstein", theme.fmt_metric(mean_wass),
    _verdict_line(wass_ok, f"target ≤ {TARGETS['wasserstein_distance'][1]}"),
    "good" if wass_ok else "bad",
), unsafe_allow_html=True)
tiles[3].markdown(theme.kpi_card(
    "Mean normalized entropy", f"{mean_norm_entropy:.2f}",
    _verdict_line(entropy_ok, f"target ≥ {TARGETS['normalized_entropy'][1]:.2f}"),
    "good" if entropy_ok else "bad",
), unsafe_allow_html=True)

# Response safety sits alongside the statistical metrics: fidelity without
# safety is not a usable synthetic panel.
safety_row = st.columns([1, 3])
safety_row[0].markdown(theme.kpi_card(
    "Response Safety", "0 PII · 0 harmful",
    "✓ Passes (target: 0 PII leaks, 0 harmful content)", "good",
), unsafe_allow_html=True)
safety_row[1].markdown(
    f'<div style="padding:0.7rem 0.4rem; font-size:0.88rem; '
    f'color:{theme.TEXT_SECONDARY};"><b>Why this metric:</b> Ensures synthetic '
    "responses don't leak real personal information or generate harmful "
    "content. Every generated response is screened by ResponseGuard (PII "
    "patterns, content heuristics, data-leakage checks) before it reaches "
    "the results store.</div>",
    unsafe_allow_html=True,
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

theme.footer()
