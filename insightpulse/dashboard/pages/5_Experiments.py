"""Experiments — ten research evaluation tabs backed by recorded results.

Every number shown here comes from the result files under
``insightpulse/analytics/experiments/results`` (single source of truth
shared with the API frontend and the test suite); charts come from the
experiment classes themselves.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, theme
from components.charts import PLOTLY_CONFIG

sys.path.insert(0, str(data_loader.REPO_ROOT / "src"))
from insightpulse.analytics.experiments import get_experiment, load_result_file

user = auth.require_page("experiments")
theme.page_header(
    "Experiments",
    "Embedding, generation, and calibration tuning; multi-model, ablation, "
    "drift, timing, and benchmark comparisons.",
    "Experiments",
)

_SELECTED_BG = "background-color:#E7F6DF; color:#2E6318; font-weight:600;"


def _chart(fig) -> None:
    """Render a Plotly figure with the shared modebar config."""
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _finding(text: str) -> None:
    """Render an experiment finding under its chart."""
    st.markdown(
        f"<div style='border-left:4px solid #00A4E4; background:#F4FAFD; "
        f"padding:0.5rem 0.9rem; border-radius:6px; font-size:0.88rem;'>"
        f"<b>Finding.</b> {text}</div>",
        unsafe_allow_html=True,
    )


def _results_table(rows: list[dict], drop: tuple[str, ...] = ()) -> None:
    """Show a results table with the selected row highlighted in green."""
    frame = pd.DataFrame(rows)
    if "selected" in frame:
        selected_mask = frame["selected"].fillna(False)
    else:
        selected_mask = pd.Series(False, index=frame.index)
    frame = frame.drop(columns=[c for c in ("selected", *drop) if c in frame])
    frame = theme.humanize_columns(frame).reset_index(drop=True)
    styler = frame.style.apply(
        lambda row: [_SELECTED_BG if selected_mask.iloc[row.name] else ""] * len(row),
        axis=1,
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)


def _experiment_block(name: str, drop: tuple[str, ...] = ()) -> None:
    """Standard sub-section: table + chart + finding for one experiment."""
    exp = get_experiment(name)
    payload = exp.load_results()
    if isinstance(payload.get("results"), list):
        _results_table(payload["results"], drop=drop)
    _chart(exp.generate_chart())
    _finding(exp.get_summary())


tabs = st.tabs([
    "Embedding Tuning", "Generation Tuning", "Calibration Analysis",
    "Multi-Model Comparison", "Ablation Study", "Sequential Dependency",
    "Drift Detection", "Timing & Scalability", "SotA Comparison",
    "Hyperparameter Search",
])

# ---------------------------------------------------------------------------
# Tab 1 · Embedding tuning
# ---------------------------------------------------------------------------
with tabs[0]:
    for title, name in [
        ("Chunking strategy", "chunking_strategy"),
        ("Encoder architecture", "encoder_architecture"),
        ("Embedding dimension", "embedding_dimension"),
        ("Clustering algorithm", "clustering_algorithm"),
    ]:
        st.subheader(title)
        _experiment_block(name)
        st.divider()

# ---------------------------------------------------------------------------
# Tab 2 · Generation tuning
# ---------------------------------------------------------------------------
with tabs[1]:
    for title, name in [
        ("Prompting strategy", "prompting_strategy"),
        ("Temperature sweep", "temperature_sweep"),
        ("Retrieval strategy", "retrieval_strategy"),
        ("Response parsing", "response_parsing"),
        ("Fine-tuning vs prompt conditioning", "finetuning_comparison"),
    ]:
        st.subheader(title)
        _experiment_block(name)
        st.divider()

# ---------------------------------------------------------------------------
# Tab 3 · Calibration analysis
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("BDCL calibration impact")
    bdcl = load_result_file("bdcl_before_after")["results"]
    cols = st.columns(4)
    for col, (label, key, fmt) in zip(cols, [
        ("JS divergence", "js", "{:.3f}"),
        ("Wasserstein", "wasserstein", "{:.3f}"),
        ("Hallucination", "hallucination", "{:.1f}%"),
        ("Shannon entropy", "entropy", "{:.2f} bits"),
    ], strict=True):
        with col:
            st.markdown(theme.kpi_card(
                label, fmt.format(bdcl["after"][key]),
                f"was {fmt.format(bdcl['before'][key])}", "good",
            ), unsafe_allow_html=True)
    _chart(get_experiment("bdcl_before_after").generate_chart())
    _finding(get_experiment("bdcl_before_after").get_summary())
    st.divider()

    st.subheader("Sinkhorn convergence")
    _chart(get_experiment("sinkhorn_convergence").generate_chart())
    _finding(get_experiment("sinkhorn_convergence").get_summary())
    st.divider()

    st.subheader("Regularisation sensitivity (ε)")
    _experiment_block("epsilon_sensitivity")
    st.divider()

    st.subheader("Behavioural weight sensitivity (λb)")
    _experiment_block("behavioural_weight")
    st.divider()

    st.subheader("Fairness weight sensitivity (λf)")
    _experiment_block("fairness_weight")

# ---------------------------------------------------------------------------
# Tab 4 · Multi-model comparison
# ---------------------------------------------------------------------------
with tabs[3]:
    multi = get_experiment("multi_model")
    payload = multi.load_results()
    st.caption(
        f"Fixed cohort of {payload['cohort_size']} respondents, seed "
        f"{payload['seed']}, {payload['survey_instrument']}. "
        f"{payload['measurement_note']}"
    )
    frame = pd.DataFrame(payload["results"])
    # Best value per metric column (direction-aware) rendered in green.
    best = {
        "js_infit": frame["js_infit"].min(),
        "hallucination": frame["hallucination"].min(),
        "consistency": frame["consistency"].max(),
        "entropy": frame["entropy"].max(),
        "cost_usd": frame["cost_usd"].min(),
        "latency_sec": frame["latency_sec"].min(),
    }
    human = theme.humanize_columns(frame)
    rename_map = dict(zip(frame.columns, human.columns, strict=True))
    best_by_label = {rename_map[key]: value for key, value in best.items()}
    styler = human.style.apply(
        lambda col: [
            _SELECTED_BG if col.name in best_by_label and value == best_by_label[col.name]
            else ""
            for value in col
        ],
        axis=0,
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        _chart(multi.generate_chart())
    with right:
        _chart(multi.generate_cost_quality_chart())
    _finding(multi.get_summary())

# ---------------------------------------------------------------------------
# Tab 5 · Ablation study
# ---------------------------------------------------------------------------
with tabs[4]:
    st.success(
        "**BDCL contributes 78% of the quality improvement** — the largest "
        "single component effect (JS 0.078 → 0.017)."
    )
    _experiment_block("ablation_study")

# ---------------------------------------------------------------------------
# Tab 6 · Sequential dependency
# ---------------------------------------------------------------------------
with tabs[5]:
    seq = get_experiment("sequential_dependency")
    seq_results = seq.load_results()
    st.success(
        f"**+{seq_results['improvement_pp']} pp consistency improvement** "
        "from sequential inter-question conditioning."
    )
    rows = [
        {"strategy": "Independent", **seq_results["results"]["independent"]},
        {"strategy": "Sequential conditioning", **seq_results["results"]["conditioned"]},
        {"strategy": "Human panel reference", **seq_results["results"]["empirical"]},
    ]
    _results_table(rows)
    _chart(seq.generate_chart())
    _finding(seq.get_summary())

# ---------------------------------------------------------------------------
# Tab 7 · Drift detection
# ---------------------------------------------------------------------------
with tabs[6]:
    drift = get_experiment("drift_detection")
    drift_payload = drift.load_results()
    if not drift_payload["triggered"]:
        st.success("**No retraining triggered** across the nine-month evaluation period.")
    else:
        st.error("Retraining threshold crossed — see the retraining pipeline below.")
    st.caption(f"Alert threshold: {drift_payload['formula']}")
    _chart(drift.generate_chart())
    _finding(drift.get_summary())
    st.markdown(
        "**Retraining pipeline (drift-triggered):** "
        + " → ".join(drift_payload["retraining_pipeline"])
    )

# ---------------------------------------------------------------------------
# Tab 8 · Timing & scalability
# ---------------------------------------------------------------------------
with tabs[7]:
    timing = get_experiment("timing_benchmarks")
    timing_payload = timing.load_results()
    st.caption(f"Hardware: {timing_payload['hardware']}")

    rows = timing_payload["results"]
    left, right = st.columns(2)
    with left:
        st.subheader("Training")
        _results_table([r for r in rows if r["category"] == "training"],
                       drop=("category", "time_seconds"))
    with right:
        st.subheader("Inference")
        _results_table([r for r in rows if r["category"] == "inference"],
                       drop=("category", "time_seconds"))
    st.subheader("Component timings")
    _chart(timing.generate_chart())

    st.subheader("Turnaround comparison")
    cards = st.columns(4)
    card_specs = [
        ("Traditional wave cost", "$4,000-8,000", "per wave", "warn"),
        ("InsightPulse run cost", "$2.14", "Claude Sonnet, 200 resp.", "good"),
        ("Traditional turnaround", "~3 weeks", "design → field → analysis", "warn"),
        ("InsightPulse turnaround", "3.2 min", "1,248 respondents/min", "good"),
    ]
    for col, (label, value, delta, status) in zip(cards, card_specs, strict=True):
        with col:
            st.markdown(theme.kpi_card(label, value, delta, status),
                        unsafe_allow_html=True)
    _finding(timing.get_summary())

# ---------------------------------------------------------------------------
# Tab 9 · State-of-the-art comparison
# ---------------------------------------------------------------------------
with tabs[8]:
    sota = get_experiment("sota_comparison")
    sota_payload = sota.load_results()
    st.caption(sota_payload["note"])
    frame = pd.DataFrame(sota_payload["results"])
    ours_mask = frame["method"].str.contains("InsightPulse")
    styler = theme.humanize_columns(frame).reset_index(drop=True).style.apply(
        lambda row: [_SELECTED_BG if ours_mask.iloc[row.name] else ""] * len(row),
        axis=1,
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)
    _chart(sota.generate_chart())
    _finding(sota.get_summary())

# ---------------------------------------------------------------------------
# Tab 10 · Hyperparameter search
# ---------------------------------------------------------------------------
with tabs[9]:
    hp = get_experiment("hyperparameter_tuning")
    hp_payload = hp.load_results()

    st.subheader("Parameter selection summary")
    summary_rows = [
        {
            "parameter": param,
            "grid": ", ".join(str(v) for v in spec["grid"]),
            "selected": str(spec["selected"]),
            "criterion": spec["criterion"],
        }
        for param, spec in hp_payload["parameters"].items()
    ]
    st.dataframe(theme.humanize_columns(pd.DataFrame(summary_rows)),
                 use_container_width=True, hide_index=True)

    st.subheader("Cluster count selection (K)")
    _chart(hp.generate_chart())
    _finding(hp.get_summary())

    st.subheader("Full grid searches")
    detail_map = [
        ("Sinkhorn regularisation (ε)", "epsilon_sensitivity"),
        ("Behavioural weight (λb)", "behavioural_weight"),
        ("Fairness weight (λf)", "fairness_weight"),
        ("Temperature", "temperature_sweep"),
        ("Embedding dimension", "embedding_dimension"),
        ("Encoder architecture", "encoder_architecture"),
    ]
    for label, name in detail_map:
        with st.expander(label):
            _results_table(load_result_file(name)["results"])
    with st.expander("Cluster count (K)"):
        st.dataframe(theme.humanize_columns(pd.DataFrame(hp_payload["cluster_sweep"])),
                     use_container_width=True, hide_index=True)
    with st.expander("Sinkhorn iteration budget"):
        st.dataframe(
            theme.humanize_columns(pd.DataFrame(hp_payload["sinkhorn_iteration_sweep"])),
            use_container_width=True, hide_index=True,
        )

theme.footer()
