"""Survey Runner — configure and execute a synthetic survey.

Two execution modes:
- Demo Mode: full pipeline against the synthetic sample data, simulated
  responses, no API or LLM keys needed.
- Production Mode: submits to the FastAPI service running the real
  8-agent LangGraph DAG (live LLM calls).
"""

import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, nav, simulation, theme
from components.charts import PLOTLY_CONFIG, distribution_chart

user = auth.require_page("survey-runner")
theme.page_header(
    "Survey Runner",
    "Configure a pulse survey, select a target cohort, and execute it "
    "against a panel of digital-twin respondents.",
    "Survey Runner",
)

can_run = auth.has_permission(user, "run")

# Step pills: the active step is NIQ blue, inactive steps light gray.
_STEPS = ("1 · Questions", "2 · Cohort", "3 · Configure", "4 · Run")
active_step = 3 if simulation.get_last_run() is not None else 0
step_pills = "".join(
    f'<span class="niq-badge" style="background:{theme.BLUE};">{step}</span>'
    if i == active_step else
    f'<span class="niq-badge" style="background:#E5E7EB;'
    f' color:{theme.TEXT_SECONDARY};">{step}</span>'
    for i, step in enumerate(_STEPS)
)
st.markdown(
    f'<div style="display:flex; gap:0.5rem; margin-bottom:0.8rem;'
    f' flex-wrap:wrap;">{step_pills}</div>',
    unsafe_allow_html=True,
)

if not data_loader.require_data():
    theme.footer()
    st.stop()

panelists = data_loader.load_panelists()
catalog = data_loader.question_catalog()
question_by_text = {q["text"]: q for q in catalog}

# ---------------------------------------------------------------------------
# Configuration form
# ---------------------------------------------------------------------------

with st.form("survey_config"):
    st.subheader("1 · Questions")
    selected_texts = st.multiselect(
        "Survey questions (from the validated question bank)",
        options=list(question_by_text),
        default=list(question_by_text)[:2],
        help="Questions come from the historical survey bank so synthetic "
             "results can be validated against empirical ground truth.",
    )

    st.subheader("2 · Cohort")
    col1, col2 = st.columns(2)
    with col1:
        age_filter = st.multiselect(
            "Age groups", sorted(panelists["age_group"].unique()), default=[]
        )
        income_filter = st.multiselect(
            "Income groups", sorted(panelists["income_group"].unique()), default=[]
        )
    with col2:
        region_filter = st.multiselect(
            "Regions", sorted(panelists["region"].unique()), default=[]
        )
        archetype_filter = st.multiselect(
            "Behavioral archetypes",
            sorted(panelists["behavioral_archetype"].unique()),
            default=[],
        )
    max_cohort = min(500, user["max_cohort_size"])
    cohort_size = st.slider(
        "Cohort size", min_value=10, max_value=max_cohort,
        value=min(100, max_cohort), step=10,
        help=f"Your {user['tier']} tier allows up to {user['max_cohort_size']:,} respondents.",
    )

    st.subheader("3 · Execution")
    col3, col4 = st.columns(2)
    with col3:
        model_options = list(simulation.MODEL_PROFILES)
        default_model = st.session_state.get("user_prefs", {}).get(
            "default_model", model_options[0]
        )
        model = st.selectbox(
            "LLM model", model_options,
            index=model_options.index(default_model) if default_model in model_options else 0,
        )
    with col4:
        calibrate = st.toggle("Apply BDCL calibration", value=True)

    mode = st.radio(
        "Execution mode",
        [
            "Demo Mode — simulated responses, no API key required",
            "Production Mode — live LLM calls via API",
        ],
        horizontal=True,
    )

    with st.expander("⚙ Advanced settings"):
        seed = st.number_input(
            "Random seed", value=42, min_value=0, step=1,
            help="Fixes the sampling so a run can be reproduced exactly.",
        )

    submitted = st.form_submit_button(
        "Run Survey",
        use_container_width=True,
        disabled=not can_run,
        help=None if can_run else "Read-only access — contact administrator",
    )

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

if submitted and can_run:
    if not selected_texts:
        st.error("Select at least one question.")
        st.stop()

    questions = [question_by_text[t] for t in selected_texts]
    cohort_pool = data_loader.filter_cohort(
        panelists, age_filter, income_filter, region_filter, archetype_filter
    )

    if cohort_pool.empty:
        st.error("No panelists match the cohort filters — relax the criteria.")
        st.stop()

    effective_size = min(cohort_size, len(cohort_pool))
    if effective_size < cohort_size:
        st.info(
            f"Only {len(cohort_pool)} panelists match the filters; "
            f"running with {effective_size} instead of {cohort_size}."
        )
    cohort = cohort_pool.sample(n=effective_size, random_state=int(seed))

    if mode.startswith("Demo"):
        with st.spinner("Running the 8-agent pipeline (demo mode)..."):
            run = simulation.simulate_survey_run(
                questions, cohort, model, seed=int(seed), calibrate=calibrate
            )
        simulation.store_run(run)
        auth.record_activity(
            "Ran survey",
            f"{len(questions)} question(s), {effective_size} respondents, {model}",
        )
    else:
        payload = {
            "questions": [q["text"] for q in questions],
            "cohort_size": effective_size,
            "context": "US consumer goods market",
            "models": [model],
            "seed": int(seed),
        }
        with st.spinner("Running survey through the API pipeline..."):
            try:
                result = data_loader.post_survey_run(payload)
                st.success(f"API run complete: {result.get('total_responses', 0)} responses.")
                st.session_state["last_api_result"] = result
                st.json(result)
                st.stop()
            except httpx.ConnectError:
                st.error(
                    f"Could not connect to the API at `{data_loader.API_URL}`. "
                    "Start it with `make demo`, or use Demo Mode."
                )
                st.stop()
            except httpx.HTTPStatusError as exc:
                st.error(f"API error {exc.response.status_code}: {exc.response.text[:500]}")
                st.stop()

# ---------------------------------------------------------------------------
# Run summary (demo mode)
# ---------------------------------------------------------------------------

run = simulation.get_last_run()
if run is None:
    st.info(
        "No results yet — configure a survey above and press **Run Survey** "
        "to get started.",
        icon="✨",
    )
    theme.footer()
    st.stop()

totals = run["totals"]
st.divider()
st.markdown(
    f'### <span title="Full run ID: {run["run_id"]}">{theme.run_label(run["run_id"])}</span>'
    f' — {run["config"]["model"]}',
    unsafe_allow_html=True,
)

halluc_pct = totals["hallucination_rate"] * 100
consistency_pct = totals["consistency_score"] * 100
valid_share = totals["valid_responses"] / max(totals["total_responses"], 1)

tiles = st.columns(5)
tiles[0].markdown(theme.kpi_card(
    "Responses", f"{totals['total_responses']:,}", "generated this run", "info",
), unsafe_allow_html=True)
tiles[1].markdown(theme.kpi_card(
    "Valid", f"{totals['valid_responses']:,}", f"{valid_share:.1%} of total",
    "good" if valid_share > 0.93 else "warn",
), unsafe_allow_html=True)
tiles[2].markdown(theme.kpi_card(
    "Hallucination rate", f"{halluc_pct:.1f}%", "target < 5%",
    "good" if halluc_pct < 5 else ("warn" if halluc_pct <= 8 else "bad"),
), unsafe_allow_html=True)
tiles[3].markdown(theme.kpi_card(
    "Consistency", f"{consistency_pct:.1f}%", "target > 90%",
    "good" if consistency_pct > 90 else ("warn" if consistency_pct >= 80 else "bad"),
), unsafe_allow_html=True)
tiles[4].markdown(theme.kpi_card(
    "Est. cost", f"${totals['total_cost_usd']:.2f}", "this run", "neutral",
), unsafe_allow_html=True)

first = run["question_results"][0]
raw_total = max(sum(first["raw_counts"]), 1)
emp_total = max(sum(first["empirical_counts"]), 1)
series = {
    "Synthetic (raw)": [100 * c / raw_total for c in first["raw_counts"]],
    "Empirical": [100 * c / emp_total for c in first["empirical_counts"]],
}
st.markdown("")
st.markdown(f"#### Preview — {first['text']}")
st.plotly_chart(
    distribution_chart(first["options"], series),
    use_container_width=True,
    config=PLOTLY_CONFIG,
)
nav.page_link("pages/2_📊_Results.py", label="→ Full results, metrics, and breakdowns")

theme.footer()
