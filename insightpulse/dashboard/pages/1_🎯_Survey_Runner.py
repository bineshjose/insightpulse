"""Survey Runner — configure and execute a synthetic survey.

Two execution modes:
- Local simulation: full pipeline demo against the synthetic sample data,
  no API or LLM keys needed.
- API pipeline: submits to the FastAPI service running the real 8-agent
  LangGraph DAG.
"""

import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, nav, simulation, theme
from components.charts import distribution_chart

st.set_page_config(page_title="Survey Runner | InsightPulse", page_icon="🎯", layout="wide")

user = auth.require_auth("run")
theme.apply()
auth.render_sidebar(user)
theme.page_header(
    "🎯 Survey Runner",
    "Configure a pulse survey, select a target cohort, and execute it "
    "against a panel of digital-twin respondents.",
    "Survey Runner",
)

st.markdown(
    f"""
<div style="display:flex; gap:0.5rem; margin-bottom:0.8rem; flex-wrap:wrap;">
  {"".join(f'<span class="niq-badge" style="background:{theme.NAVY};">{step}</span>'
           for step in ("1 · Questions", "2 · Cohort", "3 · Configure", "4 · Run"))}
</div>
""",
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
    col3, col4, col5 = st.columns(3)
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
        seed = st.number_input("Random seed", value=42, min_value=0, step=1)
    with col5:
        calibrate = st.toggle("Apply BDCL calibration", value=True)

    mode = st.radio(
        "Execution mode",
        ["Local simulation (no API required)", "API pipeline (LangGraph agents)"],
        horizontal=True,
    )

    submitted = st.form_submit_button("🚀 Run Survey", use_container_width=True)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

if submitted:
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

    if mode.startswith("Local"):
        with st.spinner("Running the 8-agent pipeline (local simulation)..."):
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
                    "Start it with `make demo`, or use Local simulation mode."
                )
                st.stop()
            except httpx.HTTPStatusError as exc:
                st.error(f"API error {exc.response.status_code}: {exc.response.text[:500]}")
                st.stop()

# ---------------------------------------------------------------------------
# Run summary (local simulation)
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
st.subheader(f"Run `{run['run_id']}` — {run['config']['model']}")

tiles = st.columns(5)
tiles[0].metric("Responses", f"{totals['total_responses']:,}")
tiles[1].metric(
    "Valid",
    f"{totals['valid_responses']:,}",
    f"{totals['valid_responses'] / max(totals['total_responses'], 1):.1%} of total",
    delta_color="off",
)
tiles[2].metric("Hallucination rate", f"{totals['hallucination_rate']:.1%}")
tiles[3].metric("Consistency", f"{totals['consistency_score']:.1%}")
tiles[4].metric("Est. cost", f"${totals['total_cost_usd']:.2f}")

first = run["question_results"][0]
raw_total = max(sum(first["raw_counts"]), 1)
emp_total = max(sum(first["empirical_counts"]), 1)
series = {
    "Synthetic (raw)": [100 * c / raw_total for c in first["raw_counts"]],
    "Empirical": [100 * c / emp_total for c in first["empirical_counts"]],
}
st.plotly_chart(
    distribution_chart(first["options"], series, title=f"Preview — {first['text']}"),
    use_container_width=True,
)
nav.page_link("pages/2_📊_Results.py", label="→ Full results, metrics, and breakdowns")

theme.footer()
