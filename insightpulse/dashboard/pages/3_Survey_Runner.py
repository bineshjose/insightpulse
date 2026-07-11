"""Survey Runner — set up and execute a pulse survey against the twin panel."""

import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, demo_engine, nav, theme
from components.charts import PLOTLY_CONFIG, distribution_chart

sys.path.insert(0, str(data_loader.REPO_ROOT / "src"))
from insightpulse.core.models.survey import (
    CATEGORIES,
    CLIENTS,
    PRIORITIES,
    make_survey_id,
    suggest_contract_id,
)

user = auth.require_page("survey-runner")
theme.page_header(
    "Survey Runner",
    "Set up a pulse survey, select a target cohort, and execute it "
    "against a panel of digital-twin respondents.",
    "Survey Runner",
)

can_run = auth.has_permission(user, "run")

# Placeholder under the header — filled once live step state is known.
step_pills_slot = st.empty()

if not data_loader.require_data():
    theme.footer()
    st.stop()

panelists = data_loader.load_panelists()
catalog = data_loader.question_catalog()
question_by_text = {q["text"]: q for q in catalog}


def _humanize(value: str) -> str:
    """Display form for coded filter values: 'upper_middle' → 'Upper Middle'."""
    return str(value).replace("_", " ").title()


# ---------------------------------------------------------------------------
# 1 · Setup — business metadata
# ---------------------------------------------------------------------------

st.subheader("1 · Setup")

setup_col1, setup_col2 = st.columns([1.4, 1])
with setup_col1:
    survey_name = st.text_input(
        "Survey name", key="meta_name", placeholder="e.g. Organic Labeling Importance",
    )
    client_name = st.selectbox("Client", CLIENTS, key="meta_client")
    contract_id = st.text_input(
        "Contract ID",
        value=suggest_contract_id(client_name, 47),
        key=f"meta_contract_{client_name}",
    )
with setup_col2:
    category = st.selectbox("Category", CATEGORIES, key="meta_category")
    priority = st.selectbox("Priority", PRIORITIES, index=1, key="meta_priority")
    due_date = st.date_input("Due date (optional)", value=None, key="meta_due")

notes = st.text_area("Notes (optional)", key="meta_notes", height=68)
st.caption(
    f"Region: {', '.join(user['regions'])} · Executor: {user['name']} "
    f"({user['email']})"
)

# ---------------------------------------------------------------------------
# 2 · Questions
# ---------------------------------------------------------------------------

st.subheader("2 · Questions")
selected_texts = st.multiselect(
    "Survey questions (from the validated question bank)",
    options=list(question_by_text),
    default=list(question_by_text)[:2],
    help="Questions come from the historical survey bank so results can be "
         "validated against empirical ground truth.",
)

# ---------------------------------------------------------------------------
# 3 · Cohort
# ---------------------------------------------------------------------------

st.subheader("3 · Cohort")
col1, col2 = st.columns(2)
with col1:
    age_filter = st.multiselect(
        "Age groups", sorted(panelists["age_group"].unique()),
        default=[], format_func=_humanize,
    )
    income_filter = st.multiselect(
        "Income groups", sorted(panelists["income_group"].unique()),
        default=[], format_func=_humanize,
    )
with col2:
    region_filter = st.multiselect(
        "Regions", sorted(panelists["region"].unique()),
        default=[], format_func=_humanize,
    )
    archetype_filter = st.multiselect(
        "Behavioral archetypes",
        sorted(panelists["behavioral_archetype"].unique()),
        default=[], format_func=_humanize,
    )
max_cohort = min(500, user["max_cohort_size"])
cohort_size = st.slider(
    "Cohort size", min_value=10, max_value=max_cohort,
    value=min(100, max_cohort), step=10,
    help=f"Your {user['tier']} tier allows up to {user['max_cohort_size']:,} respondents.",
)

cohort_pool = data_loader.filter_cohort(
    panelists, age_filter, income_filter, region_filter, archetype_filter
)

# ---------------------------------------------------------------------------
# 4 · Config
# ---------------------------------------------------------------------------

st.subheader("4 · Config")
col3, col4 = st.columns(2)
with col3:
    model_options = list(demo_engine.MODEL_PROFILES)
    default_model = st.session_state.get("user_prefs", {}).get(
        "default_model", model_options[0]
    )
    model = st.selectbox(
        "LLM model", model_options,
        index=model_options.index(default_model) if default_model in model_options else 0,
    )
with col4:
    calibrate = st.toggle("Apply BDCL calibration", value=True)

mode = st.radio("Execution mode", ["Demo Mode", "Production Mode"], horizontal=True)

with st.expander("⚙ Advanced settings"):
    seed = st.number_input(
        "Random seed", value=42, min_value=0, step=1,
        help="Fixes the sampling so a run can be reproduced exactly.",
    )

# ---------------------------------------------------------------------------
# Step progression (rendered under the header, computed from live state)
# ---------------------------------------------------------------------------

step_done = [
    bool(survey_name.strip()),
    bool(selected_texts),
    not cohort_pool.empty,
    True,  # model/mode always have valid defaults
    demo_engine.get_last_run() is not None,
]
current = next((i for i, done in enumerate(step_done) if not done), 4)

_STEPS = ("1 · Setup", "2 · Questions", "3 · Cohort", "4 · Config", "5 · Review & Run")
pills = []
for i, step in enumerate(_STEPS):
    if step_done[i] and i != current:
        pills.append(
            f'<span class="niq-badge" style="background:{theme.GREEN};">✓ {step}</span>'
        )
    elif i == current:
        pills.append(
            f'<span class="niq-badge" style="background:{theme.BLUE};">{step}</span>'
        )
    else:
        pills.append(
            f'<span class="niq-badge" style="background:#E5E7EB;'
            f' color:{theme.TEXT_SECONDARY};">{step}</span>'
        )
step_pills_slot.markdown(
    f'<div class="niq-steps" style="display:flex; gap:0.5rem; flex-wrap:wrap;'
    f' margin-bottom:0.8rem;">{"".join(pills)}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 5 · Review & Run
# ---------------------------------------------------------------------------

st.subheader("5 · Review & Run")

effective_size = min(cohort_size, len(cohort_pool))
review = {
    "Survey": survey_name.strip() or "—",
    "Client": client_name,
    "Contract": contract_id,
    "Category": category,
    "Priority": priority,
    "Questions": len(selected_texts),
    "Cohort": f"{effective_size:,} respondents",
    "Model": model,
    "Mode": mode,
}
review_cells = "".join(
    f'<div style="min-width:140px;"><div style="color:{theme.TEXT_SECONDARY};'
    f' font-size:0.75rem; font-weight:600;">{label}</div>'
    f'<div style="color:{theme.TEXT}; font-weight:600;">{value}</div></div>'
    for label, value in review.items()
)
st.markdown(
    f'<div class="niq-card" style="display:flex; gap:1.4rem; flex-wrap:wrap;">'
    f"{review_cells}</div>",
    unsafe_allow_html=True,
)
st.markdown("")

ready = all(step_done[:4])
submitted = st.button(
    "Run Survey",
    use_container_width=True,
    type="primary",
    disabled=not (can_run and ready),
    help=(
        "Read-only access — contact administrator" if not can_run
        else None if ready
        else "Complete the highlighted step above to run"
    ),
)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

if submitted and can_run:
    if cohort_pool.empty:
        st.error("No panelists match the cohort filters — relax the criteria.")
        st.stop()

    questions = [question_by_text[t] for t in selected_texts]
    if effective_size < cohort_size:
        matches = len(cohort_pool)
        st.info(
            f"Only {matches} panelist{'s' if matches != 1 else ''} "
            f"{'match' if matches != 1 else 'matches'} the selected filters "
            f"(requested: {cohort_size}). Adjust filters for a larger cohort."
        )
    cohort = cohort_pool.sample(n=effective_size, random_state=int(seed))

    metadata = {
        "survey_id": make_survey_id(demo_engine.next_survey_sequence()),
        "survey_name": survey_name.strip(),
        "client_name": client_name,
        "contract_id": contract_id.strip(),
        "category": category,
        "region": ", ".join(user["regions"]),
        "priority": priority,
        "executor_name": user["name"],
        "executor_email": user["email"],
        "due_date": due_date.isoformat() if due_date else None,
        "notes": notes.strip(),
    }

    if mode == "Demo Mode":
        with st.spinner("Running the 8-agent pipeline..."):
            run = demo_engine.run_survey(
                questions, cohort, model,
                seed=int(seed), calibrate=calibrate, metadata=metadata,
            )
        demo_engine.store_run(run)
        auth.record_activity(
            "Ran survey",
            f"{metadata['survey_name']} ({effective_size} respondents, {model})",
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
                st.success(f"Run complete: {result.get('total_responses', 0)} responses.")
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
# Run summary
# ---------------------------------------------------------------------------

run = demo_engine.get_last_run()
if run is None:
    theme.footer()
    st.stop()

totals = run["totals"]
meta = run.get("metadata") or {}
run_title = meta.get("survey_name") or theme.run_label(run["run_id"])
run_sub = meta.get("survey_id", "")
st.divider()
st.markdown(
    f'### <span title="Run {run["run_id"]}">{run_title}</span>'
    f'<span style="color:{theme.TEXT_SECONDARY}; font-size:0.95rem; font-weight:500;">'
    f" &nbsp;{run_sub} · {run['config']['model']}</span>",
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
st.markdown(f"#### {first['text']}")
st.plotly_chart(
    distribution_chart(first["options"], series),
    use_container_width=True,
    config=PLOTLY_CONFIG,
)
nav.page_link("pages/4_Results.py", label="→ Full results, metrics, and breakdowns")

theme.footer()
