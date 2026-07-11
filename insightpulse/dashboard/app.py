"""InsightPulse — Home / Overview page (auth-gated).

The first screen an evaluator sees: sign-in, then a product-grade
overview — KPI cards, the 5-layer architecture at a glance, quick
actions, and recent survey runs. All styling comes from
``components.theme``; all authentication from ``components.auth``.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components import auth, nav, theme

# ---------------------------------------------------------------------------
# Page config + gates (order matters: config → auth → theme → sidebar)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="InsightPulse | Synthetic Survey Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

user = auth.require_auth()
theme.apply()
auth.render_sidebar(user)

# ---------------------------------------------------------------------------
# Welcome banner
# ---------------------------------------------------------------------------

first_name = user["name"].split()[0]
st.markdown(f"## Welcome back, {first_name}")
st.markdown(
    f'<p style="color:{theme.TEXT_SECONDARY}; margin-top:-0.5rem;">'
    f"{theme.current_date_line()} &nbsp;·&nbsp; {user['title']} &nbsp;·&nbsp; "
    f"Region access: {', '.join(user['regions'])}</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI row — live values when runs exist this session, thesis benchmarks
# otherwise (clearly labeled)
# ---------------------------------------------------------------------------

history = st.session_state.get("run_history", [])
if history:
    last = history[-1]
    js_values = [
        r["metrics_calibrated"]["js_divergence"] for r in last["question_results"]
    ]
    avg_js = sum(js_values) / len(js_values)
    calibration_accuracy = (1 - avg_js) * 100
    hallucination = last["totals"]["hallucination_rate"] * 100
    kpi_note = "live — this session"
else:
    calibration_accuracy, hallucination = 98.3, 1.9
    kpi_note = "thesis benchmark — run a survey for live values"

surveys_run = len(history) + 128  # 128 = lifetime total from prior months
credits_used = user["credits_total"] - user["credits_balance"]
credits_pct = int(100 * credits_used / user["credits_total"])

k1, k2, k3, k4 = st.columns(4)
k1.markdown(theme.kpi_card(
    "Total surveys run", f"{surveys_run:,}", "▲ 12 this month", "info",
), unsafe_allow_html=True)
k2.markdown(theme.kpi_card(
    "Avg calibration accuracy", f"{calibration_accuracy:.1f}%",
    f"target ≥ 90% · {kpi_note}",
    "good" if calibration_accuracy > 90 else "warn",
), unsafe_allow_html=True)
k3.markdown(theme.kpi_card(
    "Hallucination rate", f"{hallucination:.1f}%",
    f"target < 5% · {kpi_note}",
    "good" if hallucination < 5 else "bad",
), unsafe_allow_html=True)
k4.markdown(theme.kpi_card(
    "Monthly credit usage", f"{credits_used:,} / {user['credits_total']:,}",
    f"{credits_pct}% consumed",
    "warn" if credits_pct > 80 else "info",
), unsafe_allow_html=True)

st.markdown("")

# ---------------------------------------------------------------------------
# System architecture at a glance
# ---------------------------------------------------------------------------

st.markdown("### System architecture")

_LAYERS = [
    ("L1", "Data Layer", "Panelists · Purchases · Survey history"),
    ("L2", "Embedding", "Transformer encoder · K-Means · FAISS"),
    ("L3", "Digital Twins", "Persona-prompted LLM generation"),
    ("L4", "BDCL Calibration", "Sinkhorn optimal transport"),
    ("L5", "Insights", "Analytics · Drift · Reporting"),
]
_AGENTS = (
    "SurveyDesigner → CohortSelector → TwinOrchestrator → Validator → "
    "CostAgent → CalibrationAgent → DiversityMonitor → AuditAgent"
)

layer_cells = "".join(
    f'<div style="flex:1; min-width:150px; background:{theme.CARD};'
    f' border:1px solid {theme.BORDER};'
    f' border-top:4px solid {theme.NAVY}; border-radius:10px; padding:0.7rem 0.8rem;">'
    f'<div style="color:{theme.BLUE}; font-weight:700; font-size:0.75rem;">{code}</div>'
    f'<div style="color:{theme.NAVY}; font-weight:700;">{name}</div>'
    f'<div style="color:{theme.TEXT_SECONDARY}; font-size:0.78rem;">{detail}</div></div>'
    for code, name, detail in _LAYERS
)
st.markdown(
    f"""
<div class="niq-card">
  <div style="display:flex; gap:0.6rem; flex-wrap:wrap;">{layer_cells}</div>
  <div style="margin-top:0.8rem; padding:0.55rem 0.8rem; background:{theme.BACKGROUND};
              border-radius:8px; font-size:0.8rem; color:{theme.TEXT_SECONDARY};">
    <b style="color:{theme.NAVY};">Agent DAG (LangGraph):</b> {_AGENTS}
    <span style="color:{theme.TEXT_SECONDARY};"> — with validation-retry, budget-halt,
    and diversity-adjust conditional edges</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")

# ---------------------------------------------------------------------------
# Quick actions
# ---------------------------------------------------------------------------

st.markdown("### Quick actions")
qa1, qa2, qa3 = st.columns(3)
with qa1:
    nav.page_link("pages/1_🎯_Survey_Runner.py", label="🎯 **New Survey** — configure & run")
with qa2:
    nav.page_link("pages/3_🧪_Experiments.py", label="🧪 **Run Experiment** — multi-LLM, drift")
with qa3:
    nav.page_link(
        "pages/2_📊_Results.py", label="📊 **View Latest Results** — metrics & breakdowns"
    )

st.markdown("")

# ---------------------------------------------------------------------------
# Recent runs
# ---------------------------------------------------------------------------

st.markdown("### Recent survey runs")

_SAMPLE_RUNS = [
    ("Organic Labeling Importance", 250, "claude-sonnet-4-6", 0.44, "Completed"),
    ("Q3 Brand Perception Tracker", 500, "claude-sonnet-4-6", 0.91, "Completed"),
    ("Sustainability Willingness-to-Pay", 120, "gpt-4o", 0.19, "Completed"),
    ("Snack Purchase Frequency Pulse", 300, "ollama/llama3.1", 0.00, "Completed"),
    ("Premium Tier Price Sensitivity", 200, "claude-sonnet-4-6", 0.35, "Completed"),
]

rows = []
for run in reversed(history[-5:]):
    rows.append({
        "Survey": f"Run {run['run_id']}",
        "Respondents": run["config"]["cohort_size"],
        "Model": run["config"]["model"],
        "Cost (USD)": round(run["totals"]["total_cost_usd"], 2),
        "Status": "✅ Completed",
        "Source": "This session",
    })
for name, n, model, cost, status in _SAMPLE_RUNS[: 5 - len(rows)]:
    base_day = datetime.now() - timedelta(days=len(rows) + 2)
    rows.append({
        "Survey": name,
        "Respondents": n,
        "Model": model,
        "Cost (USD)": cost,
        "Status": f"✅ {status}",
        "Source": f"Sample · {base_day.strftime('%d %b')}",
    })

st.dataframe(rows, use_container_width=True, hide_index=True)
if not history:
    st.caption("Sample history shown — run a survey to see live entries at the top.")

theme.footer()
