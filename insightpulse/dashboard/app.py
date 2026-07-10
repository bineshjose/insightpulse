"""InsightPulse — Streamlit Dashboard (home page).

The five workflow tabs live as dedicated pages under ``dashboard/pages/``:
    1. 🎯 Survey Runner — configure and execute surveys
    2. 📊 Results — distributions, metrics, demographic breakdowns
    3. 🧪 Experiments — multi-LLM comparison, calibration, drift, sequence
    4. ✅ Validation — cross-validation against empirical ground truth
    5. 📋 Audit — agent trace, provenance hash, reproducibility

This home page shows the panel overview and system status.
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components import data_loader

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="InsightPulse",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
ENV = os.getenv("ENV", "demo")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 InsightPulse")
    st.caption("Synthetic Panelist Pulse Surveys")
    st.divider()

    st.markdown(f"**Environment:** `{ENV}`")
    st.markdown(f"**API:** `{API_URL}`")

    st.divider()

    st.markdown("""
    **Architecture:**
    - L1: Data Layer
    - L2: Embedding (ℝ¹²⁸)
    - L3: Digital Twin Generation
    - L4: BDCL Calibration
    - L5: Agent Orchestration
    """)

    st.divider()
    st.caption("M.Tech Thesis — IIT Madras + NielsenIQ")
    st.caption("Binesh Jose (CH24M521)")

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

st.title("InsightPulse Dashboard")
st.markdown(
    "Generate survey-grade synthetic consumer responses using "
    "LLM-based digital twins calibrated via optimal transport."
)

# --- Data status ---
if data_loader.data_available():
    panelists = data_loader.load_panelists()
    purchases = data_loader.load_purchases()
    responses = data_loader.load_survey_responses()

    tiles = st.columns(4)
    tiles[0].metric("Panel households", f"{len(panelists):,}")
    tiles[1].metric("Purchase records", f"{len(purchases):,}")
    tiles[2].metric("Historical responses", f"{len(responses):,}")
    tiles[3].metric("Behavioral archetypes", panelists["behavioral_archetype"].nunique())
else:
    st.warning(
        "Synthetic sample data not found — generate it with `make generate-data`.",
        icon="⚠️",
    )

st.divider()

# --- Navigation ---
st.subheader("Workflow")

col1, col2 = st.columns(2)
with col1:
    st.page_link(
        "pages/1_🎯_Survey_Runner.py", label="🎯 **Survey Runner** — "
        "configure a cohort and execute a pulse survey",
    )
    st.page_link(
        "pages/2_📊_Results.py", label="📊 **Results** — "
        "distributions, metrics, and demographic breakdowns",
    )
    st.page_link(
        "pages/3_🧪_Experiments.py", label="🧪 **Experiments** — "
        "multi-LLM comparison, calibration convergence, drift",
    )
with col2:
    st.page_link(
        "pages/4_✅_Validation.py", label="✅ **Validation** — "
        "cross-validation against empirical ground truth",
    )
    st.page_link(
        "pages/5_📋_Audit.py", label="📋 **Audit** — "
        "agent trace, provenance, reproducibility",
    )
