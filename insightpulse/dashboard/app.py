"""InsightPulse — application entrypoint.

Owns everything that runs on every rerun regardless of the current page:
page config, the sign-in gate, theme CSS, the shared sidebar, and the
role-aware navigation registry. Page content lives in ``views/home.py``
and ``pages/*.py``; each page re-checks its own access with
``auth.require_page`` so direct URLs cannot bypass the role model.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components import auth, theme

st.set_page_config(
    page_title="InsightPulse | Synthetic Survey Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "**InsightPulse** — Synthetic Survey Intelligence Platform\n\n"
            "IIT Madras × NielsenIQ · © 2025-2026"
        ),
    },
)

user = auth.require_auth()
theme.apply()
auth.render_sidebar(user)

# All pages are registered (so restricted URLs resolve to the Access
# Restricted card, not a 404); the sidebar hides what the role can't open.
pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page(
        "pages/1_🎯_Survey_Runner.py", title="Survey Runner",
        icon=":material/checklist:", url_path="survey-runner",
    ),
    st.Page(
        "pages/2_📊_Results.py", title="Results",
        icon=":material/monitoring:", url_path="results",
    ),
    st.Page(
        "pages/3_🧪_Experiments.py", title="Experiments",
        icon=":material/science:", url_path="experiments",
    ),
    st.Page(
        "pages/4_✅_Validation.py", title="Validation",
        icon=":material/verified:", url_path="validation",
    ),
    st.Page(
        "pages/5_📋_Audit.py", title="Audit",
        icon=":material/receipt_long:", url_path="audit",
    ),
    st.Page(
        "pages/6_👤_Profile.py", title="Profile",
        icon=":material/account_circle:", url_path="profile",
    ),
]

st.navigation(pages).run()
