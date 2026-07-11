"""Enterprise login and role-based access for the dashboard.

Demo-grade authentication: four hardcoded, realistic accounts held
in-memory, session-state persistence, and a permissions model that the
pages consult for role-based access. This is deliberately NOT a security
boundary — the demo deployment has no user database — but the flow (gate →
session → role checks → logout) mirrors how the production tool would sit
behind the ingress OAuth proxy (see docs/api.md, Operational notes).

Usage (top of every page):
    user = auth.require_auth()          # login gate
    theme.apply()                       # NIQ styling
    auth.render_sidebar(user)           # shared chrome
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

from components import theme

_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

# ---------------------------------------------------------------------------
# Accounts (demo-grade; a real deployment authenticates at the ingress)
# ---------------------------------------------------------------------------

USERS: dict[str, dict[str, Any]] = {
    "binesh.jose@nielseniq.com": {
        "password": "Ch24m521",
        "name": "Binesh Jose",
        "initials": "BJ",
        "role": "Platform Administrator",
        "title": "ML Engineer — Consumer Intelligence",
        "department": "Data Science & Analytics",
        "regions": ["APAC", "EMEA", "Americas"],
        "permissions": {"view", "create", "run", "analyze", "export", "calibrate"},
        "tier": "Enterprise",
        "credits_total": 3000,
        "credits_balance": 1847,
        "api_calls_remaining": 3_247,
        "api_calls_quota": 10_000,
        "max_cohort_size": 5000,
        "created": "2025-08-14",
    },
    "analyst@nielseniq.com": {
        "password": "analyst123",
        "name": "Priya Sharma",
        "initials": "PS",
        "role": "Survey Analyst",
        "title": "Consumer Research Analyst",
        "department": "Market Research — FMCG Division",
        "regions": ["APAC"],
        "permissions": {"view", "create", "run", "analyze"},
        "tier": "Professional",
        "credits_total": 3000,
        "credits_balance": 1850,
        "api_calls_remaining": 6_753,
        "api_calls_quota": 10_000,
        "max_cohort_size": 1000,
        "created": "2025-10-21",
    },
    "evaluator@iitm.ac.in": {
        "password": "eval2024",
        "name": "External Evaluator",
        "initials": "EV",
        "role": "Read-Only Evaluator",
        "title": "Faculty Reviewer",
        "department": "Academic Review Board",
        "regions": ["APAC"],
        "permissions": {"view", "analyze"},
        "tier": "Academic",
        "credits_total": 600,
        "credits_balance": 500,
        "api_calls_remaining": 1_000,
        "api_calls_quota": 1_000,
        "max_cohort_size": 200,
        "created": "2025-11-02",
    },
    "demo@insightpulse.ai": {
        "password": "demo123",
        "name": "Demo User",
        "initials": "DU",
        "role": "Survey Analyst",
        "title": "Consumer Research Analyst",
        "department": "Market Research",
        "regions": ["Americas"],
        "permissions": {"view", "create", "run", "analyze"},
        "tier": "Professional",
        "credits_total": 1200,
        "credits_balance": 1000,
        "api_calls_remaining": 5_000,
        "api_calls_quota": 5_000,
        "max_cohort_size": 1000,
        "created": "2025-09-30",
    },
}

# Pages each role may open, keyed by the page slug used in app.py's
# st.navigation registry. Restricted pages are hidden from the sidebar
# (see render_sidebar) AND gated at render time (see require_page), so a
# direct URL lands on the Access Restricted card instead of the content.
PAGE_ACCESS: dict[str, set[str]] = {
    "Platform Administrator": {
        "home", "survey-runner", "results", "experiments",
        "validation", "audit", "profile",
    },
    # Analysts (and the demo account) work with surveys and results only.
    "Survey Analyst": {"home", "survey-runner", "results", "profile"},
    # Evaluators see everything, but the Survey Runner is read-only for
    # them (the Run button is disabled — they lack the "run" permission).
    "Read-Only Evaluator": {
        "home", "survey-runner", "results", "experiments",
        "validation", "audit", "profile",
    },
}

ALL_PAGE_SLUGS: set[str] = {
    "home", "survey-runner", "results", "experiments", "validation", "audit", "profile",
}


def allowed_pages(user: dict[str, Any]) -> set[str]:
    """Page slugs this user's role may open."""
    return PAGE_ACCESS.get(user["role"], {"home", "profile"})

TIER_FEATURES: dict[str, list[str]] = {
    "Enterprise": [
        "Unlimited surveys & experiments",
        "Full calibration controls (ε, λ_b, λ_f)",
        "All LLM models incl. local Ollama",
        "CSV / JSON export & API access",
        "Priority support & audit retention",
    ],
    "Professional": [
        "Create, run & analyze surveys",
        "Standard calibration presets",
        "Claude + GPT model families",
        "Dashboards & demographic breakdowns",
    ],
    "Academic": [
        "View & analyze shared surveys",
        "Full metric & validation access",
        "Experiment result exploration",
        "Read-only audit trails",
    ],
}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def current_user() -> dict[str, Any] | None:
    """The authenticated user's profile, or None."""
    return st.session_state.get("auth_user")


def login(email: str, password: str) -> bool:
    """Validate credentials and open a session.

    Args:
        email: Account email (case-insensitive).
        password: Account password.

    Returns:
        True on success (session established).
    """
    account = USERS.get(email.strip().lower())
    if account is None or account["password"] != password:
        return False
    profile = {k: v for k, v in account.items() if k != "password"}
    profile["email"] = email.strip().lower()
    profile["last_login"] = datetime.now().strftime("%b %d, %Y · %I:%M %p")
    st.session_state["auth_user"] = profile
    # Editable preferences live separately so "Save Changes" never mutates
    # the account fixture.
    st.session_state.setdefault("user_prefs", {
        "display_name": profile["name"],
        "title": profile["title"],
        "preferred_region": profile["regions"][0],
        "default_cohort_size": min(100, profile["max_cohort_size"]),
        "default_model": "claude-sonnet-4-6",
        "notify_email": True,
        "notify_completion": True,
        "notify_digest": False,
    })
    record_activity("Signed in", f"Session started for {profile['name']}")
    return True


def logout() -> None:
    """End the session and drop per-user state."""
    for key in ("auth_user", "user_prefs", "activity_log"):
        st.session_state.pop(key, None)


def has_permission(user: dict[str, Any], permission: str) -> bool:
    """Check one permission against the user's grant set."""
    return permission in user.get("permissions", set())


_ACTIVITY_TIME_FORMAT = "%b %d, %I:%M %p"


def record_activity(action: str, details: str) -> None:
    """Append one entry to the session activity log (shown on Profile)."""
    log = st.session_state.setdefault("activity_log", [])
    log.append({
        "timestamp": datetime.now().strftime(_ACTIVITY_TIME_FORMAT),
        "action": action,
        "details": details,
    })


def activity_log() -> list[dict[str, str]]:
    """The session's activity entries, newest first (seeded for realism)."""
    seeded = _seed_activity()
    live = list(reversed(st.session_state.get("activity_log", [])))
    return (live + seeded)[:12]


def _seed_activity() -> list[dict[str, str]]:
    """Deterministic historical entries so the log never looks empty.

    Varied actions, topics, and spacing so the log reads like real usage
    rather than a repeated fixture row.
    """
    base = datetime.now()
    entries = [
        (timedelta(hours=2, minutes=41),
         "Ran survey", "Organic Labeling Importance (250 respondents, claude-sonnet-4-6)"),
        (timedelta(hours=3, minutes=58),
         "Ran experiment", "Multi-LLM Comparison (4 models)"),
        (timedelta(days=1, hours=1, minutes=12),
         "Exported results", "Q3 Brand Perception Tracker — CSV"),
        (timedelta(days=1, hours=3, minutes=47),
         "Ran survey", "Price Sensitivity Pulse (200 respondents, gpt-4o)"),
        (timedelta(days=2, hours=0, minutes=25),
         "Updated calibration settings", "Sinkhorn ε 0.05 → 0.1 for production preset"),
        (timedelta(days=2, hours=3, minutes=55),
         "Ran survey", "Sustainability Willingness-to-Pay (120 respondents)"),
        (timedelta(days=3, hours=6, minutes=30),
         "Reviewed audit trail", "SRV-2026-00138 — provenance and quality gates"),
        (timedelta(days=4, hours=2, minutes=5),
         "Ran survey", "Premium Tier Price Sensitivity (200 respondents)"),
        (timedelta(days=6, hours=4, minutes=18),
         "Generated drift report", "Category-mix drift vs 3-month baseline"),
        (timedelta(days=8, hours=1, minutes=36),
         "Adjusted cohort filters", "APAC region · Premium Loyalist archetype"),
        (timedelta(days=11, hours=5, minutes=49),
         "Updated settings", "Default model → claude-sonnet-4-6"),
    ]
    return [
        {
            "timestamp": (base - offset).strftime(_ACTIVITY_TIME_FORMAT),
            "action": action,
            "details": details,
        }
        for offset, action, details in entries
    ]


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

def _asset_data_uri(name: str) -> str:
    """Read an SVG asset as a data URI (inline, CSP-friendly)."""
    path = _ASSETS_DIR / name
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/svg+xml;base64,{encoded}"


def login_page() -> None:
    """Render the centered login page (call when unauthenticated)."""
    st.markdown(
        f"""
<style>
[data-testid="stAppViewContainer"] {{
    background:
        url('{_asset_data_uri("login_bg.svg")}') center / cover no-repeat,
        linear-gradient(135deg, {theme.NAVY} 0%, {theme.NAVY_LIGHT} 100%);
}}
[data-testid="stHeader"], [data-testid="stSidebar"] {{ display: none; }}
div[data-testid="stForm"] {{
    background: #FFFFFF;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    box-shadow: 0 12px 28px rgba(0, 20, 40, 0.35), 0 28px 80px rgba(0, 20, 40, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.65);
}}
/* No "Press Enter to submit form" hints on the credential fields. */
.stTextInput div[data-testid="InputInstructions"] {{ display: none; }}
/* One eye icon only: suppress the browser-native password reveal and any
   duplicate toggle so exactly one functional eye remains. */
input::-ms-reveal, input::-ms-clear {{ display: none !important; }}
[data-testid="stPasswordInputToggle"] button:first-child {{ display: none; }}
[data-testid="stTextInputRootElement"] button:nth-of-type(2) {{ display: none; }}
.login-tagline {{
    color: rgba(255,255,255,0.85);
    text-align: center;
    font-size: 1rem;
    letter-spacing: 0.06em;
    margin: 0.4rem 0 1.6rem 0;
}}
.login-attribution {{
    color: rgba(255,255,255,0.65);
    text-align: center;
    font-size: 0.8rem;
    margin-top: 2.2rem;
}}
.login-links {{ font-size: 0.85rem; }}
</style>
""",
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        logo = _asset_data_uri("logo.svg")
        if logo:
            st.markdown(
                f'<div style="text-align:center; padding-top:2.5rem;">'
                f'<img src="{logo}" alt="InsightPulse" style="width:280px; max-width:90%;'
                f' background:#FFFFFF; padding:14px 18px; border-radius:14px;"/></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="login-tagline">SYNTHETIC SURVEY INTELLIGENCE PLATFORM</div>',
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            email = st.text_input("Email", key="login_email",
                                  placeholder="you@nielseniq.com")
            password = st.text_input("Password", key="login_password",
                                     type="password", placeholder="••••••••")
            remember_col, forgot_col = st.columns([1, 1])
            with remember_col:
                st.checkbox("Remember me", key="login_remember")
            with forgot_col:
                st.markdown(
                    '<div class="login-links" style="text-align:right; padding-top:0.4rem;">'
                    '<a href="#" style="color:#00A4E4;">Forgot password?</a></div>',
                    unsafe_allow_html=True,
                )
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if login(email, password):
                st.rerun()
            else:
                st.error("Invalid email or password")

        st.markdown(
            '<div class="login-attribution">IIT Madras &nbsp;×&nbsp; NielsenIQ<br/>'
            "M.Tech Industrial AI Project — Binesh Jose (CH24M521)</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Gate + shared sidebar
# ---------------------------------------------------------------------------

def require_auth() -> dict[str, Any]:
    """Sign-in gate for the app entrypoint.

    Returns:
        The authenticated user profile; renders the login page and stops
        otherwise.
    """
    user = current_user()
    if user is None:
        login_page()
        st.stop()
    return user


def require_page(slug: str) -> dict[str, Any]:
    """Role gate at the top of every page.

    Args:
        slug: The page's slug in the st.navigation registry
            (e.g. "experiments").

    Returns:
        The authenticated user profile. Renders the Access Restricted card
        and stops when the user's role may not open this page.
    """
    user = require_auth()
    if slug not in allowed_pages(user):
        _access_restricted(user)
        st.stop()
    return user


def _access_restricted(user: dict[str, Any]) -> None:
    """Render the Access Restricted card (reached only via direct URL)."""
    st.markdown(
        f"""
<div class="niq-card" style="max-width:520px; margin:9vh auto 1.2rem auto;
            text-align:center; padding:2.6rem 2.4rem;">
  <div style="font-size:2.6rem; line-height:1;">🔒</div>
  <h3 style="margin:0.8rem 0 0.4rem 0;">Access Restricted</h3>
  <p style="color:{theme.TEXT_SECONDARY}; margin:0.2rem 0;">
    You don't have permission to view this page.</p>
  <p style="margin:0.5rem 0;">Your role: <b>{user["role"]}</b></p>
  <p style="color:{theme.TEXT_SECONDARY}; font-size:0.9rem; margin:0.2rem 0;">
    Contact your administrator for elevated access.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1.6, 1, 1.6])
    with center:
        if st.button("Go to Dashboard", use_container_width=True):
            st.switch_page("views/home.py")


@st.cache_data(ttl=30, show_spinner=False)
def _api_healthy(api_url: str) -> bool:
    """Probe the API health endpoint (cached 30s so pages stay snappy)."""
    try:
        response = httpx.get(f"{api_url}/health", timeout=1.5)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def hide_restricted_nav(user: dict[str, Any]) -> None:
    """Hide navigation entries for pages the user's role may not open.

    Pages stay registered in st.navigation (so a direct URL renders the
    Access Restricted card instead of a 404); this only removes them from
    the sidebar list.
    """
    hidden = ALL_PAGE_SLUGS - allowed_pages(user)
    if not hidden:
        return
    rules = "\n".join(
        f'[data-testid="stSidebarNav"] li:has(a[href$="/{slug}"]) {{ display: none; }}'
        for slug in sorted(hidden)
    )
    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)


def render_sidebar(user: dict[str, Any]) -> None:
    """Render the shared sidebar chrome (logo, user, status, logout)."""
    import os

    # Brand logo at the very top of the sidebar, above the navigation
    # (sizing and the divider underneath come from theme.apply CSS).
    logo_path = _ASSETS_DIR / "logo.svg"
    if logo_path.exists():
        st.logo(str(logo_path), size="large")
    hide_restricted_nav(user)

    with st.sidebar:
        credits_used = user["credits_total"] - user["credits_balance"]
        used_fraction = credits_used / user["credits_total"]
        bar_color = theme.usage_color(used_fraction)
        # Navy reads as "no color" on the navy sidebar — use blue there.
        if bar_color == theme.NAVY:
            bar_color = theme.BLUE
        st.markdown(
            f"""
<div style="display:flex; align-items:center; gap:0.7rem;">
  <div style="width:44px; height:44px; border-radius:50%;
              background:linear-gradient(135deg, {theme.NAVY} 0%, {theme.BLUE} 100%);
              border:2px solid rgba(255,255,255,0.7); display:flex; align-items:center;
              justify-content:center; font-weight:700; font-size:1rem;">{user["initials"]}</div>
  <div style="line-height:1.25;">
    <div style="font-weight:700;">{user["name"]}</div>
    <div style="font-size:0.75rem; opacity:0.8;">{user["role"]}</div>
  </div>
</div>
<div style="margin-top:0.6rem;">{theme.tier_badge(user["tier"])}</div>
<div style="margin-top:0.55rem; font-size:0.75rem; opacity:0.85;">
  Credits: {user["credits_balance"]:,} / {user["credits_total"]:,}
</div>
<div style="background:rgba(255,255,255,0.25); border-radius:99px; height:6px; margin-top:4px;">
  <div style="background:{bar_color}; width:{used_fraction:.0%}; height:6px;
              border-radius:99px;"></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.divider()

        env = os.getenv("ENV", "demo")
        env_dot = "🟢" if env == "demo" else "🔴"
        env_label = "Demo Mode" if env == "demo" else "Live Mode"
        api_url = os.getenv("API_URL", "http://localhost:8000")
        api_ok = _api_healthy(api_url)
        api_dot = "🟢" if api_ok else "🔷"
        api_label = "Connected" if api_ok else "Local Mode"
        st.markdown(
            f"<div style='font-size:0.85rem;'>{env_dot} <b>{env_label}</b><br/>"
            f"{api_dot} API: {api_label}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        if st.button("Sign out", key="logout_button", use_container_width=True):
            logout()
            st.rerun()
