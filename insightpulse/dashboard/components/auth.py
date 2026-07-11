"""Enterprise login and role-based access for the dashboard.

Demo-grade authentication: three hardcoded, realistic accounts held
in-memory, session-state persistence, and a permissions model that the
pages consult for role-based access. This is deliberately NOT a security
boundary — the thesis demo has no user database — but the flow (gate →
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
        "credits_balance": 2500,
        "api_calls_remaining": 10_000,
        "api_calls_quota": 10_000,
        "max_cohort_size": 5000,
        "created": "2025-08-14",
    },
    "evaluator@iitm.ac.in": {
        "password": "eval2024",
        "name": "External Evaluator",
        "initials": "EV",
        "role": "Read-Only Analyst",
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
    profile["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M")
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


def record_activity(action: str, details: str) -> None:
    """Append one entry to the session activity log (shown on Profile)."""
    log = st.session_state.setdefault("activity_log", [])
    log.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": action,
        "details": details,
    })


def activity_log() -> list[dict[str, str]]:
    """The session's activity entries, newest first (seeded for realism)."""
    seeded = _seed_activity()
    live = list(reversed(st.session_state.get("activity_log", [])))
    return (live + seeded)[:10]


def _seed_activity() -> list[dict[str, str]]:
    """Deterministic historical entries so the log never looks empty."""
    base = datetime.now()
    entries = [
        ("Ran survey", "Organic Labeling Importance (50 respondents, claude-sonnet-4-6)"),
        ("Exported results", "Q3 Brand Perception — CSV"),
        ("Ran experiment", "Multi-LLM Comparison (3 models, seed 42)"),
        ("Viewed validation", "Cross-validation vs empirical bank"),
        ("Ran survey", "Sustainability Willingness-to-Pay (120 respondents)"),
        ("Updated settings", "Default model → claude-sonnet-4-6"),
    ]
    return [
        {
            "timestamp": (base - timedelta(days=i + 1, hours=3 * i)).strftime("%Y-%m-%d %H:%M"),
            "action": action,
            "details": details,
        }
        for i, (action, details) in enumerate(entries)
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
    box-shadow: 0 20px 60px rgba(0, 20, 40, 0.45);
    border: none;
}}
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
            "M.Tech Industrial AI Thesis — Binesh Jose (CH24M521)</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Gate + shared sidebar
# ---------------------------------------------------------------------------

def require_auth(permission: str | None = None) -> dict[str, Any]:
    """Authentication gate for every page.

    Args:
        permission: Optional permission this page needs beyond sign-in
            (e.g. "run", "analyze"). None means any signed-in user.

    Returns:
        The authenticated user profile. Stops rendering (login page or
        access-restricted notice) otherwise.
    """
    user = current_user()
    if user is None:
        login_page()
        st.stop()
    if permission is not None and not has_permission(user, permission):
        theme.apply()
        render_sidebar(user)
        st.markdown("## Access Restricted")
        st.warning(
            f"Your role (**{user['role']}**) does not include the "
            f"'{permission}' capability. Contact your administrator to "
            "request access.",
            icon="🔒",
        )
        theme.footer()
        st.stop()
    return user


@st.cache_data(ttl=30, show_spinner=False)
def _api_healthy(api_url: str) -> bool:
    """Probe the API health endpoint (cached 30s so pages stay snappy)."""
    try:
        response = httpx.get(f"{api_url}/health", timeout=1.5)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def render_sidebar(user: dict[str, Any]) -> None:
    """Render the shared sidebar chrome (logo, user, status, logout)."""
    import os

    with st.sidebar:
        logo = _asset_data_uri("logo.svg")
        if logo:
            st.markdown(
                f'<img src="{logo}" alt="InsightPulse" style="width:200px;'
                f' background:#FFFFFF; padding:8px 10px; border-radius:10px;"/>',
                unsafe_allow_html=True,
            )
        st.divider()

        avatar_color = theme.TIER_COLORS.get(user["tier"], theme.BLUE)
        credits_pct = int(100 * user["credits_balance"] / user["credits_total"])
        st.markdown(
            f"""
<div style="display:flex; align-items:center; gap:0.7rem;">
  <div style="width:44px; height:44px; border-radius:50%; background:{avatar_color};
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
  <div style="background:{theme.BLUE}; width:{credits_pct}%; height:6px; border-radius:99px;"></div>
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
        api_dot = "🟢" if api_ok else "⚪"
        api_label = "Healthy" if api_ok else "Offline (local simulation)"
        st.markdown(
            f"<div style='font-size:0.85rem;'>{env_dot} <b>{env_label}</b><br/>"
            f"{api_dot} API: {api_label}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        if st.button("Sign out", key="logout_button", use_container_width=True):
            logout()
            st.rerun()
