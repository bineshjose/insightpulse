"""Profile — account details, editable preferences, usage, activity."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, theme

user = auth.require_page("profile")
theme.page_header("User Profile", "Account, preferences, and usage.", "Profile")

prefs = st.session_state["user_prefs"]

# ---------------------------------------------------------------------------
# Profile card
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<div class="niq-card" style="display:flex; align-items:center; gap:1.4rem;">
  <div style="width:88px; height:88px; border-radius:50%;
              background:linear-gradient(135deg, {theme.NAVY} 0%, {theme.BLUE} 100%);
              color:#FFFFFF; display:flex; align-items:center; justify-content:center;
              font-weight:800; font-size:2rem; flex-shrink:0;">{user["initials"]}</div>
  <div style="line-height:1.5;">
    <div style="font-size:1.35rem; font-weight:700; color:{theme.TEXT};">
      {prefs["display_name"]} &nbsp;{theme.tier_badge(user["tier"])}
    </div>
    <div style="color:{theme.TEXT_SECONDARY};">{user["email"]}</div>
    <div style="color:{theme.TEXT};">{user["role"]} · {prefs["title"]}</div>
    <div style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">
      {user["department"]} · Regions: {", ".join(user["regions"])}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("")

settings_col, info_col = st.columns([1.15, 1])

# ---------------------------------------------------------------------------
# Editable settings
# ---------------------------------------------------------------------------

with settings_col:
    st.markdown("### Settings")
    with st.form("profile_settings"):
        display_name = st.text_input("Display name", value=prefs["display_name"])
        title = st.text_input("Title", value=prefs["title"])
        region = st.selectbox(
            "Preferred region",
            user["regions"],
            index=user["regions"].index(prefs["preferred_region"]),
        )
        cohort = st.slider(
            "Default cohort size", 10, user["max_cohort_size"],
            value=prefs["default_cohort_size"], step=10,
        )
        model_options = ["claude-sonnet-4-6", "gpt-4o", "ollama/llama3.1"]
        model = st.selectbox(
            "Default LLM model", model_options,
            index=model_options.index(prefs["default_model"]),
        )
        st.markdown("**Notification preferences**")
        notify_email = st.checkbox("Email alerts", value=prefs["notify_email"])
        notify_completion = st.checkbox(
            "Survey completion notifications", value=prefs["notify_completion"]
        )
        notify_digest = st.checkbox("Weekly digest", value=prefs["notify_digest"])
        saved = st.form_submit_button("Save Changes", use_container_width=True)

    if saved:
        prefs.update({
            "display_name": display_name.strip() or user["name"],
            "title": title.strip() or user["title"],
            "preferred_region": region,
            "default_cohort_size": cohort,
            "default_model": model,
            "notify_email": notify_email,
            "notify_completion": notify_completion,
            "notify_digest": notify_digest,
        })
        auth.record_activity("Updated settings", f"Default model → {model}")
        st.toast("Profile settings saved", icon="✅")

# ---------------------------------------------------------------------------
# Read-only account information
# ---------------------------------------------------------------------------

with info_col:
    st.markdown("### Account")
    st.text_input("Email (read-only)", value=user["email"], disabled=True)
    st.text_input("Role (read-only)", value=user["role"], disabled=True)

    credits_used = user["credits_total"] - user["credits_balance"]
    st.markdown(
        f"**Credit balance** — {user['credits_balance']:,} of "
        f"{user['credits_total']:,} remaining"
    )
    st.markdown(
        theme.usage_bar(credits_used, user["credits_total"],
                        f"{credits_used:,} credits consumed"),
        unsafe_allow_html=True,
    )

    calls_used = user["api_calls_quota"] - user["api_calls_remaining"]
    st.markdown(
        f"**API calls** — {calls_used:,} of "
        f"{user['api_calls_quota']:,} used this month"
    )
    st.markdown(
        theme.usage_bar(calls_used, user["api_calls_quota"],
                        f"{user['api_calls_remaining']:,} calls remaining"),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='color:{theme.TEXT_SECONDARY}; font-size:0.85rem; margin-top:0.6rem;'>"
        f"Account created: {user['created']} &nbsp;·&nbsp; "
        f"Last login: {user['last_login']}</div>",
        unsafe_allow_html=True,
    )

    with st.expander(f"{user['tier']} tier features"):
        for feature in auth.TIER_FEATURES[user["tier"]]:
            st.markdown(f"- {feature}")

st.markdown("")

# ---------------------------------------------------------------------------
# Usage statistics
# ---------------------------------------------------------------------------

st.markdown("### Usage statistics")

_WEEKS = ["W1", "W2", "W3", "W4", "W5", "W6"]
_SURVEY_TREND = [8, 11, 9, 12, 10, 14]
_RESPONSE_TREND = [640, 1120, 890, 1710, 1980, 2560]
_MONTH_SURVEYS = _SURVEY_TREND[-1]
_MONTH_RESPONSES = _RESPONSE_TREND[-1]
history = st.session_state.get("run_history", [])
live_responses = sum(r["totals"]["total_responses"] for r in history)


def _sparkline(values: list[int], color: str) -> go.Figure:
    """Tiny trend line for a stat tile (axis-free by design)."""
    fig = go.Figure(go.Scatter(
        x=_WEEKS, y=values, mode="lines",
        line={"color": color, "width": 2.2},
        fill="tozeroy", fillcolor="rgba(0,164,228,0.10)",
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(
        height=64, margin={"l": 0, "r": 0, "t": 2, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False}, yaxis={"visible": False},
        showlegend=False,
    )
    return fig


u1, u2, u3 = st.columns(3)
with u1:
    st.markdown(theme.kpi_card(
        "Surveys this month", str(_MONTH_SURVEYS + len(history)),
        "6-week trend below", "info"), unsafe_allow_html=True)
    st.plotly_chart(_sparkline(_SURVEY_TREND, theme.BLUE),
                    use_container_width=True, config={"displayModeBar": False})
with u2:
    st.markdown(theme.kpi_card(
        "Responses generated", f"{_MONTH_RESPONSES + live_responses:,}",
        "6-week trend below", "info"), unsafe_allow_html=True)
    st.plotly_chart(_sparkline(_RESPONSE_TREND, theme.BLUE),
                    use_container_width=True, config={"displayModeBar": False})
with u3:
    credits_used = user["credits_total"] - user["credits_balance"]
    st.markdown(theme.kpi_card(
        "Credits consumed", f"{credits_used:,}",
        f"of {user['credits_total']:,} total",
        "warn" if user["credits_balance"] < 0.2 * user["credits_total"] else "info",
    ), unsafe_allow_html=True)
    st.markdown(
        theme.usage_bar(credits_used, user["credits_total"],
                        f"{credits_used / user['credits_total']:.0%} of quota"),
        unsafe_allow_html=True,
    )

f1, f2 = st.columns(2)
model_counts: dict[str, int] = {}
for run in history:
    model_counts[run["config"]["model"]] = model_counts.get(run["config"]["model"], 0) + 1
favorite = max(model_counts, key=model_counts.get) if model_counts else prefs["default_model"]
avg_cohort = (
    round(sum(r["config"]["cohort_size"] for r in history) / len(history))
    if history else prefs["default_cohort_size"]
)
f1.markdown(theme.kpi_card("Favorite model", favorite, "most used", "info"),
            unsafe_allow_html=True)
f2.markdown(theme.kpi_card("Average cohort size", str(avg_cohort), "respondents/run", "info"),
            unsafe_allow_html=True)

st.markdown("")

# ---------------------------------------------------------------------------
# Recent activity
# ---------------------------------------------------------------------------

st.markdown("### Recent activity")
st.dataframe(
    pd.DataFrame(auth.activity_log()).rename(
        columns={"timestamp": "Timestamp", "action": "Action", "details": "Details"}
    ),
    use_container_width=True, hide_index=True,
)

theme.footer()
