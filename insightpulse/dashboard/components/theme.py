"""NielsenIQ professional theme — the single home for all dashboard styling.

Every page calls :func:`apply` once (right after auth) to inject the CSS;
no other module defines styles. Chart series colors live here too and are
consumed by ``components.charts`` so Plotly output matches the UI chrome.

Palette notes
    * Brand chrome (sidebar, headers, buttons) uses NielsenIQ navy/blue.
    * Chart series use an 8-slot categorical order derived from the brand
      hues and validated with the dataviz palette checker (worst adjacent
      CVD ΔE 14.2 on white). Navy itself is deliberately NOT a series
      color — it fails the lightness band; it is chrome.
    * Status colors (green/amber/red) are reserved for pass/warn/fail
      meaning and never used as "series 4".
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------

NAVY = "#003865"          # primary chrome: sidebar, headers, buttons
NAVY_LIGHT = "#004A7C"    # gradient partner / hover
BLUE = "#00A4E4"          # links, active states, accents
GREEN = "#6CC24A"         # success
AMBER = "#F2A900"         # warning
RED = "#E03C31"           # error
BACKGROUND = "#F7F8FA"
CARD = "#FFFFFF"
TEXT = "#1A1A2E"
TEXT_SECONDARY = "#6B7280"
BORDER = "#E5E7EB"

FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)

# 8-slot categorical order for charts — validated (see module docstring).
# Amber is darkened vs brand amber to sit inside the lightness band.
CHART_SERIES = [
    "#00A4E4",  # 1 niq blue
    "#E09C00",  # 2 amber (chart-tuned)
    "#6F5AA8",  # 3 violet
    "#6CC24A",  # 4 niq green
    "#E03C31",  # 5 niq red
    "#2E6BAA",  # 6 steel blue
    "#C9508B",  # 7 magenta
    "#946200",  # 8 bronze
]

# Fixed entity → color assignments (color follows the entity, everywhere).
SERIES_COLORS = {
    "Synthetic (raw)": CHART_SERIES[0],
    "Calibrated": CHART_SERIES[1],
    "Empirical": CHART_SERIES[2],
}
MODEL_COLORS = {
    "claude-sonnet-4-6": CHART_SERIES[0],
    "gpt-4o": CHART_SERIES[1],
    "ollama/llama3.1": CHART_SERIES[2],
    "claude-haiku-4-5": CHART_SERIES[3],
}

TIER_COLORS = {
    "Enterprise": NAVY,
    "Professional": BLUE,
    "Academic": GREEN,
}

_FOOTER_TEXT = (
    "InsightPulse v1.0.0 &nbsp;|&nbsp; M.Tech Industrial AI Thesis — "
    "IIT Madras × NielsenIQ &nbsp;|&nbsp; Binesh Jose (CH24M521) "
    "&nbsp;|&nbsp; © 2025"
)


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def apply() -> None:
    """Inject the NielsenIQ theme CSS (idempotent, call once per page)."""
    st.markdown(
        f"""
<style>
/* ---- base ---- */
html, body, [data-testid="stAppViewContainer"] {{
    font-family: {FONT_STACK};
    background-color: {BACKGROUND};
    color: {TEXT};
}}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ color: {NAVY}; font-family: {FONT_STACK}; }}
a {{ color: {BLUE}; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
}}
[data-testid="stSidebar"] * {{ color: #FFFFFF; }}
[data-testid="stSidebar"] a:hover {{ color: {BLUE} !important; }}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
    border-radius: 6px;
}}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
    background: rgba(0, 164, 228, 0.18);
}}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.25); }}
[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,0.10);
    color: #FFFFFF;
    border: 1px solid rgba(255,255,255,0.35);
    width: 100%;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {BLUE};
    border-color: {BLUE};
    color: #FFFFFF;
}}

/* ---- buttons ---- */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
    background-color: {NAVY};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.25rem;
    font-weight: 600;
    transition: background-color .15s ease, transform .1s ease;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {{
    background-color: {BLUE};
    color: #FFFFFF;
    transform: translateY(-1px);
}}
.stButton > button:focus:not(:active) {{ color: #FFFFFF; }}

/* ---- metrics ---- */
[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-left: 4px solid {BLUE};
    border-radius: 10px;
    padding: 0.85rem 1rem;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
}}
[data-testid="stMetricLabel"] {{ color: {TEXT_SECONDARY}; }}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {{
    border-bottom: 2px solid {BORDER};
    gap: 0.25rem;
}}
.stTabs [data-baseweb="tab"] {{
    color: {TEXT_SECONDARY};
    font-weight: 600;
}}
.stTabs [aria-selected="true"] {{
    color: {NAVY} !important;
    border-bottom: 3px solid {NAVY};
}}

/* ---- tables / dataframes ---- */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {CARD};
}}
[data-testid="stTable"] thead th {{
    background: {NAVY};
    color: #FFFFFF;
}}
[data-testid="stTable"] tbody tr:nth-child(even) {{ background: {BACKGROUND}; }}

/* ---- inputs ---- */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="select"] > div {{
    border-radius: 8px;
    border-color: {BORDER};
}}
.stTextInput input:focus, .stNumberInput input:focus,
.stTextArea textarea:focus {{
    border-color: {BLUE};
    box-shadow: 0 0 0 1px {BLUE};
}}
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background-color: {NAVY};
    border-color: {NAVY};
}}
.stCheckbox [data-baseweb="checkbox"] span, .stToggle span {{
    border-color: {NAVY};
}}

/* ---- progress ---- */
.stProgress > div > div > div > div {{ background-color: {NAVY}; }}

/* ---- expanders ---- */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {CARD};
}}
[data-testid="stExpander"] summary {{ color: {NAVY}; font-weight: 600; }}

/* ---- alerts ---- */
[data-testid="stAlert"] {{ border-radius: 10px; }}
div[data-baseweb="notification"] {{ border-radius: 10px; }}

/* ---- custom components ---- */
.niq-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
}}
.niq-kpi {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.15rem;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    height: 100%;
}}
.niq-kpi .kpi-label {{
    color: {TEXT_SECONDARY};
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.niq-kpi .kpi-value {{
    color: {TEXT};
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 2.1rem;
}}
.niq-kpi .kpi-delta {{ font-size: 0.85rem; font-weight: 600; }}
.niq-breadcrumb {{
    color: {TEXT_SECONDARY};
    font-size: 0.85rem;
    margin-bottom: 0.25rem;
}}
.niq-breadcrumb a {{ color: {TEXT_SECONDARY}; text-decoration: none; }}
.niq-badge {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #FFFFFF;
}}
.niq-footer {{
    color: {TEXT_SECONDARY};
    font-size: 0.78rem;
    text-align: center;
    border-top: 1px solid {BORDER};
    padding-top: 0.9rem;
    margin-top: 2.5rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Reusable themed fragments
# ---------------------------------------------------------------------------

def page_header(title: str, description: str, breadcrumb: str) -> None:
    """Render the standard page header with breadcrumb.

    Args:
        title: Page title (h2).
        description: One-line purpose statement.
        breadcrumb: Trailing crumb, e.g. "Survey Runner".
    """
    st.markdown(
        f'<div class="niq-breadcrumb">Home &rsaquo; {breadcrumb}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"## {title}")
    st.markdown(
        f'<p style="color:{TEXT_SECONDARY}; margin-top:-0.4rem;">{description}</p>',
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    delta: str = "",
    status: str = "info",
) -> str:
    """Build one KPI card as HTML (compose into columns with st.markdown).

    Args:
        label: Small uppercase label.
        value: Big number/string.
        delta: Optional secondary line (trend, target note).
        status: 'good' | 'warn' | 'bad' | 'info' — colors the left border.

    Returns:
        HTML string for the card.
    """
    border = {"good": GREEN, "warn": AMBER, "bad": RED, "info": BLUE}[status]
    delta_color = {"good": GREEN, "warn": AMBER, "bad": RED, "info": TEXT_SECONDARY}[status]
    delta_html = (
        f'<div class="kpi-delta" style="color:{delta_color}">{delta}</div>'
        if delta else ""
    )
    return (
        f'<div class="niq-kpi" style="border-left: 4px solid {border};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{delta_html}</div>"
    )


def tier_badge(tier: str) -> str:
    """Subscription tier badge HTML (Enterprise/Professional/Academic)."""
    color = TIER_COLORS.get(tier, BLUE)
    return f'<span class="niq-badge" style="background:{color};">{tier}</span>'


def footer() -> None:
    """Render the standard attribution footer."""
    st.markdown(f'<div class="niq-footer">{_FOOTER_TEXT}</div>', unsafe_allow_html=True)


def current_date_line() -> str:
    """Human-readable current date for the welcome banner."""
    return datetime.now().strftime("%A, %d %B %Y")
