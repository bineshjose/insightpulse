"""Audit — provenance, agent trace, cost breakdown, and reproducibility."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, demo_engine, nav, theme
from components.charts import PLOTLY_CONFIG, agent_timeline_chart

user = auth.require_page("audit")
theme.page_header(
    "Audit Trail",
    "Provenance, agent execution timeline, quality gates, and everything "
    "needed to reproduce a run exactly.",
    "Audit",
)

# Demo-mode quality gate thresholds.
GATE_HALLUCINATION_MAX = 0.05
GATE_CONSISTENCY_MIN = 0.90
GATE_VALIDITY_MIN = 0.93

_MODE_LABELS = {
    "demo": "Demo Mode",
    "api_pipeline": "Production Mode",
}

run = demo_engine.get_last_run()
if run is None:
    st.info(
        "No survey run in this session yet — run one from the Survey Runner page.",
        icon="✨",
    )
    nav.page_link("pages/3_Survey_Runner.py", label="→ Survey Runner")
    theme.footer()
    st.stop()

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

st.subheader("Provenance")

meta = run.get("metadata") or {}
col1, col2 = st.columns([2, 1])
with col1:
    if meta.get("survey_name"):
        st.markdown(
            f'<div style="font-size:1.02rem; font-weight:700; color:{theme.NAVY};'
            f' margin-bottom:0.35rem;">{meta["survey_name"]}'
            f'<span style="color:{theme.TEXT_SECONDARY}; font-weight:500;'
            f' font-size:0.85rem;"> &nbsp;{meta.get("survey_id", "")}</span></div>',
            unsafe_allow_html=True,
        )
        detail_rows = [
            ("Client", meta.get("client_name", "—")),
            ("Contract", meta.get("contract_id", "—")),
            ("Category", meta.get("category", "—")),
            ("Priority", meta.get("priority", "—")),
            ("Region", meta.get("region", "—")),
            ("Executor", f"{meta.get('executor_name', '—')}"
                         f" ({meta.get('executor_email', '—')})"),
        ]
        st.markdown(
            "".join(
                f'<div style="font-size:0.9rem;"><span style="color:'
                f'{theme.TEXT_SECONDARY};">{label}:</span> {value}</div>'
                for label, value in detail_rows
            ),
            unsafe_allow_html=True,
        )
        st.markdown("")
    st.markdown(
        f'**Run ID:** <span title="{run["run_id"]}">{theme.run_label(run["run_id"])}</span>'
        f' &nbsp;<span style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">'
        f'(full: {run["run_id"]})</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Created:** {theme.format_timestamp(run['created_at'])}")
    st.markdown(f"**Mode:** {_MODE_LABELS.get(run['mode'], run['mode'])}")
    st.markdown("**Provenance hash:**")
    st.code(run["provenance_hash"], language=None)
with col2:
    totals = run["totals"]
    st.markdown(theme.kpi_card(
        "Total cost", f"${totals['total_cost_usd']:.2f}", "", "neutral",
    ), unsafe_allow_html=True)
    st.markdown(theme.kpi_card(
        "Total tokens", f"{totals['total_tokens']:,}", "", "neutral",
    ), unsafe_allow_html=True)
    st.markdown(theme.kpi_card(
        "Throughput", f"{totals['throughput_per_min']:,} resp/min", "", "neutral",
    ), unsafe_allow_html=True)

with st.expander("Full run configuration"):
    st.json({**run["config"], "metadata": meta})

st.divider()

# ---------------------------------------------------------------------------
# Agent execution trace
# ---------------------------------------------------------------------------

st.subheader("Agent execution trace")

trace = run["agent_trace"]
st.plotly_chart(agent_timeline_chart(trace), use_container_width=True, config=PLOTLY_CONFIG)
st.caption(
    "TwinOrchestrator generates one response per (panelist × question); "
    "every other agent runs once."
)

with st.expander("Trace detail (table view)"):
    st.dataframe(
        pd.DataFrame(trace).rename(columns={
            "agent_name": "Agent", "action": "Action",
            "output_summary": "Output", "duration_ms": "Duration (ms)",
        }),
        use_container_width=True, hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

st.subheader("Quality gates")

validity_share = totals["valid_responses"] / max(totals["total_responses"], 1)
gates = pd.DataFrame({
    "Gate": [
        f"Hallucination rate below threshold (< {GATE_HALLUCINATION_MAX:.0%})",
        f"Logical consistency above floor (> {GATE_CONSISTENCY_MIN:.0%})",
        f"Response validity (> {GATE_VALIDITY_MIN:.0%} of responses clean)",
    ],
    "Observed": [
        f"{totals['hallucination_rate']:.2%}",
        f"{totals['consistency_score']:.2%}",
        f"{totals['valid_responses']:,} / {totals['total_responses']:,}",
    ],
    "Status": [
        "✅ pass" if totals["hallucination_rate"] < GATE_HALLUCINATION_MAX else "❌ fail",
        "✅ pass" if totals["consistency_score"] > GATE_CONSISTENCY_MIN else "❌ fail",
        "✅ pass" if validity_share > GATE_VALIDITY_MIN else "⚠️ review",
    ],
})
st.dataframe(gates, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

st.subheader("Reproducibility")
question_ids = json.dumps(run["config"]["questions"])
st.code(
    f"curl -X POST {data_loader.API_URL}/api/v1/survey/run \\\n"
    '  -H "Content-Type: application/json" \\\n'
    f"  -d '{{\"questions\": {question_ids},\n"
    f"       \"seed\": {run['config']['seed']},"
    f" \"cohort_size\": {run['config']['cohort_size']},"
    f" \"models\": [\"{run['config']['model']}\"]}}'",
    language="bash",
)

# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Run history")

_now = datetime.now()
_HISTORY = [
    ("SRV-2026-00142", "Organic Labeling Importance", "Unilever",
     _now - timedelta(hours=2, minutes=41), "claude-sonnet-4-6", 250,
     "1.9%", "$0.44", "134e1767"),
    ("SRV-2026-00141", "Q3 Brand Perception Tracker", "Procter & Gamble",
     _now - timedelta(days=1, hours=1, minutes=12), "claude-sonnet-4-6", 500,
     "2.1%", "$0.91", "8a92cc04"),
    ("SRV-2026-00140", "Sustainability Willingness-to-Pay", "Nestlé",
     _now - timedelta(days=2, hours=3, minutes=55), "gpt-4o", 120,
     "2.3%", "$0.19", "27f0b9d3"),
    ("SRV-2026-00139", "Snack Purchase Frequency Pulse", "PepsiCo",
     _now - timedelta(days=3, hours=6, minutes=30), "ollama/llama3.1", 300,
     "1.5%", "$0.00", "b65a01ee"),
    ("SRV-2026-00138", "Premium Tier Price Sensitivity", "Mondelēz",
     _now - timedelta(days=4, hours=2, minutes=5), "claude-sonnet-4-6", 200,
     "1.8%", "$0.35", "40dd7a16"),
]

history = st.session_state.get("run_history", [])
history_rows = [{
    "survey_id": (h.get("metadata") or {}).get("survey_id")
                 or theme.run_label(h["run_id"]),
    "survey": (h.get("metadata") or {}).get("survey_name") or "—",
    "client": (h.get("metadata") or {}).get("client_name") or "—",
    "created": theme.format_timestamp(h["created_at"]),
    "model": h["config"]["model"],
    "cohort": h["config"]["cohort_size"],
    "hallucination": f"{h['totals']['hallucination_rate']:.1%}",
    "cost": f"${h['totals']['total_cost_usd']:.2f}",
    "hash": h["provenance_hash"][:8],
} for h in reversed(history)]
history_rows += [{
    "survey_id": sid,
    "survey": name,
    "client": client,
    "created": created.strftime("%b %d, %Y · %I:%M %p"),
    "model": model,
    "cohort": cohort,
    "hallucination": halluc,
    "cost": cost,
    "hash": digest,
} for sid, name, client, created, model, cohort, halluc, cost, digest in _HISTORY]

st.dataframe(
    theme.humanize_columns(pd.DataFrame(history_rows)),
    use_container_width=True, hide_index=True,
    column_config={
        "Provenance Hash": st.column_config.TextColumn(
            "Provenance Hash",
            help="First 8 characters of the SHA-256 config hash — the full "
                 "hash ships in the audit export.",
        ),
    },
)

# Export the complete audit record (config + trace + metrics, no raw responses).
audit_record = {
    "run_id": run["run_id"],
    "created_at": run["created_at"],
    "metadata": meta,
    "provenance_hash": run["provenance_hash"],
    "config": run["config"],
    "totals": run["totals"],
    "agent_trace": run["agent_trace"],
    "question_metrics": [
        {"question_id": r["question_id"], "metrics": r["metrics_calibrated"]}
        for r in run["question_results"]
    ],
}
if auth.has_permission(user, "export"):
    export_id = meta.get("survey_id") or run["run_id"]
    if st.download_button(
        "⬇️ Export audit record (JSON)",
        data=json.dumps(audit_record, indent=2, default=str),
        file_name=f"audit_{export_id}.json",
        mime="application/json",
    ):
        auth.record_activity("Exported audit record", f"{export_id} — JSON")
else:
    st.info(
        f"Audit export requires the 'export' capability — not included in the "
        f"{user['tier']} tier.",
        icon="🔒",
    )

theme.footer()
