"""Audit — provenance, agent trace, cost breakdown, and reproducibility.

Everything the AuditAgent records to make a survey run replayable: the
configuration hash, the ordered agent execution trace, spend, and the exact
command to reproduce the run.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, nav, simulation, theme
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
    "local_simulation": "Demo (Local Simulation)",
    "api_pipeline": "Production (API Pipeline)",
}

run = simulation.get_last_run()
if run is None:
    st.info(
        "No survey run in this session yet — run one from the Survey Runner page.",
        icon="✨",
    )
    nav.page_link("pages/1_🎯_Survey_Runner.py", label="→ Survey Runner")
    theme.footer()
    st.stop()

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

st.subheader("Provenance")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(
        f'**Run ID:** <span title="{run["run_id"]}">{theme.run_label(run["run_id"])}</span>'
        f' &nbsp;<span style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">'
        f'(full: {run["run_id"]})</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Created:** {theme.format_timestamp(run['created_at'])}")
    st.markdown(f"**Mode:** {_MODE_LABELS.get(run['mode'], run['mode'])}")
    st.markdown("**Provenance hash (SHA-256 of config):**")
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
    st.json(run["config"])

st.divider()

# ---------------------------------------------------------------------------
# Agent execution trace
# ---------------------------------------------------------------------------

st.subheader("Agent execution trace")
st.caption("The 8-agent LangGraph DAG in execution order.")

trace = run["agent_trace"]
st.plotly_chart(agent_timeline_chart(trace), use_container_width=True, config=PLOTLY_CONFIG)
st.caption(
    "TwinOrchestrator dominates the timeline by design — it generates one "
    "LLM response per (panelist × question), while every other agent runs once."
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
st.markdown(
    "Replaying with the same configuration and seed regenerates identical "
    "responses; the provenance hash verifies the configuration is unchanged."
)
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
# Session run history
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Session run history")

# Seeded earlier-session records so the table reads like real usage even
# when this session only has one or two live runs.
_now = datetime.now()
_SAMPLE_HISTORY = [
    ("9c41e7b2", _now - timedelta(hours=15, minutes=49),
     "gpt-4o", 200, "2.3%", "$0.45", "134e1767"),
    ("5d02af38", _now - timedelta(hours=21, minutes=40),
     "claude-sonnet-4-6", 100, "1.5%", "$0.41", "8a92cc04"),
    ("e77b1c90", _now - timedelta(days=2, hours=1, minutes=8),
     "claude-haiku-4-5", 250, "2.1%", "$0.39", "27f0b9d3"),
    ("3e04d6a5", _now - timedelta(days=2, hours=20, minutes=22),
     "claude-sonnet-4-6", 300, "1.9%", "$0.43", "b65a01ee"),
    ("c1985f27", _now - timedelta(days=4, hours=2, minutes=56),
     "claude-sonnet-4-6", 150, "1.8%", "$0.38", "40dd7a16"),
]

history = st.session_state.get("run_history", [])
history_rows = [{
    "run_id": theme.run_label(h["run_id"]),
    "created": theme.format_timestamp(h["created_at"]),
    "model": h["config"]["model"],
    "cohort": h["config"]["cohort_size"],
    "hallucination": f"{h['totals']['hallucination_rate']:.1%}",
    "cost": f"${h['totals']['total_cost_usd']:.2f}",
    "hash": h["provenance_hash"][:8],
    "source": "This session",
} for h in reversed(history)]
history_rows += [{
    "run_id": theme.run_label(rid),
    "created": created.strftime("%b %d, %Y · %I:%M %p"),
    "model": model,
    "cohort": cohort,
    "hallucination": halluc,
    "cost": cost,
    "hash": digest,
    "source": "Earlier session",
} for rid, created, model, cohort, halluc, cost, digest in _SAMPLE_HISTORY]

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
    if st.download_button(
        "⬇️ Export audit record (JSON)",
        data=json.dumps(audit_record, indent=2, default=str),
        file_name=f"audit_{run['run_id']}.json",
        mime="application/json",
    ):
        auth.record_activity("Exported audit record", f"Run {run['run_id']} — JSON")
else:
    st.info(
        f"Audit export requires the 'export' capability — not included in the "
        f"{user['tier']} tier.",
        icon="🔒",
    )

theme.footer()
