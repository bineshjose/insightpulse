"""Audit — provenance, agent trace, cost breakdown, and reproducibility.

Everything the AuditAgent records to make a survey run replayable: the
configuration hash, the ordered agent execution trace, spend, and the exact
command to reproduce the run.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, nav, simulation, theme
from components.charts import agent_timeline_chart

st.set_page_config(page_title="Audit | InsightPulse", page_icon="📋", layout="wide")

user = auth.require_auth("analyze")
theme.apply()
auth.render_sidebar(user)
theme.page_header(
    "📋 Audit Trail",
    "Provenance, agent execution timeline, quality gates, and everything "
    "needed to reproduce a run exactly.",
    "Audit",
)

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
    st.markdown(f"**Run ID:** `{run['run_id']}`")
    st.markdown(f"**Created:** {run['created_at']}")
    st.markdown(f"**Mode:** `{run['mode']}`")
    st.markdown("**Provenance hash (SHA-256 of config):**")
    st.code(run["provenance_hash"], language=None)
with col2:
    totals = run["totals"]
    st.metric("Total cost", f"${totals['total_cost_usd']:.2f}")
    st.metric("Total tokens", f"{totals['total_tokens']:,}")
    st.metric("Throughput", f"{totals['throughput_per_min']:,} resp/min")

with st.expander("Full run configuration"):
    st.json(run["config"])

st.divider()

# ---------------------------------------------------------------------------
# Agent execution trace
# ---------------------------------------------------------------------------

st.subheader("Agent execution trace")
st.caption("The 8-agent LangGraph DAG in execution order.")

trace = run["agent_trace"]
st.plotly_chart(agent_timeline_chart(trace), use_container_width=True)

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

gates = pd.DataFrame({
    "Gate": [
        "Hallucination rate below threshold (< 3%)",
        "Logical consistency above floor (> 90%)",
        "Response validity",
    ],
    "Observed": [
        f"{totals['hallucination_rate']:.2%}",
        f"{totals['consistency_score']:.2%}",
        f"{totals['valid_responses']:,} / {totals['total_responses']:,}",
    ],
    "Status": [
        "✅ pass" if totals["hallucination_rate"] < 0.03 else "❌ fail",
        "✅ pass" if totals["consistency_score"] > 0.90 else "❌ fail",
        "✅ pass" if totals["valid_responses"] / max(totals["total_responses"], 1) > 0.95
        else "⚠️ review",
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
st.code(
    "insightpulse replay \\\n"
    f"  --seed {run['config']['seed']} \\\n"
    f"  --model {run['config']['model']} \\\n"
    f"  --cohort-size {run['config']['cohort_size']} \\\n"
    f"  --verify-hash {run['provenance_hash'][:16]}…",
    language="bash",
)

# ---------------------------------------------------------------------------
# Session run history
# ---------------------------------------------------------------------------

history = st.session_state.get("run_history", [])
if len(history) > 1:
    st.divider()
    st.subheader("Session run history")
    st.dataframe(pd.DataFrame([{
        "run_id": h["run_id"],
        "created": h["created_at"],
        "model": h["config"]["model"],
        "cohort": h["config"]["cohort_size"],
        "hallucination": f"{h['totals']['hallucination_rate']:.1%}",
        "cost": f"${h['totals']['total_cost_usd']:.2f}",
        "hash": h["provenance_hash"][:12] + "…",
    } for h in reversed(history)]), use_container_width=True, hide_index=True)

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
