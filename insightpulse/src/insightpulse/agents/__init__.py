"""Agent modules for the LangGraph orchestration DAG.

Eight specialized agents coordinate the synthetic survey pipeline:

    SurveyDesigner     → Interprets questions, infers types and options
    CohortSelector     → Selects respondent cohort from embedding space
    TwinOrchestrator   → Generates synthetic responses via digital twins
    Validator          → Checks consistency, detects hallucinations
    CalibrationAgent   → Applies BDCL optimal transport calibration
    DiversityMonitor   → Monitors response entropy, adjusts temperature
    CostAgent          → Tracks cost, enforces budget limits
    AuditAgent         → Logs provenance, ensures reproducibility

The orchestrator module (orchestrator.py) wires these agents into a
LangGraph StateGraph with conditional edges for retry loops.
"""

from insightpulse.agents.audit_agent import audit_agent_node
from insightpulse.agents.calibration_agent import calibration_agent_node
from insightpulse.agents.cohort_selector import cohort_selector_node
from insightpulse.agents.cost_agent import cost_agent_node
from insightpulse.agents.diversity_monitor import diversity_monitor_node
from insightpulse.agents.orchestrator import build_survey_pipeline, run_survey
from insightpulse.agents.survey_designer import survey_designer_node
from insightpulse.agents.twin_orchestrator import twin_orchestrator_node
from insightpulse.agents.validator import validator_node

__all__ = [
    "audit_agent_node",
    "build_survey_pipeline",
    "calibration_agent_node",
    "cohort_selector_node",
    "cost_agent_node",
    "diversity_monitor_node",
    "run_survey",
    "survey_designer_node",
    "twin_orchestrator_node",
    "validator_node",
]
