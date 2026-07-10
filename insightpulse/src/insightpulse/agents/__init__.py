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
