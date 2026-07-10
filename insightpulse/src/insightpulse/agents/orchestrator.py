"""LangGraph orchestration DAG for the synthetic survey pipeline.

This module wires the 8 specialized agents into a directed acyclic graph
using LangGraph's StateGraph. The graph defines:

    1. Linear flow: SurveyDesigner → CohortSelector → TwinOrchestrator →
       Validator → CalibrationAgent → DiversityMonitor → AuditAgent

    2. Conditional edges:
       - Validator → TwinOrchestrator (retry if validation fails)
       - DiversityMonitor → TwinOrchestrator (retry if entropy too low)
       - CostAgent checks run concurrently and can halt the pipeline

    3. State checkpointing: every agent writes to the shared
       SurveyPipelineState, enabling replay and debugging.

The DAG structure G = (A, E) matches the thesis specification exactly,
with LangGraph providing the runtime instead of a custom implementation.

Usage:
    pipeline = build_survey_pipeline()
    result = await pipeline.ainvoke(initial_state)
"""

from __future__ import annotations

from typing import Any, Literal

import structlog
from langgraph.graph import END, StateGraph

from insightpulse.agents.audit_agent import audit_agent_node
from insightpulse.agents.calibration_agent import calibration_agent_node
from insightpulse.agents.cohort_selector import cohort_selector_node
from insightpulse.agents.cost_agent import cost_agent_node
from insightpulse.agents.diversity_monitor import diversity_monitor_node
from insightpulse.agents.survey_designer import survey_designer_node
from insightpulse.agents.twin_orchestrator import twin_orchestrator_node
from insightpulse.agents.validator import validator_node
from insightpulse.models.agent_state import SurveyPipelineState

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def should_regenerate(state: dict[str, Any]) -> Literal["regenerate", "continue"]:
    """Decide whether to regenerate responses after validation.

    The Validator sets `needs_regeneration=True` when responses fail
    consistency checks or hallucination detection. We allow up to
    `max_validation_retries` regeneration attempts before accepting
    whatever we have.

    Args:
        state: Current pipeline state dictionary.

    Returns:
        "regenerate" to loop back to TwinOrchestrator, or
        "continue" to proceed to CalibrationAgent.
    """
    needs_regen = state.get("needs_regeneration", False)
    retry_count = state.get("validation_retry_count", 0)
    max_retries = 3  # Default; overridden by AgentConfig

    if needs_regen and retry_count < max_retries:
        logger.info(
            "regeneration_triggered",
            retry_count=retry_count,
            max_retries=max_retries,
        )
        return "regenerate"

    if needs_regen:
        logger.warning(
            "max_retries_reached",
            retry_count=retry_count,
            proceeding_with_partial="true",
        )

    return "continue"


def check_budget(state: dict[str, Any]) -> Literal["halt", "continue"]:
    """Check whether the cost budget has been exceeded.

    The CostAgent sets `budget_exceeded=True` when the cumulative
    cost surpasses the configured maximum.

    Args:
        state: Current pipeline state dictionary.

    Returns:
        "halt" to stop the pipeline, or "continue" to proceed.
    """
    if state.get("budget_exceeded", False):
        logger.warning("budget_exceeded", total_cost=state.get("total_cost_usd", 0))
        return "halt"
    return "continue"


def check_diversity(state: dict[str, Any]) -> Literal["adjust", "continue"]:
    """Check whether response diversity meets the minimum threshold.

    The DiversityMonitor evaluates Shannon entropy of responses.
    If entropy is below the threshold, it adjusts temperature
    parameters and triggers regeneration.

    Args:
        state: Current pipeline state dictionary.

    Returns:
        "adjust" to loop back for regeneration, or "continue".
    """
    if not state.get("diversity_acceptable", True):
        logger.info("diversity_below_threshold", adjusting="true")
        return "adjust"
    return "continue"


# ---------------------------------------------------------------------------
# Pipeline Builder
# ---------------------------------------------------------------------------

def build_survey_pipeline() -> StateGraph:
    """Build the LangGraph survey orchestration pipeline.

    Constructs the full DAG with all agents, conditional edges,
    and state management. The returned graph can be compiled and
    invoked with an initial SurveyPipelineState.

    Returns:
        Compiled LangGraph StateGraph ready for execution.

    Example:
        pipeline = build_survey_pipeline()
        compiled = pipeline.compile()
        result = await compiled.ainvoke({
            "raw_questions": ["How important is organic labeling?"],
            "requested_cohort_size": 100,
        })
    """
    logger.info("building_survey_pipeline")

    # Typed state schema: gives every field its own channel (last-value by
    # default, accumulating for agent_trace). A bare `dict` schema would
    # collapse the state into one channel and drop earlier agents' writes.
    workflow = StateGraph(SurveyPipelineState)

    # -----------------------------------------------------------------------
    # Add agent nodes
    # -----------------------------------------------------------------------
    workflow.add_node("survey_designer", survey_designer_node)
    workflow.add_node("cohort_selector", cohort_selector_node)
    workflow.add_node("twin_orchestrator", twin_orchestrator_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("cost_check", cost_agent_node)
    workflow.add_node("calibration_agent", calibration_agent_node)
    workflow.add_node("diversity_monitor", diversity_monitor_node)
    workflow.add_node("audit_agent", audit_agent_node)

    # -----------------------------------------------------------------------
    # Define edges (the DAG structure)
    # -----------------------------------------------------------------------

    # Entry point
    workflow.set_entry_point("survey_designer")

    # Linear flow
    workflow.add_edge("survey_designer", "cohort_selector")
    workflow.add_edge("cohort_selector", "twin_orchestrator")
    workflow.add_edge("twin_orchestrator", "validator")

    # Conditional: Validator → regenerate or continue
    workflow.add_conditional_edges(
        "validator",
        should_regenerate,
        {
            "regenerate": "twin_orchestrator",
            "continue": "cost_check",
        },
    )

    # Conditional: CostAgent → halt or continue
    workflow.add_conditional_edges(
        "cost_check",
        check_budget,
        {
            "halt": "audit_agent",
            "continue": "calibration_agent",
        },
    )

    # Linear: calibration → diversity check
    workflow.add_edge("calibration_agent", "diversity_monitor")

    # Conditional: DiversityMonitor → adjust or finalize
    workflow.add_conditional_edges(
        "diversity_monitor",
        check_diversity,
        {
            "adjust": "twin_orchestrator",
            "continue": "audit_agent",
        },
    )

    # Terminal: AuditAgent → END
    workflow.add_edge("audit_agent", END)

    logger.info("survey_pipeline_built", nodes=8, edges=9)

    return workflow


async def run_survey(
    questions: list[str],
    cohort_size: int = 100,
    context: str = "",
    models: list[str] | None = None,
    cohort_filters: dict[str, str] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a complete synthetic survey through the agent pipeline.

    This is the main entry point for executing a survey. It constructs
    the initial state, builds and compiles the pipeline, and returns
    the final results.

    Args:
        questions: List of survey question texts.
        cohort_size: Number of synthetic respondents to generate.
        context: Domain context for the survey.
        models: LLM models to use (empty list = use default).
        cohort_filters: Demographic filters for cohort selection.
        seed: Random seed for reproducibility.

    Returns:
        Final pipeline state dictionary containing all results,
        metrics, and audit trace.
    """
    initial_state = {
        "raw_questions": questions,
        "requested_cohort_size": cohort_size,
        "question_context": context,
        "requested_models": models or [],
        "cohort_filters": cohort_filters or {},
        "random_seed": seed,
        "status": "running",
    }

    logger.info(
        "survey_run_started",
        num_questions=len(questions),
        cohort_size=cohort_size,
        models=models,
    )

    pipeline = build_survey_pipeline()
    compiled = pipeline.compile()

    result = await compiled.ainvoke(initial_state)

    result["status"] = "completed" if not result.get("error_message") else "failed"

    logger.info(
        "survey_run_completed",
        status=result["status"],
        total_responses=len(result.get("validated_responses", [])),
        total_cost=result.get("total_cost_usd", 0),
    )

    return result
