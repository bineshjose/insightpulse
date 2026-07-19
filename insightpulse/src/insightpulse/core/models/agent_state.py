"""LangGraph shared state for the agent orchestration DAG.

This is the central data structure that flows through the LangGraph
pipeline. Each agent reads the fields it needs and returns a partial
update; LangGraph merges updates into the shared state per the channel
reducers declared here.

Design decisions
    * ``SurveyPipelineState`` is a **TypedDict** (not a Pydantic model):
      LangGraph hands nodes the state with dict semantics, which is what
      every agent's ``state.get(...)`` access assumes. Pydantic validation
      belongs at the API boundary, not inside the hot loop.
    * ``total=False`` — states are built up incrementally; most keys are
      absent until their owning agent runs.
    * ``agent_trace`` carries an **accumulating reducer** so every agent's
      trace entry survives to the end (audit requirement). All other
      channels are last-value-wins: agents own their keys exclusively, and
      regeneration retries must *replace* stale responses, not append to
      them.

State flow (owner -> fields):
    SurveyDesigner   -> parsed_questions
    CohortSelector   -> selected_panelist_ids, selected_panelists,
                        cohort_demographics_summary, embedding_info
    TwinOrchestrator -> raw_responses, generation_metadata
    Validator        -> validated_responses, rejected_responses,
                        needs_regeneration, validation_retry_count
    CostAgent        -> total_cost_usd, total_tokens, cost_per_response,
                        budget_exceeded, cost_breakdown
    CalibrationAgent -> calibrated_distributions, calibration_metrics,
                        calibration_converged, calibration_convergence
    DiversityMonitor -> response_entropy, diversity_acceptable,
                        temperature_adjustments
    AuditAgent       -> results, insight_report, provenance_hash,
                        hallucination_rate, consistency_score, status
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


def _merge_lists(left: list, right: list) -> list:
    """Accumulating reducer: concatenate state updates for list channels."""
    return left + right


class AgentTraceEntry(BaseModel):
    """A single entry in the agent execution trace.

    Provides full provenance for every agent decision, enabling
    reproducibility and debugging. Required by the AuditAgent
    for governance compliance.
    """

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_name: str
    action: str = Field(description="What the agent did (e.g., 'selected_cohort')")
    input_summary: str = Field(default="", description="Summary of input received")
    output_summary: str = Field(default="", description="Summary of output produced")
    duration_ms: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SurveyPipelineState(TypedDict, total=False):
    """Shared state for the LangGraph survey orchestration pipeline.

    Every field has one owning agent (writer) and explicit consumers;
    see the module docstring for the ownership map.
    """

    # --- Input (set by the API caller / run_survey) ---
    raw_questions: list[str]
    question_context: str
    requested_cohort_size: int
    requested_models: list[str]
    cohort_filters: dict[str, str]
    random_seed: int | None
    status: str

    # --- SurveyDesigner ---
    parsed_questions: list[dict]

    # --- CohortSelector ---
    selected_panelist_ids: list[str]
    selected_panelists: list[dict]
    cohort_demographics_summary: dict[str, Any]
    embedding_info: dict[str, Any]

    # --- TwinOrchestrator ---
    raw_responses: list[dict]
    generation_metadata: dict[str, Any]

    # --- Validator ---
    validated_responses: list[dict]
    rejected_responses: list[dict]
    validation_retry_count: int
    needs_regeneration: bool

    # --- RedTeamAgent (adversarial validation, §4.5.15) ---
    red_team_rejected: list[dict]
    red_team_flag_summary: dict[str, int]

    # --- CalibrationAgent ---
    calibrated_distributions: dict[str, list[float]]
    calibration_metrics: list[dict]
    calibration_converged: bool
    calibration_convergence: dict[str, list[float]]

    # --- DiversityMonitor ---
    response_entropy: dict[str, float]
    diversity_acceptable: bool
    temperature_adjustments: dict[str, float]

    # --- CostAgent ---
    total_cost_usd: float
    total_tokens: int
    cost_per_response: float
    budget_exceeded: bool
    cost_breakdown: dict[str, float]

    # --- AuditAgent ---
    agent_trace: Annotated[list[dict], _merge_lists]
    provenance_hash: str
    results: list[dict]
    insight_report: dict[str, Any]
    hallucination_rate: float
    consistency_score: float
    error_message: str
