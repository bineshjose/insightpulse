"""LangGraph shared state for the agent orchestration DAG.

This is the central data structure that flows through the LangGraph
pipeline. Each agent reads from and writes to this state, enabling
typed, validated communication between agents without custom protocols.

The state replaces the custom A2A (Agent-to-Agent) protocol from the
original thesis with LangGraph's built-in typed state management —
achieving the same functionality with better tooling support.

State Flow:
    SurveyDesigner → writes parsed questions + context
    CohortSelector → writes selected panelist cohort
    TwinOrchestrator → writes raw synthetic responses
    Validator → writes validation results, may flag for regeneration
    CalibrationAgent → writes calibrated distributions
    DiversityMonitor → writes entropy metrics, may adjust temperature
    CostAgent → writes cost tracking, may halt if over budget
    AuditAgent → writes complete provenance log
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _merge_lists(left: list, right: list) -> list:
    """Merge strategy for list fields in LangGraph state.

    LangGraph uses reducer functions to merge state updates from
    parallel agent executions. For lists, we concatenate.
    """
    return left + right


def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge strategy for dict fields — right-side wins on conflicts."""
    return {**left, **right}


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


class SurveyPipelineState(BaseModel):
    """Shared state for the LangGraph survey orchestration pipeline.

    This TypedDict-style state is the single source of truth for the
    entire agent DAG. Each agent reads what it needs and writes its
    outputs back. LangGraph handles state persistence, checkpointing,
    and replay.

    Design principle: every field has a clear owner (the agent that
    writes it) and clear consumers (the agents that read it).
    """

    # --- Run Identity ---
    run_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(
        default="initialized",
        description="Pipeline status: initialized → running → completed | failed",
    )

    # --- Input (set by API caller) ---
    raw_questions: list[str] = Field(
        default_factory=list,
        description="Raw survey question texts from the user",
    )
    question_context: str = Field(
        default="",
        description="Domain context for the survey (e.g., 'snack food market in the US')",
    )
    requested_cohort_size: int = Field(default=100, ge=1)
    requested_models: list[str] = Field(
        default_factory=list,
        description="LLM models to use (empty = use default)",
    )
    cohort_filters: dict[str, str] = Field(
        default_factory=dict,
        description="Demographic filters for cohort selection",
    )
    random_seed: int | None = Field(default=None)

    # --- SurveyDesigner Output ---
    parsed_questions: list[dict] = Field(
        default_factory=list,
        description="Structured SurveyQuestion objects (serialized)",
    )

    # --- CohortSelector Output ---
    selected_panelist_ids: list[str] = Field(
        default_factory=list,
        description="IDs of selected synthetic respondents",
    )
    cohort_demographics_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics of the selected cohort",
    )

    # --- TwinOrchestrator Output ---
    raw_responses: Annotated[list[dict], _merge_lists] = Field(
        default_factory=list,
        description="Raw synthetic responses before validation",
    )
    generation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Token counts, costs, latencies from generation",
    )

    # --- Validator Output ---
    validated_responses: list[dict] = Field(
        default_factory=list,
        description="Responses that passed validation",
    )
    rejected_responses: list[dict] = Field(
        default_factory=list,
        description="Responses that failed validation with reasons",
    )
    validation_retry_count: int = Field(default=0, ge=0)
    needs_regeneration: bool = Field(
        default=False,
        description="Flag for TwinOrchestrator to regenerate rejected responses",
    )

    # --- CalibrationAgent Output ---
    calibrated_distributions: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Post-BDCL calibrated distributions keyed by question_id",
    )
    calibration_metrics: list[dict] = Field(
        default_factory=list,
        description="CalibrationMetrics for each question",
    )
    calibration_converged: bool = Field(default=True)

    # --- DiversityMonitor Output ---
    response_entropy: dict[str, float] = Field(
        default_factory=dict,
        description="Shannon entropy per question",
    )
    diversity_acceptable: bool = Field(
        default=True,
        description="Whether diversity meets minimum threshold",
    )
    temperature_adjustments: dict[str, float] = Field(
        default_factory=dict,
        description="Temperature adjustments applied per question",
    )

    # --- CostAgent Output ---
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    cost_per_response: float = Field(default=0.0, ge=0.0)
    budget_exceeded: bool = Field(default=False)
    cost_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by model and agent",
    )

    # --- AuditAgent Output ---
    agent_trace: Annotated[list[dict], _merge_lists] = Field(
        default_factory=list,
        description="Ordered execution trace for reproducibility",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of inputs + config for provenance verification",
    )

    # --- Final Results ---
    results: list[dict] = Field(
        default_factory=list,
        description="Final SurveyResult objects (serialized)",
    )
    error_message: str = Field(default="")
