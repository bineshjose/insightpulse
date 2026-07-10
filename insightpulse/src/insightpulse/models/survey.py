"""Survey question, response, and result data models.

These models define the complete lifecycle of a synthetic survey:
from question definition through response generation to aggregated
results with calibration and validation metrics.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Question Types
# ---------------------------------------------------------------------------

class QuestionType(StrEnum):
    """Supported survey question types.

    Each type determines how the LLM is prompted and how responses
    are parsed, validated, and aggregated.
    """

    SINGLE_CHOICE = "single_choice"       # Pick one from options
    MULTIPLE_CHOICE = "multiple_choice"   # Pick multiple from options
    LIKERT_5 = "likert_5"                 # 1-5 scale agreement
    LIKERT_7 = "likert_7"                 # 1-7 scale agreement
    OPEN_ENDED = "open_ended"             # Free-text response
    RANKING = "ranking"                   # Rank options in order
    NET_PROMOTER = "net_promoter"         # 0-10 NPS scale


# ---------------------------------------------------------------------------
# Survey Question
# ---------------------------------------------------------------------------

class SurveyQuestion(BaseModel):
    """A single survey question to be answered by synthetic panelists.

    The question definition includes the text, type, options (if applicable),
    and optional context that helps the LLM understand the domain.
    Sequential questions include prior_responses to model dependencies
    (addresses evaluator feedback #4: sequential question dependency).
    """

    question_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique question identifier",
    )
    text: str = Field(description="The survey question text")
    question_type: QuestionType = Field(default=QuestionType.SINGLE_CHOICE)
    options: list[str] = Field(
        default_factory=list,
        description="Response options (for choice-based questions)",
    )
    category: str = Field(
        default="general",
        description="Question category (e.g., 'brand_perception', 'purchase_intent')",
    )
    context: str = Field(
        default="",
        description=(
            "Additional context for the LLM about the survey domain, "
            "product category, or market situation."
        ),
    )
    is_sequential: bool = Field(
        default=False,
        description="Whether this question depends on previous answers in the survey",
    )
    prior_questions: list[SurveyQuestion] = Field(
        default_factory=list,
        description=(
            "Previous questions in the sequence, used to maintain "
            "logical consistency across multi-question surveys."
        ),
    )


# ---------------------------------------------------------------------------
# Individual Response
# ---------------------------------------------------------------------------

class SurveyResponse(BaseModel):
    """A single synthetic response from one digital twin.

    Contains the generated answer along with metadata about the
    generation process — which model was used, the confidence score,
    the panelist's demographic profile, and the behavioral cluster.
    """

    response_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique response identifier",
    )
    question_id: str = Field(description="ID of the question being answered")
    panelist_id: str = Field(description="ID of the synthetic respondent household")
    answer: str = Field(description="The generated response text or selected option")
    answer_index: int | None = Field(
        default=None,
        description="Index of selected option (for choice-based questions)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence in this response",
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning chain explaining why this answer was chosen",
    )

    # --- Generation Metadata ---
    model_used: str = Field(default="", description="LLM model that generated this response")
    generation_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to generate this response in milliseconds",
    )
    token_count: int = Field(default=0, ge=0, description="Total tokens consumed")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated cost in USD")

    # --- Respondent Context ---
    behavioral_cluster: int = Field(
        default=-1,
        description="K-Means cluster assignment (0-4 for K=5)",
    )
    demographic_summary: str = Field(
        default="",
        description="Natural-language summary of respondent demographics",
    )

    # --- Validation Flags ---
    is_valid: bool = Field(default=True, description="Whether this response passed validation")
    validation_flags: list[str] = Field(
        default_factory=list,
        description="List of validation issues (e.g., 'hallucination_detected', 'inconsistent')",
    )

    # --- Sequential Context ---
    prior_responses: list[str] = Field(
        default_factory=list,
        description="Responses to prior questions in a sequential survey",
    )


# ---------------------------------------------------------------------------
# Aggregated Results
# ---------------------------------------------------------------------------

class ResponseDistribution(BaseModel):
    """Distribution of responses across options for a single question.

    Used for calibration alignment and validation against empirical
    survey distributions.
    """

    option: str = Field(description="The response option")
    count: int = Field(default=0, ge=0, description="Number of respondents who chose this")
    percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    calibrated_percentage: float | None = Field(
        default=None,
        description="Percentage after BDCL calibration",
    )


class SurveyResult(BaseModel):
    """Aggregated results for a single survey question.

    Contains the response distribution (before and after calibration),
    demographic breakdowns, and all evaluation metrics required by
    the thesis and evaluator feedback.
    """

    question_id: str
    question_text: str
    total_responses: int = Field(ge=0)
    valid_responses: int = Field(ge=0)

    # --- Response Distribution ---
    distribution: list[ResponseDistribution] = Field(default_factory=list)

    # --- Evaluation Metrics ---
    # (addresses evaluator feedback #2: justify these metrics)
    cosine_similarity: float | None = Field(
        default=None,
        description=(
            "Cosine similarity between synthetic and empirical behavioral-response "
            "coupling vectors. Chosen because behavioral embeddings are unit-normalized "
            "vectors where angular distance captures behavioral alignment."
        ),
    )
    js_divergence: float | None = Field(
        default=None,
        description=(
            "Jensen-Shannon divergence between synthetic and empirical response "
            "distributions. Chosen because it is symmetric, bounded [0,1], and "
            "well-defined for distributions with zero-probability bins — unlike KL "
            "divergence which diverges to infinity."
        ),
    )
    wasserstein_distance: float | None = Field(
        default=None,
        description=(
            "Wasserstein-1 distance measuring the minimum 'work' to transform "
            "the synthetic distribution into the empirical one. Chosen because "
            "it respects the ordinal structure of Likert scales and captures "
            "distributional shift even when supports don't overlap."
        ),
    )
    shannon_entropy: float | None = Field(
        default=None,
        description=(
            "Shannon entropy of the response distribution, measuring diversity. "
            "Chosen because it quantifies whether synthetic responses exhibit "
            "the natural variability of real populations vs. mode collapse."
        ),
    )

    # --- Fairness Metrics ---
    demographic_parity: dict[str, float] = Field(
        default_factory=dict,
        description="Parity differences across demographic groups (age, income, region)",
    )

    # --- Demographic Breakdowns ---
    by_age_group: dict[str, list[ResponseDistribution]] = Field(default_factory=dict)
    by_income_group: dict[str, list[ResponseDistribution]] = Field(default_factory=dict)
    by_region: dict[str, list[ResponseDistribution]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Survey Run (Top-Level Container)
# ---------------------------------------------------------------------------

class SurveyRun(BaseModel):
    """A complete survey execution run with full provenance.

    This is the top-level container that the AuditAgent logs for
    reproducibility. Contains everything needed to replay the
    exact same survey under the same conditions.
    """

    run_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # --- Input ---
    questions: list[SurveyQuestion]
    cohort_size: int = Field(ge=1)
    cohort_criteria: dict[str, str] = Field(
        default_factory=dict,
        description="Criteria used for cohort selection (e.g., age_group='25-34')",
    )

    # --- Configuration ---
    models_used: list[str] = Field(
        default_factory=list,
        description="LLM models used in this run",
    )
    calibration_applied: bool = Field(default=True)
    random_seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )

    # --- Output ---
    responses: list[SurveyResponse] = Field(default_factory=list)
    results: list[SurveyResult] = Field(default_factory=list)

    # --- Run Metrics ---
    total_time_seconds: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    hallucination_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Agent Trace ---
    agent_trace: list[dict] = Field(
        default_factory=list,
        description="Ordered log of agent executions with inputs/outputs",
    )
