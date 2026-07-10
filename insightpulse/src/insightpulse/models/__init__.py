"""Data models for InsightPulse.

All data structures flowing through the system are defined as Pydantic v2
BaseModel subclasses. This provides runtime validation, serialization,
and clear documentation of every field.

Modules:
    panelist: Household and panelist demographic/behavioral data.
    survey: Survey questions, responses, and results.
    embedding: Behavioral embedding vectors and cluster assignments.
    calibration: BDCL calibration inputs, outputs, and metrics.
    agent_state: LangGraph shared state for agent orchestration.
"""

from insightpulse.models.agent_state import SurveyPipelineState
from insightpulse.models.calibration import (
    CalibrationInput,
    CalibrationMetrics,
    CalibrationOutput,
)
from insightpulse.models.embedding import (
    BehavioralEmbedding,
    ClusterAssignment,
    ConditioningVector,
)
from insightpulse.models.panelist import (
    DemographicProfile,
    Household,
    Panelist,
    PurchaseRecord,
)
from insightpulse.models.survey import (
    SurveyQuestion,
    SurveyResponse,
    SurveyResult,
    SurveyRun,
)

__all__ = [
    "BehavioralEmbedding",
    "CalibrationInput",
    "CalibrationMetrics",
    "CalibrationOutput",
    "ClusterAssignment",
    "ConditioningVector",
    "DemographicProfile",
    "Household",
    "Panelist",
    "PurchaseRecord",
    "SurveyPipelineState",
    "SurveyQuestion",
    "SurveyResponse",
    "SurveyResult",
    "SurveyRun",
]
