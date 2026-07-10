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

from insightpulse.models.panelist import (
    Panelist,
    Household,
    PurchaseRecord,
    DemographicProfile,
)
from insightpulse.models.survey import (
    SurveyQuestion,
    SurveyResponse,
    SurveyResult,
    SurveyRun,
)
from insightpulse.models.embedding import (
    BehavioralEmbedding,
    ClusterAssignment,
    ConditioningVector,
)
from insightpulse.models.calibration import (
    CalibrationInput,
    CalibrationOutput,
    CalibrationMetrics,
)
from insightpulse.models.agent_state import SurveyPipelineState

__all__ = [
    "Panelist",
    "Household",
    "PurchaseRecord",
    "DemographicProfile",
    "SurveyQuestion",
    "SurveyResponse",
    "SurveyResult",
    "SurveyRun",
    "BehavioralEmbedding",
    "ClusterAssignment",
    "ConditioningVector",
    "CalibrationInput",
    "CalibrationOutput",
    "CalibrationMetrics",
    "SurveyPipelineState",
]
