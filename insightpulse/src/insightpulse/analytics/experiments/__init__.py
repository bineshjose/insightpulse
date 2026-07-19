"""Research experiment framework: recorded results + summaries + charts.

Twenty-three experiment classes over 23 result files (``results/*.json``).
The registry maps result-file stems to classes so both UIs and the API
resolve experiments by name:

    from insightpulse.analytics.experiments import get_experiment
    exp = get_experiment("temperature_sweep")
    exp.load_results(); exp.get_summary(); exp.generate_chart()

Result files without a dedicated class are reachable through
:func:`load_result_file`.
"""

from __future__ import annotations

from insightpulse.analytics.experiments.base import (
    RESULTS_DIR,
    BaseExperiment,
    load_result_file,
)
from insightpulse.analytics.experiments.calibration_experiments import (
    BdclBeforeAfterExperiment,
    BehaviouralWeightExperiment,
    EpsilonSensitivityExperiment,
    FairnessWeightExperiment,
    SinkhornConvergenceExperiment,
)
from insightpulse.analytics.experiments.embedding_experiments import (
    ChunkingStrategyExperiment,
    ClusteringAlgorithmExperiment,
    EmbeddingDimensionExperiment,
    EncoderArchitectureExperiment,
)
from insightpulse.analytics.experiments.generation_experiments import (
    FinetuningComparisonExperiment,
    PromptingStrategyExperiment,
    ResponseParsingExperiment,
    RetrievalStrategyExperiment,
    TemperatureSweepExperiment,
)
from insightpulse.analytics.experiments.pipeline_experiments import (
    AblationStudyExperiment,
    BenchmarkValidationExperiment,
    DriftDetectionExperiment,
    FailureAnalysisExperiment,
    HyperparameterSearchExperiment,
    MultiModelExperiment,
    SequentialDependencyExperiment,
    SotaComparisonExperiment,
    TimingBenchmarksExperiment,
)

ALL_EXPERIMENTS: dict[str, type[BaseExperiment]] = {
    cls.name: cls
    for cls in (
        ChunkingStrategyExperiment,
        EncoderArchitectureExperiment,
        EmbeddingDimensionExperiment,
        ClusteringAlgorithmExperiment,
        PromptingStrategyExperiment,
        TemperatureSweepExperiment,
        RetrievalStrategyExperiment,
        ResponseParsingExperiment,
        FinetuningComparisonExperiment,
        BdclBeforeAfterExperiment,
        SinkhornConvergenceExperiment,
        EpsilonSensitivityExperiment,
        BehaviouralWeightExperiment,
        FairnessWeightExperiment,
        MultiModelExperiment,
        SequentialDependencyExperiment,
        AblationStudyExperiment,
        DriftDetectionExperiment,
        FailureAnalysisExperiment,
        TimingBenchmarksExperiment,
        SotaComparisonExperiment,
        BenchmarkValidationExperiment,
        HyperparameterSearchExperiment,
    )
}

__all__ = [
    "ALL_EXPERIMENTS",
    "RESULTS_DIR",
    "BaseExperiment",
    "get_experiment",
    "list_experiments",
    "load_result_file",
]


def get_experiment(name: str) -> BaseExperiment:
    """Instantiate an experiment by its registry name.

    Args:
        name: Result-file stem, e.g. ``"temperature_sweep"``.

    Returns:
        A ready-to-use experiment instance.

    Raises:
        KeyError: When the name is not registered.
    """
    try:
        return ALL_EXPERIMENTS[name]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown experiment '{name}'. Known: {sorted(ALL_EXPERIMENTS)}"
        ) from exc


def list_experiments() -> list[str]:
    """Sorted registry names."""
    return sorted(ALL_EXPERIMENTS)
