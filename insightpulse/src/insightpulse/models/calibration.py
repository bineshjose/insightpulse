"""Calibration data models for the Behavioral-Demographic Calibration Layer (BDCL).

The BDCL uses optimal transport (Sinkhorn algorithm) to align synthetic
response distributions P_syn with empirical distributions P_real, subject
to behavioral regularization and demographic fairness constraints.

These models capture the inputs, outputs, and quality metrics of the
calibration process.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CalibrationInput(BaseModel):
    """Input to the BDCL calibration process.

    Contains the raw synthetic distribution to be calibrated, the
    target empirical distribution, and any constraints.
    """

    question_id: str = Field(description="Question being calibrated")

    # Distribution arrays — each entry corresponds to a response option
    synthetic_distribution: list[float] = Field(
        description="P_syn: raw synthetic response distribution (sums to 1.0)"
    )
    empirical_distribution: list[float] = Field(
        description="P_real: target empirical distribution from real surveys"
    )
    option_labels: list[str] = Field(
        description="Labels for each distribution bin"
    )

    # Behavioral constraint: similarity matrix between respondents
    behavioral_similarity_matrix: list[list[float]] | None = Field(
        default=None,
        description=(
            "Pairwise cosine similarity matrix between behavioral embeddings. "
            "Used as regularization to preserve behavioral structure during transport."
        ),
    )

    # Demographic constraints: target marginals per group
    demographic_targets: dict[str, list[float]] | None = Field(
        default=None,
        description=(
            "Target distribution per demographic group. Keys are group names "
            "(e.g., 'age_18_24'), values are target distributions."
        ),
    )

    # Hyperparameters (can override config defaults per-question)
    lambda_behavioral: float | None = Field(
        default=None,
        description="Override for behavioral regularization weight",
    )
    lambda_fairness: float | None = Field(
        default=None,
        description="Override for fairness constraint weight",
    )


class CalibrationOutput(BaseModel):
    """Output of the BDCL calibration process.

    Contains the calibrated distribution, the optimal transport plan,
    and convergence diagnostics.
    """

    question_id: str
    calibrated_distribution: list[float] = Field(
        description="P_calibrated: distribution after optimal transport alignment"
    )

    # Transport plan (the coupling matrix)
    transport_plan: list[list[float]] | None = Field(
        default=None,
        description="Optimal transport coupling matrix Γ* from Sinkhorn",
    )

    # Calibration weights per respondent
    respondent_weights: list[float] = Field(
        default_factory=list,
        description="Post-calibration weights w_i for each synthetic respondent",
    )

    # Convergence diagnostics
    converged: bool = Field(default=True, description="Whether Sinkhorn converged")
    iterations_used: int = Field(default=0, ge=0)
    final_error: float = Field(default=0.0, ge=0.0)
    convergence_history: list[float] = Field(
        default_factory=list,
        description="Sinkhorn error at each iteration (for convergence plots)",
    )


class CalibrationMetrics(BaseModel):
    """Quality metrics for calibration evaluation.

    These metrics quantify how well the calibrated synthetic distribution
    matches the empirical distribution, and are used in the thesis
    experiments and evaluator feedback responses.
    """

    question_id: str

    # --- Distribution Alignment ---
    wasserstein_before: float = Field(
        description="Wasserstein distance before calibration"
    )
    wasserstein_after: float = Field(
        description="Wasserstein distance after calibration"
    )
    js_divergence_before: float = Field(
        description="JS divergence before calibration"
    )
    js_divergence_after: float = Field(
        description="JS divergence after calibration"
    )

    # --- Improvement ---
    wasserstein_improvement_pct: float = Field(
        description="Percentage improvement in Wasserstein distance"
    )
    js_improvement_pct: float = Field(
        description="Percentage improvement in JS divergence"
    )

    # --- Fairness (per demographic group) ---
    demographic_parity_before: dict[str, float] = Field(
        default_factory=dict,
        description="Group parity difference before calibration",
    )
    demographic_parity_after: dict[str, float] = Field(
        default_factory=dict,
        description="Group parity difference after calibration",
    )

    @property
    def calibration_effective(self) -> bool:
        """Whether calibration meaningfully improved alignment.

        A calibration is considered effective if it reduced the
        Wasserstein distance by at least 20%.
        """
        return self.wasserstein_improvement_pct >= 20.0
