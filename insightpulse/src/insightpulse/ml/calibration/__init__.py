"""L4 calibration — Sinkhorn OT, fairness, regularization, strategies.

:func:`get_calibration_engine` is the composition-root factory (Strategy
pattern): demo/test resolve to :class:`SimpleCalibrationEngine`,
production to :class:`SinkhornCalibrationEngine`.
"""

from __future__ import annotations

from insightpulse.config.settings import Environment, get_settings
from insightpulse.data.repositories import get_data_repository
from insightpulse.ml.calibration.fairness import FairnessConstraintManager
from insightpulse.ml.calibration.pipeline import (
    CalibrationEngine,
    EmpiricalDistributionLoader,
    SimpleCalibrationEngine,
    SinkhornCalibrationEngine,
)
from insightpulse.ml.calibration.regularizer import BehavioralRegularizer
from insightpulse.ml.calibration.sinkhorn import SinkhornSolver, ordinal_cost_matrix

__all__ = [
    "BehavioralRegularizer",
    "CalibrationEngine",
    "EmpiricalDistributionLoader",
    "FairnessConstraintManager",
    "SimpleCalibrationEngine",
    "SinkhornCalibrationEngine",
    "SinkhornSolver",
    "get_calibration_engine",
    "ordinal_cost_matrix",
]


def get_calibration_engine(env: Environment | None = None) -> CalibrationEngine:
    """L4 factory: the environment's BDCL calibration engine.

    Both strategies calibrate against empirical targets loaded through the
    environment's own data repository.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use CalibrationEngine.
    """
    loader = EmpiricalDistributionLoader(get_data_repository(env))
    effective = env if env is not None else get_settings().env
    if effective == Environment.PRODUCTION:
        return SinkhornCalibrationEngine(target_loader=loader)
    return SimpleCalibrationEngine(target_loader=loader)
