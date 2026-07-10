"""Reproducible thesis experiments.

Each module is runnable via ``python -m experiments.<name>`` (or
``make experiments``), takes ``--seed``, and writes JSON results plus a
300-dpi figure to ``experiments/results/``:

    multi_llm_comparison    — evaluator feedback #1, #6, #8
    calibration_convergence — Sinkhorn ε sweep (speed vs plan sharpness)
    drift_detection         — evaluator feedback #2 (retraining trigger)
    sequential_dependency   — evaluator feedback #3

Shared figure style and output helpers live in :mod:`experiments.common`.
"""

__all__: list[str] = []
