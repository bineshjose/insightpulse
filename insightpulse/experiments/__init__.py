"""Reproducible thesis experiments.

Each module is runnable via ``python -m experiments.<name>`` (or
``make experiments``), takes ``--seed``, and writes JSON results plus a
300-dpi figure to ``experiments/results/``:

    multi_llm_comparison    — evaluator feedback #1, #6, #8
    calibration_convergence — Sinkhorn ε sweep (speed vs plan sharpness)
    drift_detection         — evaluator feedback #2 (retraining trigger)
    sequential_dependency   — evaluator feedback #3

The hyperparameter/ablation sweep scripts reproduce the thesis result tables
and take ``--mode {demo,production}`` (demo = synthetic + simulated, the
default; production = real ml/ modules). They write CSVs to ``data/demo/``:

    embedding_sweeps        — chunking, encoder arch, dimension d, clustering, K
    generation_sweeps       — prompting, temperature, retrieval, parsing
    calibration_sweeps      — ε, λ_b, λ_f, Sinkhorn iteration budget
    ablation_study          — six-component removal + progressive build-up
    finetuning_comparison   — full FT / LoRA / QLoRA / prompt conditioning
    cross_validation        — NIQ / Pew / ESS / Twin-2K benchmarks

Shared figure style and output helpers live in :mod:`experiments.common`.
"""

__all__: list[str] = []
