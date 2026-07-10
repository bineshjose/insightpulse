# Evaluator Feedback Matrix

Each Term-2 evaluation point mapped to where it is addressed in code,
experiments, and documentation. Every referenced artifact is in this
repository and reproducible (`make generate-data && make test &&
make experiments`).

| # | Feedback | Code | Experiments / evidence | Docs |
|---|---|---|---|---|
| 1 | **Multi-LLM comparison** — document performance differences between model versions | `llm/router.py` (per-call cost/latency accounting, `get_performance_summary`); `layers/generative_layer.py` (per-family sampling profiles); `simulation.py` (per-model bias profiles) | `experiments/multi_llm_comparison.py` — identical survey/cohort/seed across Claude Sonnet, GPT-4o, Llama 3.1; hallucination 1.8% vs 2.8% vs 7.4%; `--live` flag for real-provider runs | [ADR-002](adr/002-litellm-multi-model-routing.md) |
| 2 | **Retraining pipeline** — time window, drift detection, trigger criteria | `layers/insight_layer.py::DriftDetector` (rolling JS vs 3-month baseline, trigger = noise mean + 3σ, consecutive + hard rules); `layers/data_layer.py::get_data_version` (change signal); version-keyed embedding cache invalidation in L2 | `experiments/drift_detection.py` — trigger derived from the stationary noise floor (0.0055), zero false alarms, injected drift caught in month 2 | [architecture.md](architecture.md) §cross-cutting |
| 3 | **Sequential question dependency** — incorporate into the generation model | `layers/generative_layer.py` — prior answers injected into subsequent prompts (`PersonaPromptBuilder.build(prior_responses=...)`); questions processed strictly in order | `experiments/sequential_dependency.py` — within-person Spearman 0.28 → 0.65, contradictions 10.4% → 1.8%, marginals provably unchanged | tested in `tests/test_layers.py::TestLLMGenerationEngine::test_sequential_prior_answers_reach_prompts` |
| 4 | **Metric justification** — why cosine, JS, Wasserstein for this data | `utils/metrics.py` — every metric's docstring states the property that fits survey data (bounded/symmetric JS for empty bins; ordinal-aware Wasserstein for Likert; angular cosine for unit-norm embeddings; entropy against mode collapse) | metric behavior pinned by `tests/` assertion suite (ordinality, symmetry, bounds) | dashboard Validation tab ships the justification table |
| 5 | **Real-world validation** — cross-validate against Pew/ESS ground truth | `layers/calibration_layer.py::EmpiricalDistributionLoader` — calibration targets come from the empirical response bank through L1; benchmark CSVs drop into the same schema | dashboard Validation tab: per-question pass/fail vs thesis targets; Pew ATP / ESS / Twin-2K-500 integration slots | [README](../README.md) §validation |
| 6 | **Calibration parameters when switching LLMs** — document what changes | Each model's raw distribution is calibrated separately; `CalibrationMetrics` (before/after per model) quantifies the transport work; per-family sampling profiles in `MODEL_PARAMETER_PROFILES` | multi-LLM experiment's raw-vs-calibrated split shows model-specific mode collapse ⇒ model-specific transport work | [ADR-003](adr/003-sinkhorn-optimal-transport.md) |
| 7 | **Readable figures** — high-res, properly labeled | `experiments/common.py` — 300-dpi savefig, labeled axes, direction-of-better annotations, CVD-validated palette shared with the dashboard | all four experiment figures in `experiments/results/` regenerate via `make experiments` | — |
| 8 | **Concise reporting** — highlight the problem gap clearly | `layers/insight_layer.py::ReportGenerator` — versioned, structured JSON per run; `README.md` leads with the problem gap (36% → single-digit response rates) | experiment JSON summaries carry one-paragraph interpretations | this matrix; [architecture.md](architecture.md) |

## Where to look during evaluation

1. **Contract → production → demo** reading order per layer:
   `src/insightpulse/layers/<layer>.py` (ABC first, then strategies).
2. **Evidence chain** per feedback point: the experiment script, its JSON
   output, and its figure share a name under `experiments/`.
3. **Reproducibility**: every experiment takes `--seed`; survey runs are
   provenance-hashed and replay identically (proven by
   `tests/test_integration.py::test_run_survey_is_reproducible`).
