# CLAUDE.md — InsightPulse Project Context

## Project Overview

**InsightPulse** is an Agentic AI framework for Synthetic Panelist Pulse Surveys,
developed as an M.Tech thesis project at IIT Madras in collaboration with NielsenIQ.

It creates AI-driven "digital twins" of real consumers that can answer survey questions
as synthetic panelists — replacing/complementing traditional surveys that suffer from
declining response rates (~36% in the 1990s → single digits today), high costs, and
respondent fatigue.

**Author:** Binesh Jose (CH24M521), M.Tech Industrial AI, IIT Madras
**Mentors:** Noah Tilzer, Vijayakumar Sivagnanam
**Phase:** Term 3 (Final) — Validation & Reporting

---

## Architecture (5 Layers)

1. **L1 — Data Layer**: Ingests panelist demographics, purchase data, survey history,
   product metadata. Supports NIQ proprietary data and public benchmarks (Pew ATP,
   ESS, Kaggle, Twin-2K-500).

2. **L2 — Feature Engineering & Embedding Layer**: Transformer-based sequence encoder
   produces behavioral embeddings B_i ∈ ℝ¹²⁸ from purchase sequences. K-Means
   clustering (K=5) discovers behavioral archetypes. Concatenates with demographic
   vectors to form conditioning inputs u_i = [z_i; d_i].

3. **L3 — Digital Twin Generative Layer**: LoRA fine-tuned LLM conditioned on u_i
   generates survey responses via structured persona prompts.

4. **L4 — BDCL (Behavioral-Demographic Calibration Layer)**: Optimal transport
   (Sinkhorn algorithm) aligns synthetic distributions P_syn with empirical P_real,
   subject to behavioral regularization and demographic fairness constraints.

5. **L5 — Agentic Orchestration (LangGraph DAG)**: 8 specialized agents coordinated
   as a directed acyclic graph:
   - SurveyDesigner → CohortSelector → TwinOrchestrator → Validator
   - CalibrationAgent, DiversityMonitor, CostAgent, AuditAgent

---

## Evaluator Feedback (MUST ADDRESS in Term 3)

These points must be answered in code, experiments, and report:

1. Multi-LLM comparison: document performance differences between model versions
2. Retraining pipeline: define time window, drift detection, trigger criteria
3. Sequential question dependency: incorporate into generation model
4. Metric justification: why cosine similarity, JS divergence, Wasserstein for this data
5. Real-world validation: cross-validate against Pew/ESS ground truth
6. Calibration parameters: document what changes when switching LLMs
7. Readable figures: all charts must be high-res with proper labels
8. Concise report: highlight problem gap clearly

---

## Tech Stack

- **Python 3.11+** with type hints everywhere
- **LangGraph** for agent orchestration (DAG with state checkpointing)
- **LiteLLM** for multi-model routing (Claude, OpenAI, Ollama)
- **FastAPI** for REST API
- **Streamlit** for dashboard (5 tabs: survey runner, results, experiments, validation, audit)
- **PyTorch** for transformer-based behavioral encoder
- **POT** (Python Optimal Transport) for BDCL Sinkhorn calibration
- **FAISS** for vector similarity search in cohort selection
- **scikit-learn** for K-Means clustering
- **Docker Compose** for containerization
- **SQLite** (demo) / **PostgreSQL** (production) for persistence
- **Pydantic v2** for all data models

---

## Coding Standards

- **Style**: Google Python Style Guide
- **Type hints**: Required on ALL function signatures (params + return)
- **Docstrings**: Google-style, required on all public classes and functions
- **Imports**: stdlib → third-party → local, separated by blank lines
- **Error handling**: Never bare `except:`; always catch specific exceptions
- **Logging**: Use `structlog` with context binding, never `print()`
- **Models**: Pydantic v2 BaseModel for all data structures
- **Tests**: pytest with fixtures in conftest.py, >80% coverage target
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Constants**: UPPER_SNAKE_CASE, defined in config, never hardcoded
- **Comments**: Explain WHY, not WHAT — code should be self-documenting
- **Line length**: 100 characters max
- **No magic numbers**: All thresholds and hyperparameters in config

---

## Environment Profiles

Switch with `ENV` environment variable:
- `demo`: SQLite, in-memory cache, synthetic data, single Docker Compose
- `production`: PostgreSQL, Redis, NIQ data via API, Kubernetes, Terraform
- `test`: Mock LLMs, pytest fixtures, CI runner

---

## Key Metrics (from thesis results)

- Behavioral fidelity: cosine similarity = 0.84
- Calibration accuracy: JS divergence = 0.017, Wasserstein = 0.041
- Hallucination rate: 1.9% (vs 7.8% baseline)
- Logical consistency: 94.6%
- Throughput: 1,248 respondents/min
- Response diversity: Shannon entropy = 2.31

---

## Commands

```bash
# Start demo environment
make demo

# Run tests
make test

# Run experiments
make experiments

# Generate sample data
make generate-data

# Lint and format
make lint
```
