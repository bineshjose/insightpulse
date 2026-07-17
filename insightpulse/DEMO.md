# InsightPulse — Code Demo Reference Document
## M.Tech Thesis | CH24M521 | IIT Madras + NielsenIQ

> This document serves as the complete reference for  code demo video.

---

## 1. Project Overview

### 1a. Problem Statement

Traditional consumer survey panels face a structural crisis: response rates have declined below 5%, each survey wave costs $4,000–$8,000, turnaround takes 3–6 weeks, and demographic gaps are growing as younger and minority populations drop out. NielsenIQ, the world's largest consumer intelligence company, needs a scalable alternative that maintains statistical rigour.

### 1b. Objectives and Application Domain

InsightPulse generates synthetic survey respondents using AI-driven consumer digital twins. The system jointly integrates:
- Behavioural representation learning (purchase data → 128-dim embeddings)
- LLM-based response generation (persona + CoT prompting)
- Optimal transport calibration (BDCL — aligns synthetic distributions with real benchmarks)
- Multi-agent orchestration (8 LangGraph agents with adversarial validation)

**Domain:** Consumer research / FMCG market intelligence for NielsenIQ

**What it delivers:** A 200-respondent synthetic pulse survey in 3.2 minutes at $2.14, versus $4K–$8K and 3–6 weeks traditionally. Throughput: 1,248 respondents/minute.

---

## 2. Programming Language and Frameworks

### 2a. Languages

| Language | Usage | Version |
|----------|-------|---------|
| Python | Backend, ML, API, agents, calibration | 3.11.7 |
| TypeScript | Next.js production frontend | 5.x |
| HCL | Terraform infrastructure-as-code | 1.6+ |
| YAML | Configuration profiles, CI/CD workflows | — |
| SQL | Snowflake queries, PostgreSQL schemas | — |

### 2b. Libraries and Frameworks (with versions)

**ML and Embeddings:**

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.2 (CPU build) | Transformer encoder training (InfoNCE contrastive loss) |
| scikit-learn | 1.4.0 | K-Means clustering (K=5), silhouette score, metrics |
| FAISS-cpu | 1.7.4 | IVFFlat nearest-neighbour index (256 cells, 1.2ms query) |

**LLM and Agentic:**

| Library | Version | Purpose |
|---------|---------|---------|
| LangGraph | 0.2.1 | Directed acyclic graph for 8-agent orchestration |
| LiteLLM | 1.35 | Unified routing to 4 LLM providers (Claude, GPT-4o, Haiku, Llama) |
| vLLM | 0.4.1 | Local Llama 3.1 8B inference serving |

**Optimal Transport:**

| Library | Version | Purpose |
|---------|---------|---------|
| POT | 0.9.1 | Sinkhorn optimal transport solver for BDCL calibration |

**API and Data Validation:**

| Library | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.139.0 | REST API server with async support |
| Pydantic | 2.6 | Data validation, TypedDict state schemas, config models |
| SQLAlchemy | 2.0.51 | ORM for PostgreSQL metadata and audit trail |
| Alembic | 1.15.2 | Database migration management |

**Data Connectors:**

| Library | Version | Purpose |
|---------|---------|---------|
| snowflake-connector-python | 3.15.0 | NIQ panel data (PB-scale, 39,305 HH) |
| azure-storage-blob | 12.24.1 | ADLS access for Pew, ESS, Twin-2K datasets |
| redis | 5.2.1 | Working-set cache (cohort data, 1-hour TTL) |
| psycopg2 | 2.9.10 | PostgreSQL connection (benchmarks, audit trail) |

**Frontend and Visualisation:**

| Library | Version | Purpose |
|---------|---------|---------|
| Streamlit | 1.45.1 | 9-page analytics dashboard |
| Next.js 15 / React 18 | 15.3.3 | Production frontend mirror |
| Plotly | 6.1.2 | Interactive charts in dashboard |

**Observability:**

| Library | Version | Purpose |
|---------|---------|---------|
| prometheus-client | 0.22.1 | 20 custom metrics (latency, throughput, token usage) |
| opentelemetry-sdk | 1.33.0 | Distributed tracing (1 trace/run, child span/agent) |
| structlog | 25.4.0 | Structured JSON logging (PII-redacted) |

**Testing:**

| Library | Version | Purpose |
|---------|---------|---------|
| pytest | 8.4.1 | 181 tests, 86% line coverage |
| pytest-asyncio | 1.0.0 | Async agent and API tests |
| pytest-cov | 6.1.1 | Coverage reporting |

**DevOps:**

| Tool | Purpose |
|------|---------|
| Docker | Multi-stage container builds |
| Terraform | Azure infrastructure provisioning (AKS, ACR, KeyVault) |
| GitHub Actions | CI/CD pipeline (lint → test → build → deploy) |
| Helm | Kubernetes chart for AKS deployment |

---

## 3. Compute Resources

### 3a. CPU/GPU Configuration

| Resource | Specification | Usage |
|----------|---------------|-------|
| Primary compute | Azure D4s_v3 (4 vCPU, 16 GB RAM) | All experiments, API serving, pipeline execution |
| GPU (optional) | NVIDIA V100 (single) | Local Llama 3.1 8B inference via vLLM |
| Embedding training | CPU-only (PyTorch 2.2 CPU build) | 45 min for full encoder training |

### 3b. RAM and Storage

| Component | Size | Details |
|-----------|------|---------|
| FAISS index | 19.2 MB | 39,305 × 128-dim, IVFFlat with 256 cells |
| PostgreSQL | ~500 MB | Benchmarks, metadata, audit trail (Alembic migrations) |
| Redis cache | ~200 MB | Working-set cohort cache, 1-hour TTL |
| Snowflake | PB-scale | NIQ panel data queried in-place (no local copy) |
| Docker image | ~1.2 GB | Multi-stage build, Python 3.11 slim base |

### 3c. Cloud / Execution Environment

| Service | Purpose |
|---------|---------|
| Azure AKS | Kubernetes cluster for API and pipeline |
| Azure ACR | Container registry for Docker images |
| Azure Key Vault | Secrets management (API keys, DB credentials) |
| Azure VNet + NSGs | Network security (HTTPS:443 only) |
| Snowflake | NIQ data warehouse (queried in-place) |
| Azure Data Lake (ADLS) | Storage for Pew, ESS, Twin-2K datasets |

**Local development:** Ubuntu 22.04 LTS, Python 3.11.7 virtualenv, seed 42 for reproducibility.

---

## 4. Dataset Details

### 4a. Dataset Sources and Size

| Dataset | Records | Source | Format |
|---------|---------|--------|--------|
| NIQ Consumer Panel | 6,482,173 transactions | Snowflake | Parquet |
| NIQ Panelist Demographics | 39,305 households | Snowflake | Parquet |
| NIQ RMS Store-Item-Week | 2,148,552 records | Snowflake | Parquet |
| NIQ Survey Dataset | 218,674 responses (~4,200 respondents) | Snowflake | Parquet |
| Pew American Trends Panel | 12,400 responses | ADLS | .sav |
| European Social Survey R11 | 8,200 responses | ADLS | .sav |
| Twin-2K-500 | 500 persona profiles | ADLS | Parquet |
| Kaggle Consumer Behaviour | Variable | CSV download | CSV |

**Observation period:** January 2024 – December 2025 (24 months)

### 4b. Number of Samples / Classes / Tokens

| Metric | Value |
|--------|-------|
| Households | 39,305 |
| Total transactions | 6,482,173 |
| Product categories (vocabulary) | 247 |
| Behavioural archetypes (K) | 5 |
| Embedding dimension (d) | 128 |
| Evaluation cohort | 200 respondents (archetype-stratified) |
| Evaluation questions | 10 (4 Likert, 3 categorical, 3 mixed with skip logic) |
| Benchmark respondents | 2,547 (resolution ±3.2 pp per Proposition 2) |

### 4c. Preprocessing

**8-Rule Quality Framework** (implemented in `src/insightpulse/data/`):

| # | Gate | What it checks |
|---|------|----------------|
| 1 | Duplicates | Exact and near-duplicate records |
| 2 | Referential integrity | Foreign keys valid across datasets |
| 3 | Type constraints | Column types match schema_map.yaml |
| 4 | Value ranges | Numeric values within plausible bounds |
| 5 | Missingness | Fields required vs optional, imputation rules |
| 6 | Outliers | Grubbs test for extreme values |
| 7 | Temporal order | Purchase timestamps monotonically increasing |
| 8 | Drift baseline | Distribution shift vs previous load |

**Tokenisation:** 60-day sliding windows (≥5 events/window, covers 96% of panel). Each purchase event → triple: category (247 vocab) · price quintile · promotional flag.

**No traditional train-test split:** The embedding encoder trains on the full 39,305-household panel. Evaluation uses a 200-respondent archetype-stratified cohort. Cross-validation uses 3 external benchmarks with zero training overlap (Pew, ESS, Twin-2K). Results reported over 5 runs (seeds 42–46).

---

## 5. Tools and Execution Environment

### ML Frameworks
- PyTorch 2.2 (CPU): Transformer encoder, InfoNCE loss, cosine annealing
- scikit-learn 1.4.0: K-Means, silhouette, Davies-Bouldin index

### LLM Frameworks
- LiteLLM 1.35: Unified API for Claude Sonnet 4.6, GPT-4o, Claude Haiku 4.5, Llama 3.1 8B
- LangGraph 0.2.1: DAG orchestration for 8 agents with shared TypedDict state
- vLLM 0.4.1: Local serving for Llama 3.1 8B on V100

### Vector Database
- FAISS-cpu 1.7.4: IVFFlat index, 256 cells, 39,305 × 128-dim, 1.2ms query latency

### Optimal Transport
- POT 0.9.1: Log-domain Sinkhorn solver for BDCL calibration

### Infrastructure
- Terraform 1.6+: Azure AKS, ACR, PostgreSQL, Redis, Key Vault provisioning
- Docker: Multi-stage builds (builder + runtime stages)
- GitHub Actions: CI/CD (lint → test → build → push ACR → deploy AKS)
- Kubernetes: HPA, liveness/readiness probes, ServiceMonitor for Prometheus

### Hardware
- Development: Ubuntu 22.04, D4s_v3, Python 3.11.7
- Production: AKS cluster with auto-scaling, blue-green deployment via DNS switch

---

## 6. Code Organisation

### 6a. Production-style modular architecture (NOT notebook-based)

```
insightpulse/
├── src/insightpulse/                 # Core application
│   ├── __init__.py
│   ├── api/                          # FastAPI REST endpoints
│   │   ├── main.py                   # App factory, middleware, lifespan
│   │   ├── routes/                   # Survey, results, health routes
│   │   └── middleware/               # Auth, rate-limit, CORS
│   ├── agents/                       # LangGraph agent definitions
│   │   ├── graph.py                  # DAG wiring (SurveyDesigner → ... → AuditAgent)
│   │   ├── survey_designer.py        # Parses instrument, builds dependency graph
│   │   ├── cohort_selector.py        # Snowflake query, archetype-stratified sampling
│   │   ├── twin_orchestrator.py      # Embedding retrieval + LLM generation
│   │   ├── validator.py              # Hallucination, consistency, format checks
│   │   ├── cost_agent.py             # Budget tracking, model routing decisions
│   │   ├── calibration_agent.py      # BDCL Sinkhorn execution
│   │   ├── diversity_monitor.py      # Shannon entropy check, temperature adjustment
│   │   └── audit_agent.py            # Provenance hashing, seed logging, metrics
│   ├── ml/                           # Machine learning modules
│   │   ├── embeddings.py             # Transformer encoder, InfoNCE, K-Means
│   │   ├── generation.py             # Prompt construction, CoT, sequential conditioning
│   │   ├── calibration.py            # BDCL Sinkhorn solver (POT), 3 parameters
│   │   └── llm.py                    # LiteLLM routing, model registry, cost tracking
│   ├── data/                         # Data access layer
│   │   ├── connectors/               # Snowflake, ADLS, PostgreSQL, Redis, CSV
│   │   ├── etl/                      # Extract-Transform-Load pipelines
│   │   ├── quality/                  # 8-rule quality gate framework
│   │   └── repository.py             # Repository pattern for data access
│   ├── security/                     # Defence-in-depth
│   │   ├── prompt_guard.py           # 47 injection patterns, encoding analysis
│   │   ├── response_guard.py         # PII regex, NER scanning, content safety
│   │   ├── auth.py                   # JWT validation, RBAC (4 roles)
│   │   └── rate_limiter.py           # 100 req/min sliding window
│   ├── observability/                # Three-pillar monitoring
│   │   ├── metrics.py                # 20 Prometheus counters/histograms
│   │   ├── tracing.py                # OpenTelemetry spans (1 trace/run)
│   │   ├── logging.py                # Structured JSON, PII redaction
│   │   └── alerts.py                 # 9-rule alert manager with cooldown
│   └── config/                       # Configuration
│       ├── settings.py               # Pydantic BaseSettings (env vars + YAML)
│       └── profiles/                 # demo.yaml, production.yaml, test.yaml
├── experiments/                      # Reproducible experiment scripts
│   ├── multi_llm_comparison.py       # 4-model evaluation on 200-respondent cohort
│   ├── calibration_convergence.py    # Sinkhorn convergence tracking
│   ├── drift_detection.py            # 9-month weekly JS monitoring
│   └── sequential_dependency.py      # Independent vs sequential generation
├── demo_engine.py                    # Local demo 
├── dashboard/                        # Streamlit frontend (9 pages)
├── frontend/                         # Next.js/React production frontend
├── infra/                            # Infrastructure-as-code
│   ├── terraform/                    # Azure AKS, ACR, PostgreSQL, Redis, KeyVault
│   ├── k8s/                          # Kubernetes manifests (HPA, ingress, probes)
│   ├── docker/                       # Multi-stage Dockerfile
│   └── ci/workflows/                 # GitHub Actions CI/CD
├── tests/                            # 181 tests, 86% coverage
│   ├── unit/                         # Module-level tests with mocked LLM
│   ├── integration/                  # Cross-module pipeline tests
│   └── conftest.py                   # Shared fixtures, mock LLM router
├── pyproject.toml                    # Project metadata and dependencies
└── README.md                         # Project documentation
```

### 6b. Key Design Principles
- **Production-style:** No Jupyter notebooks. All code in `.py` modules with type hints.
- **Repository pattern:** Data access abstracted behind interfaces.
- **Config-driven:** All hyperparameters in `config/settings.py` + profile YAMLs. Nothing hardcoded.
- **Dependency injection:** Connectors and models injected via config, enabling test mocking.
- **Seed control:** Seed 42 propagated to NumPy, Python random, PyTorch generator for full reproducibility.

---

## 7. Pipeline Design

### 7a. Fully Automatic Workflow

The pipeline is fully automatic once triggered. No manual intervention between stages. A single API call or CLI command executes the entire survey lifecycle.

### 7b. Data Flow

```
[Snowflake/ADLS/CSV]
        ↓
  ETL Pipeline (data/etl/)
        ↓
  8 Quality Gates (data/quality/)
        ↓
  Embedding Training (ml/embeddings.py)
  → Transformer 2L/4H, InfoNCE
  → K-Means K=5
  → FAISS IVFFlat index
        ↓
  Survey Execution (agents/graph.py)
  ├── SurveyDesigner: parse instrument
  ├── CohortSelector: Snowflake + Redis cache
  ├── TwinOrchestrator: FAISS k-NN → persona prompt → LLM call
  │       ↕ retry ≤ 3
  ├── Validator: hallucination + consistency + format
  ├── CostAgent: budget check, model routing
  ├── CalibrationAgent: BDCL Sinkhorn per question
  ├── DiversityMonitor: entropy ≥ 1.5 bits, T adjustment
  └── AuditAgent: provenance hash, metrics, cost log
        ↓
  [Dashboard / API / CSV-SPSS export]
```

### 7c. Entry Points

| Command / File | What it does |
|---------------|--------------|
| `python -m insightpulse.api.main` | Starts FastAPI server |
| `streamlit run dashboard/app.py` | Starts Streamlit dashboard |
| `python demo_engine.py` | Runs full pipeline locally  |
| `python experiments/multi_llm_comparison.py` | Runs 4-model comparison experiment |
| `python experiments/drift_detection.py` | Runs drift monitoring experiment |
| `pytest tests/ -v --cov` | Runs all 181 tests with coverage |
| `terraform apply` (in infra/terraform/) | Provisions Azure infrastructure |

---

## 8. Deployment Mode

### 8a. API (FastAPI)
- `src/insightpulse/api/main.py` — app factory with lifespan management
- Routes: `/survey` (CRUD), `/results` (query), `/health` (liveness)
- Middleware: JWT auth, rate limiting (100 req/min), CORS
- Async endpoints with 256-way concurrency for batch generation

### 8b. Web Apps
- **Streamlit:** 9-page dashboard — survey wizard, execution monitor, results explorer, admin
- **Next.js/React 18:** Production frontend mirror with WebSocket real-time updates

### 8c. Cloud Deployment
- Terraform → Azure AKS cluster with HPA auto-scaling
- Docker multi-stage build → Azure Container Registry
- Blue-green deployment via DNS switch, smoke test: 50-respondent end-to-end survey
- CI/CD: GitHub Actions → lint → test → build → push ACR → deploy AKS
- Commit to production in ~12 minutes

### 8d. Local Demo
- `demo_engine.py` — runs complete pipeline locally or external services
- Generates all 6 acceptance metrics and visualisations

---

## 9. Code Walkthrough Guide

### 9a. Key Modules — Exact File, Class, Function

**Embedding Pipeline** (`src/insightpulse/ml/embeddings.py`):
- Class: `BehaviouralEncoder` — 2-layer, 4-head transformer
- `train()` — InfoNCE contrastive training, 50 epochs, cosine annealing
- `encode(sequences)` → 128-dim vectors
- `cluster(embeddings, k=5)` → K-Means archetypes
- `build_index(embeddings)` → FAISS IVFFlat (256 cells)
- `query(embedding, k=5)` → nearest neighbours in 1.2ms

**BDCL Calibration** (`src/insightpulse/ml/calibration.py`):
- Class: `BDCLCalibrator`
- `__init__(epsilon=0.1, lambda_b=0.3, lambda_f=0.2)` — 3-parameter config
- `build_cost_matrix(m)` — ordinal Likert cost: c_ij = (|i−j|/(m−1))²
- `calibrate(p_syn, p_real, embeddings)` — log-domain Sinkhorn, returns calibrated distribution
- `_sinkhorn_iteration()` — single iteration with fairness projection
- Uses POT library for the core OT solver
- Convergence: ~80 iterations, tolerance 1e-6, 2.1s per question

**LLM Routing** (`src/insightpulse/ml/llm.py`):
- Class: `LLMRouter`
- `route(model_name, prompt)` — dispatches to LiteLLM
- Supported: `claude-sonnet-4-6`, `gpt-4o`, `claude-haiku-4-5`, `llama-3.1-8b`
- Cost tracking per call, token counting, retry with exponential backoff

**Generation** (`src/insightpulse/ml/generation.py`):
- Class: `TwinGenerator`
- `build_persona(demographic, behavioural_context)` — constructs persona narrative
- `generate_response(persona, question, prior_responses=[])` — CoT + sequential
- `build_cot_prompt()` — chain-of-thought reasoning template
- Temperature: 0.7 (configurable), JSON output with regex fallback

**Agent Definitions** (`src/insightpulse/agents/`):

| Agent | File | Class | Key Method |
|-------|------|-------|------------|
| SurveyDesigner | `survey_designer.py` | `SurveyDesignerAgent` | `parse_instrument()` — extracts metadata, dependency graph |
| CohortSelector | `cohort_selector.py` | `CohortSelectorAgent` | `select_cohort()` — Snowflake query, stratified sampling, Redis cache |
| TwinOrchestrator | `twin_orchestrator.py` | `TwinOrchestratorAgent` | `generate_batch()` — FAISS retrieval → persona → LLM, batches of 50 |
| Validator | `validator.py` | `ValidatorAgent` | `validate()` — hallucination, consistency, format; retry ≤ 3 |
| CostAgent | `cost_agent.py` | `CostAgent` | `check_budget()` — running cost, model routing, HALT if exceeded |
| CalibrationAgent | `calibration_agent.py` | `CalibrationAgent` | `calibrate()` — invokes BDCLCalibrator per question |
| DiversityMonitor | `diversity_monitor.py` | `DiversityMonitorAgent` | `check_entropy()` — Shannon ≥ 1.5 bits, ADJUST T if low |
| AuditAgent | `audit_agent.py` | `AuditAgent` | `log_provenance()` — seeds, versions, cost, metrics, hashing |

**DAG Wiring** (`src/insightpulse/agents/graph.py`):
- `build_graph()` — constructs LangGraph StateGraph with edges:
  ```
  SurveyDesigner → CohortSelector → TwinOrchestrator → Validator
                                          ↑ retry           ↓
                                          └─────── CostAgent → CalibrationAgent
                                                                    ↓
                                              DiversityMonitor → AuditAgent
  ```
- Shared state: TypedDict with survey config, cohort, responses, calibrated results, audit log
- Conditional edges: retry (validation failure), HALT (budget), ADJUST (low entropy)

**Security** (`src/insightpulse/security/`):
- `PromptGuard.scan(prompt)` — 47 injection patterns, encoding analysis, token budget 8,192
- `ResponseGuard.scan(response)` — PII regex (email/SSN), NER, content safety, format check
- `auth.py` — JWT validation via Azure AD, 4 RBAC roles
- `rate_limiter.py` — 100 req/min sliding window per API key
- Stats: 23 injections intercepted, 0 breaches (9 months)

**Observability** (`src/insightpulse/observability/`):
- `metrics.py` — 20 Prometheus metrics: latency p50/p95/p99, throughput, token usage, Sinkhorn iterations, error rates
- `tracing.py` — OpenTelemetry: 1 trace per survey run, child span per agent, attribute propagation
- `logging.py` — structlog JSON, PII redaction, Azure Monitor sink
- `alerts.py` — 9-rule alert manager: cooldown, firing, resolved states

---

## 10. Performance Metrics

### 10a. Training Time

| Component | Time | Hardware |
|-----------|------|----------|
| Behavioural encoder | 45 min | CPU (D4s_v3) |
| K-Means clustering | 12 s | CPU |
| FAISS index build | 3 s | CPU |

### 10b. Inference Latency

| Component | Latency |
|-----------|---------|
| FAISS k-NN query (k=5) | 1.2 ms |
| LLM generation (Claude Sonnet) | 0.8 s/response |
| LLM generation (Llama 3.1 8B) | 0.3 s/response |
| BDCL calibration (Sinkhorn) | 2.1 s/question |
| Validation checks | 5 ms/response |
| Embedding inference | 0.3 ms/embedding |

### 10c. End-to-End Pipeline Runtime

| Scale | Time | Throughput |
|-------|------|-----------|
| 200 respondents (evaluation) | 3.2 min | — |
| 10,000 respondents (scale test) | 8 min | 1,248 resp/min |

Achieved through 256-way asynchronous concurrency.

### 10d. Acceptance Metrics (final system)

| Metric | Pre-BDCL | Post-BDCL | Threshold | Status |
|--------|----------|-----------|-----------|--------|
| Cosine similarity | 0.87 | 0.84 | ≥ 0.80 | ✓ |
| JS divergence | 0.078 | 0.017 | ≤ 0.05 | ✓ |
| Wasserstein distance | 0.136 | 0.041 | ≤ 0.15 | ✓ |
| Hallucination rate | 7.8% | 1.9% | < 5% | ✓ |
| Logical consistency | 81.2% | 94.6% | > 90% | ✓ |
| Shannon entropy | 1.89 | 2.31 | ≥ 1.5 bits | ✓ |

---

## 11. Resource Utilisation

### CPU/GPU Usage

| Process | CPU | GPU | RAM |
|---------|-----|-----|-----|
| Embedding training | 4 cores (100%) | None | ~4 GB |
| LLM inference (API) | <1 core | None | ~1 GB |
| LLM inference (Llama local) | 2 cores | V100 (100%) | ~14 GB |
| BDCL calibration | 1 core | None | ~500 MB |
| FAISS query | 1 core | None | ~200 MB |
| Full pipeline (200 resp) | 4 cores | None | ~8 GB |

### Memory Consumption

| Component | Memory |
|-----------|--------|
| FAISS index (loaded) | 19.2 MB |
| Embedding model (loaded) | ~50 MB |
| LangGraph DAG state | ~100 MB per run |
| Redis cache | ~200 MB |
| Total working set | ~2 GB |

---

## 12. Model and Training Parameters

### Embedding Encoder

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | Transformer | — |
| Layers | 2 | Architecture sweep (Table 7.6) |
| Attention heads | 4 | Architecture sweep |
| Embedding dimension (d) | 128 | Dimension sweep (Table 7.7) |
| Sequence length | 256 tokens | Chunking comparison (Table 7.5) |
| Contrastive temperature (τ_cl) | 0.07 | Standard InfoNCE |
| Dropout | 0.1 | Validation monitoring |
| Batch size | 256 | Memory constraint |
| Training epochs | 50 | Cosine annealing |
| Optimiser | Adam | lr = 3e-4 |
| LR schedule | Cosine annealing | — |
| Loss function | InfoNCE | Contrastive |
| Parameters | 1.2M | — |

### BDCL Calibration

| Parameter | Value | Source |
|-----------|-------|--------|
| Regularisation (ε) | 0.1 | ε sweep (Table 7.15) |
| Behavioural weight (λb) | 0.3 | λb sweep (Table 7.16) |
| Fairness weight (λf) | 0.2 | λf sweep (Table 7.17) |
| Uncertainty weight (η) | 0 (disabled) | Not tuned |
| Max Sinkhorn iterations | 200 | Budget with early stopping |
| Convergence tolerance | 1e-6 | Log-domain precision |
| Cost matrix | c_ij = (|i−j|/(m−1))² | Ordinal Likert |
| Typical convergence | ~80 iterations | Empirical |

### Generation

| Parameter | Value | Source |
|-----------|-------|--------|
| Default model | Claude Sonnet 4.6 | Multi-model comparison |
| Prompting strategy | CoT with persona | Strategy comparison (Table 7.9) |
| Temperature (T) | 0.7 | Temperature sweep (Table 7.10) |
| Retrieval k-NN | k = 5 | Retrieval comparison (Table 7.11) |
| Sequential conditioning | Enabled | Sequential experiment (§7.11) |
| Output format | JSON with regex fallback | Parsing comparison (Table 7.12) |
| Batch size | 50 respondents | Concurrency optimisation |

---

## 13. LLM / Agentic AI Specifics

### 13a. Base LLMs Used

| Model | Provider | Halluc. | Consist. | Cost/run |
|-------|----------|---------|----------|----------|
| Claude Sonnet 4.6 | Anthropic | 1.7% | 98.6% | $2.14 |
| GPT-4o | OpenAI | 3.3% | 98.0% | $1.36 |
| Claude Haiku 4.5 | Anthropic | 3.5% | 97.7% | $0.54 |
| Llama 3.1 8B | Meta (local) | 7.1% | 96.4% | $0.00 |

### 13b. Prompt Engineering Strategy

Persona + Chain-of-Thought + Sequential conditioning:
1. Behavioural persona: constructed from FAISS k-NN retrieval (k=5 neighbours)
2. Demographic context: age, income, region, household size
3. CoT instruction: "Think step by step about how this consumer would respond"
4. Sequential context: all prior responses for the same respondent appended
5. Output format: structured JSON with Pydantic validation + regex fallback

### 13c. RAG Pipeline

Not traditional RAG. Instead: FAISS k-NN retrieval of behavioural embeddings.
- Query: respondent's embedding vector z_i
- Index: 39,305 × 128-dim IVFFlat (256 cells)
- Returns: k=5 nearest neighbours' purchase patterns
- These are converted to natural-language behavioural context (not document chunks)

### 13d. Embedding Model and Vector Database

| Component | Details |
|-----------|---------|
| Embedding model | Custom transformer (2L/4H, 1.2M params) — NOT a pretrained model |
| Training data | 39,305 household purchase sequences (6.4M transactions) |
| Training objective | InfoNCE contrastive loss |
| Vector DB | FAISS IVFFlat, 256 cells |
| Index size | 19.2 MB |
| Query latency | 1.2 ms |

### 13e. Agent Workflow / Tools Integration

8 agents in LangGraph DAG with shared TypedDict state:

```
SurveyDesigner → CohortSelector → TwinOrchestrator → Validator
                                       ↑ retry ≤ 3         ↓
                                       └────────── CostAgent → CalibrationAgent
                                                                     ↓
                           ADJUST (raise T) ← DiversityMonitor → AuditAgent
```

Conditional edges: retry on validation failure, HALT on budget exceeded or 3 rejects, ADJUST temperature if entropy < 1.5 bits.

Agent interactions per respondent: 5.3 average (1.2 generation calls = 20% retry rate).

### 13f. Token Usage and Cost Estimation

| Model | Tokens/response | Cost/response | Cost/200-resp run |
|-------|----------------|---------------|-------------------|
| Claude Sonnet | ~800 | $0.0107 | $2.14 |
| GPT-4o | ~750 | $0.0068 | $1.36 |
| Claude Haiku | ~700 | $0.0027 | $0.54 |
| Llama 3.1 | ~900 | $0.00 | $0.00 |

### 13g. Safety / Guardrails / Hallucination Handling

**PromptGuard** (`security/prompt_guard.py`):
- 47 injection patterns (SQL, prompt override, jailbreak, encoding attacks)
- Token budget enforcement: 8,192 tokens max
- Template validation: prompts must match approved templates
- Detection rate: 94.2% | 23 injections intercepted, 0 breaches (9 months)

**ResponseGuard** (`security/response_guard.py`):
- PII detection: email, SSN, phone regex patterns
- NER scanning for named entities that should not appear
- Content safety: toxicity, bias, inappropriate content
- Format validation: JSON schema compliance

**Hallucination handling:**
- Validator agent checks every response for factual consistency
- Retry ≤ 3 cycles with rejection-regeneration
- Red-Team agent runs 47 adversarial patterns in parallel
- Residual 1.9% concentrates in 4 structured failure modes

### 13h. Evaluation Methodology

6 pre-registered acceptance criteria with quantitative thresholds (Chapter 3).
18 experiments (Chapter 7): 4 embedding sweeps, 5 generation sweeps, 3 calibration sweeps, multi-model comparison, 6-component ablation, 9-month drift monitoring, 3 cross-benchmark validations, qualitative failure analysis.

---

## 14. Experiments (Reproducible)

### Experiment Files

| File | What it does | How to run |
|------|-------------|-----------|
| `experiments/multi_llm_comparison.py` | Evaluates 4 LLMs on 200-respondent cohort | `python experiments/multi_llm_comparison.py` |
| `experiments/calibration_convergence.py` | Tracks Sinkhorn iteration convergence | `python experiments/calibration_convergence.py` |
| `experiments/drift_detection.py` | 9-month weekly JS monitoring | `python experiments/drift_detection.py` |
| `experiments/sequential_dependency.py` | Independent vs sequential generation | `python experiments/sequential_dependency.py` |
| `demo_engine.py` | Full pipeline for demo  | `python demo_engine.py` |

### Running the Demo

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run Local  demo 
python demo_engine.py

# 3. Run specific experiment
python experiments/multi_llm_comparison.py

# 4. Run tests
pytest tests/ -v --cov=src/insightpulse --cov-report=term

# 5. Start dashboard
streamlit run dashboard/app.py

# 6. Start API server
python -m insightpulse.api.main
```

---

## 15. External / Open-Source Acknowledgements

All external libraries are open-source and documented in `pyproject.toml`:
- PyTorch (BSD), scikit-learn (BSD), FAISS (MIT), POT (MIT)
- LangGraph (Apache 2.0), LiteLLM (Apache 2.0)
- FastAPI (MIT), Pydantic (MIT), SQLAlchemy (MIT)
- Streamlit (Apache 2.0), Plotly (MIT)
- Prometheus client (Apache 2.0), OpenTelemetry (Apache 2.0)

The LLM providers (Anthropic, OpenAI) are accessed via their commercial APIs. Llama 3.1 is used under Meta's community licence.

All custom code (145 files, 16 modules) is original work for this thesis.

---

*Document generated for code demo video recording. All values verified against thesis and codebase.*
