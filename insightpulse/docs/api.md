# API Reference

Base URL: `http://localhost:8000` (demo) · `https://api.insightpulse.example.com` (production).
Interactive OpenAPI docs are served at `/docs`.

All endpoints are rate-limited per client (default **60 requests/minute**,
sliding window; `/health` and `/docs` exempt). Exceeding the limit returns
`429` with a `Retry-After` header. Domain failures (provider or database
outages) return `503` and are safe to retry; validation failures return
`422` with a specific message.

---

## `GET /health`

Liveness/readiness probe for Docker, Kubernetes, and load balancers.

**Response `200`**

```json
{"status": "healthy", "env": "demo", "version": "1.0.0"}
```

---

## `POST /api/v1/survey/run`

Execute a synthetic survey through the full 8-agent pipeline.

**Request body**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `questions` | `string[]` | 1–20 items, each 1–500 chars, non-blank | Survey question texts |
| `cohort_size` | `int` | 1–5000 (default 50) | Number of synthetic respondents |
| `context` | `string` | ≤ 2000 chars | Domain context (e.g. "US snack market") |
| `models` | `string[]` | each must be a supported model | LLM models (empty = profile default) |
| `cohort_filters` | `object` | key/value demographic filters | e.g. `{"age_group": "25-34"}` |
| `seed` | `int` | ≥ 0, optional | Reproducibility seed |

**Example**

```bash
curl -X POST http://localhost:8000/api/v1/survey/run \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["How important is organic labeling when purchasing snacks?"],
    "cohort_size": 100,
    "context": "US snack food market",
    "models": ["claude-sonnet-4-6"],
    "cohort_filters": {"age_group": "25-34"},
    "seed": 42
  }'
```

**Response `200`**

```json
{
  "run_id": "3f8a1c2e",
  "status": "completed",
  "total_responses": 100,
  "total_cost_usd": 0.0,
  "hallucination_rate": 0.02,
  "results": [
    {
      "question_id": "q_1",
      "question_text": "How important is organic labeling when purchasing snacks?",
      "options": ["Not at all important", "...", "Extremely important"],
      "total_responses": 100,
      "valid_responses": 98,
      "distribution": [{"option": "Very important", "count": 31, "percentage": 31.6}],
      "calibrated_distribution": [0.08, 0.19, 0.27, 0.28, 0.18],
      "entropy": 2.21,
      "calibration_metrics": {
        "js_divergence_before": 0.0102,
        "js_divergence_after": 0.0009,
        "wasserstein_improvement_pct": 71.3,
        "converged": true,
        "iterations": 1
      }
    }
  ],
  "agent_trace": [{"agent_name": "SurveyDesigner", "duration_ms": 812.4}],
  "provenance_hash": "9c1d2e3f4a5b6c7d"
}
```

`run_id` is bound into every structured log line of the run — use it to
correlate with Log Analytics. `provenance_hash` fingerprints the request:
identical requests (same seed) replay identically.

**Errors**

| Status | Meaning | Example detail |
|---|---|---|
| `422` | Validation failed — nothing executed | `"Question 1 is empty or whitespace-only"`, `"Unknown model(s) ['gpt-99-ultra']"` |
| `429` | Rate limit exceeded (Retry-After set) | `"Rate limit exceeded: 60 requests per 60s"` |
| `503` | Retryable domain failure | `"DataLayerError: database unreachable"` |
| `500` | Unexpected pipeline error | `"Pipeline error: ..."` |

---

## `GET /api/v1/models`

The models accepted by `/survey/run` (the validation source of truth).

**Response `200`**

```json
{
  "default": "claude-sonnet-4-6",
  "supported": ["claude-haiku-4-5", "claude-sonnet-4-6", "gpt-4o", "ollama/llama3.1"]
}
```

---

## `GET /api/v1/config`

The active configuration — a safe subset only (never secrets).

**Response `200`**

```json
{
  "env": "demo",
  "default_model": "claude-sonnet-4-6",
  "default_cohort_size": 100,
  "calibration": {"epsilon": 0.1, "lambda_behavioral": 0.3, "lambda_fairness": 0.2},
  "embedding": {"dim": 128, "num_clusters": 5},
  "limits": {
    "rate_limit_per_minute": 60,
    "max_questions": 20,
    "max_question_length": 500,
    "max_cohort_size": 5000
  }
}
```

---

## Operational notes

- **Timeouts**: large cohorts take minutes; the ingress proxy-read-timeout
  is 600s (`k8s/ingress.yaml`). Clients should use ≥ 300s timeouts.
- **Idempotency**: runs with the same request body and seed produce the
  same provenance hash and identical responses (demo strategies are fully
  deterministic; production LLM runs are provider-stochastic).
- **Authentication**: the demo API is unauthenticated by design. In
  production, terminate auth at the ingress (OAuth2 proxy / API management)
  — the app deliberately contains no credential-handling code.
