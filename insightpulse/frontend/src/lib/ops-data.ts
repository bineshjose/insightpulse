/**
 * Operations page fixtures — system health, security, performance, alerts,
 * model registry, data lineage, and project metrics. Values mirror the
 * backend's /health, /metrics, and observability surfaces so the demo
 * environment renders the same numbers the production stack reports.
 */

// ---------------------------------------------------------------------------
// 1 · System health
// ---------------------------------------------------------------------------

export type StatusTone = "green" | "blue" | "amber" | "red";

export interface ComponentHealth {
  component: string;
  status: string;
  detail: string;
  tone: StatusTone;
}

export const SYSTEM_HEALTH: readonly ComponentHealth[] = [
  { component: "API Server", status: "Healthy", detail: "12ms response time", tone: "green" },
  { component: "Database", status: "Local Mode", detail: "SQLite — demo profile", tone: "blue" },
  { component: "Cache", status: "Local Mode", detail: "In-memory — demo profile", tone: "blue" },
  { component: "LLM Provider", status: "Available", detail: "claude-sonnet-4-6", tone: "green" },
  { component: "Embedding Service", status: "Loaded", detail: "500 vectors (128-dim)", tone: "green" },
  { component: "ETL Scheduler", status: "Idle", detail: "next: benchmark refresh in 6h", tone: "blue" },
] as const;

// ---------------------------------------------------------------------------
// 2 · Security dashboard
// ---------------------------------------------------------------------------

export interface SecurityKpi {
  label: string;
  value: string;
  note: string;
  status: "good" | "warn" | "info";
}

export const SECURITY_KPIS: readonly SecurityKpi[] = [
  { label: "Prompt Injection Attempts", value: "0", note: "last 24h", status: "good" },
  { label: "PII Detections", value: "2", note: "auto-redacted", status: "warn" },
  { label: "Failed Auth Attempts", value: "0", note: "last 24h", status: "good" },
  { label: "Rate Limit Hits", value: "0", note: "last 24h", status: "good" },
  { label: "Input Validation Rejections", value: "1", note: "last 7 days", status: "good" },
] as const;

export type SecuritySeverity = "MEDIUM" | "LOW" | "INFO";

export interface SecurityEvent {
  timestamp: string;
  eventType: string;
  severity: SecuritySeverity;
  details: string;
  action: string;
}

export const SECURITY_EVENTS: readonly SecurityEvent[] = [
  {
    timestamp: "Jul 11, 10:42 AM",
    eventType: "PII Detected",
    severity: "MEDIUM",
    details: "Email pattern in response R-0047",
    action: "Auto-redacted",
  },
  {
    timestamp: "Jul 10, 3:15 PM",
    eventType: "Input Validation",
    severity: "LOW",
    details: "Survey name exceeded 200 chars",
    action: "Rejected (422)",
  },
  {
    timestamp: "Jul 9, 11:30 AM",
    eventType: "Rate Limit",
    severity: "INFO",
    details: "User demo@insightpulse.ai hit 100 req/min",
    action: "Throttled",
  },
  {
    timestamp: "Jul 8, 9:45 AM",
    eventType: "Auth Attempt",
    severity: "LOW",
    details: "Invalid token format on /api/v1/survey/run",
    action: "Rejected (401)",
  },
  {
    timestamp: "Jul 7, 2:20 PM",
    eventType: "Prompt Pattern",
    severity: "MEDIUM",
    details: "Detected 'ignore previous' in question text",
    action: "Blocked",
  },
] as const;

export const SECURITY_MONITORING_NOTE =
  "12 injection patterns monitored, 3 encoding checks active";

export const SECURITY_CONFIG = [
  { setting: "PromptGuard", value: "Enabled" },
  { setting: "PII redaction", value: "On" },
  { setting: "Input validation", value: "Strict" },
  { setting: "JWT authentication", value: "Ready" },
] as const;

// ---------------------------------------------------------------------------
// 3 · Performance metrics
// ---------------------------------------------------------------------------

export const PERF_KPIS = [
  { label: "Avg Latency", value: "450ms", note: "per LLM call", status: "info" },
  { label: "Throughput", value: "774 resp/min", note: "current window", status: "good" },
  { label: "Error Rate", value: "0.2%", note: "target < 1%", status: "good" },
  { label: "Cache Hit Rate", value: "87%", note: "embedding lookups", status: "good" },
] as const;

export const LATENCY_PERCENTILES = [
  { percentile: "p50", ms: 320 },
  { percentile: "p75", ms: 450 },
  { percentile: "p90", ms: 680 },
  { percentile: "p95", ms: 890 },
  { percentile: "p99", ms: 1200 },
] as const;

/** Responses per minute, last 7 days (Jul 5 – Jul 11). */
export const THROUGHPUT_7D = [
  { day: "Jul 5", rpm: 712 },
  { day: "Jul 6", rpm: 748 },
  { day: "Jul 7", rpm: 767 },
  { day: "Jul 8", rpm: 731 },
  { day: "Jul 9", rpm: 794 },
  { day: "Jul 10", rpm: 818 },
  { day: "Jul 11", rpm: 774 },
] as const;

export interface ModelPerf {
  model: string;
  avgLatency: string;
  avgTokens: number;
  costPerResponse: string;
  errorRate: string;
  qualityScore: string;
}

export const MODEL_PERF: readonly ModelPerf[] = [
  { model: "claude-sonnet-4-6", avgLatency: "420ms", avgTokens: 180, costPerResponse: "$0.0022", errorRate: "0.1%", qualityScore: "98.3%" },
  { model: "gpt-4o", avgLatency: "380ms", avgTokens: 165, costPerResponse: "$0.0018", errorRate: "0.3%", qualityScore: "97.1%" },
  { model: "claude-haiku-4-5", avgLatency: "190ms", avgTokens: 140, costPerResponse: "$0.0004", errorRate: "0.2%", qualityScore: "96.5%" },
  { model: "ollama/llama3.1", avgLatency: "1,200ms", avgTokens: 210, costPerResponse: "$0.00", errorRate: "0.5%", qualityScore: "94.9%" },
] as const;

export const ENDPOINT_PERF = [
  { endpoint: "/api/v1/survey/run", avg: "18s", note: "full pipeline (L1–L5)" },
  { endpoint: "/health", avg: "5ms", note: "component breakdown" },
  { endpoint: "/api/v1/config", avg: "12ms", note: "active configuration" },
  { endpoint: "/metrics", avg: "8ms", note: "Prometheus scrape" },
] as const;

// ---------------------------------------------------------------------------
// 4 · Alerts
// ---------------------------------------------------------------------------

export const ALERT_STATUS = [
  { severity: "CRITICAL", count: 0, tone: "red" as StatusTone },
  { severity: "WARNING", count: 0, tone: "amber" as StatusTone },
  { severity: "INFO", count: 1, tone: "blue" as StatusTone },
] as const;

export const ALERT_HISTORY = [
  { timestamp: "Jul 8, 2:30 PM", message: "Budget utilization reached 62%", severity: "INFO", resolution: "Auto-resolved" },
  { timestamp: "Jul 5, 9:15 AM", message: "Calibration non-convergence on 2/10 questions", severity: "WARNING", resolution: "Resolved (ε adjusted)" },
  { timestamp: "Jul 1, 11:00 AM", message: "Model gpt-4o latency spike (p95 > 5s)", severity: "WARNING", resolution: "Resolved (provider recovered)" },
] as const;

export const ALERT_RULES = [
  { rule: "Hallucination rate", threshold: ">10% for 3 consecutive runs", severity: "CRITICAL", cooldown: "15m" },
  { rule: "LLM API error rate", threshold: ">20% within 5 min", severity: "CRITICAL", cooldown: "15m" },
  { rule: "Prompt injection", threshold: "any HIGH/CRITICAL detection", severity: "CRITICAL", cooldown: "0m" },
  { rule: "Calibration non-convergence", threshold: ">30% of questions", severity: "WARNING", cooldown: "30m" },
  { rule: "Budget utilization", threshold: ">80% of monthly budget", severity: "WARNING", cooldown: "60m" },
  { rule: "Data quality gate", threshold: "CRITICAL gate failure", severity: "WARNING", cooldown: "30m" },
  { rule: "Latency p95", threshold: ">30s", severity: "WARNING", cooldown: "15m" },
  { rule: "Behavioral drift", threshold: "above retraining trigger", severity: "INFO", cooldown: "1440m" },
  { rule: "Throughput drop", threshold: ">50% vs 7-day average", severity: "INFO", cooldown: "60m" },
] as const;

// ---------------------------------------------------------------------------
// 5 · Model registry
// ---------------------------------------------------------------------------

export const MODEL_REGISTRY = [
  { modelId: "claude-sonnet-4-6", provider: "Anthropic", version: "4.6", status: "Active", registered: "Jun 15", lastUsed: "Jul 11", runs: 89, avgQuality: "98.3%" },
  { modelId: "gpt-4o", provider: "OpenAI", version: "2024-08-06", status: "Active", registered: "Jun 20", lastUsed: "Jul 10", runs: 34, avgQuality: "97.1%" },
  { modelId: "claude-haiku-4-5", provider: "Anthropic", version: "4.5", status: "Active", registered: "Jun 25", lastUsed: "Jul 9", runs: 12, avgQuality: "96.5%" },
  { modelId: "ollama/llama3.1", provider: "Meta (local)", version: "3.1-8B", status: "Active", registered: "Jul 1", lastUsed: "Jul 8", runs: 8, avgQuality: "94.9%" },
] as const;

export const LIFECYCLE_POLICY =
  "Evaluate monthly. Retire if quality < 90% for 3 consecutive runs.";

// ---------------------------------------------------------------------------
// 6 · Data lineage
// ---------------------------------------------------------------------------

export interface DataSource {
  source: string;
  type: string;
  lastRefreshed: string;
  records: string;
  quality: string;
  pipeline: string;
  schema: readonly { field: string; dtype: string }[];
  qualityDetails: readonly string[];
}

export const DATA_SOURCES: readonly DataSource[] = [
  {
    source: "Snowflake (NIQ Panel)",
    type: "Primary",
    lastRefreshed: "Continuous",
    records: "39.3K HH",
    quality: "Healthy",
    pipeline: "cohort_extraction",
    schema: [
      { field: "household_id", dtype: "VARCHAR (PK)" },
      { field: "demographics", dtype: "VARIANT (age, income, region, size)" },
      { field: "purchase_events", dtype: "ARRAY<STRUCT> (ts, sku, price, promo)" },
      { field: "panel_tenure_months", dtype: "INTEGER" },
    ],
    qualityDetails: [
      "Null rate 0.0% across key columns",
      "Primary keys unique — no duplicate households",
      "Purchase timestamps within panel tenure window",
    ],
  },
  {
    source: "ADLS (Pew ATP)",
    type: "Benchmark",
    lastRefreshed: "Jul 5",
    records: "12.4K resp",
    quality: "Healthy",
    pipeline: "benchmark_pipeline",
    schema: [
      { field: "respondent_id", dtype: "string (PK)" },
      { field: "wave", dtype: "string (2023–2025)" },
      { field: "question_code", dtype: "string" },
      { field: "response", dtype: "string / ordinal" },
      { field: "weight", dtype: "float (post-stratification)" },
    ],
    qualityDetails: [
      "Wave coverage complete for matched questions",
      "Weights sum to population totals within 0.1%",
      "No orphaned question codes after mapping",
    ],
  },
  {
    source: "ADLS (ESS R11)",
    type: "Benchmark",
    lastRefreshed: "Jul 3",
    records: "8.2K resp",
    quality: "Healthy",
    pipeline: "benchmark_pipeline",
    schema: [
      { field: "idno", dtype: "string (PK)" },
      { field: "cntry", dtype: "string (ISO country)" },
      { field: "variable", dtype: "string (ESS codebook)" },
      { field: "value", dtype: "ordinal / categorical" },
      { field: "anweight", dtype: "float (analysis weight)" },
    ],
    qualityDetails: [
      "Round 11 extract validated against ESS codebook",
      "Country coverage: 12 markets used for validation",
      "Missing-value codes normalized to NULL",
    ],
  },
  {
    source: "ADLS (Twin-2K-500)",
    type: "Benchmark",
    lastRefreshed: "Jun 28",
    records: "500 personas",
    quality: "Healthy",
    pipeline: "benchmark_pipeline",
    schema: [
      { field: "persona_id", dtype: "string (PK)" },
      { field: "persona_profile", dtype: "json (demographics + traits)" },
      { field: "question_id", dtype: "string" },
      { field: "human_answer", dtype: "string / ordinal" },
    ],
    qualityDetails: [
      "Full published panel ingested (500/500 personas)",
      "Answer coverage 100% on comparison question set",
      "Checksums match published release",
    ],
  },
  {
    source: "PostgreSQL (App)",
    type: "Metadata",
    lastRefreshed: "Live",
    records: "143 runs",
    quality: "Healthy",
    pipeline: "direct write",
    schema: [
      { field: "run_id", dtype: "uuid (PK)" },
      { field: "survey_metadata", dtype: "jsonb (client, contract, executor)" },
      { field: "results", dtype: "jsonb (distributions + metrics)" },
      { field: "provenance_hash", dtype: "varchar(64)" },
      { field: "created_at", dtype: "timestamptz" },
    ],
    qualityDetails: [
      "All 143 runs carry a provenance hash",
      "Foreign keys valid — no orphaned results",
      "Write latency p95 under 20ms",
    ],
  },
  {
    source: "Redis (Cache)",
    type: "Hot cache",
    lastRefreshed: "Live",
    records: "500 embeddings",
    quality: "Healthy",
    pipeline: "cohort_extraction",
    schema: [
      { field: "emb:{household_id}", dtype: "bytes (float32[128])" },
      { field: "cohort:{filter_hash}", dtype: "list<household_id> (TTL 1h)" },
      { field: "config:active", dtype: "json" },
    ],
    qualityDetails: [
      "Hit rate 87% on embedding lookups",
      "All 500 panel embeddings resident",
      "Evictions: 0 in last 7 days",
    ],
  },
] as const;

/** Pipeline flow rows rendered by the lineage diagram. */
export const LINEAGE_FLOWS = [
  {
    name: "Benchmark path",
    nodes: ["External Sources", "Ingestion", "ADLS", "Benchmark ETL", "PostgreSQL"],
  },
  {
    name: "Panel path",
    nodes: ["Snowflake", "Cohort Extraction", "Redis", "L2 Embeddings"],
  },
  {
    name: "Shared downstream",
    nodes: ["L3 Generation", "L4 Calibration", "L5 Insights", "Results"],
  },
] as const;

// ---------------------------------------------------------------------------
// 7 · Project & code metrics
// ---------------------------------------------------------------------------

export const PROJECT_STATS = [
  { label: "Total Modules", value: "16+", note: "src/insightpulse" },
  { label: "Python Files", value: "135+", note: "typed, docstringed" },
  { label: "Lines of Code", value: "27K+", note: "including tests" },
  { label: "Tests", value: "315", note: "Coverage 83%+" },
  { label: "Design Patterns", value: "8", note: "documented below" },
] as const;

export const DESIGN_PATTERNS = [
  "Repository",
  "Strategy",
  "Factory",
  "Pipeline",
  "Observer",
  "Decorator",
  "Template Method",
  "Circuit Breaker",
] as const;

/** Top backend dependencies (versions pinned in pyproject.toml). */
export const DEPENDENCIES = [
  { pkg: "fastapi", version: "≥0.115.0", purpose: "REST API framework" },
  { pkg: "langgraph", version: "≥0.2.0", purpose: "Agent DAG orchestration" },
  { pkg: "litellm", version: "≥1.50.0", purpose: "Multi-model LLM routing" },
  { pkg: "torch", version: "≥2.4.0", purpose: "Behavioral sequence encoder" },
  { pkg: "scikit-learn", version: "≥1.5.0", purpose: "K-Means archetype clustering" },
  { pkg: "faiss-cpu", version: "≥1.8.0", purpose: "Cohort vector similarity search" },
  { pkg: "pot", version: "≥0.9.4", purpose: "Sinkhorn optimal transport (BDCL)" },
  { pkg: "streamlit", version: "≥1.40.0", purpose: "Analyst dashboard" },
  { pkg: "plotly", version: "≥5.24.0", purpose: "Dashboard charting" },
  { pkg: "pydantic", version: "≥2.10.0", purpose: "Typed data models" },
  { pkg: "sqlalchemy", version: "≥2.0.0", purpose: "Persistence layer" },
  { pkg: "structlog", version: "≥24.4.0", purpose: "Structured logging" },
  { pkg: "pandas", version: "≥2.2.0, <3.0", purpose: "Panel data processing" },
  { pkg: "numpy", version: "≥1.26.0", purpose: "Numerical computing" },
  { pkg: "httpx", version: "≥0.27.0", purpose: "Async HTTP client" },
] as const;

export const ARCHITECTURE_LAYERS = [
  { code: "L1", name: "Data Layer", detail: "Panelists · Purchases · Benchmarks" },
  { code: "L2", name: "Embedding", detail: "Transformer · K-Means · FAISS" },
  { code: "L3", name: "Digital Twins", detail: "Persona-prompted LLMs" },
  { code: "L4", name: "BDCL Calibration", detail: "Sinkhorn optimal transport" },
  { code: "L5", name: "Insights", detail: "Analytics · Drift · Reports" },
] as const;

export const API_ENDPOINTS = [
  { method: "POST", path: "/api/v1/survey/run", description: "Execute a complete survey pipeline" },
  { method: "GET", path: "/api/v1/config", description: "Current active configuration" },
  { method: "GET", path: "/health", description: "System health with component breakdown" },
  { method: "GET", path: "/health/ready", description: "Kubernetes readiness probe" },
  { method: "GET", path: "/health/live", description: "Kubernetes liveness probe" },
  { method: "GET", path: "/metrics", description: "Prometheus metrics endpoint" },
] as const;

export const CURL_EXAMPLE = [
  "curl -X POST http://localhost:8000/api/v1/survey/run \\",
  '  -H "Content-Type: application/json" \\',
  '  -d \'{"questions": ["How important is organic labeling?"], "cohort_size": 100}\'',
].join("\n");
