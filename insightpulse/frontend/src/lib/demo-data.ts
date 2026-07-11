/**
 * Deterministic seed data mirroring the Python demo engine, so every page
 * renders meaningfully before (or without) a live backend. Values match
 * experiments/results/*.json.
 */

import type { AgentTraceEntry, SurveyMetadata, SurveyResult, SurveyRunResponse } from "@/lib/types";

/** FMCG clients available on the survey Setup step. */
export const CLIENTS = [
  "Unilever",
  "Procter & Gamble",
  "Nestlé",
  "PepsiCo",
  "Coca-Cola",
  "Mondelēz",
  "Mars",
  "Colgate-Palmolive",
  "Reckitt",
  "Internal Research",
] as const;

export const CATEGORIES = [
  "FMCG — Food",
  "FMCG — Beverages",
  "FMCG — Personal Care",
  "FMCG — Household",
  "Consumer Electronics",
  "Health & Wellness",
] as const;

export const PRIORITIES = ["High", "Medium", "Low"] as const;

/** Short client codes used in auto-suggested contract IDs. */
const CLIENT_CODES: Record<string, string> = {
  Unilever: "UNI",
  "Procter & Gamble": "PG",
  "Nestlé": "NES",
  PepsiCo: "PEP",
  "Coca-Cola": "KO",
  "Mondelēz": "MDLZ",
  Mars: "MARS",
  "Colgate-Palmolive": "CL",
  Reckitt: "RKT",
  "Internal Research": "INT",
};

/** Auto-suggest a contract ID for a client, e.g. "NIQ-UNI-2026-Q3-047". */
export function suggestContractId(client: string, sequence = 47): string {
  const now = new Date();
  const quarter = Math.floor(now.getMonth() / 3) + 1;
  const code = CLIENT_CODES[client] ?? "GEN";
  return `NIQ-${code}-${now.getFullYear()}-Q${quarter}-${String(sequence).padStart(3, "0")}`;
}

/** Display survey ID, e.g. "SRV-2026-00142". */
export function makeSurveyId(sequence: number): string {
  return `SRV-${new Date().getFullYear()}-${String(sequence).padStart(5, "0")}`;
}

/** "Jul 11, 2026 · 10:34 AM" from a Date. */
export function formatRunDate(date: Date): string {
  const day = date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  const time = date.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  return `${day} · ${time}`;
}

/** A recent timestamp `days` back at the given business hour. */
function recentDate(days: number, hour: number, minute: number): Date {
  const date = new Date();
  date.setDate(date.getDate() - days);
  date.setHours(hour, minute, 0, 0);
  return date;
}

export const LIKERT_OPTIONS = [
  "Not at all important",
  "Slightly important",
  "Moderately important",
  "Very important",
  "Extremely important",
] as const;

export const SUPPORTED_MODELS = [
  "claude-sonnet-4-6",
  "gpt-4o",
  "claude-haiku-4-5",
  "ollama/llama3.1",
] as const;

/** NIQ chart series — same validated order as the Streamlit theme. */
export const CHART_SERIES = [
  "#00A4E4",
  "#E09C00",
  "#6F5AA8",
  "#6CC24A",
  "#E03C31",
  "#2E6BAA",
  "#C9508B",
  "#946200",
] as const;

export const SERIES_COLORS = {
  raw: CHART_SERIES[0],
  calibrated: CHART_SERIES[1],
  empirical: CHART_SERIES[2],
} as const;

const LATEST_TRACE: AgentTraceEntry[] = [
  { agent_name: "SurveyDesigner", action: "parse_questions", output_summary: "Parsed and structured 1 question", duration_ms: 843 },
  { agent_name: "CohortSelector", action: "select_cohort", output_summary: "Selected 250 households via FAISS similarity", duration_ms: 317 },
  { agent_name: "TwinOrchestrator", action: "generate_responses", output_summary: "Generated 250 responses with claude-sonnet-4-6", duration_ms: 19375 },
  { agent_name: "Validator", action: "validate_responses", output_summary: "Flagged 1.6% responses for hallucination", duration_ms: 641 },
  { agent_name: "CostAgent", action: "cost_check", output_summary: "Verified spend within budget", duration_ms: 44 },
  { agent_name: "CalibrationAgent", action: "bdcl_calibration", output_summary: "Aligned P_syn to P_real via Sinkhorn OT", duration_ms: 706 },
  { agent_name: "DiversityMonitor", action: "check_entropy", output_summary: "Checked Shannon entropy against diversity floor", duration_ms: 162 },
  { agent_name: "AuditAgent", action: "finalize_results", output_summary: "Wrote provenance record", duration_ms: 118 },
];

const LATEST_RESULT: SurveyResult = {
  question_id: "q_organic",
  question_text: "How important is organic labeling when purchasing snacks?",
  options: [...LIKERT_OPTIONS],
  total_responses: 250,
  valid_responses: 246,
  distribution: [
    { option: LIKERT_OPTIONS[0], count: 44, percentage: 17.6 },
    { option: LIKERT_OPTIONS[1], count: 58, percentage: 23.2 },
    { option: LIKERT_OPTIONS[2], count: 69, percentage: 27.6 },
    { option: LIKERT_OPTIONS[3], count: 51, percentage: 20.4 },
    { option: LIKERT_OPTIONS[4], count: 28, percentage: 11.2 },
  ],
  calibrated_distribution: [0.166, 0.216, 0.262, 0.219, 0.137],
  entropy: 2.21,
  calibration_metrics: {
    question_id: "q_organic",
    wasserstein_before: 0.0782,
    wasserstein_after: 0.0233,
    js_divergence_before: 0.0102,
    js_divergence_after: 0.0009,
    wasserstein_improvement_pct: 70.2,
    js_improvement_pct: 91.2,
    converged: true,
    iterations: 46,
  },
};

const LATEST_METADATA: SurveyMetadata = {
  survey_id: "SRV-2026-00142",
  survey_name: "Organic Labeling Importance",
  client_name: "Unilever",
  contract_id: "NIQ-UNI-2026-Q3-047",
  category: "FMCG — Food",
  region: "APAC, EMEA, Americas",
  priority: "High",
  executor_name: "Binesh Jose",
  executor_email: "binesh.jose@nielseniq.com",
  created_at: recentDate(0, 10, 34).toISOString(),
};

/** The most recent completed run, shown on Results/Audit before a live run. */
export const LATEST_RUN: SurveyRunResponse = {
  run_id: "a3f8c2d1",
  status: "completed",
  total_responses: 250,
  total_cost_usd: 0.44,
  hallucination_rate: 0.016,
  results: [LATEST_RESULT],
  agent_trace: LATEST_TRACE,
  provenance_hash: "9c1d2e3f4a5b6c7d",
  metadata: LATEST_METADATA,
};

/** Multi-LLM comparison — matches experiments/multi_llm_comparison.json. */
export const MODEL_COMPARISON = [
  { model: "claude-sonnet-4-6", jsDivergence: 0.0002, wasserstein: 0.0233, hallucination: 0.018, consistency: 0.99, costUsd: 0.65, throughput: 1248 },
  { model: "gpt-4o", jsDivergence: 0.0003, wasserstein: 0.0253, hallucination: 0.028, consistency: 0.98, costUsd: 0.44, throughput: 1420 },
  { model: "ollama/llama3.1", jsDivergence: 0.0004, wasserstein: 0.0368, hallucination: 0.074, consistency: 0.953, costUsd: 0.0, throughput: 530 },
] as const;

/** Sinkhorn convergence — iterations to threshold per ε (experiment data). */
export const CONVERGENCE_BY_EPSILON: Record<string, { iteration: number; error: number }[]> = {
  "0.01": Array.from({ length: 40 }, (_, i) => ({ iteration: i * 8 + 1, error: 0.3 * Math.exp(-0.055 * (i * 8 + 1)) })),
  "0.05": Array.from({ length: 40 }, (_, i) => ({ iteration: i * 2 + 1, error: 0.25 * Math.exp(-0.29 * (i * 2 + 1)) })),
  "0.1": Array.from({ length: 40 }, (_, i) => ({ iteration: i + 1, error: 0.22 * Math.exp(-0.56 * (i + 1)) })),
  "0.5": Array.from({ length: 12 }, (_, i) => ({ iteration: i + 1, error: 0.18 * Math.exp(-2.1 * (i + 1)) })),
};

/** Drift series — matches experiments/drift_detection.json shape. */
export const DRIFT_SERIES = [
  { month: "25-10", stationary: 0.0025, drifted: 0.0025 },
  { month: "25-11", stationary: 0.0014, drifted: 0.0014 },
  { month: "25-12", stationary: 0.0013, drifted: 0.0013 },
  { month: "26-01", stationary: 0.0021, drifted: 0.0021 },
  { month: "26-02", stationary: 0.0008, drifted: 0.0008 },
  { month: "26-03", stationary: 0.0007, drifted: 0.0018 },
  { month: "26-04", stationary: 0.0032, drifted: 0.0098 },
  { month: "26-05", stationary: 0.0045, drifted: 0.0107 },
  { month: "26-06", stationary: 0.0017, drifted: 0.0248 },
] as const;

export const DRIFT_TRIGGER = 0.0055;

/** Sequential-dependency experiment summary. */
export const SEQUENTIAL_SUMMARY = {
  spearman: { independent: 0.28, conditioned: 0.65, empirical: 0.24 },
  contradictions: { independent: 0.104, conditioned: 0.018, empirical: 0.116 },
} as const;

/** Validation targets vs measured (acceptance criteria). */
export const VALIDATION_CHECKS = [
  { metric: "Cosine similarity", target: "≥ 0.80", actual: "0.84", pass: true, why: "Behavioral embeddings are unit-normalized; angular distance captures alignment." },
  { metric: "JS divergence", target: "≤ 0.05", actual: "0.017", pass: true, why: "Symmetric, bounded, finite on empty answer bins (unlike KL)." },
  { metric: "Wasserstein distance", target: "≤ 0.15", actual: "0.041", pass: true, why: "Respects Likert ordinality — far transports cost more." },
  { metric: "Hallucination rate", target: "< 5%", actual: "1.9%", pass: true, why: "Fraction of twins citing non-existent facts (Validator flags)." },
  { metric: "Logical consistency", target: "> 90%", actual: "94.6%", pass: true, why: "Cross-question coherence in sequential surveys." },
  { metric: "Shannon entropy", target: "≥ 1.5 bits", actual: "2.31", pass: true, why: "Diversity floor — guards against LLM mode collapse." },
] as const;

/** Recent runs table for the dashboard overview and audit history. */
export const RECENT_RUNS = [
  { surveyId: "SRV-2026-00142", survey: "Organic Labeling Importance", client: "Unilever", respondents: 250, model: "claude-sonnet-4-6", cost: 0.44, hallucination: "1.9%", status: "completed", date: formatRunDate(recentDate(0, 10, 34)) },
  { surveyId: "SRV-2026-00141", survey: "Q3 Brand Perception Tracker", client: "Procter & Gamble", respondents: 500, model: "claude-sonnet-4-6", cost: 0.91, hallucination: "2.1%", status: "completed", date: formatRunDate(recentDate(1, 14, 15)) },
  { surveyId: "SRV-2026-00140", survey: "Sustainability Willingness-to-Pay", client: "Nestlé", respondents: 120, model: "gpt-4o", cost: 0.19, hallucination: "2.3%", status: "completed", date: formatRunDate(recentDate(2, 11, 20)) },
  { surveyId: "SRV-2026-00139", survey: "Snack Purchase Frequency Pulse", client: "PepsiCo", respondents: 300, model: "ollama/llama3.1", cost: 0.0, hallucination: "1.5%", status: "completed", date: formatRunDate(recentDate(3, 16, 45)) },
  { surveyId: "SRV-2026-00138", survey: "Premium Tier Price Sensitivity", client: "Mondelēz", respondents: 200, model: "claude-sonnet-4-6", cost: 0.35, hallucination: "1.8%", status: "completed", date: formatRunDate(recentDate(4, 9, 30)) },
] as const;

const LAST_RUN_KEY = "insightpulse.lastRun";

/** Persist the latest live run so Results/Audit pages survive navigation. */
export function storeLastRun(run: SurveyRunResponse): void {
  try {
    window.localStorage.setItem(LAST_RUN_KEY, JSON.stringify(run));
  } catch {
    // storage full/blocked: page state still holds the run
  }
}

/** The latest live run, or null (callers fall back to SAMPLE_RUN). */
export function loadLastRun(): SurveyRunResponse | null {
  try {
    const stored = window.localStorage.getItem(LAST_RUN_KEY);
    return stored ? (JSON.parse(stored) as SurveyRunResponse) : null;
  } catch {
    return null;
  }
}
