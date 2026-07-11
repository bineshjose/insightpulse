/**
 * TypeScript types mirroring the Python Pydantic models (src/insightpulse/models)
 * and the FastAPI request/response schemas (src/insightpulse/main.py).
 * Field names match the JSON wire format exactly.
 */

// ---------------------------------------------------------------------------
// Survey domain (models/survey.py)
// ---------------------------------------------------------------------------

export type QuestionType =
  | "single_choice"
  | "multiple_choice"
  | "likert_5"
  | "likert_7"
  | "open_ended"
  | "ranking"
  | "net_promoter";

export interface SurveyQuestion {
  question_id: string;
  text: string;
  question_type: QuestionType;
  options: string[];
  category?: string;
  context?: string;
  is_sequential?: boolean;
}

export interface SurveyResponse {
  response_id: string;
  question_id: string;
  panelist_id: string;
  answer: string;
  answer_index?: number | null;
  confidence: number;
  reasoning?: string;
  model_used: string;
  is_valid: boolean;
  validation_flags: string[];
}

export interface ResponseDistribution {
  option: string;
  count: number;
  percentage: number;
}

export interface CalibrationMetrics {
  question_id: string;
  wasserstein_before: number;
  wasserstein_after: number;
  js_divergence_before: number;
  js_divergence_after: number;
  wasserstein_improvement_pct: number;
  js_improvement_pct: number;
  converged?: boolean;
  iterations?: number;
}

export interface SurveyResult {
  question_id: string;
  question_text: string;
  options: string[];
  total_responses: number;
  valid_responses: number;
  distribution: ResponseDistribution[];
  calibrated_distribution: number[] | null;
  entropy: number;
  calibration_metrics: CalibrationMetrics | null;
}

export interface AgentTraceEntry {
  agent_name: string;
  action: string;
  output_summary: string;
  duration_ms: number;
}

// ---------------------------------------------------------------------------
// API schemas (main.py)
// ---------------------------------------------------------------------------

export interface SurveyRequest {
  questions: string[];
  cohort_size: number;
  context?: string;
  models?: string[];
  cohort_filters?: Record<string, string>;
  seed?: number;
}

/** Business provenance attached to a survey run (client, contract, executor). */
export interface SurveyMetadata {
  survey_id: string;
  survey_name: string;
  client_name: string;
  contract_id: string;
  category: string;
  region: string;
  priority: string;
  executor_name: string;
  executor_email: string;
  created_at: string;
  due_date?: string | null;
  notes?: string;
}

export interface SurveyRunResponse {
  run_id: string;
  status: string;
  total_responses: number;
  total_cost_usd: number;
  hallucination_rate: number;
  results: SurveyResult[];
  agent_trace: AgentTraceEntry[];
  provenance_hash: string;
  metadata?: SurveyMetadata;
}

export interface HealthResponse {
  status: string;
  env: string;
  version: string;
}

export interface ConfigResponse {
  env: string;
  default_model: string;
  default_cohort_size: number;
  calibration: {
    epsilon: number;
    lambda_behavioral: number;
    lambda_fairness: number;
  };
  embedding: { dim: number; num_clusters: number };
  limits: {
    rate_limit_per_minute: number;
    max_questions: number;
    max_question_length: number;
    max_cohort_size: number;
  };
}

export interface ModelsResponse {
  default: string;
  supported: string[];
}

// ---------------------------------------------------------------------------
// Users & access (mirrors dashboard/components/auth.py)
// ---------------------------------------------------------------------------

export type SubscriptionTier = "Enterprise" | "Professional" | "Academic";

export type Permission =
  | "view"
  | "create"
  | "run"
  | "analyze"
  | "export"
  | "calibrate";

export type UserRole =
  | "Platform Administrator"
  | "Read-Only Evaluator"
  | "Survey Analyst";

export interface User {
  email: string;
  name: string;
  initials: string;
  role: UserRole;
  title: string;
  department: string;
  regions: string[];
  permissions: Permission[];
  tier: SubscriptionTier;
  creditsTotal: number;
  creditsBalance: number;
  apiCallsRemaining: number;
  apiCallsQuota: number;
  maxCohortSize: number;
  created: string;
}
