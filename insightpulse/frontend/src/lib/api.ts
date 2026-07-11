/**
 * Typed API client for the InsightPulse backend.
 *
 * All calls go through the Next.js rewrite (/api/backend/* → FastAPI) so the
 * browser stays same-origin. Every helper returns typed data or throws a
 * typed ApiError; callers own loading/error/empty states.
 */

import type {
  ConfigResponse,
  HealthResponse,
  ModelsResponse,
  SurveyRequest,
  SurveyRunResponse,
} from "@/lib/types";

/** Structured error carrying the HTTP status and backend detail message. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

const IS_DEV = process.env.NODE_ENV === "development";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (IS_DEV) {
    // Request logging in development only — never in production builds.
    console.info(`[api] ${init?.method ?? "GET"} ${path}`);
  }
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, "Backend unreachable — is the API running?");
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail !== undefined) detail = JSON.stringify(body.detail);
    } catch {
      // non-JSON error body: keep statusText
    }
    throw new ApiError(response.status, detail);
  }

  const data = (await response.json()) as T;
  if (IS_DEV) console.info(`[api] ${path} ←`, data);
  return data;
}

/** Execute a synthetic survey through the full agent pipeline. */
export function runSurvey(body: SurveyRequest): Promise<SurveyRunResponse> {
  return request<SurveyRunResponse>("/api/backend/v1/survey/run", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Safe subset of the active backend configuration. */
export function getConfig(): Promise<ConfigResponse> {
  return request<ConfigResponse>("/api/backend/v1/config");
}

/** Models accepted by the survey endpoint. */
export function getModels(): Promise<ModelsResponse> {
  return request<ModelsResponse>("/api/backend/v1/models");
}

/** Backend liveness (also used for the header environment badge). */
export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}
