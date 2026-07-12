# ADR-007: Layered LLM security with a dependency-free JWT implementation

## Status

Accepted (Final Stage).

## Context

An LLM-backed survey platform has attack surfaces a conventional CRUD
API does not: prompt injection (direct in the question text, or hidden
behind base64/hex/unicode obfuscation), PII leakage in generated
responses, training-data regurgitation, and token-exhaustion abuse.
Reviewers and enterprise customers expect these threats to be addressed
explicitly, not deferred to the LLM provider.

Constraints:

1. Demo mode must run on a laptop with zero API keys and no extra
   dependency footprint (the demo venv deliberately excludes heavy or
   niche packages).
2. Detection must run *inside* the pipeline, not only at the API edge —
   the dashboard's Demo Mode bypasses HTTP entirely.
3. Every threshold and pattern must be tunable from configuration.

Options considered for authentication:

1. **python-jose / PyJWT dependency** — battle-tested, but adds a
   required dependency for a demo that may never verify a token.
2. **Session-only auth** — no API story for programmatic clients.
3. **Stdlib HS256 JWT (hmac + base64 + json) behind a stable
   interface** — RFC 7519-conformant for the one algorithm we use,
   zero dependencies, swappable for python-jose in production.

Options considered for guarding:

1. **Provider-side moderation only** — invisible to auditors, varies by
   model, unavailable for local Ollama.
2. **A single regex filter at the API boundary** — misses the Demo Mode
   path and response-side leakage.
3. **Layered guards at both ends of L3** — PromptGuard before any
   persona prompt is constructed, ResponseGuard before any response is
   stored, plus boundary input validation for cheap 422s.

## Decision

Build `src/insightpulse/security/` as a cross-cutting layer with:

- **PromptGuard** (Facade over a Chain of Responsibility of detectors:
  patterns, encodings, length, template structure) screening every
  question inside the `TwinOrchestrator`, so both HTTP and Demo Mode
  paths are covered. Unsafe questions are dropped and recorded in the
  audit trace with `prompt_injection_blocked`.
- **ResponseGuard** redacting PII at the source and re-checked by the
  `Validator` agent (defense in depth) with `security_check:*` flags.
- **Stdlib HS256 JWT** (`JWTAuthenticator`) with constant-time
  signature comparison and a pinned algorithm (rejects `none`);
  python-jose remains available via the `security` optional extra and
  can replace the implementation without interface changes (Strategy).
- **Input validation** as framework-agnostic pure functions reused by
  the FastAPI validators (HTML/script vectors, SQL-injection patterns,
  allowlisted models/filters/clients, contract-ID format).

All thresholds, pattern toggles, rate limits, and allowlists live in
`SecurityConfig` (`config/settings.py`).

## Consequences

- Demo mode carries the full security posture with no new dependencies;
  the same code paths run in production with stricter settings (HSTS,
  explicit CORS origins, mandatory JWT-or-API-key on `/api/`).
- Rule-based detection is auditable and deterministic, but weaker than
  an ML classifier against novel injections — mitigated by layering
  (patterns + encodings + structure + response-side checks) and by
  logging every detection for review.
- Implementing JWT in-tree means owning ~100 lines of crypto-adjacent
  code; scope is deliberately limited to HS256 and covered by
  roundtrip/expiry/tamper tests.
