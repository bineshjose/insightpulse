# ADR-004: Strategy pattern + factories for demo/production separation

## Status

Accepted (Stage 4).

## Context

The system must run in three environments with radically different
dependency budgets: `demo` (laptop, no API keys, no database, no GPU
stack), `test` (CI, fully mocked externals), and `production`
(PostgreSQL, Redis, real LLM providers, transformer + FAISS). Earlier
stages leaned toward `if settings.is_demo():` branches inside shared code
paths — which scatter environment logic across the codebase, get missed
in reviews, and make it impossible to read "the production behavior" as
one coherent unit.

Options considered:

1. **Inline `if env == ...` branching** — lowest ceremony, but N call
   sites × M environments of hidden coupling.
2. **Separate codebases / build flavors** — total separation, total
   duplication.
3. **Strategy pattern per layer + a factory composition root** — one
   abstract contract per layer, sibling implementations, environment
   resolved exactly once.

## Decision

Every layer (L1–L5) is an **ABC contract** with a demo Strategy and a
production Strategy, selected by **factory functions** in
`layers/__init__.py` — the only place that reads the environment for
wiring purposes. Agents depend on contracts, never on concrete classes.
The `test` profile maps to the demo strategies so CI can never touch a
provider or database. Heavy imports (torch, faiss, litellm, SQL drivers)
live inside production classes, keeping demo deployments light.

## Consequences

- (+) A reviewer reads the ABC for the contract, the production class for
  the industrial logic, the demo class for the testing strategy — each is
  self-contained.
- (+) Constructor injection everywhere makes each layer unit-testable
  with fakes (`test_layers.py` runs both strategies with no network).
- (+) `test_factories.py` pins the environment → implementation mapping,
  so a wiring regression is a failing test, not a production surprise.
- (−) More files and indirection than inline branches; the factory is one
  more concept to learn. Accepted: the codebase is read far more often
  than it is written, especially by thesis evaluators.
- (−) Contracts must stay strategy-agnostic; anything provider-specific
  (e.g. circuit-breaker state) stays inside the production class.
