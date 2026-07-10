# ADR-002: LiteLLM for multi-model routing

## Status

Accepted (Stage 1).

## Context

The evaluators explicitly require multi-LLM comparison (feedback #1, #8):
the same survey must run on Claude, OpenAI, and local Ollama models with
per-model cost, latency, and quality accounting. Each provider has a
different SDK, auth scheme, streaming shape, and pricing table.

Options considered:

1. **Per-provider SDKs behind our own adapter layer** — three SDKs to
   track, our own cost tables to maintain.
2. **LiteLLM** — one `completion()` API over 100+ providers, built-in
   cost calculation, provider-prefixed model names (`ollama/llama3.1`).
3. **LangChain chat-model abstractions** — heavier dependency surface for
   the same routing need; we already keep LangChain to `langchain-core`.

## Decision

Use **LiteLLM** behind our own thin `LLMRouter` (`llm/router.py`). The
router owns what LiteLLM does not: structured call metadata
(`LLMCallResult`), per-model performance summaries for the comparison
experiments, and structlog integration. Model families get their own
sampling profiles in L3 (temperature/top_p per provider).

## Consequences

- (+) Adding a model is a config string, not an integration project —
  exactly what the multi-LLM experiment needs.
- (+) Built-in cost tracking feeds the CostAgent's budget enforcement.
- (+) Ollama support gives a zero-cost local baseline for the comparison.
- (−) LiteLLM abstracts away provider-specific capabilities (e.g. native
  structured outputs); our `ResponseParser` fallback chain compensates.
- (−) Cost tables for brand-new models can lag; the router falls back to
  configured per-million rates when LiteLLM cannot price a response.
