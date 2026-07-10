# ADR-001: LangGraph for agent orchestration instead of a custom DAG runtime

## Status

Accepted (Stage 1; state schema hardened in Stage 4).

## Context

The thesis specifies an 8-agent pipeline as a directed acyclic graph
G = (A, E) with conditional edges: validation failures loop back to
regeneration (bounded retries), budget overruns short-circuit to audit,
and low diversity triggers temperature-adjusted regeneration. The original
thesis design proposed a custom agent-to-agent (A2A) message protocol and
a hand-rolled scheduler.

Options considered:

1. **Custom asyncio scheduler + A2A protocol** — full control, but we own
   scheduling, retry semantics, state merging, checkpointing, and replay.
2. **LangGraph** — declarative `StateGraph` with typed state channels,
   conditional edges, and built-in checkpointing.
3. **Prefect / Airflow style workflow engines** — mature, but built for
   batch data pipelines; per-request DAG execution inside an API process
   is not their model, and their operational footprint is large.

## Decision

Use **LangGraph**. Agents are plain async functions over a shared
`SurveyPipelineState` TypedDict; the DAG topology and conditional edges
are declared in one place (`agents/orchestrator.py`). State merging is
governed by channel reducers (accumulating `agent_trace`, last-value
everything else).

## Consequences

- (+) Conditional retry edges are one `add_conditional_edges` call each —
  the thesis DAG maps 1:1 onto the framework's primitives.
- (+) Typed state channels give exactly-once, ordered trace accumulation,
  which the AuditAgent's provenance guarantees depend on.
- (+) Checkpointing/replay comes with the framework when needed later.
- (−) A framework dependency with a fast release cadence; mitigated by
  pinning `langgraph>=0.2` and keeping agents framework-agnostic (plain
  functions over dict-like state, trivially portable).
- (−) The state schema is load-bearing and easy to get wrong: passing a
  bare `dict` collapsed all channels into one and silently dropped agent
  outputs — found by the Stage 4 integration tests and fixed by the typed
  TypedDict schema. This ADR records the rule: **never** build the graph
  over an untyped state.
