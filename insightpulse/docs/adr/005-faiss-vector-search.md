# ADR-005: FAISS for behavioral similarity search in cohort selection

## Status

Accepted (Stage 1 design; implemented in Stage 4).

## Context

Cohort selection needs nearest-neighbor search over behavioral embeddings
B_i ∈ ℝ¹²⁸: "expand this seed cohort with behaviorally similar
households." The demo panel is 500 households, but the production target
is NIQ-scale panels (10⁵–10⁷ households), where brute-force distance
computation per query stops being viable.

Options considered:

1. **Brute-force NumPy** — exact, trivial, perfect at demo scale; O(N·d)
   per query at panel scale.
2. **FAISS** — the standard ANN library: exact flat indexes, IVF/HNSW
   approximate indexes, disk persistence, battle-tested at billion scale.
3. **A vector database (pgvector / Qdrant / Milvus)** — operationally
   heavier; pgvector was a serious contender since PostgreSQL is already
   in the stack, but it couples index lifecycle to the database and adds
   little at current scale.

## Decision

Use **FAISS** behind the `find_similar` contract method. The demo
strategy keeps brute-force NumPy (correct choice at ≤ thousands of
vectors); the production `FAISSIndexManager` builds IVFFlat when the
panel is large enough to train it and **falls back to the exact flat
index — logged, never silent — below that threshold**. Indexes persist
to disk beside the embedding cache. Distance thresholds and `nprobe`
live in `EmbeddingConfig`.

## Consequences

- (+) The same contract covers 500 and 10⁷ households; scaling is a
  config change (index type), not an architecture change.
- (+) Index persistence means cohort expansion never re-embeds the panel.
- (−) IVF is approximate: recall depends on `nprobe`; acceptable because
  cohort expansion needs representative neighbors, not exact top-k.
- (−) faiss-cpu is a binary dependency with occasional platform quirks;
  isolated inside the production strategy (lazy import), so demo and CI
  environments never load it.
- Revisit: if the production database standardizes on PostgreSQL 16+,
  re-evaluate pgvector to remove one moving part.
