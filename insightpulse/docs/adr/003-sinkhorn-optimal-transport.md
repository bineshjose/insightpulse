# ADR-003: Sinkhorn entropic optimal transport for BDCL calibration

## Status

Accepted (thesis core; log-domain implementation since Stage 4).

## Context

Raw LLM response distributions are systematically biased — mode collapse
toward agreeable/modal answers is the dominant artifact, and its severity
is model-specific. The calibration layer must align the synthetic
distribution P_syn with the empirical distribution P_real while
(a) respecting the ordinal structure of Likert/NPS scales,
(b) preserving genuine behavioral signal (λ_b), and
(c) keeping demographic groups within bounded parity (λ_f).

Options considered:

1. **Post-stratification / raking (IPF)** — the survey-industry standard;
   reweights marginals but is blind to the ordinal geometry: moving mass
   from "agree" to "neutral" and to "strongly disagree" cost the same.
2. **Direct distribution replacement** — trivially matches P_real but
   destroys all behavioral signal (equivalent to not running twins).
3. **Optimal transport (exact, linear program)** — respects the ground
   cost, but O(n³ log n) and unstable for repeated per-question solves.
4. **Entropic OT via Sinkhorn** — ε-regularized transport: fast
   (matrix-scaling iterations), differentiable, with a tunable
   sharpness/speed trade-off.

## Decision

Use **Sinkhorn entropic optimal transport** with a squared index-distance
ground cost (ordinal-aware), behavioral regularization λ_b blending back
toward P_syn, and post-hoc fairness verification per demographic group.
Implement the solver **in the log domain** — the multiplicative scaling
form underflows (`exp(-C/ε)` → 0) for the small ε values in the thesis
sweep; log-sum-exp updates are stable across the full ε range.

## Consequences

- (+) The ground cost encodes Likert ordinality — far transports are
  penalized super-linearly, matching how misread a respondent would be.
- (+) ε gives an explicit, measured trade-off (experiments): convergence
  in 10–426 iterations vs transport-plan entropy 0.93–0.58. Production
  uses ε = 0.1.
- (+) The transport plan itself is exported — auditable evidence of *how*
  mass moved, not just the final marginal.
- (−) At convergence Sinkhorn matches the target marginal exactly for any
  ε, so marginal-quality metrics cannot differentiate ε; plan-level
  metrics are required (documented in the convergence experiment).
- (−) One more hyperparameter family (ε, iterations, threshold) — all
  surfaced in `CalibrationConfig`, never hardcoded.
