"""L3 generation pipeline — contract, demo and LLM strategies.

:class:`GenerationEngine` is the L3 contract (Strategy pattern).
:class:`DemoGenerationEngine` samples archetype-conditional
statistical twins; :class:`LLMGenerationEngine` drives real LLM
calls behind a circuit breaker.
"""


from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from insightpulse.config.settings import GenerationConfig, get_settings
from insightpulse.core.exceptions import CircuitBreakerOpenError, GenerationError
from insightpulse.ml.generation import uncertainty as uncertainty_mod
from insightpulse.ml.generation.persona_builder import PersonaPromptBuilder
from insightpulse.ml.generation.response_parser import ResponseParser
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


# Sampling profiles per model family: temperature/top_p tuned per provider
# (evaluator feedback #6 — parameters are model-specific, never shared).
MODEL_PARAMETER_PROFILES: dict[str, dict[str, float]] = {
    "claude": {"temperature": 0.7, "top_p": 0.95},
    "gpt": {"temperature": 0.8, "top_p": 0.9},
    "ollama": {"temperature": 0.9, "top_p": 0.9},
    "default": {"temperature": 0.7, "top_p": 0.95},
}




def _parameter_profile(model: str) -> dict[str, float]:
    """Sampling parameters for a model, resolved by family prefix."""
    lowered = model.lower()
    for family, profile in MODEL_PARAMETER_PROFILES.items():
        if family != "default" and family in lowered:
            return profile
    return MODEL_PARAMETER_PROFILES["default"]


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class GenerationEngine(ABC):
    """Contract for L3 response generation (Strategy pattern).

    Single responsibility: (questions x panelists) -> raw response dicts
    carrying full generation provenance. Collaborators: L2 conditioning
    vectors (persona construction) and :class:`GenerationConfig`.

    Example:
        >>> engine = get_generation_engine()               # factory
        >>> responses = await engine.generate_responses(
        ...     questions, cohort, model="claude-sonnet-4-6", seed=42
        ... )
    """

    @abstractmethod
    async def generate_responses(
        self,
        questions: list[dict[str, Any]],
        panelists: pd.DataFrame,
        model: str,
        seed: int = 42,
        conditioning_vectors: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate one response per (question, panelist) pair.

        Questions are processed in order; for sequential surveys each
        panelist's earlier answers condition the later ones.

        Args:
            questions: Structured question dicts (id, text, type, options).
            panelists: Selected cohort rows (demographics + archetype).
            model: LLM identifier (also recorded on every response).
            seed: Random seed for reproducibility where applicable.
            conditioning_vectors: Optional u_i vectors from L2, used to
                enrich persona construction.

        Returns:
            Response dicts with answer, reasoning, confidence, and full
            generation metadata (model, latency, tokens, cost, parse path).

        Raises:
            GenerationError: When generation fails irrecoverably for the
                whole batch (single-response failures are logged/skipped).
        """


# ---------------------------------------------------------------------------
# Demo implementation — statistical twins
# ---------------------------------------------------------------------------

class DemoGenerationEngine(GenerationEngine):
    """Demo strategy: archetype-conditional statistical twins.

    Single responsibility: reproducible survey responses without any LLM.
    Answers are sampled from each archetype's empirical conditional
    distribution, distorted by the target model's documented bias profile
    (mode collapse + sampling noise) — the same engine validated by the
    Stage 3 experiments, exposed here behind the L3 contract.

    Example:
        >>> engine = DemoGenerationEngine()
        >>> responses = await engine.generate_responses(qs, cohort, "gpt-4o")
    """

    def __init__(self, config: GenerationConfig | None = None) -> None:
        """Create the demo engine.

        Args:
            config: Generation settings (used for provenance fields only —
                the demo engine makes no network calls).
        """
        self._config = config or get_settings().generation

    async def generate_responses(
        self,
        questions: list[dict[str, Any]],
        panelists: pd.DataFrame,
        model: str,
        seed: int = 42,
        conditioning_vectors: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, Any]]:
        """See :meth:`GenerationEngine.generate_responses` (demo strategy)."""
        from insightpulse import demo_engine

        if model not in demo_engine.MODEL_PROFILES:
            raise GenerationError(
                f"No demo profile for model '{model}'. "
                f"Known: {list(demo_engine.MODEL_PROFILES)}"
            )
        profile = demo_engine.MODEL_PROFILES[model]
        rng = np.random.default_rng(seed)
        started = time.perf_counter()

        responses: list[dict[str, Any]] = []
        for question in questions:
            options = question.get("options") or []
            if not options:
                logger.warning(
                    "demo_generation_skipped_open_question",
                    question_id=question.get("question_id"),
                )
                continue
            conditionals = demo_engine.archetype_conditionals(
                question["question_id"], options
            )
            distorted = {
                arch: demo_engine.model_distribution(base, profile, rng)
                for arch, base in conditionals.items()
            }
            fallback = np.full(len(options), 1.0 / len(options))

            for row in panelists.to_dict(orient="records"):
                archetype = str(row.get("behavioral_archetype", ""))
                dist = distorted.get(archetype, fallback)
                answer_index = int(rng.choice(len(options), p=dist))
                responses.append({
                    "response_id": f"{row['panelist_id']}_{question['question_id']}",
                    "question_id": question["question_id"],
                    "panelist_id": str(row["panelist_id"]),
                    "answer": options[answer_index],
                    "answer_index": answer_index,
                    "reasoning": (
                        f"Archetype-conditional response "
                        f"({archetype or 'panel-average'} profile)"
                    ),
                    "confidence": round(
                        float(np.clip(rng.normal(0.78, 0.12), 0.05, 0.99)), 2
                    ),
                    # Analytic form of the k-sample entropy estimator
                    # (§4.4.5): the conditional distribution is known here.
                    "uncertainty": round(
                        uncertainty_mod.distribution_uncertainty(dist), 4
                    ),
                    # Synthetic units inherit their panelist's expansion
                    # weight (§3.1.3) for expansion-weighted calibration.
                    "expansion_factor": float(row.get("expansion_factor", 1.0)),
                    "model_used": model,
                    "generation_time_ms": float(profile["latency_ms"]),
                    "token_count": int(profile["tokens_per_response"]),
                    "cost_usd": 0.0,  # the demo engine never spends
                    "behavioral_cluster": int(row.get("cluster_id", -1)),
                    # Demographic attributes ride along for L5 breakdowns.
                    "age_group": str(row.get("age_group", "")),
                    "income_group": str(row.get("income_group", "")),
                    "region": str(row.get("region", "")),
                    "demographic_summary": (
                        f"{row.get('age_group', '')} | "
                        f"{row.get('income_group', '')} | {row.get('region', '')}"
                    ),
                    "parse_method": "demo",
                })

        logger.info(
            "generation_complete",
            strategy="demo",
            model=model,
            responses=len(responses),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return responses


# ---------------------------------------------------------------------------
# Production collaborators
# ---------------------------------------------------------------------------



class CircuitBreaker:
    """Circuit Breaker for the LLM provider (closed → open → half-open).

    Single responsibility: shed calls fast when the provider is failing.
    After ``failure_threshold`` consecutive failures the breaker opens and
    :meth:`before_call` raises immediately (no network I/O). Once the
    recovery window passes, a single probe call is allowed (half-open);
    its outcome closes or re-opens the breaker.

    Example:
        >>> breaker = CircuitBreaker(threshold=5, recovery_seconds=30)
        >>> breaker.before_call()      # raises CircuitBreakerOpenError when open
        >>> breaker.record_success()   # closes / resets
    """

    def __init__(
        self,
        threshold: int,
        recovery_seconds: float,
        clock: Any = time.monotonic,
    ) -> None:
        """Create a closed breaker.

        Args:
            threshold: Consecutive failures that open the breaker.
            recovery_seconds: Open duration before a half-open probe.
            clock: Injectable time source (tests advance it manually).
        """
        self._threshold = threshold
        self._recovery_seconds = recovery_seconds
        self._clock = clock
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        """Current state: 'closed', 'open', or 'half_open'."""
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self._recovery_seconds:
            return "half_open"
        return "open"

    def before_call(self) -> None:
        """Gate a call attempt.

        Raises:
            CircuitBreakerOpenError: When the breaker is open, or when it
                is half-open and another probe is already in flight.
        """
        state = self.state
        if state == "closed":
            return
        if state == "half_open" and not self._probing:
            self._probing = True  # admit exactly one probe
            logger.info("circuit_breaker_half_open_probe")
            return
        raise CircuitBreakerOpenError(
            f"LLM circuit breaker is {state}; call shed before network I/O"
        )

    def record_success(self) -> None:
        """Close the breaker after a successful call."""
        if self._opened_at is not None:
            logger.info("circuit_breaker_closed")
        self._consecutive_failures = 0
        self._opened_at = None
        self._probing = False

    def record_failure(self) -> None:
        """Count a failure; open the breaker at the threshold."""
        self._consecutive_failures += 1
        self._probing = False
        if (
            self._consecutive_failures >= self._threshold
            and self._opened_at is None
        ) or self.state == "half_open":
            self._opened_at = self._clock()
            logger.warning(
                "circuit_breaker_opened",
                consecutive_failures=self._consecutive_failures,
                recovery_seconds=self._recovery_seconds,
            )


# ---------------------------------------------------------------------------
# Production implementation — real LLM calls
# ---------------------------------------------------------------------------

class LLMGenerationEngine(GenerationEngine):
    """Production strategy: concurrent, resilient LLM twin generation.

    Single responsibility: orchestrate prompt building, semaphore-limited
    concurrent LLM calls, retries, circuit breaking, parsing, and token
    budgeting into the L3 contract. Collaborators:
    :class:`PersonaPromptBuilder`, :class:`ResponseParser`,
    :class:`CircuitBreaker`, and the LiteLLM router (injected — tests pass
    a fake).

    Example:
        >>> engine = LLMGenerationEngine()            # router built lazily
        >>> responses = await engine.generate_responses(
        ...     questions, cohort, "claude-sonnet-4-6"
        ... )
    """

    def __init__(
        self,
        config: GenerationConfig | None = None,
        router: Any = None,
        prompt_builder: PersonaPromptBuilder | None = None,
        parser: ResponseParser | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Create the production engine with injectable collaborators.

        Args:
            config: Concurrency/resilience settings. Defaults to profile's.
            router: LLM router exposing ``async generate(...)``. Defaults
                to a lazily constructed :class:`LLMRouter` (kept lazy so
                importing this module never imports litellm).
            prompt_builder: Persona prompt builder override.
            parser: Response parser override.
            circuit_breaker: Breaker override (tests inject a fake clock).
        """
        self._config = config or get_settings().generation
        self._router = router
        self._prompt_builder = prompt_builder or PersonaPromptBuilder(self._config)
        self._parser = parser or ResponseParser()
        self._breaker = circuit_breaker or CircuitBreaker(
            self._config.circuit_failure_threshold,
            self._config.circuit_recovery_seconds,
        )
        self._semaphore = asyncio.Semaphore(self._config.max_concurrency)

    def _get_router(self) -> Any:
        """Build the default router on first use (lazy litellm import)."""
        if self._router is None:
            from insightpulse.ml.llm.router import LLMRouter

            self._router = LLMRouter()
        return self._router

    async def generate_responses(
        self,
        questions: list[dict[str, Any]],
        panelists: pd.DataFrame,
        model: str,
        seed: int = 42,
        conditioning_vectors: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, Any]]:
        """See :meth:`GenerationEngine.generate_responses` (LLM strategy).

        Questions run strictly in order (sequential conditioning);
        panelists within a question run concurrently under the semaphore.
        """
        started = time.perf_counter()
        rows = panelists.to_dict(orient="records")
        all_responses: list[dict[str, Any]] = []
        priors: dict[str, list[str]] = {str(r["panelist_id"]): [] for r in rows}
        failure_count = 0

        for question_index, question in enumerate(questions):
            tasks = [
                self._generate_single(
                    row,
                    question,
                    model,
                    priors[str(row["panelist_id"])] if question_index > 0 else None,
                )
                for row in rows
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for row, result in zip(rows, results, strict=True):
                if isinstance(result, BaseException):
                    failure_count += 1
                    logger.error(
                        "response_generation_failed",
                        panelist_id=row["panelist_id"],
                        question_id=question["question_id"],
                        error=str(result)[:300],
                    )
                    continue
                all_responses.append(result)
                priors[str(row["panelist_id"])].append(result["answer"])

        if not all_responses and failure_count:
            raise GenerationError(
                f"All {failure_count} generation attempts failed for model "
                f"'{model}' — provider unreachable or misconfigured"
            )

        logger.info(
            "generation_complete",
            strategy="llm",
            model=model,
            responses=len(all_responses),
            failures=failure_count,
            template_version=self._prompt_builder.template_version,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return all_responses

    async def _generate_single(
        self,
        panelist: dict[str, Any],
        question: dict[str, Any],
        model: str,
        prior_responses: list[str] | None,
    ) -> dict[str, Any]:
        """Generate one response: breaker -> semaphore -> retry -> parse.

        Args:
            panelist: Panelist attributes.
            question: Structured question dict.
            model: LLM identifier.
            prior_responses: Sequential conditioning answers, if any.

        Returns:
            Response dict with full generation provenance.

        Raises:
            CircuitBreakerOpenError: When calls are being shed.
            GenerationError: When retries are exhausted for this response.
        """
        from tenacity import (
            AsyncRetrying,
            retry_if_exception_type,
            stop_after_attempt,
            wait_fixed,
        )

        self._breaker.before_call()
        system, user = self._prompt_builder.build(panelist, question, prior_responses)
        params = _parameter_profile(model)

        async with self._semaphore:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self._config.retry_attempts),
                    wait=wait_fixed(self._config.retry_wait_seconds),
                    retry=retry_if_exception_type(Exception),
                    reraise=True,
                ):
                    with attempt:
                        result = await self._get_router().generate(
                            prompt=user,
                            system=system,
                            model=model,
                            temperature=params["temperature"],
                            top_p=params["top_p"],
                            max_tokens=self._config.max_tokens_per_response,
                        )
            except Exception as exc:
                self._breaker.record_failure()
                raise GenerationError(
                    f"LLM call failed after {self._config.retry_attempts} "
                    f"attempts: {exc}"
                ) from exc

        self._breaker.record_success()
        parsed = self._parser.parse(result.content, question)

        over_budget = result.total_tokens > self._config.max_tokens_per_response
        if over_budget:
            logger.warning(
                "token_budget_exceeded",
                panelist_id=panelist["panelist_id"],
                tokens=result.total_tokens,
                budget=self._config.max_tokens_per_response,
            )

        return {
            "response_id": f"{panelist['panelist_id']}_{question['question_id']}",
            "question_id": question["question_id"],
            "panelist_id": str(panelist["panelist_id"]),
            "answer": parsed["answer"],
            "reasoning": parsed["reasoning"],
            "confidence": parsed["confidence"],
            # Synthetic units inherit the panelist's expansion weight
            # (§3.1.3); k-sample uncertainty is estimated downstream when
            # η-aware calibration is enabled.
            "expansion_factor": float(panelist.get("expansion_factor", 1.0)),
            "model_used": model,
            "generation_time_ms": result.latency_ms,
            "token_count": result.total_tokens,
            "cost_usd": result.cost_usd,
            "behavioral_cluster": int(panelist.get("cluster_id", -1) or -1),
            # Demographic attributes ride along for L5 breakdowns.
            "age_group": str(panelist.get("age_group", "")),
            "income_group": str(panelist.get("income_group", "")),
            "region": str(panelist.get("region", "")),
            "demographic_summary": (
                f"{panelist.get('age_group', '')} | "
                f"{panelist.get('income_group', '')} | {panelist.get('region', '')}"
            ),
            "parse_method": parsed["parse_method"],
            "over_token_budget": over_budget,
        }
