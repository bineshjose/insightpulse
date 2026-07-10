"""L3 — Digital Twin Generative Layer.

Architectural role
    Implements thesis layer L3: each selected panelist becomes a digital
    twin that answers survey questions. The twin is conditioned on
    u_i = [z_i; d_i] from L2 — in the LLM strategy through a structured
    persona prompt, in the simulated strategy through archetype-conditional
    response distributions.

Design decisions
    * **Strategy pattern** — :class:`SimulatedGenerationEngine` (demo:
      statistical simulation, zero network dependencies, reproducible) and
      :class:`LLMGenerationEngine` (production: real LLM calls) share the
      :class:`GenerationEngine` contract, so the pipeline cannot tell them
      apart.
    * **Circuit Breaker pattern** — the production engine sheds load
      *before* network I/O once the provider is failing, so a dead API
      degrades a run in seconds instead of minutes of timeout stacking.
    * **Fallback-chain parsing** — LLM output goes through
      JSON -> regex -> option-scan -> raw extraction; every response
      records which parser succeeded, making output-quality regressions
      observable per model.
    * Sequential surveys feed each twin's **prior answers into subsequent
      prompts** (evaluator feedback #3) — the measured effect is
      +0.37 within-person Spearman (see experiments/sequential_dependency).
    * Concurrency, retries, breaker thresholds, token budgets, and the
      prompt template version all come from :class:`GenerationConfig`.

Evaluator feedback addressed
    #1/#8 (model differences) via per-family parameter profiles and
    per-response provenance; #3 (sequential dependency) via prior-answer
    conditioning; #6 (calibration per model) is enabled by recording the
    generating model on every response.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from insightpulse.config.settings import GenerationConfig, get_settings
from insightpulse.exceptions import CircuitBreakerOpenError, GenerationError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Behavioral archetype descriptions used in persona prompts. Keys match the
# K=5 clusters from L2 (thesis Results-1) and the sample-data archetypes.
CLUSTER_PERSONAS: dict[int, str] = {
    0: (
        "You are a price-sensitive shopper. You carefully compare prices, "
        "frequently buy store brands, wait for sales, and use coupons regularly. "
        "You prioritize getting the best value for money and are willing to "
        "switch brands to save money."
    ),
    1: (
        "You are a premium brand loyalist. You prefer well-known premium brands "
        "and are willing to pay more for perceived quality. You rarely switch "
        "brands based on price and value consistency and brand reputation."
    ),
    2: (
        "You are a category explorer. You enjoy trying new products and brands, "
        "purchase across many different categories, and are open to innovation. "
        "You are influenced by product novelty and variety."
    ),
    3: (
        "You are a promotion-driven buyer. You plan purchases around promotions, "
        "stock up during deals, and are highly responsive to advertising and "
        "in-store promotions. You may switch brands for a good promotion."
    ),
    4: (
        "You are a convenience-oriented consumer. You value convenience and "
        "time-saving, prefer ready-to-use products, shop frequently at nearby "
        "stores, and make quick purchasing decisions without extensive comparison."
    ),
}

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
# Demo implementation — statistical simulation
# ---------------------------------------------------------------------------

class SimulatedGenerationEngine(GenerationEngine):
    """Demo strategy: archetype-conditional statistical twins.

    Single responsibility: reproducible survey responses without any LLM.
    Answers are sampled from each archetype's empirical conditional
    distribution, distorted by the target model's documented bias profile
    (mode collapse + sampling noise) — the same engine validated by the
    Stage 3 experiments, exposed here behind the L3 contract.

    Example:
        >>> engine = SimulatedGenerationEngine()
        >>> responses = await engine.generate_responses(qs, cohort, "gpt-4o")
    """

    def __init__(self, config: GenerationConfig | None = None) -> None:
        """Create the simulated engine.

        Args:
            config: Generation settings (used for provenance fields only —
                the simulation makes no network calls).
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
        """See :meth:`GenerationEngine.generate_responses` (simulated)."""
        from insightpulse import simulation

        if model not in simulation.MODEL_PROFILES:
            raise GenerationError(
                f"No simulation profile for model '{model}'. "
                f"Known: {list(simulation.MODEL_PROFILES)}"
            )
        profile = simulation.MODEL_PROFILES[model]
        rng = np.random.default_rng(seed)
        started = time.perf_counter()

        responses: list[dict[str, Any]] = []
        for question in questions:
            options = question.get("options") or []
            if not options:
                logger.warning(
                    "simulated_generation_skipped_open_question",
                    question_id=question.get("question_id"),
                )
                continue
            conditionals = simulation.archetype_conditionals(
                question["question_id"], options
            )
            distorted = {
                arch: simulation.model_distribution(base, profile, rng)
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
                        f"Simulated {archetype or 'panel-average'} response "
                        f"(archetype-conditional distribution)"
                    ),
                    "confidence": round(
                        float(np.clip(rng.normal(0.78, 0.12), 0.05, 0.99)), 2
                    ),
                    "model_used": model,
                    "generation_time_ms": float(profile["latency_ms"]),
                    "token_count": int(profile["tokens_per_response"]),
                    "cost_usd": 0.0,  # simulation never spends
                    "behavioral_cluster": int(row.get("cluster_id", -1)),
                    # Demographic attributes ride along for L5 breakdowns.
                    "age_group": str(row.get("age_group", "")),
                    "income_group": str(row.get("income_group", "")),
                    "region": str(row.get("region", "")),
                    "demographic_summary": (
                        f"{row.get('age_group', '')} | "
                        f"{row.get('income_group', '')} | {row.get('region', '')}"
                    ),
                    "parse_method": "simulated",
                })

        logger.info(
            "generation_complete",
            strategy="simulated",
            model=model,
            responses=len(responses),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return responses


# ---------------------------------------------------------------------------
# Production collaborators
# ---------------------------------------------------------------------------

class PersonaPromptBuilder:
    """Builds versioned persona prompts from panelist context.

    Single responsibility: (panelist, question, priors) -> (system, user)
    prompt pair. The template version is stamped by config and recorded in
    provenance, so prompt changes are visible in run comparisons — prompt
    drift is a real confound in LLM systems.

    Example:
        >>> builder = PersonaPromptBuilder(config)
        >>> system, user = builder.build(row, question, prior_responses)
    """

    def __init__(self, config: GenerationConfig) -> None:
        """Store the template version and limits."""
        self._config = config

    @property
    def template_version(self) -> str:
        """Version tag stamped into every run's provenance."""
        return self._config.prompt_template_version

    def build(
        self,
        panelist: dict[str, Any],
        question: dict[str, Any],
        prior_responses: list[str] | None = None,
    ) -> tuple[str, str]:
        """Construct the (system, user) prompt pair for one twin response.

        Args:
            panelist: Panelist attributes (demographics + cluster).
            question: Structured question dict.
            prior_responses: The twin's earlier answers in this survey
                (sequential conditioning — evaluator feedback #3).

        Returns:
            (system_prompt, user_prompt).
        """
        cluster_id = int(panelist.get("cluster_id", 0) or 0)
        behavioral = CLUSTER_PERSONAS.get(cluster_id, CLUSTER_PERSONAS[0])

        demographics = (
            f"You are a {panelist.get('age_group', '25-34')} year old person "
            f"from the {panelist.get('region', 'suburban')} United States. "
            f"Your household has {panelist.get('household_size', '2')} members"
        )
        if panelist.get("has_children"):
            demographics += " including children under 18"
        demographics += (
            f". Your education level is {panelist.get('education_level', 'bachelors')} "
            f"and your household income is {panelist.get('income_group', 'middle')}."
        )

        system = f"""You are a synthetic survey respondent — a digital twin of a real consumer.
Answer the survey question AS THIS PERSON, based on the profile below.
[persona template {self.template_version}]

DEMOGRAPHIC PROFILE:
{demographics}

BEHAVIORAL PROFILE:
{behavioral}

RESPONSE RULES:
1. Answer ONLY from the provided options (if options are given).
2. Stay consistent with your demographic and behavioral profile.
3. Be realistic — real consumers don't always choose the most "correct" answer.
4. Give brief reasoning (1-2 sentences) for why this person answers this way.
5. Rate your confidence from 0.0 to 1.0.

Respond ONLY in this JSON format:
{{"answer": "...", "reasoning": "...", "confidence": 0.X}}"""

        user = f"Survey Question: {question['text']}"
        if question.get("options"):
            option_lines = "\n".join(
                f"  {i + 1}. {opt}" for i, opt in enumerate(question["options"])
            )
            user += f"\n\nResponse Options:\n{option_lines}"
        if question.get("context"):
            user += f"\n\nContext: {question['context']}"
        if prior_responses:
            prior_lines = "\n".join(
                f"  Q{i + 1}: {answer}" for i, answer in enumerate(prior_responses)
            )
            user += (
                f"\n\nYour previous answers in this survey:\n{prior_lines}"
                "\n\nEnsure your answer is logically consistent with your "
                "previous responses."
            )
        return system, user


class ResponseParser:
    """Parses LLM output through a fallback chain, never raising.

    Single responsibility: raw model text -> {answer, reasoning,
    confidence, parse_method}. Chain (Chain-of-Responsibility, ordered by
    trust): strict JSON -> regex field extraction -> option scan ->
    raw-text fallback. The successful stage is recorded per response so
    per-model output quality is observable.

    Example:
        >>> parsed = ResponseParser().parse(llm_text, question)
        >>> parsed["parse_method"]
        'json'
    """

    _FIELD_PATTERN = re.compile(
        r'"answer"\s*:\s*"(?P<answer>[^"]*)"', re.IGNORECASE
    )
    _CONFIDENCE_PATTERN = re.compile(
        r'"confidence"\s*:\s*(?P<confidence>[0-9.]+)', re.IGNORECASE
    )
    _REASONING_PATTERN = re.compile(
        r'"reasoning"\s*:\s*"(?P<reasoning>[^"]*)"', re.IGNORECASE
    )
    _FALLBACK_CONFIDENCE = 0.3  # low trust marker for non-JSON parses
    _MAX_RAW_ANSWER_CHARS = 500

    def parse(self, content: str, question: dict[str, Any]) -> dict[str, Any]:
        """Parse one model response.

        Args:
            content: Raw LLM output text.
            question: The question (options guide the option-scan stage).

        Returns:
            Dict with answer, reasoning, confidence, parse_method. The
            answer may be empty only when the model returned nothing at
            all — the Validator then rejects it downstream.
        """
        cleaned = self._strip_fences(content)

        parsed = self._try_json(cleaned)
        if parsed is not None:
            return {**parsed, "parse_method": "json"}

        parsed = self._try_regex(cleaned)
        if parsed is not None:
            return {**parsed, "parse_method": "regex"}

        parsed = self._try_option_scan(cleaned, question.get("options") or [])
        if parsed is not None:
            return {**parsed, "parse_method": "option_scan"}

        return {
            "answer": cleaned.strip()[: self._MAX_RAW_ANSWER_CHARS],
            "reasoning": "Parsing fallback — raw model output used",
            "confidence": self._FALLBACK_CONFIDENCE,
            "parse_method": "raw",
        }

    @staticmethod
    def _strip_fences(content: str) -> str:
        """Remove surrounding markdown code fences, if any."""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _try_json(self, cleaned: str) -> dict[str, Any] | None:
        """Stage 1: strict JSON object."""
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or not str(data.get("answer", "")).strip():
            return None
        return {
            "answer": str(data.get("answer", "")),
            "reasoning": str(data.get("reasoning", "")),
            "confidence": self._clamp_confidence(data.get("confidence")),
        }

    def _try_regex(self, cleaned: str) -> dict[str, Any] | None:
        """Stage 2: extract JSON-ish fields from malformed output."""
        answer_match = self._FIELD_PATTERN.search(cleaned)
        if answer_match is None or not answer_match.group("answer").strip():
            return None
        reasoning_match = self._REASONING_PATTERN.search(cleaned)
        confidence_match = self._CONFIDENCE_PATTERN.search(cleaned)
        return {
            "answer": answer_match.group("answer"),
            "reasoning": reasoning_match.group("reasoning") if reasoning_match else "",
            "confidence": self._clamp_confidence(
                confidence_match.group("confidence") if confidence_match else None
            ),
        }

    def _try_option_scan(
        self, cleaned: str, options: list[str]
    ) -> dict[str, Any] | None:
        """Stage 3: find exactly one known option mentioned in the text."""
        lowered = cleaned.lower()
        mentioned = [opt for opt in options if opt.lower() in lowered]
        if len(mentioned) != 1:
            return None  # zero or ambiguous mentions -> next stage
        return {
            "answer": mentioned[0],
            "reasoning": "Option identified by scan of unstructured output",
            "confidence": self._FALLBACK_CONFIDENCE,
        }

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        """Coerce a confidence value into [0, 1] with a neutral default."""
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.5


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
            from insightpulse.llm.router import LLMRouter

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
