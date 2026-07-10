"""TwinOrchestrator agent — generates synthetic survey responses.

The core generative agent in the pipeline. For each selected panelist,
constructs a persona prompt from their demographic profile and behavioral
cluster, then generates a survey response via the LLM.

Responsibilities:
    - Build persona prompts from conditioning vectors u_i = [z_i; d_i]
    - Generate responses via the LLM router (multi-model support)
    - Handle sequential question dependency (evaluator feedback #4)
    - Track generation metadata (tokens, cost, latency)
    - Support batch and parallel generation

Key design decision: each digital twin is a PROMPT, not a fine-tuned model.
The LoRA fine-tuning approach from the thesis is represented here as
carefully crafted persona prompts with behavioral context. For production
with NIQ data, the prompt can be augmented with actual purchase summaries
from the behavioral embedding layer.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from insightpulse.config.settings import get_settings
from insightpulse.llm.router import LLMRouter

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Cluster-specific behavioral descriptions
# ---------------------------------------------------------------------------
# These correspond to the 5 behavioral archetypes from K-Means clustering
# (thesis Results-1, K=5). In production, these would be generated from
# the actual behavioral embeddings via the embedding layer.

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


def _build_persona_prompt(
    panelist: dict[str, Any],
    question: dict[str, Any],
    prior_responses: list[str] | None = None,
) -> tuple[str, str]:
    """Build the system and user prompts for a digital twin response.

    Constructs a detailed persona prompt that conditions the LLM to
    respond as the described consumer. The prompt incorporates:
    - Demographic attributes (from D_i)
    - Behavioral archetype (from B_i via cluster assignment)
    - Sequential context (prior answers, if applicable)

    This addresses evaluator feedback #4: sequential question dependency
    is handled by including prior_responses in the prompt context.

    Args:
        panelist: Panelist dictionary with demographics and cluster.
        question: Structured question dictionary from SurveyDesigner.
        prior_responses: Responses to prior questions in the survey.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    cluster_id = panelist.get("cluster_id", 0)
    behavioral_desc = CLUSTER_PERSONAS.get(cluster_id, CLUSTER_PERSONAS[0])

    # Build demographic description
    demo_desc = (
        f"You are a {panelist.get('age_group', '25-34')} year old person "
        f"from the {panelist.get('region', 'suburban')} United States. "
        f"Your household has {panelist.get('household_size', '2')} members"
    )
    if panelist.get("has_children"):
        demo_desc += " including children under 18"
    demo_desc += (
        f". Your education level is {panelist.get('education_level', 'bachelors')} "
        f"and your household income is {panelist.get('income_group', 'middle')}."
    )

    system_prompt = f"""You are a synthetic survey respondent — a digital twin of a real consumer.
Answer the survey question AS THIS PERSON, based on the demographic and behavioral profile below.

DEMOGRAPHIC PROFILE:
{demo_desc}

BEHAVIORAL PROFILE:
{behavioral_desc}

RESPONSE RULES:
1. Answer ONLY from the provided options (if options are given).
2. Your answer should be consistent with your demographic and behavioral profile.
3. Be realistic — real consumers don't always choose the most "correct" answer.
4. Provide brief reasoning (1-2 sentences) explaining why this person would choose this answer.
5. Rate your confidence from 0.0 to 1.0.

Respond ONLY in this JSON format:
{{"answer": "...", "reasoning": "...", "confidence": 0.X}}"""

    # Build user prompt with question and options
    user_prompt = f"Survey Question: {question['text']}"

    if question.get("options"):
        options_text = "\n".join(
            f"  {i + 1}. {opt}" for i, opt in enumerate(question["options"])
        )
        user_prompt += f"\n\nResponse Options:\n{options_text}"

    if question.get("context"):
        user_prompt += f"\n\nContext: {question['context']}"

    # Sequential dependency: include prior responses
    if prior_responses:
        prior_text = "\n".join(
            f"  Q{i + 1}: {resp}" for i, resp in enumerate(prior_responses)
        )
        user_prompt += (
            f"\n\nYour previous answers in this survey:\n{prior_text}"
            "\n\nEnsure your answer is logically consistent with your previous responses."
        )

    return system_prompt, user_prompt


async def twin_orchestrator_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: generate synthetic survey responses.

    For each selected panelist × question combination, constructs a
    persona prompt and generates a response via the LLM router.
    Supports parallel generation for efficiency.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with raw responses and generation metadata.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    panelist_ids = state.get("selected_panelist_ids", [])
    questions = state.get("parsed_questions", [])
    models = state.get("requested_models", []) or [settings.default_llm_model]

    if not panelist_ids or not questions:
        logger.warning("twin_orchestrator_no_input")
        return {
            "raw_responses": [],
            "agent_trace": [_trace_entry("No panelists or questions", 0)],
        }

    logger.info(
        "twin_orchestrator_start",
        num_panelists=len(panelist_ids),
        num_questions=len(questions),
        models=models,
    )

    # Load panelist data for persona construction
    panelists = await _load_panelists_by_ids(panelist_ids)
    router = LLMRouter()

    # Generate responses for each panelist × question
    all_responses: list[dict[str, Any]] = []
    model = models[0]  # Primary model for generation

    # Process questions sequentially to maintain dependencies
    for q_idx, question in enumerate(questions):
        question_responses: list[dict[str, Any]] = []

        # Batch panelists for parallel generation
        tasks = []
        for panelist in panelists:
            # Collect prior responses for this panelist (sequential dependency)
            prior_responses = [
                r["answer"]
                for r in all_responses
                if r["panelist_id"] == panelist["panelist_id"]
            ]

            tasks.append(
                _generate_single_response(
                    router=router,
                    panelist=panelist,
                    question=question,
                    model=model,
                    prior_responses=prior_responses if q_idx > 0 else None,
                )
            )

        # Execute in batches to avoid overwhelming the API
        batch_size = 10
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start : batch_start + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error("response_generation_failed", error=str(result))
                    continue
                if result:
                    question_responses.append(result)

        all_responses.extend(question_responses)

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Compute generation metadata
    generation_metadata = {
        "total_responses": len(all_responses),
        "total_tokens": router.total_tokens,
        "total_cost_usd": router.total_cost,
        "avg_latency_ms": (
            duration_ms / len(all_responses) if all_responses else 0
        ),
        "model_used": model,
    }

    logger.info(
        "twin_orchestrator_complete",
        responses_generated=len(all_responses),
        total_cost=f"${router.total_cost:.4f}",
        duration_ms=f"{duration_ms:.1f}",
    )

    return {
        "raw_responses": all_responses,
        "generation_metadata": generation_metadata,
        "total_cost_usd": state.get("total_cost_usd", 0) + router.total_cost,
        "total_tokens": state.get("total_tokens", 0) + router.total_tokens,
        "agent_trace": [_trace_entry(
            f"Generated {len(all_responses)} responses using {model}",
            duration_ms,
            metadata=generation_metadata,
        )],
    }


async def _generate_single_response(
    router: LLMRouter,
    panelist: dict[str, Any],
    question: dict[str, Any],
    model: str,
    prior_responses: list[str] | None = None,
) -> dict[str, Any] | None:
    """Generate a single survey response for one panelist-question pair.

    Args:
        router: LLM router instance.
        panelist: Panelist data dictionary.
        question: Structured question dictionary.
        model: LLM model identifier.
        prior_responses: Prior answers for sequential consistency.

    Returns:
        Response dictionary or None if generation fails.
    """
    import json

    system_prompt, user_prompt = _build_persona_prompt(
        panelist, question, prior_responses
    )

    try:
        result = await router.generate(
            prompt=user_prompt,
            system=system_prompt,
            model=model,
            temperature=0.7,
        )

        # Parse the structured JSON response
        parsed = _parse_response(result.content, question)

        return {
            "response_id": f"{panelist['panelist_id']}_{question['question_id']}",
            "question_id": question["question_id"],
            "panelist_id": panelist["panelist_id"],
            "answer": parsed.get("answer", ""),
            "reasoning": parsed.get("reasoning", ""),
            "confidence": parsed.get("confidence", 0.5),
            "model_used": model,
            "generation_time_ms": result.latency_ms,
            "token_count": result.total_tokens,
            "cost_usd": result.cost_usd,
            "behavioral_cluster": panelist.get("cluster_id", -1),
            "demographic_summary": (
                f"{panelist.get('age_group', '')} | "
                f"{panelist.get('income_group', '')} | "
                f"{panelist.get('region', '')}"
            ),
        }

    except Exception as e:
        logger.error(
            "single_response_failed",
            panelist_id=panelist["panelist_id"],
            question_id=question["question_id"],
            error=str(e),
        )
        return None


def _parse_response(content: str, question: dict[str, Any]) -> dict[str, Any]:
    """Parse the LLM's JSON response, with fallback for malformed output.

    Args:
        content: Raw LLM output text.
        question: Question dict for context.

    Returns:
        Dictionary with answer, reasoning, and confidence.
    """
    import json

    # Clean markdown fences
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        return {
            "answer": str(parsed.get("answer", "")),
            "reasoning": str(parsed.get("reasoning", "")),
            "confidence": min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
        }
    except (json.JSONDecodeError, ValueError):
        # Fallback: use the raw content as the answer
        return {
            "answer": content.strip()[:500],
            "reasoning": "Response parsing fallback — raw output used",
            "confidence": 0.3,
        }


async def _load_panelists_by_ids(panelist_ids: list[str]) -> list[dict[str, Any]]:
    """Load panelist data for the selected cohort.

    In demo mode, regenerates from the minimal sample.
    In production, queries the database.

    Args:
        panelist_ids: List of panelist identifiers to load.

    Returns:
        List of panelist dictionaries.
    """
    from insightpulse.agents.cohort_selector import _load_synthetic_panelists

    # Load all and filter — in production this would be a targeted DB query
    all_panelists = _load_synthetic_panelists()
    id_set = set(panelist_ids)
    return [p for p in all_panelists if p.get("panelist_id") in id_set]


def _trace_entry(
    summary: str,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an agent trace entry for audit logging."""
    return {
        "agent_name": "TwinOrchestrator",
        "action": "generate_responses",
        "output_summary": summary,
        "duration_ms": duration_ms,
        "metadata": metadata or {},
    }
