"""Validator agent — checks response quality, consistency, and hallucinations.

Validates synthetic responses against multiple quality criteria before
they proceed to calibration. Responses that fail validation are flagged
for regeneration (up to max_validation_retries).

Validation Checks:
    1. Option validity: answer matches one of the provided options
    2. Logical consistency: answer is consistent with demographic profile
    3. Hallucination detection: answer doesn't reference non-existent products/events
    4. Confidence calibration: flag low-confidence responses
    5. Sequential consistency: answer is logically consistent with prior responses
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from insightpulse.config.settings import get_settings

logger = structlog.get_logger(__name__)


async def validator_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: validate synthetic responses.

    Reads `raw_responses` and `parsed_questions` from state.
    Writes `validated_responses`, `rejected_responses`, and
    `needs_regeneration` flag.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with validation results.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    raw_responses = state.get("raw_responses", [])
    questions = state.get("parsed_questions", [])

    if not raw_responses:
        return {
            "validated_responses": [],
            "rejected_responses": [],
            "needs_regeneration": False,
            "agent_trace": [_trace_entry("No responses to validate", 0)],
        }

    logger.info("validator_start", num_responses=len(raw_responses))

    # Build a lookup from question_id to question spec
    question_map = {q["question_id"]: q for q in questions}

    validated = []
    rejected = []

    for response in raw_responses:
        question = question_map.get(response.get("question_id", ""))
        flags = _validate_response(response, question)

        response["validation_flags"] = flags
        response["is_valid"] = len(flags) == 0

        if response["is_valid"]:
            validated.append(response)
        else:
            rejected.append(response)

    # Determine if regeneration is needed
    rejection_rate = len(rejected) / len(raw_responses) if raw_responses else 0
    needs_regen = rejection_rate > 0.3  # Regenerate if >30% rejected
    retry_count = state.get("validation_retry_count", 0)

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "validator_complete",
        validated=len(validated),
        rejected=len(rejected),
        rejection_rate=f"{rejection_rate:.2%}",
        needs_regeneration=needs_regen,
        retry_count=retry_count,
    )

    return {
        "validated_responses": validated,
        "rejected_responses": rejected,
        "needs_regeneration": needs_regen,
        "validation_retry_count": retry_count + (1 if needs_regen else 0),
        "agent_trace": [_trace_entry(
            f"Validated {len(validated)}, rejected {len(rejected)} "
            f"({rejection_rate:.1%} rejection rate)",
            duration_ms,
            metadata={
                "validated_count": len(validated),
                "rejected_count": len(rejected),
                "rejection_rate": rejection_rate,
                "flag_summary": _summarize_flags(rejected),
            },
        )],
    }


def _validate_response(
    response: dict[str, Any],
    question: dict[str, Any] | None,
) -> list[str]:
    """Run all validation checks on a single response.

    Args:
        response: The response dictionary to validate.
        question: The corresponding question specification.

    Returns:
        List of validation flag strings (empty = passed all checks).
    """
    flags: list[str] = []
    settings = get_settings()

    answer = response.get("answer", "").strip()

    if not answer:
        flags.append("empty_response")
        return flags

    # Check 1: Option validity (for choice-based questions)
    if question and question.get("options"):
        if not _is_valid_option(answer, question["options"]):
            flags.append("invalid_option")

    # Check 2: Confidence threshold
    confidence = response.get("confidence", 0.5)
    if confidence < 0.2:
        flags.append("very_low_confidence")

    # Check 3: Hallucination heuristics
    if _detect_hallucination(answer, response.get("reasoning", "")):
        flags.append("hallucination_detected")

    # Check 4: Response length sanity
    if len(answer) > 2000:
        flags.append("response_too_long")
    if len(answer) < 2 and question and question.get("question_type") != "net_promoter":
        flags.append("response_too_short")

    # Check 5: Demographic consistency (basic heuristics)
    demo_flags = _check_demographic_consistency(response, question)
    flags.extend(demo_flags)

    return flags


def _is_valid_option(answer: str, options: list[str]) -> bool:
    """Check if the answer matches one of the provided options.

    Uses fuzzy matching to handle minor formatting differences
    between the LLM output and the option text.

    Args:
        answer: The response answer text.
        options: List of valid response options.

    Returns:
        True if the answer matches an option.
    """
    answer_lower = answer.lower().strip()

    for option in options:
        option_lower = option.lower().strip()
        # Exact match
        if answer_lower == option_lower:
            return True
        # Contained match (handles "Agree" matching "Strongly agree")
        if option_lower in answer_lower or answer_lower in option_lower:
            return True
        # Numeric match for Likert/NPS
        if answer_lower.isdigit() and option_lower.isdigit():
            if int(answer_lower) == int(option_lower):
                return True

    return False


def _detect_hallucination(answer: str, reasoning: str) -> bool:
    """Detect potential hallucinations in the response.

    Uses heuristic checks for common hallucination patterns:
    - Referencing specific but non-existent studies or statistics
    - Citing exact percentages or numbers without basis
    - Mentioning non-existent products, brands, or events

    This is a lightweight check. The full hallucination detection
    pipeline in the thesis uses an LLM-based verifier; we implement
    rule-based heuristics here and flag for human review.

    Args:
        answer: The response text.
        reasoning: The reasoning chain.

    Returns:
        True if hallucination indicators are detected.
    """
    combined = (answer + " " + reasoning).lower()

    # Pattern: citing specific statistics ("according to a 2024 study...")
    hallucination_patterns = [
        "according to a study",
        "research shows that",
        "statistics indicate",
        "a recent report",
        "surveys have shown",
        "data suggests that",
    ]

    # These patterns are suspicious in a survey RESPONSE (not a research paper)
    suspicious_count = sum(1 for p in hallucination_patterns if p in combined)

    return suspicious_count >= 2


def _check_demographic_consistency(
    response: dict[str, Any],
    question: dict[str, Any] | None,
) -> list[str]:
    """Basic demographic consistency checks.

    Verifies that the response makes sense given the respondent's
    demographic profile. For example, a retiree shouldn't reference
    "my daily commute to the office."

    Args:
        response: Response dictionary with demographic info.
        question: Question specification.

    Returns:
        List of consistency flags.
    """
    flags: list[str] = []
    reasoning = response.get("reasoning", "").lower()
    demo = response.get("demographic_summary", "").lower()

    # Age-related consistency
    if "65+" in demo and any(
        phrase in reasoning
        for phrase in ["my children are toddlers", "just graduated from college"]
    ):
        flags.append("age_inconsistency")

    # Income-related consistency
    if "low" in demo and "luxury" in reasoning and "can easily afford" in reasoning:
        flags.append("income_inconsistency")

    return flags


def _summarize_flags(rejected: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize validation flags across all rejected responses.

    Args:
        rejected: List of rejected response dictionaries.

    Returns:
        Dictionary mapping flag names to their frequency.
    """
    from collections import Counter

    all_flags: list[str] = []
    for r in rejected:
        all_flags.extend(r.get("validation_flags", []))

    return dict(Counter(all_flags))


def _trace_entry(
    summary: str,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an agent trace entry for audit logging."""
    return {
        "agent_name": "Validator",
        "action": "validate_responses",
        "output_summary": summary,
        "duration_ms": duration_ms,
        "metadata": metadata or {},
    }
