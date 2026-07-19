"""Red-teaming agent — adversarial validation of generated responses.

Screens every response the Validator passed against a library of known
failure patterns (thesis §4.5.15). Responses it flags are rejected and
regenerated regardless of the primary Validator's verdict — the two
checks are independent lines of defence.

Adversarial categories:
    1. Demographic stereotyping — responses that justify an answer by a
       demographic identity rather than the behavioural profile.
    2. Brand hallucination — plausible portmanteaus: invented brand-like
       names composed from morphemes common in FMCG naming, which slip
       past the Validator's verbatim checks.
    3. Temporal inconsistency — references to events outside the
       behavioural observation window (future launches, "next year's").
    4. Prompt leakage — fragments of the persona instruction surfacing
       (verbatim or paraphrased) in the response text.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

import structlog

from insightpulse.observability.metrics import get_metrics_collector

logger = structlog.get_logger(__name__)

# Regeneration triggers when more than this share of screened responses
# is flagged; below it, flagged responses are dropped without a retry
# cycle (heuristic screens must not be able to loop the pipeline).
_REGENERATION_THRESHOLD = 0.05

# --- Category 1: demographic stereotyping -----------------------------------
# Answers justified by demographic identity instead of behaviour.
_STEREOTYPE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bas a (?:woman|man|female|male|senior|elderly person|young person)\b",
        r"\bpeople (?:of my|from my) (?:age|gender|background|ethnicity)\b",
        r"\b(?:women|men|seniors|young people|poor people|rich people) "
        r"(?:always|never|all|typically) (?:buy|prefer|choose|want)\b",
        r"\btypical for (?:my|our) (?:age|gender|income|demographic)\b",
        r"\bbecause of my (?:gender|ethnicity|race|age group)\b",
    )
]

# --- Category 2: brand hallucination (plausible portmanteaus) ---------------
# Morphemes common in FMCG brand naming; an intercapped compound built
# from two of them (e.g. "CrispJoy") reads like a brand but isn't one.
_BRAND_MORPHEMES = (
    "crisp", "snack", "fresh", "gold", "joy", "pure", "sun", "farm",
    "nutri", "vita", "glow", "wave", "bite", "brew", "dairy", "green",
    "silk", "spark", "cream", "oven", "polar", "frost", "quick", "meadow",
)
_INTERCAP_TOKEN = re.compile(r"\b([A-Z][a-z]+){2,}\b")

# --- Category 3: temporal inconsistency -------------------------------------
_TEMPORAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bnext year'?s?\b",
        r"\bupcoming (?:launch|release|product)\b",
        r"\bwill be (?:launched|released) (?:next|later this)\b",
        r"\bpre-?order(?:ed|ing)?\b",
        r"\bwhen it (?:comes out|launches|releases)\b",
    )
]

# --- Category 4: prompt leakage ---------------------------------------------
# Verbatim and paraphrased fragments of persona instructions.
_LEAKAGE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\byou are a\b",
        r"\brespond as\b",
        r"\bmy persona\b",
        r"\bas instructed\b",
        r"\bthe (?:persona|instruction|prompt) (?:says|describes|tells)\b",
        r"\bi (?:was|am) (?:told|asked|instructed) to (?:answer|respond|act)\b",
        r"\bsurvey instrument\b",
        r"\bconditioning (?:context|profile|vector)\b",
        r"\bbehavioural embedding\b",
    )
]

FLAG_PREFIX = "red_team"


async def red_team_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: adversarially screen validated responses.

    Reads ``validated_responses``; writes the screened
    ``validated_responses``, ``red_team_rejected``, and (when the flag
    rate exceeds the regeneration threshold) ``needs_regeneration``.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with the screened responses and rejection summary.
    """
    start_time = time.perf_counter()
    responses = state.get("validated_responses", [])

    if not responses:
        return {
            "red_team_rejected": [],
            "red_team_flag_summary": {},
            "agent_trace": [_trace_entry("No responses to screen", 0)],
        }

    logger.info("red_team_start", num_responses=len(responses))
    collector = get_metrics_collector()

    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for response in responses:
        flags = screen_response(response)
        collector.record_validation_check(
            "red_team_screen", "pass" if not flags else "fail"
        )
        if flags:
            response["validation_flags"] = (
                response.get("validation_flags", []) + flags
            )
            response["is_valid"] = False
            rejected.append(response)
        else:
            passed.append(response)

    flag_rate = len(rejected) / len(responses)
    needs_regen = flag_rate > _REGENERATION_THRESHOLD
    retry_count = state.get("validation_retry_count", 0)
    flag_summary = dict(Counter(
        flag for entry in rejected for flag in entry.get("validation_flags", [])
        if flag.startswith(FLAG_PREFIX)
    ))
    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "red_team_complete",
        passed=len(passed),
        rejected=len(rejected),
        flag_rate=f"{flag_rate:.2%}",
        needs_regeneration=needs_regen,
        flag_summary=flag_summary,
    )

    updates: dict[str, Any] = {
        "validated_responses": passed,
        "red_team_rejected": rejected,
        "red_team_flag_summary": flag_summary,
        "agent_trace": [_trace_entry(
            f"Screened {len(responses)} responses, rejected {len(rejected)} "
            f"({flag_rate:.1%}) across {len(flag_summary)} adversarial categories",
            duration_ms,
            metadata={
                "rejected_count": len(rejected),
                "flag_rate": flag_rate,
                "flag_summary": flag_summary,
            },
        )],
    }
    if needs_regen:
        updates["needs_regeneration"] = True
        updates["validation_retry_count"] = retry_count + 1
    return updates


def screen_response(response: dict[str, Any]) -> list[str]:
    """Screen one response against all four adversarial categories.

    Args:
        response: Response dict (answer, reasoning, demographic fields).

    Returns:
        ``red_team:*`` flags (empty when the response is clean).
    """
    text = f"{response.get('answer', '')} {response.get('reasoning', '')}"
    flags: list[str] = []

    if any(pattern.search(text) for pattern in _STEREOTYPE_PATTERNS):
        flags.append(f"{FLAG_PREFIX}:demographic_stereotyping")
    if _detect_brand_portmanteau(text):
        flags.append(f"{FLAG_PREFIX}:brand_hallucination")
    if any(pattern.search(text) for pattern in _TEMPORAL_PATTERNS):
        flags.append(f"{FLAG_PREFIX}:temporal_inconsistency")
    if any(pattern.search(text) for pattern in _LEAKAGE_PATTERNS):
        flags.append(f"{FLAG_PREFIX}:prompt_leakage")

    return flags


def _detect_brand_portmanteau(text: str) -> bool:
    """Detect invented brand-like compounds built from FMCG morphemes.

    An intercapped token (``CrispJoy``) whose camel-case parts are both
    common brand morphemes is treated as a fabricated brand reference.
    Real brands are either single words or don't decompose this way.

    Args:
        text: Combined answer + reasoning text.

    Returns:
        True when a plausible portmanteau is present.
    """
    for match in _INTERCAP_TOKEN.finditer(text):
        parts = re.findall(r"[A-Z][a-z]+", match.group(0))
        if len(parts) >= 2 and all(
            part.lower() in _BRAND_MORPHEMES for part in parts
        ):
            return True
    return False


def _trace_entry(
    summary: str,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an agent trace entry for audit logging."""
    return {
        "agent_name": "RedTeamAgent",
        "action": "adversarial_validation",
        "output_summary": summary,
        "duration_ms": duration_ms,
        "metadata": metadata or {},
    }
