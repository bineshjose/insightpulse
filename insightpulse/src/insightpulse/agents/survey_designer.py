"""SurveyDesigner agent — interprets and structures survey questions.

The first agent in the DAG. Takes raw question text from the user and
produces structured SurveyQuestion objects with inferred question types,
response options, and domain context.

Responsibilities:
    - Parse raw question text into structured format
    - Infer question type (Likert, single-choice, open-ended, etc.)
    - Generate appropriate response options when not provided
    - Attach domain context for downstream agents
    - Handle sequential question dependencies
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from insightpulse.core.models.survey import QuestionType
from insightpulse.ml.llm.router import LLMRouter

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the SurveyDesigner LLM call
# ---------------------------------------------------------------------------

SURVEY_DESIGNER_SYSTEM_PROMPT = """You are a survey science expert at NielsenIQ.
Your task is to take a raw survey question and produce a structured survey specification.

For each question, determine:
1. question_type: one of [single_choice, multiple_choice, likert_5, likert_7,
   open_ended, ranking, net_promoter]
2. options: appropriate response options (for choice/likert/ranking types)
3. category: the survey category (brand_perception, purchase_intent,
   product_satisfaction, lifestyle, media_consumption, price_sensitivity, general)

IMPORTANT RULES:
- For Likert-5 scales, always use:
  ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
- For Likert-7 scales, use the 7-point variant with "Somewhat" options
- For NPS, options are "0" through "10"
- For single_choice, generate 4-6 mutually exclusive, exhaustive options
- Always include an "Other" or "Not applicable" option where appropriate

Respond ONLY with a JSON object (no markdown, no explanation):
{
    "question_type": "...",
    "options": ["..."],
    "category": "...",
    "requires_context": true/false
}"""


async def survey_designer_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: parse and structure raw survey questions.

    Reads `raw_questions` from state and writes `parsed_questions`
    with fully structured question specifications.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with parsed questions.
    """
    start_time = time.perf_counter()
    raw_questions = state.get("raw_questions", [])
    context = state.get("question_context", "")

    if not raw_questions:
        logger.warning("survey_designer_no_questions")
        return {
            "parsed_questions": [],
            "agent_trace": [_trace_entry("No questions provided", 0)],
        }

    logger.info("survey_designer_start", num_questions=len(raw_questions))

    router = LLMRouter()
    parsed_questions = []

    for i, question_text in enumerate(raw_questions):
        prompt = f"Survey question: \"{question_text}\""
        if context:
            prompt += f"\nSurvey context: {context}"

        try:
            result = await router.generate(
                prompt=prompt,
                system=SURVEY_DESIGNER_SYSTEM_PROMPT,
                temperature=0.1,  # Low temperature for structured output
                max_tokens=512,
            )

            # Parse the LLM's structured response
            spec = _parse_question_spec(result.content, question_text)

        except Exception as e:
            logger.error(
                "survey_designer_parse_failed",
                question=question_text,
                error=str(e),
            )
            # Fallback: treat as single-choice with default options
            spec = _default_question_spec(question_text)

        # Build the full question object
        question_dict = {
            "question_id": f"q_{i + 1}",
            "text": question_text,
            "question_type": spec["question_type"],
            "options": spec["options"],
            "category": spec.get("category", "general"),
            "context": context,
            "is_sequential": i > 0 and len(raw_questions) > 1,
            "prior_questions": [q["text"] for q in parsed_questions],
        }
        parsed_questions.append(question_dict)

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "survey_designer_complete",
        num_parsed=len(parsed_questions),
        duration_ms=f"{duration_ms:.1f}",
    )

    return {
        "parsed_questions": parsed_questions,
        "agent_trace": [_trace_entry(
            f"Parsed {len(parsed_questions)} questions",
            duration_ms,
            metadata={"question_types": [q["question_type"] for q in parsed_questions]},
        )],
    }


def _parse_question_spec(llm_output: str, question_text: str) -> dict[str, Any]:
    """Parse the LLM's structured output into a question specification.

    Args:
        llm_output: Raw text from the LLM (expected JSON).
        question_text: Original question for fallback context.

    Returns:
        Dictionary with question_type, options, and category.
    """
    # Strip markdown code fences if present
    cleaned = llm_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        spec = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("question_spec_json_parse_failed", output=llm_output[:200])
        return _default_question_spec(question_text)

    # Validate question_type
    valid_types = {qt.value for qt in QuestionType}
    if spec.get("question_type") not in valid_types:
        spec["question_type"] = "single_choice"

    # Ensure options is a list
    if not isinstance(spec.get("options"), list):
        spec["options"] = []

    return spec


def _default_question_spec(question_text: str) -> dict[str, Any]:
    """Generate a sensible default specification when parsing fails.

    Uses keyword heuristics to guess the question type.

    Args:
        question_text: The original question text.

    Returns:
        Default question specification dictionary.
    """
    text_lower = question_text.lower()

    # Heuristic: detect common patterns
    if any(word in text_lower for word in ["how likely", "recommend", "nps"]):
        return {
            "question_type": "net_promoter",
            "options": [str(i) for i in range(11)],
            "category": "general",
        }
    elif any(word in text_lower for word in ["agree", "disagree", "important", "satisfied"]):
        return {
            "question_type": "likert_5",
            "options": [
                "Strongly disagree",
                "Disagree",
                "Neutral",
                "Agree",
                "Strongly agree",
            ],
            "category": "general",
        }
    elif any(word in text_lower for word in ["rank", "order", "prioritize"]):
        return {
            "question_type": "ranking",
            "options": [],
            "category": "general",
        }
    else:
        return {
            "question_type": "single_choice",
            "options": [
                "Very likely",
                "Somewhat likely",
                "Neutral",
                "Somewhat unlikely",
                "Very unlikely",
            ],
            "category": "general",
        }


def _trace_entry(
    summary: str,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an agent trace entry for audit logging."""
    return {
        "agent_name": "SurveyDesigner",
        "action": "parse_questions",
        "output_summary": summary,
        "duration_ms": duration_ms,
        "metadata": metadata or {},
    }
