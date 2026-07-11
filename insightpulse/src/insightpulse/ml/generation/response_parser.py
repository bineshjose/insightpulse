"""L3 response parsing — JSON → regex → option-scan → raw fallback."""


from __future__ import annotations

import json
import re
from typing import Any

from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)






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
