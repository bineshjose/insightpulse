"""Response safety filtering for synthetic survey answers.

Synthetic panelist responses must never leak personally identifiable
information (a twin is conditioned on a real panelist's data), harmful
content, or memorized training data. :class:`ResponseGuard` screens every
generated answer before it is persisted or returned by the API.

Detection is deliberately conservative: honest negative product opinions
("I hate this product") and numeric answers ("9" on an NPS scale) must
pass untouched. All redaction toggles come from ``get_settings().security``.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel

from insightpulse.config.settings import SecurityConfig, get_settings
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


class PIIType(StrEnum):
    """Categories of personally identifiable information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    NAME = "name"


class PIIMatch(BaseModel):
    """One detected PII span inside a response.

    Attributes:
        pii_type: The category of PII detected.
        value: The matched text (kept in memory only — never logged).
        start: Start offset of the span in the source text.
        end: End offset (exclusive) of the span in the source text.
        redacted_text: The replacement label for this span.
    """

    pii_type: PIIType
    value: str
    start: int
    end: int
    redacted_text: str


# Replacement labels per PII category.
REDACTION_LABELS: dict[PIIType, str] = {
    PIIType.EMAIL: "[EMAIL REDACTED]",
    PIIType.PHONE: "[PHONE REDACTED]",
    PIIType.SSN: "[SSN REDACTED]",
    PIIType.CREDIT_CARD: "[CARD REDACTED]",
    PIIType.ADDRESS: "[ADDRESS REDACTED]",
    PIIType.NAME: "[NAME REDACTED]",
}

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Phones require 10+ digits WITH separators (or an international +prefix)
# so plain integers, years, and scale answers ("9") never match.
_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b"
    r"|\+\d{10,14}\b"
)

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Card candidates: 13-19 digits, optionally space/hyphen separated, not
# embedded in a longer digit run. Every candidate must also pass Luhn.
_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")

_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z]+\s+){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
    r"Court|Ct|Way|Place|Pl|Terrace|Ter|Circle|Cir)\b\.?",
)

# Names only via explicit self-identification or signature lines — never
# bare capitalized words (survey answers mention brand names constantly).
_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i:\bmy\s+name\s+is\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"),
    re.compile(r"(?i:\b(?:regards|sincerely|thanks|best),?\s*\n\s*)([A-Z][a-z]+\s+[A-Z][a-z]+)"),
)

# Harmful-content heuristics. Conservative by design: they require a
# violent/dehumanizing verb or claim TARGETED AT A GROUP OF PEOPLE, so
# negative product sentiment never matches.
_HARMFUL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:kill|hurt|attack|eliminate)\s+(?:all\s+|those\s+)?"
        r"(?:people|women|men|foreigners|immigrants|minorit\w+)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdeserve(?:s)?\s+to\s+die\b", re.IGNORECASE),
    re.compile(r"\bgo\s+back\s+to\s+(?:your|their)\s+country\b", re.IGNORECASE),
    re.compile(
        r"\b(?:all|those)\s+(?:women|men|foreigners|immigrants|minorit\w+)\s+"
        r"are\s+(?:stupid|criminals?|inferior|subhuman)\b",
        re.IGNORECASE,
    ),
)

# Data-leakage heuristics: assistant boilerplate, training references,
# long verbatim URLs, and copyright lines betray memorized model output
# rather than a persona-grounded survey answer.
_LEAKAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bas an ai(?: language)? model\b", re.IGNORECASE),
    re.compile(r"\bi (?:was|am|have been) trained (?:by|on)\b", re.IGNORECASE),
    re.compile(r"https?://\S{40,}"),
    re.compile(r"(?:©|\(c\)|copyright)\s*\d{4}", re.IGNORECASE),
    re.compile(r"\ball rights reserved\b", re.IGNORECASE),
)

# A passage repeated verbatim at this length suggests regurgitation.
VERBATIM_BLOCK_CHARS = 200
_VERBATIM_SCAN_STRIDE = 50


def _luhn_valid(digits: str) -> bool:
    """Check a digit string with the Luhn algorithm (ISO/IEC 7812).

    Args:
        digits: The candidate card number, digits only.

    Returns:
        True when the checksum holds.
    """
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class ResponseGuard:
    """Screens generated survey responses for PII, harm, and leakage.

    Sits between the L3 generation engine and persistence: every synthetic
    answer is scanned, PII spans are redacted, and harmful or memorized
    content is flagged for the Validator agent to reject.

    Example:
        >>> guard = ResponseGuard()
        >>> guard.sanitize_response("Email me at a@b.com")
        'Email me at [EMAIL REDACTED]'
    """

    def __init__(self, config: SecurityConfig | None = None) -> None:
        """Bind the security configuration.

        Args:
            config: Security settings; defaults to the application config.
        """
        self._config = config or get_settings().security

    def detect_pii(self, text: str) -> list[PIIMatch]:
        """Find all PII spans in a response.

        Args:
            text: The generated response text.

        Returns:
            Non-overlapping matches sorted by start offset. When spans
            overlap, the earlier-detected (more specific) type wins:
            email, SSN, card, phone, address, name.
        """
        matches: list[PIIMatch] = []
        matches += self._regex_matches(text, _EMAIL_RE, PIIType.EMAIL)
        matches += self._regex_matches(text, _SSN_RE, PIIType.SSN)
        matches += self._card_matches(text)
        matches += self._regex_matches(text, _PHONE_RE, PIIType.PHONE)
        matches += self._regex_matches(text, _ADDRESS_RE, PIIType.ADDRESS)
        matches += self._name_matches(text)

        selected: list[PIIMatch] = []
        for match in sorted(matches, key=lambda m: (m.start, -(m.end - m.start))):
            if all(match.start >= s.end or match.end <= s.start for s in selected):
                selected.append(match)
                # Position and type only — never the value itself.
                logger.warning(
                    "pii_detected",
                    pii_type=str(match.pii_type),
                    start=match.start,
                    end=match.end,
                )
        return sorted(selected, key=lambda m: m.start)

    def detect_harmful_content(self, text: str) -> bool:
        """Flag toxic or discriminatory content in a survey answer.

        Deliberately conservative: ordinary negative product opinions
        ("I hate this product") never match — only violence or
        dehumanization targeted at groups of people does.

        Args:
            text: The generated response text.

        Returns:
            True when a harmful pattern matches (and detection is enabled).
        """
        if not self._config.harmful_content_detection_enabled:
            return False
        for pattern in _HARMFUL_PATTERNS:
            if pattern.search(text):
                logger.warning("harmful_content_detected", pattern=pattern.pattern[:60])
                return True
        return False

    def detect_data_leakage(self, text: str) -> bool:
        """Flag responses that look like memorized training data.

        Heuristics: assistant boilerplate ("As an AI language model"),
        training references, long verbatim URLs, copyright lines, and any
        passage of :data:`VERBATIM_BLOCK_CHARS` characters repeated
        verbatim within the response.

        Args:
            text: The generated response text.

        Returns:
            True when a leakage heuristic fires.
        """
        for pattern in _LEAKAGE_PATTERNS:
            if pattern.search(text):
                logger.warning("data_leakage_detected", pattern=pattern.pattern[:60])
                return True
        if self._has_repeated_block(text):
            logger.warning("data_leakage_detected", pattern="repeated_verbatim_block")
            return True
        return False

    def sanitize_response(self, text: str) -> str:
        """Redact all detected PII spans from a response.

        Args:
            text: The generated response text.

        Returns:
            The text with each PII span replaced by its redaction label
            (e.g. ``[EMAIL REDACTED]``). Returns the input unchanged when
            ``pii_redaction_enabled`` is False.
        """
        if not self._config.pii_redaction_enabled:
            return text
        result = text
        for match in sorted(self.detect_pii(text), key=lambda m: m.start, reverse=True):
            result = result[: match.start] + match.redacted_text + result[match.end:]
        return result

    @staticmethod
    def _regex_matches(text: str, pattern: re.Pattern[str], pii_type: PIIType) -> list[PIIMatch]:
        """Collect matches of one PII regex as :class:`PIIMatch` records."""
        return [
            PIIMatch(
                pii_type=pii_type,
                value=m.group(0),
                start=m.start(),
                end=m.end(),
                redacted_text=REDACTION_LABELS[pii_type],
            )
            for m in pattern.finditer(text)
        ]

    @staticmethod
    def _card_matches(text: str) -> list[PIIMatch]:
        """Collect card-number candidates that pass the Luhn check."""
        matches: list[PIIMatch] = []
        for m in _CARD_RE.finditer(text):
            digits = re.sub(r"[ -]", "", m.group(0))
            if 13 <= len(digits) <= 19 and _luhn_valid(digits):
                matches.append(
                    PIIMatch(
                        pii_type=PIIType.CREDIT_CARD,
                        value=m.group(0),
                        start=m.start(),
                        end=m.end(),
                        redacted_text=REDACTION_LABELS[PIIType.CREDIT_CARD],
                    )
                )
        return matches

    @staticmethod
    def _name_matches(text: str) -> list[PIIMatch]:
        """Collect names from self-identification and signature phrases."""
        matches: list[PIIMatch] = []
        for pattern in _NAME_PATTERNS:
            matches += [
                PIIMatch(
                    pii_type=PIIType.NAME,
                    value=m.group(1),
                    start=m.start(1),
                    end=m.end(1),
                    redacted_text=REDACTION_LABELS[PIIType.NAME],
                )
                for m in pattern.finditer(text)
            ]
        return matches

    @staticmethod
    def _has_repeated_block(text: str) -> bool:
        """True when a long block of text appears verbatim more than once."""
        if len(text) < 2 * VERBATIM_BLOCK_CHARS:
            return False
        for offset in range(0, len(text) - VERBATIM_BLOCK_CHARS, _VERBATIM_SCAN_STRIDE):
            block = text[offset: offset + VERBATIM_BLOCK_CHARS]
            if text.count(block) > 1:
                return True
        return False
