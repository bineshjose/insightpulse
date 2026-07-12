"""API input validation helpers for the survey endpoints.

Framework-agnostic pure functions (plus the :class:`InputValidator`
Facade bundling them) that the API layer calls before any request data
reaches the pipeline. Each check returns a list of
:class:`ValidationIssue` records; :func:`to_422_detail` converts them to
the shape FastAPI uses for 422 responses, keeping this module importable
without FastAPI.

Bounds come from settings: question count/length from the root API
hardening fields (``api_max_questions``, ``api_max_question_length``) and
everything else from ``get_settings().security``.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, field_validator

from insightpulse.config.settings import Settings, get_settings
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Truncation applied to echoed-back input in issues (and logs).
RECEIVED_MAX_CHARS = 100

# Cohort bounds: a single panelist up to the full demo panel size.
MIN_COHORT_SIZE = 1
MAX_COHORT_SIZE = 5000

# Demographic dimensions the CohortSelector can filter on. Includes both
# the panel schema names (income_group, behavioral_archetype) and the NIQ
# API aliases (income_bracket) so either vocabulary validates.
KNOWN_DEMOGRAPHIC_DIMS: frozenset[str] = frozenset({
    "age_group",
    "income_group",
    "income_bracket",
    "region",
    "household_size",
    "education",
    "urbanicity",
    "behavioral_archetype",
})
MAX_FILTER_VALUE_LENGTH = 100

# Survey names: alphanumeric plus hyphen/space/period, starting alnum.
_SURVEY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .\-]*$")

# HTML/script vectors rejected outright in question text.
_HTML_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r"<\s*img\b", re.IGNORECASE),
    re.compile(r"<\s*iframe\b", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"\bon(?:error|load|click|mouseover)\s*=", re.IGNORECASE),
)

# SQL-injection vectors: statement chaining, UNION probes, and quote+comment
# combos. Anchored so ordinary prose ("drop", "union") never matches.
_SQL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r";\s*drop\s+(?:table|database)\b", re.IGNORECASE),
    re.compile(r"\bunion\s+(?:all\s+)?select\b", re.IGNORECASE),
    re.compile(r"'\s*--"),
    re.compile(r"'\s*(?:or|and)\s*'?\d*'?\s*=\s*'?\d*'?", re.IGNORECASE),
)


class ValidationIssue(BaseModel):
    """One rejected input field with the reason and expectation.

    Attributes:
        field: Dotted/indexed path of the offending field.
        message: Human-readable rejection reason.
        expected: What valid input looks like.
        received: The offending input, truncated to
            :data:`RECEIVED_MAX_CHARS` characters.
    """

    field: str
    message: str
    expected: str
    received: str

    @field_validator("received")
    @classmethod
    def _truncate_received(cls, v: str) -> str:
        """Never echo more than RECEIVED_MAX_CHARS of attacker input."""
        return v[:RECEIVED_MAX_CHARS]


def _issue(field: str, message: str, expected: str, received: object) -> ValidationIssue:
    """Build and log one validation issue.

    Args:
        field: The offending field path.
        message: Rejection reason.
        expected: Description of valid input.
        received: The raw offending value (stringified and truncated).

    Returns:
        The constructed issue.
    """
    issue = ValidationIssue(
        field=field, message=message, expected=expected, received=str(received)
    )
    logger.warning(
        "input_validation_failed",
        field=issue.field,
        message=issue.message,
        received=issue.received,
    )
    return issue


def to_422_detail(issues: list[ValidationIssue]) -> list[dict[str, Any]]:
    """Convert issues to FastAPI 422 ``detail`` entries.

    Args:
        issues: Validation issues to convert.

    Returns:
        Dicts shaped like FastAPI/Pydantic validation errors
        (``loc`` / ``msg`` / ``type`` / ``ctx``).
    """
    return [
        {
            "loc": ["body", issue.field],
            "msg": issue.message,
            "type": "value_error",
            "ctx": {"expected": issue.expected, "received": issue.received},
        }
        for issue in issues
    ]


def validate_questions(
    questions: list[str], settings: Settings | None = None
) -> list[ValidationIssue]:
    """Validate a batch of survey questions.

    Checks: batch size (``api_max_questions``), per-question length
    (``api_max_question_length``), non-empty text, HTML/script vectors,
    and SQL-injection vectors.

    Args:
        questions: The raw question strings.
        settings: Application settings; defaults to the singleton.

    Returns:
        All detected issues (empty when the batch is valid).
    """
    settings = settings or get_settings()
    issues: list[ValidationIssue] = []

    if len(questions) > settings.api_max_questions:
        issues.append(_issue(
            "questions",
            f"Too many questions: {len(questions)}",
            f"at most {settings.api_max_questions} questions per survey",
            len(questions),
        ))

    for i, question in enumerate(questions):
        field = f"questions[{i}]"
        if not question.strip():
            issues.append(_issue(field, "Question is empty", "non-empty text", question))
            continue
        if len(question) > settings.api_max_question_length:
            issues.append(_issue(
                field,
                f"Question too long: {len(question)} characters",
                f"at most {settings.api_max_question_length} characters",
                question,
            ))
        if any(p.search(question) for p in _HTML_PATTERNS):
            issues.append(_issue(
                field, "HTML/script content is not allowed", "plain text", question
            ))
        if any(p.search(question) for p in _SQL_PATTERNS):
            issues.append(_issue(
                field, "SQL injection pattern detected", "plain text", question
            ))
    return issues


def validate_cohort_size(n: object) -> list[ValidationIssue]:
    """Validate the requested cohort size.

    Args:
        n: The requested size (any type — non-integers are rejected).

    Returns:
        Issues when ``n`` is not an integer in
        [:data:`MIN_COHORT_SIZE`, :data:`MAX_COHORT_SIZE`].
    """
    expected = f"integer between {MIN_COHORT_SIZE} and {MAX_COHORT_SIZE}"
    if isinstance(n, bool) or not isinstance(n, int):
        return [_issue("cohort_size", "Cohort size must be an integer", expected, n)]
    if not MIN_COHORT_SIZE <= n <= MAX_COHORT_SIZE:
        return [_issue("cohort_size", f"Cohort size {n} out of range", expected, n)]
    return []


def validate_model_name(name: str, settings: Settings | None = None) -> list[ValidationIssue]:
    """Validate an LLM model name against the allowlist.

    Args:
        name: The requested model identifier.
        settings: Application settings; defaults to the singleton.

    Returns:
        Issues when the model is not in ``security.allowed_models``.
    """
    settings = settings or get_settings()
    allowed = settings.security.allowed_models
    if name not in allowed:
        return [_issue(
            "model", f"Model '{name[:RECEIVED_MAX_CHARS]}' is not allowed",
            f"one of {allowed}", name,
        )]
    return []


def validate_filters(filters: dict[str, Any]) -> list[ValidationIssue]:
    """Validate cohort demographic filters.

    Args:
        filters: Mapping of demographic dimension to requested value.

    Returns:
        Issues for unknown dimensions and empty/oversized/non-string
        values.
    """
    issues: list[ValidationIssue] = []
    for key, value in filters.items():
        field = f"filters.{key}"
        if key not in KNOWN_DEMOGRAPHIC_DIMS:
            issues.append(_issue(
                field,
                f"Unknown demographic dimension '{key[:RECEIVED_MAX_CHARS]}'",
                f"one of {sorted(KNOWN_DEMOGRAPHIC_DIMS)}",
                key,
            ))
            continue
        if not isinstance(value, str) or not value.strip():
            issues.append(_issue(field, "Filter value must be a non-empty string",
                                 "non-empty string", value))
        elif len(value) > MAX_FILTER_VALUE_LENGTH:
            issues.append(_issue(
                field,
                f"Filter value too long: {len(value)} characters",
                f"at most {MAX_FILTER_VALUE_LENGTH} characters",
                value,
            ))
    return issues


def validate_survey_name(name: str, settings: Settings | None = None) -> list[ValidationIssue]:
    """Validate a survey name.

    Args:
        name: The requested survey name.
        settings: Application settings; defaults to the singleton.

    Returns:
        Issues when the name is empty, exceeds
        ``security.max_survey_name_length``, or contains characters
        outside alphanumerics, hyphen, space, and period.
    """
    settings = settings or get_settings()
    max_length = settings.security.max_survey_name_length
    expected = f"1-{max_length} chars: letters, digits, hyphen, space, period"
    if not name or len(name) > max_length:
        return [_issue("survey_name", "Survey name length out of range", expected, name)]
    if not _SURVEY_NAME_RE.match(name):
        return [_issue("survey_name", "Survey name contains invalid characters",
                       expected, name)]
    return []


def validate_contract_id(cid: str, settings: Settings | None = None) -> list[ValidationIssue]:
    """Validate a client contract identifier.

    Args:
        cid: The contract ID.
        settings: Application settings; defaults to the singleton.

    Returns:
        Issues when the ID does not match
        ``security.contract_id_pattern``.
    """
    settings = settings or get_settings()
    pattern = settings.security.contract_id_pattern
    if not re.match(pattern, cid):
        return [_issue("contract_id", "Contract ID has invalid format",
                       f"pattern {pattern}", cid)]
    return []


def validate_client_name(name: str, settings: Settings | None = None) -> list[ValidationIssue]:
    """Validate a client name against the configured client registry.

    Args:
        name: The client name.
        settings: Application settings; defaults to the singleton.

    Returns:
        Issues when the client is not in ``security.known_clients``.
    """
    settings = settings or get_settings()
    known = settings.security.known_clients
    if name not in known:
        return [_issue("client_name",
                       f"Unknown client '{name[:RECEIVED_MAX_CHARS]}'",
                       f"one of {known}", name)]
    return []


class InputValidator:
    """Facade bundling every input check behind one object (Facade pattern).

    The API layer constructs one validator and calls its methods; each
    method delegates to the framework-agnostic module functions above so
    the checks stay individually testable and reusable.

    Example:
        >>> validator = InputValidator()
        >>> validator.cohort_size(100)
        []
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Bind the settings used for all bounds.

        Args:
            settings: Application settings; defaults to the singleton.
        """
        self._settings = settings or get_settings()

    def questions(self, questions: list[str]) -> list[ValidationIssue]:
        """Validate survey questions (see :func:`validate_questions`).

        Args:
            questions: The raw question strings.

        Returns:
            All detected issues.
        """
        return validate_questions(questions, self._settings)

    def cohort_size(self, n: object) -> list[ValidationIssue]:
        """Validate cohort size (see :func:`validate_cohort_size`).

        Args:
            n: The requested cohort size.

        Returns:
            All detected issues.
        """
        return validate_cohort_size(n)

    def model_name(self, name: str) -> list[ValidationIssue]:
        """Validate a model name (see :func:`validate_model_name`).

        Args:
            name: The requested model identifier.

        Returns:
            All detected issues.
        """
        return validate_model_name(name, self._settings)

    def filters(self, filters: dict[str, Any]) -> list[ValidationIssue]:
        """Validate cohort filters (see :func:`validate_filters`).

        Args:
            filters: Demographic filter mapping.

        Returns:
            All detected issues.
        """
        return validate_filters(filters)

    def survey_name(self, name: str) -> list[ValidationIssue]:
        """Validate a survey name (see :func:`validate_survey_name`).

        Args:
            name: The requested survey name.

        Returns:
            All detected issues.
        """
        return validate_survey_name(name, self._settings)

    def contract_id(self, cid: str) -> list[ValidationIssue]:
        """Validate a contract ID (see :func:`validate_contract_id`).

        Args:
            cid: The contract identifier.

        Returns:
            All detected issues.
        """
        return validate_contract_id(cid, self._settings)

    def client_name(self, name: str) -> list[ValidationIssue]:
        """Validate a client name (see :func:`validate_client_name`).

        Args:
            name: The client name.

        Returns:
            All detected issues.
        """
        return validate_client_name(name, self._settings)
