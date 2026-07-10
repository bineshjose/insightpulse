"""Structured logging configuration with structlog.

Provides a single ``configure_logging()`` entry point that sets up structlog
with environment-appropriate rendering:

- **demo / test**: colored, human-readable console output for development.
- **production**: single-line JSON for ingestion by log aggregators.

Agents bind run-level context once (run_id, agent name) and every subsequent
log line carries it automatically — essential for tracing a survey run
through the 8-agent LangGraph pipeline.

Usage:
    from insightpulse.utils.logging import configure_logging, get_logger

    configure_logging()  # once, at application startup
    logger = get_logger(__name__)
    logger = logger.bind(run_id=str(run.run_id), agent="Validator")
    logger.info("validation_complete", validated=42, rejected=3)
"""

from __future__ import annotations

import logging
import sys

import structlog

from insightpulse.config.settings import Environment, get_settings

# Guard against double-configuration when multiple entry points
# (API, dashboard, experiments) import this module.
_CONFIGURED = False


def configure_logging(
    log_level: str | None = None,
    json_output: bool | None = None,
    force: bool = False,
) -> None:
    """Configure structlog and the stdlib logging bridge.

    Idempotent: repeated calls are no-ops unless ``force=True``, so every
    entry point (API server, dashboard, experiment scripts) can safely call
    this at import time.

    Args:
        log_level: Logging level name (DEBUG, INFO, ...). Defaults to the
            ``log_level`` setting from the active environment profile.
        json_output: Render logs as JSON lines. Defaults to True in
            production, False elsewhere.
        force: Reconfigure even if logging was already configured (used by
            tests that need to swap renderers).
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    level_name = (log_level or settings.log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    if json_output is None:
        json_output = settings.env == Environment.PRODUCTION

    # Processors shared by both renderers. Order matters: context merging
    # must run before formatting, and exception info before rendering.
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: structlog.typing.Processor
    if json_output:
        shared_processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, httpx, litellm, ...) through the same
    # renderer so all output shares one format.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Quiet noisy third-party loggers regardless of app level.
    for noisy in ("httpx", "httpcore", "urllib3", "litellm", "faiss"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger, configuring logging on first use if needed.

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.

    Returns:
        A bound structlog logger.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


def bind_run_context(run_id: str, **extra: str | int | float) -> None:
    """Bind survey-run context to all subsequent log lines in this task.

    Uses contextvars so the binding follows async execution through the
    LangGraph pipeline without threading a logger object through state.

    Args:
        run_id: The SurveyRun identifier.
        **extra: Additional context (e.g., model="claude-sonnet-4-6").
    """
    structlog.contextvars.bind_contextvars(run_id=run_id, **extra)


def clear_run_context() -> None:
    """Clear per-run context bindings (call at the end of a survey run)."""
    structlog.contextvars.clear_contextvars()
