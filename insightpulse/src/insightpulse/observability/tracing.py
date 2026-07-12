"""Distributed tracing: request spans, agent spans, and LLM call spans.

Two execution paths behind one public API (Strategy pattern, selected at
import):

- **OpenTelemetry** — when the ``opentelemetry`` SDK is importable (install
  the ``observability`` extra), spans are created through the global OTel
  tracer and exported per ``settings.observability.tracing_exporter``.
- **Fallback** — a lightweight :class:`Span` that records name, attributes,
  and duration, and emits one structured JSON log line via structlog on
  exit. Zero external dependencies, used by the demo profile and CI.

Public surface (identical either way):

- :class:`TracingMiddleware` — root span per HTTP request + ``X-Request-ID``
  propagation bound into structlog contextvars.
- :func:`trace_agent` — decorator for async LangGraph node functions.
- :func:`trace_llm_call` — context manager around a single LLM call.
- :func:`trace_etl_step` — decorator for sync or async ETL steps.
- :func:`start_span` — general-purpose span context manager.
"""

from __future__ import annotations

import functools
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from insightpulse.config.settings import get_settings
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Header used for request correlation across services.
REQUEST_ID_HEADER = "X-Request-ID"
# Short-form request IDs keep log lines readable while staying unique
# enough for a single service's correlation window.
_REQUEST_ID_LENGTH = 12

F = TypeVar("F", bound=Callable[..., Any])

# --- OpenTelemetry availability (checked once at import) --------------------

try:
    from opentelemetry import trace as _otel_trace

    _OTEL_AVAILABLE = True
except ImportError:
    _otel_trace = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# structlog context helpers
# ---------------------------------------------------------------------------

def bind_trace_id(request_id: str, **extra: str | int | float) -> None:
    """Bind a request/trace ID into structlog contextvars.

    Every subsequent log line in the current task carries the ID, so a
    request can be followed through middleware, agents, and the data layer.

    Args:
        request_id: Correlation ID for the current request.
        **extra: Additional trace-scoped context (e.g. ``endpoint=...``).
    """
    structlog.contextvars.bind_contextvars(request_id=request_id, **extra)


def clear_trace_context() -> None:
    """Clear trace-scoped structlog context (call when the request ends)."""
    structlog.contextvars.unbind_contextvars("request_id")


# ---------------------------------------------------------------------------
# Fallback span (no OpenTelemetry required)
# ---------------------------------------------------------------------------

class Span:
    """Minimal span: name + attributes + duration, logged on exit.

    API-compatible with the subset of the OpenTelemetry span interface the
    codebase uses (``set_attribute``), so callers never branch on which
    tracing path is active.
    """

    def __init__(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Create the span and start its clock.

        Args:
            name: Span name (dot-namespaced, e.g. ``agent.Validator``).
            attributes: Initial span attributes.
        """
        self.name = name
        self.attributes: dict[str, Any] = dict(attributes or {})
        self.start_time = time.perf_counter()
        self.duration_ms: float | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach one attribute to the span.

        Args:
            key: Attribute name.
            value: Attribute value (kept JSON-serialisable by convention).
        """
        self.attributes[key] = value

    def end(self, error: str | None = None) -> None:
        """Close the span and emit one structured log line.

        Args:
            error: Exception summary when the traced block raised.
        """
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        log = logger.bind(**self.attributes) if self.attributes else logger
        if error is not None:
            log.warning(
                "span_completed", span=self.name, duration_ms=round(self.duration_ms, 3),
                error=error, status="error",
            )
        else:
            log.info(
                "span_completed", span=self.name, duration_ms=round(self.duration_ms, 3),
                status="ok",
            )


@contextmanager
def start_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a span (OpenTelemetry when available, fallback otherwise).

    Args:
        name: Span name.
        **attributes: Initial attributes set on the span.

    Yields:
        A span object exposing ``set_attribute(key, value)``.
    """
    if not get_settings().observability.tracing_enabled:
        yield Span(name, attributes)  # inert: never logged
        return

    if _OTEL_AVAILABLE:
        tracer = _otel_trace.get_tracer("insightpulse")
        with tracer.start_as_current_span(name) as otel_span:
            for key, value in attributes.items():
                otel_span.set_attribute(key, value)
            yield otel_span
        return

    span = Span(name, attributes)
    try:
        yield span
    except Exception as exc:
        span.end(error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        span.end()


# ---------------------------------------------------------------------------
# Exporter configuration (production)
# ---------------------------------------------------------------------------

def configure_tracing() -> None:
    """Configure the trace exporter from ``settings.observability``.

    ``console`` (the demo default) needs no setup — the fallback span logs
    directly, and OTel (when installed) uses its console exporter.
    ``azure_monitor`` and ``jaeger`` require the matching SDK from the
    ``observability`` extras; a clear ``RuntimeError`` is raised when the
    SDK is absent rather than silently dropping spans.

    Raises:
        RuntimeError: If the configured exporter's SDK is not installed.
    """
    settings = get_settings()
    exporter = settings.observability.tracing_exporter

    if not settings.observability.tracing_enabled or exporter == "console":
        if _OTEL_AVAILABLE and exporter == "console":
            _configure_otel_console()
        logger.info("tracing_configured", exporter=exporter, otel=_OTEL_AVAILABLE)
        return

    if not _OTEL_AVAILABLE:
        raise RuntimeError(
            f"Tracing exporter '{exporter}' requires the OpenTelemetry SDK. "
            "Install it with: pip install 'insightpulse[observability]'"
        )

    if exporter == "azure_monitor":
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
        except ImportError as exc:
            raise RuntimeError(
                "Exporter 'azure_monitor' requires azure-monitor-opentelemetry. "
                "Install it with: pip install azure-monitor-opentelemetry"
            ) from exc
        configure_azure_monitor()
    elif exporter == "jaeger":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Exporter 'jaeger' requires the OTLP exporter (Jaeger ingests "
                "OTLP natively). Install it with: "
                "pip install opentelemetry-exporter-otlp"
            ) from exc
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        _otel_trace.set_tracer_provider(provider)
    else:
        raise RuntimeError(
            f"Unknown tracing exporter '{exporter}' "
            "(expected: console | azure_monitor | jaeger)"
        )
    logger.info("tracing_configured", exporter=exporter, otel=True)


def _configure_otel_console() -> None:
    """Attach the OTel console exporter (only called when OTel is present)."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    _otel_trace.set_tracer_provider(provider)


# ---------------------------------------------------------------------------
# HTTP middleware
# ---------------------------------------------------------------------------

class TracingMiddleware(BaseHTTPMiddleware):
    """Root span + request-ID propagation for every HTTP request.

    Single responsibility: request correlation. Design pattern: decorator
    over the ASGI app (Starlette middleware chain), mirroring the rate
    limiter's structure. Accepts an inbound ``X-Request-ID`` (so upstream
    gateways can correlate) or generates a short uuid4-derived ID, binds it
    into structlog contextvars for the request's lifetime, and returns it
    on the response.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Wrap one request in a root span with correlation context.

        Args:
            request: Incoming request.
            call_next: Continuation into the app.

        Returns:
            The app's response with ``X-Request-ID`` set.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[
            :_REQUEST_ID_LENGTH
        ]
        bind_trace_id(request_id)
        try:
            with start_span(
                "http.request",
                request_id=request_id,
                http_method=request.method,
                http_path=request.url.path,
            ) as span:
                response = await call_next(request)
                span.set_attribute("http_status_code", response.status_code)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            clear_trace_context()


# ---------------------------------------------------------------------------
# Domain-specific tracing helpers
# ---------------------------------------------------------------------------

def trace_agent(name: str) -> Callable[[F], F]:
    """Trace an async LangGraph node function as one span.

    Records the node's duration and a brief output summary (the keys of the
    returned state-update dict) — enough to follow a run through the DAG
    without logging response payloads.

    Args:
        name: Agent name (e.g. ``"Validator"``).

    Returns:
        A decorator preserving the wrapped coroutine's signature and result.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with start_span(f"agent.{name}", agent=name) as span:
                result = await fn(*args, **kwargs)
                if isinstance(result, dict):
                    span.set_attribute("output_keys", ",".join(sorted(map(str, result))))
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def trace_llm_call(model: str) -> Iterator[Any]:
    """Trace one LLM API call as a span.

    The caller sets token/cost attributes on the yielded span once the
    provider response is available:

        with trace_llm_call("claude-sonnet-4-6") as span:
            response = await client.complete(...)
            span.set_attribute("input_tokens", response.usage.input_tokens)
            span.set_attribute("output_tokens", response.usage.output_tokens)
            span.set_attribute("cost_usd", cost)

    Latency is recorded automatically as the span duration.

    Args:
        model: Model identifier for the call.

    Yields:
        The span (set ``input_tokens``/``output_tokens``/``cost_usd`` on it).
    """
    start = time.perf_counter()
    with start_span("llm.call", model=model) as span:
        yield span
        span.set_attribute("latency_seconds", round(time.perf_counter() - start, 4))


def trace_etl_step(name: str) -> Callable[[F], F]:
    """Trace an ETL pipeline step (works on sync and async functions).

    Args:
        name: Step name (e.g. ``"benchmark_refresh"``).

    Returns:
        A decorator preserving the wrapped function's signature and result.
    """

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with start_span(f"etl.{name}", etl_step=name):
                    return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with start_span(f"etl.{name}", etl_step=name):
                return fn(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator
