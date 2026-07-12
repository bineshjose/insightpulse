"""Application metrics: counters, histograms, and gauges with Prometheus output.

Two interchangeable backends (Strategy pattern), selected once at import:

- **InMemoryBackend** — a pure-Python, thread-safe registry that renders the
  Prometheus text exposition format itself. Zero external dependencies, so
  the demo profile and CI never need ``prometheus-client`` installed.
- **PrometheusClientBackend** — a thin delegate to ``prometheus_client``
  when it is importable (install the ``observability`` extra), so production
  scraping benefits from the battle-tested reference implementation.

Callers never touch raw metric names: :class:`MetricsCollector` exposes
typed convenience methods (``record_survey_run``, ``record_llm_call``, ...)
and notifies registered observers (Observer pattern) so middleware and the
alerting layer can react to recorded events without coupling to metric
internals.

Usage:
    from insightpulse.observability.metrics import get_metrics_collector

    metrics = get_metrics_collector()
    metrics.record_llm_call(
        model="claude-sonnet-4-6", status="success",
        latency_seconds=0.42, input_tokens=350, output_tokens=120,
    )
    exposition = metrics.render_prometheus()
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Bucket definitions (seconds unless stated otherwise)
# ---------------------------------------------------------------------------

# General-purpose latency buckets (mirrors the prometheus_client defaults).
DEFAULT_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)

# Survey runs span seconds to minutes (cohort size × questions × model).
SURVEY_DURATION_BUCKETS: tuple[float, ...] = (
    1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
)

# Sinkhorn iteration counts (dimensionless, bounded by sinkhorn_max_iter).
CALIBRATION_ITERATION_BUCKETS: tuple[float, ...] = (
    10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
)

# Cohort extraction is I/O bound: sub-second for cache hits, tens of
# seconds when Snowflake extraction runs.
COHORT_EXTRACTION_BUCKETS: tuple[float, ...] = (
    0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)

# Cohort-size label buckets for survey_run_duration_seconds: upper bound
# (inclusive) mapped to its label. Sizes beyond the last bound overflow.
_COHORT_SIZE_LABEL_BOUNDS: tuple[tuple[int, str], ...] = (
    (50, "1-50"),
    (100, "51-100"),
    (500, "101-500"),
)
_COHORT_SIZE_OVERFLOW_LABEL = "500+"


def cohort_size_bucket(cohort_size: int) -> str:
    """Map a cohort size onto its label bucket for histogram labelling.

    Args:
        cohort_size: Number of synthetic panelists in the run.

    Returns:
        The bucket label (e.g. ``"51-100"``).
    """
    for bound, label in _COHORT_SIZE_LABEL_BOUNDS:
        if cohort_size <= bound:
            return label
    return _COHORT_SIZE_OVERFLOW_LABEL


# ---------------------------------------------------------------------------
# Prometheus text exposition helpers
# ---------------------------------------------------------------------------

def _escape_help(text: str) -> str:
    """Escape a HELP line per the Prometheus text exposition format."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_value(value: float) -> str:
    """Render a sample value the way Prometheus expects (ints unpadded)."""
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value):
        return str(int(value))
    return repr(value)


def _format_labels(labelnames: Sequence[str], labelvalues: Sequence[str]) -> str:
    """Render a ``{name="value",...}`` label block (empty string if none)."""
    if not labelnames:
        return ""
    pairs = ",".join(
        f'{name}="{_escape_label_value(str(value))}"'
        for name, value in zip(labelnames, labelvalues, strict=True)
    )
    return "{" + pairs + "}"


# ---------------------------------------------------------------------------
# In-memory metric primitives
# ---------------------------------------------------------------------------

class _InMemoryMetric:
    """Base class for in-memory metrics: label handling + thread safety."""

    metric_type: str = ""

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> None:
        """Create the metric.

        Args:
            name: Prometheus metric name (snake_case, unit-suffixed).
            documentation: HELP text describing the metric.
            labelnames: Ordered label names; children are keyed by values.
        """
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()

    def _label_key(self, labels: dict[str, Any]) -> tuple[str, ...]:
        """Validate and order label values into the internal storage key."""
        if set(labels) != set(self.labelnames):
            raise ValueError(
                f"Metric '{self.name}' expects labels {self.labelnames}, "
                f"got {tuple(labels)}"
            )
        return tuple(str(labels[name]) for name in self.labelnames)

    def render(self) -> str:
        """Render this metric's exposition block (HELP, TYPE, samples)."""
        lines = [
            f"# HELP {self.name} {_escape_help(self.documentation)}",
            f"# TYPE {self.name} {self.metric_type}",
        ]
        lines.extend(self._sample_lines())
        return "\n".join(lines)

    def _sample_lines(self) -> list[str]:
        """Render the sample lines; implemented by each metric type."""
        raise NotImplementedError


class _CounterChild:
    """A counter bound to one label-value combination."""

    def __init__(self, parent: InMemoryCounter, key: tuple[str, ...]) -> None:
        """Bind to the parent counter and label key."""
        self._parent = parent
        self._key = key

    def inc(self, amount: float = 1.0) -> None:
        """Increment the bound series by ``amount`` (must be >= 0)."""
        self._parent._inc(self._key, amount)


class InMemoryCounter(_InMemoryMetric):
    """Thread-safe monotonically increasing counter with labels."""

    metric_type = "counter"

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> None:
        """Create the counter; see :class:`_InMemoryMetric`."""
        super().__init__(name, documentation, labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def labels(self, **labels: Any) -> _CounterChild:
        """Return the child counter for one label-value combination."""
        return _CounterChild(self, self._label_key(labels))

    def inc(self, amount: float = 1.0) -> None:
        """Increment an unlabelled counter by ``amount``."""
        if self.labelnames:
            raise ValueError(f"Counter '{self.name}' requires labels(); use .labels(...).inc()")
        self._inc((), amount)

    def _inc(self, key: tuple[str, ...], amount: float) -> None:
        """Apply the increment under the metric lock."""
        if amount < 0:
            raise ValueError(f"Counter '{self.name}' cannot decrease (amount={amount})")
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: Any) -> float:
        """Read the current value for a label combination (testing/queries)."""
        key = self._label_key(labels) if self.labelnames else ()
        with self._lock:
            return self._values.get(key, 0.0)

    def samples(self) -> dict[tuple[str, ...], float]:
        """Snapshot all series as ``{label_values: value}``."""
        with self._lock:
            return dict(self._values)

    def _sample_lines(self) -> list[str]:
        with self._lock:
            return [
                f"{self.name}{_format_labels(self.labelnames, key)} {_format_value(value)}"
                for key, value in sorted(self._values.items())
            ]


class _GaugeChild:
    """A gauge bound to one label-value combination."""

    def __init__(self, parent: InMemoryGauge, key: tuple[str, ...]) -> None:
        """Bind to the parent gauge and label key."""
        self._parent = parent
        self._key = key

    def set(self, value: float) -> None:
        """Set the bound series to ``value``."""
        self._parent._set(self._key, value)

    def inc(self, amount: float = 1.0) -> None:
        """Increment the bound series by ``amount``."""
        self._parent._add(self._key, amount)

    def dec(self, amount: float = 1.0) -> None:
        """Decrement the bound series by ``amount``."""
        self._parent._add(self._key, -amount)


class InMemoryGauge(_InMemoryMetric):
    """Thread-safe gauge (can go up and down) with labels."""

    metric_type = "gauge"

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> None:
        """Create the gauge; see :class:`_InMemoryMetric`."""
        super().__init__(name, documentation, labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def labels(self, **labels: Any) -> _GaugeChild:
        """Return the child gauge for one label-value combination."""
        return _GaugeChild(self, self._label_key(labels))

    def set(self, value: float) -> None:
        """Set an unlabelled gauge to ``value``."""
        if self.labelnames:
            raise ValueError(f"Gauge '{self.name}' requires labels(); use .labels(...).set()")
        self._set((), value)

    def inc(self, amount: float = 1.0) -> None:
        """Increment an unlabelled gauge by ``amount``."""
        if self.labelnames:
            raise ValueError(f"Gauge '{self.name}' requires labels(); use .labels(...).inc()")
        self._add((), amount)

    def dec(self, amount: float = 1.0) -> None:
        """Decrement an unlabelled gauge by ``amount``."""
        self.inc(-amount)

    def _set(self, key: tuple[str, ...], value: float) -> None:
        """Set a series value under the metric lock."""
        with self._lock:
            self._values[key] = float(value)

    def _add(self, key: tuple[str, ...], amount: float) -> None:
        """Add to a series value under the metric lock."""
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: Any) -> float:
        """Read the current value for a label combination (testing/queries)."""
        key = self._label_key(labels) if self.labelnames else ()
        with self._lock:
            return self._values.get(key, 0.0)

    def samples(self) -> dict[tuple[str, ...], float]:
        """Snapshot all series as ``{label_values: value}``."""
        with self._lock:
            return dict(self._values)

    def _sample_lines(self) -> list[str]:
        with self._lock:
            return [
                f"{self.name}{_format_labels(self.labelnames, key)} {_format_value(value)}"
                for key, value in sorted(self._values.items())
            ]


class _HistogramSeries:
    """Mutable bucket counts, sum, and count for one label combination."""

    __slots__ = ("bucket_counts", "count", "total")

    def __init__(self, num_buckets: int) -> None:
        """Initialise zeroed bucket counts (excludes the +Inf bucket)."""
        self.bucket_counts = [0] * num_buckets
        self.total = 0.0
        self.count = 0


class _HistogramChild:
    """A histogram bound to one label-value combination."""

    def __init__(self, parent: InMemoryHistogram, key: tuple[str, ...]) -> None:
        """Bind to the parent histogram and label key."""
        self._parent = parent
        self._key = key

    def observe(self, value: float) -> None:
        """Record one observation into the bound series."""
        self._parent._observe(self._key, value)


class InMemoryHistogram(_InMemoryMetric):
    """Thread-safe histogram with cumulative buckets, labels, +Inf handling."""

    metric_type = "histogram"

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS,
    ) -> None:
        """Create the histogram.

        Args:
            name: Prometheus metric name.
            documentation: HELP text.
            labelnames: Ordered label names.
            buckets: Ascending finite upper bounds; +Inf is added implicitly.
        """
        super().__init__(name, documentation, labelnames)
        finite = tuple(float(b) for b in buckets if not math.isinf(b))
        if list(finite) != sorted(finite):
            raise ValueError(f"Histogram '{name}' buckets must be ascending: {buckets}")
        self.buckets = finite
        self._series: dict[tuple[str, ...], _HistogramSeries] = {}

    def labels(self, **labels: Any) -> _HistogramChild:
        """Return the child histogram for one label-value combination."""
        return _HistogramChild(self, self._label_key(labels))

    def observe(self, value: float) -> None:
        """Record one observation into an unlabelled histogram."""
        if self.labelnames:
            raise ValueError(
                f"Histogram '{self.name}' requires labels(); use .labels(...).observe()"
            )
        self._observe((), value)

    def _observe(self, key: tuple[str, ...], value: float) -> None:
        """Apply one observation under the metric lock."""
        with self._lock:
            series = self._series.get(key)
            if series is None:
                series = self._series[key] = _HistogramSeries(len(self.buckets))
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    series.bucket_counts[i] += 1
            series.total += value
            series.count += 1

    def series(self, **labels: Any) -> _HistogramSeries | None:
        """Read the series for a label combination (testing/queries)."""
        key = self._label_key(labels) if self.labelnames else ()
        with self._lock:
            return self._series.get(key)

    def all_series(self) -> dict[tuple[str, ...], _HistogramSeries]:
        """Snapshot all series keyed by label values."""
        with self._lock:
            return dict(self._series)

    def _sample_lines(self) -> list[str]:
        lines: list[str] = []
        bucket_labelnames = (*self.labelnames, "le")
        with self._lock:
            for key, series in sorted(self._series.items()):
                cumulative = 0
                for bound, in_bucket in zip(self.buckets, series.bucket_counts, strict=True):
                    # bucket_counts are already cumulative per bound because
                    # _observe increments every bound >= value.
                    cumulative = in_bucket
                    label_block = _format_labels(bucket_labelnames, (*key, _format_value(bound)))
                    lines.append(f"{self.name}_bucket{label_block} {cumulative}")
                inf_block = _format_labels(bucket_labelnames, (*key, "+Inf"))
                lines.append(f"{self.name}_bucket{inf_block} {series.count}")
                plain = _format_labels(self.labelnames, key)
                lines.append(f"{self.name}_sum{plain} {_format_value(series.total)}")
                lines.append(f"{self.name}_count{plain} {series.count}")
        return lines


# ---------------------------------------------------------------------------
# Backends (Strategy pattern)
# ---------------------------------------------------------------------------

class MetricsBackend(Protocol):
    """Interface every metrics backend implements (Strategy pattern)."""

    def counter(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> Any:
        """Create and register a counter."""
        ...

    def gauge(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> Any:
        """Create and register a gauge."""
        ...

    def histogram(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS,
    ) -> Any:
        """Create and register a histogram."""
        ...

    def render(self) -> str:
        """Render all registered metrics as Prometheus exposition text."""
        ...


class InMemoryBackend:
    """Dependency-free backend backed by the in-memory metric primitives."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._metrics: list[_InMemoryMetric] = []
        self._lock = threading.Lock()

    def counter(
        self, name: str, documentation: str, labelnames: Sequence[str] = ()
    ) -> InMemoryCounter:
        """Create and register an :class:`InMemoryCounter`."""
        metric = InMemoryCounter(name, documentation, labelnames)
        self._register(metric)
        return metric

    def gauge(
        self, name: str, documentation: str, labelnames: Sequence[str] = ()
    ) -> InMemoryGauge:
        """Create and register an :class:`InMemoryGauge`."""
        metric = InMemoryGauge(name, documentation, labelnames)
        self._register(metric)
        return metric

    def histogram(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS,
    ) -> InMemoryHistogram:
        """Create and register an :class:`InMemoryHistogram`."""
        metric = InMemoryHistogram(name, documentation, labelnames, buckets)
        self._register(metric)
        return metric

    def _register(self, metric: _InMemoryMetric) -> None:
        """Add a metric to the registry, rejecting duplicate names."""
        with self._lock:
            if any(existing.name == metric.name for existing in self._metrics):
                raise ValueError(f"Metric '{metric.name}' already registered")
            self._metrics.append(metric)

    def render(self) -> str:
        """Render the full Prometheus text exposition for all metrics."""
        with self._lock:
            metrics = list(self._metrics)
        blocks = [metric.render() for metric in metrics]
        return "\n".join(blocks) + "\n"


class PrometheusClientBackend:
    """Backend delegating to ``prometheus_client`` (production scraping).

    Uses a dedicated ``CollectorRegistry`` so the collector's metrics never
    collide with other registries in the process (e.g. multiprocess mode).
    """

    def __init__(self) -> None:
        """Create the delegate registry; raises ImportError if SDK missing."""
        from prometheus_client import CollectorRegistry

        self._registry = CollectorRegistry()

    def counter(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> Any:
        """Create and register a ``prometheus_client.Counter``."""
        from prometheus_client import Counter

        return Counter(name, documentation, labelnames, registry=self._registry)

    def gauge(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> Any:
        """Create and register a ``prometheus_client.Gauge``."""
        from prometheus_client import Gauge

        return Gauge(name, documentation, labelnames, registry=self._registry)

    def histogram(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS,
    ) -> Any:
        """Create and register a ``prometheus_client.Histogram``."""
        from prometheus_client import Histogram

        return Histogram(name, documentation, labelnames, registry=self._registry, buckets=buckets)

    def render(self) -> str:
        """Render exposition text via ``prometheus_client.generate_latest``."""
        from prometheus_client import generate_latest

        return generate_latest(self._registry).decode("utf-8")


def _default_backend() -> MetricsBackend:
    """Select the backend: prometheus_client when importable, else in-memory."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        logger.debug("metrics_backend_selected", backend="in_memory")
        return InMemoryBackend()
    logger.debug("metrics_backend_selected", backend="prometheus_client")
    return PrometheusClientBackend()


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

# Observers receive (event_name, payload) for each recorded event.
MetricsObserver = Callable[[str, dict[str, Any]], None]


class MetricsCollector:
    """Central registry of every InsightPulse metric plus typed recorders.

    Single responsibility: own metric definitions and translate domain
    events into metric updates. Middleware and agents call the typed
    ``record_*``/``set_*`` methods and never touch raw metric names.
    Registered observers (Observer pattern) are notified of every recorded
    event so the alerting layer can evaluate rules in near-real-time.
    """

    def __init__(self, backend: MetricsBackend | None = None) -> None:
        """Create the collector and define all metrics on the backend.

        Args:
            backend: Metrics backend; defaults to prometheus_client when
                installed, otherwise the in-memory implementation.
        """
        self._backend: MetricsBackend = backend or _default_backend()
        self._observers: list[MetricsObserver] = []
        self._observer_lock = threading.Lock()

        b = self._backend
        # --- Counters ---
        self.survey_runs_total = b.counter(
            "survey_runs_total",
            "Completed survey runs by model, terminal status, and client.",
            ("model", "status", "client"),
        )
        self.llm_calls_total = b.counter(
            "llm_calls_total",
            "Individual LLM API calls by model and outcome.",
            ("model", "status"),
        )
        self.llm_tokens_total = b.counter(
            "llm_tokens_total",
            "Tokens exchanged with LLM providers (direction: input|output).",
            ("model", "direction"),
        )
        self.validation_checks_total = b.counter(
            "validation_checks_total",
            "Validator agent checks by check type and result.",
            ("check_type", "result"),
        )
        self.prompt_injection_detected_total = b.counter(
            "prompt_injection_detected_total",
            "Prompt injection attempts detected by the PromptGuard.",
            ("risk_level",),
        )
        self.pii_detections_total = b.counter(
            "pii_detections_total",
            "PII patterns detected (and redacted) in generated responses.",
            ("pii_type",),
        )
        self.calibration_runs_total = b.counter(
            "calibration_runs_total",
            "BDCL Sinkhorn calibration runs by convergence outcome.",
            ("converged",),
        )
        self.api_requests_total = b.counter(
            "api_requests_total",
            "HTTP requests served by endpoint, method, and status code.",
            ("endpoint", "method", "status_code"),
        )

        # --- Histograms ---
        self.survey_run_duration_seconds = b.histogram(
            "survey_run_duration_seconds",
            "End-to-end survey run duration through the LangGraph pipeline.",
            ("model", "cohort_size_bucket"),
            buckets=SURVEY_DURATION_BUCKETS,
        )
        self.llm_call_latency_seconds = b.histogram(
            "llm_call_latency_seconds",
            "Latency of individual LLM API calls.",
            ("model",),
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.calibration_convergence_iterations = b.histogram(
            "calibration_convergence_iterations",
            "Sinkhorn iterations until convergence, by epsilon setting.",
            ("epsilon_bucket",),
            buckets=CALIBRATION_ITERATION_BUCKETS,
        )
        self.response_generation_time_seconds = b.histogram(
            "response_generation_time_seconds",
            "Time to generate one synthetic panelist response.",
            ("model",),
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.cohort_extraction_duration_seconds = b.histogram(
            "cohort_extraction_duration_seconds",
            "Time to select and extract a panelist cohort (FAISS + data layer).",
            (),
            buckets=COHORT_EXTRACTION_BUCKETS,
        )
        self.api_request_duration_seconds = b.histogram(
            "api_request_duration_seconds",
            "HTTP request handling duration by endpoint.",
            ("endpoint",),
            buckets=DEFAULT_LATENCY_BUCKETS,
        )

        # --- Gauges ---
        self.active_survey_runs = b.gauge(
            "active_survey_runs",
            "Survey runs currently executing in the pipeline.",
        )
        self.llm_token_budget_remaining = b.gauge(
            "llm_token_budget_remaining",
            "Remaining token budget per model (CostAgent enforcement).",
            ("model",),
        )
        self.total_cost_usd_accumulated = b.gauge(
            "total_cost_usd_accumulated",
            "Accumulated LLM spend in USD per model.",
            ("model",),
        )
        self.cache_hit_rate_ratio = b.gauge(
            "cache_hit_rate_ratio",
            "Cache hit rate over the recent window (0-1).",
        )
        self.embedding_cache_entries = b.gauge(
            "embedding_cache_entries",
            "Number of behavioral embeddings held in the cache.",
        )
        self.hallucination_rate_current_ratio = b.gauge(
            "hallucination_rate_current_ratio",
            "Hallucination rate of the most recent survey run (0-1).",
        )

    # --- Observer pattern -------------------------------------------------

    def add_observer(self, observer: MetricsObserver) -> None:
        """Register an observer notified of every recorded event.

        Args:
            observer: Callable receiving ``(event_name, payload)``.
        """
        with self._observer_lock:
            self._observers.append(observer)

    def remove_observer(self, observer: MetricsObserver) -> None:
        """Deregister a previously added observer (no-op if absent)."""
        with self._observer_lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def _notify(self, event: str, **payload: Any) -> None:
        """Fan an event out to observers; observer errors never propagate."""
        with self._observer_lock:
            observers = list(self._observers)
        for observer in observers:
            try:
                observer(event, payload)
            except Exception:  # observers must never break metric recording
                logger.exception("metrics_observer_failed", observer_event=event)

    # --- Typed recorders ----------------------------------------------------

    def record_survey_run(
        self,
        model: str,
        status: str,
        client: str,
        duration_seconds: float,
        cohort_size: int,
    ) -> None:
        """Record one completed survey run.

        Args:
            model: LLM model identifier used for generation.
            status: Terminal status (``success`` | ``failed`` | ``cancelled``).
            client: Client organisation the survey ran for.
            duration_seconds: End-to-end pipeline duration.
            cohort_size: Number of synthetic panelists in the cohort.
        """
        self.survey_runs_total.labels(model=model, status=status, client=client).inc()
        self.survey_run_duration_seconds.labels(
            model=model, cohort_size_bucket=cohort_size_bucket(cohort_size)
        ).observe(duration_seconds)
        self._notify(
            "survey_run",
            model=model,
            status=status,
            client=client,
            duration_seconds=duration_seconds,
            cohort_size=cohort_size,
        )

    def record_llm_call(
        self,
        model: str,
        status: str,
        latency_seconds: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record one LLM API call.

        Args:
            model: Model identifier.
            status: Call outcome (``success`` | ``error`` | ``timeout``).
            latency_seconds: Wall-clock latency of the call.
            input_tokens: Prompt tokens sent.
            output_tokens: Completion tokens received.
            cost_usd: Cost attributed to this call (added to the model total).
        """
        self.llm_calls_total.labels(model=model, status=status).inc()
        self.llm_call_latency_seconds.labels(model=model).observe(latency_seconds)
        if input_tokens:
            self.llm_tokens_total.labels(model=model, direction="input").inc(input_tokens)
        if output_tokens:
            self.llm_tokens_total.labels(model=model, direction="output").inc(output_tokens)
        if cost_usd:
            self.total_cost_usd_accumulated.labels(model=model).inc(cost_usd)
        self._notify(
            "llm_call",
            model=model,
            status=status,
            latency_seconds=latency_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    def record_validation_check(self, check_type: str, result: str) -> None:
        """Record one Validator agent check.

        Args:
            check_type: Check category (consistency, hallucination, ...).
            result: Outcome (``pass`` | ``fail`` | ``critical_failure``).
        """
        self.validation_checks_total.labels(check_type=check_type, result=result).inc()
        self._notify("validation_check", check_type=check_type, result=result)

    def record_prompt_injection(self, risk_level: str) -> None:
        """Record a prompt injection detection.

        Args:
            risk_level: PromptGuard risk classification (low ... critical).
        """
        self.prompt_injection_detected_total.labels(risk_level=risk_level).inc()
        self._notify("prompt_injection", risk_level=risk_level)

    def record_pii_detection(self, pii_type: str) -> None:
        """Record a PII detection (and redaction) in a generated response.

        Args:
            pii_type: PII category (email, phone, ssn, ...).
        """
        self.pii_detections_total.labels(pii_type=pii_type).inc()
        self._notify("pii_detection", pii_type=pii_type)

    def record_calibration_run(self, converged: bool, iterations: int, epsilon: float) -> None:
        """Record one BDCL calibration run.

        Args:
            converged: Whether Sinkhorn converged within the iteration cap.
            iterations: Iterations performed.
            epsilon: Sinkhorn regularization strength used.
        """
        self.calibration_runs_total.labels(converged=str(converged).lower()).inc()
        self.calibration_convergence_iterations.labels(
            epsilon_bucket=f"{epsilon:g}"
        ).observe(float(iterations))
        self._notify(
            "calibration_run", converged=converged, iterations=iterations, epsilon=epsilon
        )

    def record_api_request(
        self, endpoint: str, method: str, status_code: int, duration_seconds: float
    ) -> None:
        """Record one handled HTTP request.

        Args:
            endpoint: Route template (e.g. ``/api/v1/survey/run``).
            method: HTTP method.
            status_code: Response status code.
            duration_seconds: Handling duration.
        """
        self.api_requests_total.labels(
            endpoint=endpoint, method=method, status_code=str(status_code)
        ).inc()
        self.api_request_duration_seconds.labels(endpoint=endpoint).observe(duration_seconds)
        self._notify(
            "api_request",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_seconds=duration_seconds,
        )

    def record_response_generation(self, model: str, duration_seconds: float) -> None:
        """Record the generation time of one synthetic response.

        Args:
            model: Model identifier.
            duration_seconds: Generation duration.
        """
        self.response_generation_time_seconds.labels(model=model).observe(duration_seconds)
        self._notify("response_generation", model=model, duration_seconds=duration_seconds)

    def record_cohort_extraction(self, duration_seconds: float) -> None:
        """Record the duration of one cohort selection + extraction.

        Args:
            duration_seconds: Extraction duration.
        """
        self.cohort_extraction_duration_seconds.observe(duration_seconds)
        self._notify("cohort_extraction", duration_seconds=duration_seconds)

    # --- Gauge setters -------------------------------------------------------

    def survey_run_started(self) -> None:
        """Increment the active-run gauge when a run enters the pipeline."""
        self.active_survey_runs.inc()

    def survey_run_finished(self) -> None:
        """Decrement the active-run gauge when a run leaves the pipeline."""
        self.active_survey_runs.dec()

    def set_token_budget_remaining(self, model: str, tokens: float) -> None:
        """Set the remaining token budget gauge for a model."""
        self.llm_token_budget_remaining.labels(model=model).set(tokens)

    def set_cache_hit_rate(self, ratio: float) -> None:
        """Set the cache hit-rate gauge (0-1)."""
        self.cache_hit_rate_ratio.set(ratio)

    def set_embedding_cache_entries(self, entries: int) -> None:
        """Set the embedding cache size gauge."""
        self.embedding_cache_entries.set(float(entries))

    def set_hallucination_rate(self, ratio: float) -> None:
        """Set the current hallucination-rate gauge (0-1) and notify observers."""
        self.hallucination_rate_current_ratio.set(ratio)
        self._notify("hallucination_rate", ratio=ratio)

    # --- Export ----------------------------------------------------------------

    def render_prometheus(self) -> str:
        """Render the full Prometheus text exposition for the /metrics endpoint.

        Returns:
            Exposition text (``# HELP``/``# TYPE`` blocks plus samples).
        """
        return self._backend.render()


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get the process-wide :class:`MetricsCollector` singleton.

    Returns:
        The shared collector, created on first use.
    """
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector


def reset_metrics_collector() -> None:
    """Discard the singleton so the next access builds a fresh registry.

    Intended for tests that need isolated metric state.
    """
    global _collector
    with _collector_lock:
        _collector = None
