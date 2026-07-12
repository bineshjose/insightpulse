"""Alert rules, firing/resolved state machine, and alert history.

Rule thresholds all come from ``settings.observability`` — tightening an
alert never requires a code change. The :class:`AlertManager` implements a
firing/resolved state machine with per-rule cooldown so a flapping metric
produces one actionable notification instead of a page storm; the clock is
injectable for deterministic tests.

Notification channels:

- **demo** — structured log lines via structlog (visible in the console
  and the Operations dashboard's alert feed).
- **production** — hook points for Azure Monitor, PagerDuty, and Slack are
  stubbed on the manager (``_send_azure_monitor_alert`` etc.); wire them to
  the real SDKs/webhooks during rollout without touching the state machine.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from insightpulse.config.settings import get_settings
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Default capacity of the in-memory alert history ring buffer: roughly a
# month of alerts at the expected demo/staging alert rate.
DEFAULT_HISTORY_CAPACITY = 500

_SECONDS_PER_MINUTE = 60.0


class AlertSeverity(StrEnum):
    """Alert severity levels, ordered by escalation urgency."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertRule(BaseModel):
    """One alert rule: what fires, at which threshold, and how often.

    ``threshold`` is the numeric trigger for value-based rules; rules
    evaluated on a boolean condition (e.g. a HIGH-risk injection attempt)
    use ``threshold=0`` and are passed ``True``/``False`` directly.
    """

    name: str
    condition_description: str
    severity: AlertSeverity
    threshold: float
    cooldown_minutes: int = Field(ge=0)
    # Consecutive breaching evaluations required before firing (>=1).
    consecutive_required: int = Field(default=1, ge=1)


class AlertEvent(BaseModel):
    """One firing (and optional resolution) of an alert rule."""

    rule: str
    severity: AlertSeverity
    message: str
    fired_at: datetime
    resolved_at: datetime | None = None


def build_default_rules() -> dict[str, AlertRule]:
    """Build the standard rule set from the active settings profile.

    Returns:
        Rules keyed by name, thresholds sourced from
        ``settings.observability``.
    """
    obs = get_settings().observability
    cooldown = obs.default_alert_cooldown_minutes
    rules = [
        AlertRule(
            name="hallucination_rate_critical",
            condition_description=(
                f"Hallucination rate > {obs.hallucination_alert_threshold:.0%} for "
                f"{obs.hallucination_consecutive_runs} consecutive runs"
            ),
            severity=AlertSeverity.CRITICAL,
            threshold=obs.hallucination_alert_threshold,
            cooldown_minutes=cooldown,
            consecutive_required=obs.hallucination_consecutive_runs,
        ),
        AlertRule(
            name="llm_api_error_rate",
            condition_description=(
                f"LLM API error rate > {obs.llm_error_rate_threshold:.0%} in a 5-minute window"
            ),
            severity=AlertSeverity.CRITICAL,
            threshold=obs.llm_error_rate_threshold,
            cooldown_minutes=cooldown,
        ),
        AlertRule(
            name="prompt_injection_high_risk",
            condition_description="Prompt injection attempt at HIGH or CRITICAL risk level",
            severity=AlertSeverity.CRITICAL,
            threshold=0.0,
            cooldown_minutes=cooldown,
        ),
        AlertRule(
            name="calibration_nonconvergence",
            condition_description=(
                "Calibration non-convergence rate > "
                f"{obs.calibration_nonconvergence_threshold:.0%}"
            ),
            severity=AlertSeverity.WARNING,
            threshold=obs.calibration_nonconvergence_threshold,
            cooldown_minutes=cooldown,
        ),
        AlertRule(
            name="budget_utilization",
            condition_description=(
                f"Budget utilization > {obs.budget_warning_utilization:.0%} of monthly limit"
            ),
            severity=AlertSeverity.WARNING,
            threshold=obs.budget_warning_utilization,
            cooldown_minutes=cooldown,
        ),
        AlertRule(
            name="data_quality_critical",
            condition_description="Data quality check failed at CRITICAL severity",
            severity=AlertSeverity.WARNING,
            threshold=0.0,
            cooldown_minutes=cooldown,
        ),
        AlertRule(
            name="latency_p95",
            condition_description=(
                f"Survey response latency p95 > {obs.latency_p95_threshold_seconds:.0f}s"
            ),
            severity=AlertSeverity.WARNING,
            threshold=obs.latency_p95_threshold_seconds,
            cooldown_minutes=cooldown,
        ),
        AlertRule(
            name="behavioral_drift",
            condition_description="Behavioral drift above the retraining trigger",
            severity=AlertSeverity.INFO,
            threshold=0.0,
            cooldown_minutes=cooldown,
        ),
        AlertRule(
            name="throughput_drop",
            condition_description=(
                f"Survey throughput dropped > {obs.throughput_drop_threshold:.0%} "
                "vs the 7-day average"
            ),
            severity=AlertSeverity.INFO,
            threshold=obs.throughput_drop_threshold,
            cooldown_minutes=cooldown,
        ),
    ]
    return {rule.name: rule for rule in rules}


class AlertHistory:
    """Bounded ring buffer of alert events with time-window queries."""

    def __init__(self, capacity: int = DEFAULT_HISTORY_CAPACITY) -> None:
        """Create the buffer.

        Args:
            capacity: Maximum retained events (oldest evicted first).
        """
        self._events: deque[AlertEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, event: AlertEvent) -> None:
        """Append one event (evicting the oldest at capacity)."""
        with self._lock:
            self._events.append(event)

    def query(self, since: datetime | None = None) -> list[AlertEvent]:
        """Return events fired at or after ``since`` (all when None).

        Args:
            since: Window start (inclusive), timezone-aware.

        Returns:
            Matching events, oldest first.
        """
        with self._lock:
            events = list(self._events)
        if since is None:
            return events
        return [event for event in events if event.fired_at >= since]

    def active(self) -> list[AlertEvent]:
        """Return events that fired and have not resolved."""
        with self._lock:
            return [event for event in self._events if event.resolved_at is None]


class _RuleState:
    """Mutable evaluation state for one rule (internal to the manager)."""

    __slots__ = ("consecutive_breaches", "event", "firing", "last_fired_at")

    def __init__(self) -> None:
        """Initialise a non-firing state."""
        self.firing = False
        self.last_fired_at: float | None = None
        self.consecutive_breaches = 0
        self.event: AlertEvent | None = None


class AlertManager:
    """Evaluates rules against observations with a firing state machine.

    Transitions per evaluation:

    - not firing + condition met (``consecutive_required`` times in a row)
      + cooldown elapsed → **fires** (notified once, recorded in history);
    - firing + condition still met → suppressed (no duplicate notification);
    - firing + condition cleared → **resolves** (resolution recorded);
    - not firing + condition met inside cooldown → suppressed.
    """

    def __init__(
        self,
        rules: dict[str, AlertRule] | None = None,
        history: AlertHistory | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Create the manager.

        Args:
            rules: Rule set; defaults to :func:`build_default_rules`.
            history: Event sink; a fresh :class:`AlertHistory` by default.
            clock: Injectable epoch-seconds time source (tests advance it).
        """
        self.rules = rules if rules is not None else build_default_rules()
        self.history = history if history is not None else AlertHistory()
        self._clock = clock
        self._states: dict[str, _RuleState] = {name: _RuleState() for name in self.rules}
        self._lock = threading.Lock()

    def evaluate(self, rule_name: str, observed: float | bool) -> AlertEvent | None:
        """Evaluate one rule against an observation.

        Args:
            rule_name: Name of a registered rule.
            observed: Metric value (compared against ``rule.threshold``) or
                a boolean condition for event-style rules.

        Returns:
            The :class:`AlertEvent` when this evaluation *fires* or
            *resolves* the alert; None when nothing changed (still firing
            within cooldown, condition still clear, or breach streak not
            yet long enough).

        Raises:
            KeyError: If ``rule_name`` is not registered.
        """
        rule = self.rules[rule_name]
        condition_met = observed if isinstance(observed, bool) else observed > rule.threshold
        now = self._clock()

        with self._lock:
            state = self._states[rule_name]
            if condition_met:
                state.consecutive_breaches += 1
                if state.firing:
                    return None  # already notified; suppress duplicates
                if state.consecutive_breaches < rule.consecutive_required:
                    return None  # streak not long enough yet
                cooldown_seconds = rule.cooldown_minutes * _SECONDS_PER_MINUTE
                if (
                    state.last_fired_at is not None
                    and now - state.last_fired_at < cooldown_seconds
                ):
                    logger.debug("alert_suppressed_cooldown", rule=rule_name)
                    return None
                return self._fire(rule, state, now, observed)

            state.consecutive_breaches = 0
            if state.firing:
                return self._resolve(rule, state, now)
            return None

    def _fire(
        self, rule: AlertRule, state: _RuleState, now: float, observed: float | bool
    ) -> AlertEvent:
        """Transition a rule into the firing state and notify."""
        event = AlertEvent(
            rule=rule.name,
            severity=rule.severity,
            message=f"{rule.condition_description} (observed: {observed})",
            fired_at=datetime.fromtimestamp(now, tz=UTC),
        )
        state.firing = True
        state.last_fired_at = now
        state.event = event
        self.history.record(event)
        self._notify(event)
        return event

    def _resolve(self, rule: AlertRule, state: _RuleState, now: float) -> AlertEvent | None:
        """Transition a rule out of the firing state."""
        event = state.event
        if event is not None:
            event.resolved_at = datetime.fromtimestamp(now, tz=UTC)
            logger.info("alert_resolved", rule=rule.name, severity=str(rule.severity))
        state.firing = False
        state.event = None
        return event

    # --- Notification channels ---------------------------------------------

    def _notify(self, event: AlertEvent) -> None:
        """Route a fired alert to the configured channels.

        Demo: one structured log line. Production: the stub channel hooks
        below are invoked in addition to the log line.
        """
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
        }[event.severity]
        log_method(
            "alert_fired",
            rule=event.rule,
            severity=str(event.severity),
            message=event.message,
        )
        if get_settings().is_production():
            self._send_azure_monitor_alert(event)
            self._send_pagerduty_event(event)
            self._send_slack_webhook(event)

    def _send_azure_monitor_alert(self, event: AlertEvent) -> None:
        """Production hook: push a custom event to Azure Monitor.

        Wire-up: emit via ``azure-monitor-opentelemetry`` custom events or
        an Action Group webhook; the connection string comes from Key Vault
        (see infra/k8s/keyvault-csi.yaml).
        """

    def _send_pagerduty_event(self, event: AlertEvent) -> None:
        """Production hook: page on CRITICAL via the PagerDuty Events API v2.

        Wire-up: POST to the Events API with the service routing key;
        deduplicate on ``event.rule`` so resolve calls close the incident.
        """

    def _send_slack_webhook(self, event: AlertEvent) -> None:
        """Production hook: post WARNING/INFO alerts to the ops Slack channel.

        Wire-up: POST the event summary to the incoming-webhook URL stored
        in Key Vault; keep CRITICAL routing on PagerDuty to avoid dual paging.
        """


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_manager: AlertManager | None = None
_manager_lock = threading.Lock()


def get_alert_manager() -> AlertManager:
    """Get the process-wide :class:`AlertManager` singleton."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = AlertManager()
    return _manager


def reset_alert_manager() -> None:
    """Discard the singleton (test isolation helper)."""
    global _manager
    with _manager_lock:
        _manager = None
