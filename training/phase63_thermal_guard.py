"""Strict, injectable PHASE63 thermal runtime guard.

The state machine in this module deliberately has no torch, CUDA, subprocess,
or lock dependency.  Production code supplies the telemetry provider; tests
supply deterministic CPU-only samples.  A telemetry failure is an abort, never
an assumption that the GPU is safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Any, Callable, Mapping


POLICY_VERSION = "phase63-thermal-runtime-v1"
PRECHECK_MAX_C = 60.0
PAUSE_AT_C = 80.0
RESUME_AT_OR_BELOW_C = 65.0
ABORT_AT_C = 85.0
COOLDOWN_TIMEOUT_SECONDS = 300.0
# Querying nvidia-smi may take up to five seconds in the legacy provider.  Six
# seconds allows that bounded query plus scheduling jitter, while remaining a
# separate execution-safety setting rather than a scientific parameter.
MAX_SAMPLE_AGE_SECONDS = 6.0
MAX_FUTURE_SKEW_SECONDS = 0.5
WATCHDOG_INTERVAL_SECONDS = 0.25


class ThermalState(str, Enum):
    PRECHECK = "PRECHECK"
    RUNNING = "RUNNING"
    PAUSE_PENDING = "PAUSE_PENDING"
    PAUSED = "PAUSED"
    RESUMING = "RESUMING"
    ABORTED = "ABORTED"
    COMPLETE = "COMPLETE"


class ThermalGuardError(RuntimeError):
    """Fail-closed error for a blocked PHASE63 thermal transition."""


class ThermalAbort(ThermalGuardError):
    """The guard observed an unsafe or unverifiable thermal condition."""


@dataclass(frozen=True)
class ThermalSample:
    timestamp_monotonic: float
    gpu_temperature_c: float
    hardware_thermal_slowdown: bool
    software_thermal_slowdown: bool
    gpu_utilization_percent: float | None = None
    power_w: float | None = None


def _finite_number(value: Any, field: str, *, minimum: float | None = None,
                   maximum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ThermalGuardError("THERMAL_SAMPLE_INVALID:" + field)
    value = float(value)
    if minimum is not None and value < minimum:
        raise ThermalGuardError("THERMAL_SAMPLE_INVALID:" + field)
    if maximum is not None and value > maximum:
        raise ThermalGuardError("THERMAL_SAMPLE_INVALID:" + field)
    return value


def validate_sample(value: Mapping[str, Any], *, now: float, previous_timestamp: float | None) -> ThermalSample:
    """Validate a single selected-GPU sample before it influences execution."""
    if not isinstance(value, Mapping) or not value:
        raise ThermalGuardError("THERMAL_SAMPLE_EMPTY")
    required = ("timestamp_monotonic", "gpu_temperature_c", "hardware_thermal_slowdown",
                "software_thermal_slowdown")
    if any(field not in value for field in required):
        raise ThermalGuardError("THERMAL_SAMPLE_MISSING_REQUIRED_FIELD")
    timestamp = _finite_number(value["timestamp_monotonic"], "timestamp_monotonic", minimum=0.0)
    temperature = _finite_number(value["gpu_temperature_c"], "gpu_temperature_c", minimum=0.0, maximum=150.0)
    if timestamp > now + MAX_FUTURE_SKEW_SECONDS:
        raise ThermalGuardError("THERMAL_SAMPLE_TIMESTAMP_FUTURE")
    if now - timestamp > MAX_SAMPLE_AGE_SECONDS:
        raise ThermalGuardError("THERMAL_SAMPLE_STALE")
    if previous_timestamp is not None and timestamp < previous_timestamp:
        raise ThermalGuardError("THERMAL_SAMPLE_TIMESTAMP_REGRESSION")
    # A fresh clock with an unchanged source timestamp is an unchanging sensor
    # value, not evidence that it remains safe.  Treat it as stale fail-closed.
    if previous_timestamp is not None and timestamp == previous_timestamp:
        raise ThermalGuardError("THERMAL_SAMPLE_TIMESTAMP_NOT_ADVANCING")
    for field in ("hardware_thermal_slowdown", "software_thermal_slowdown"):
        if type(value[field]) is not bool:
            raise ThermalGuardError("THERMAL_SAMPLE_INVALID:" + field)
    optional: dict[str, float | None] = {"gpu_utilization_percent": None, "power_w": None}
    if "gpu_utilization_percent" in value:
        optional["gpu_utilization_percent"] = _finite_number(value["gpu_utilization_percent"], "gpu_utilization_percent", minimum=0.0, maximum=100.0)
    if "power_w" in value:
        optional["power_w"] = _finite_number(value["power_w"], "power_w", minimum=0.0, maximum=1000.0)
    return ThermalSample(timestamp, temperature, value["hardware_thermal_slowdown"],
                         value["software_thermal_slowdown"], **optional)


class ThermalGuard:
    """Boundary gate and receipt accumulator for a single PHASE63 action."""
    def __init__(self, sample_provider: Callable[[], Mapping[str, Any]], *,
                 clock_provider: Callable[[], float] = time.monotonic,
                 sleep_provider: Callable[[float], None] = time.sleep) -> None:
        self.sample_provider = sample_provider
        self.clock_provider = clock_provider
        self.sleep_provider = sleep_provider
        self._lock = threading.RLock()
        self.state = ThermalState.PRECHECK
        self.previous_timestamp: float | None = None
        self.precheck_temperature: float | None = None
        self.max_temperature: float | None = None
        self.sample_count = 0
        self.pause_events: list[dict[str, Any]] = []
        self.resume_events: list[dict[str, Any]] = []
        self.sensor_failure_count = 0
        self.stale_count = 0
        self.hardware_thermal_slowdown_seen = False
        self.software_thermal_slowdown_seen = False
        self.abort_reason: str | None = None

    def _transition(self, target: ThermalState, *, allowed: tuple[ThermalState, ...]) -> None:
        if self.state not in allowed:
            raise ThermalGuardError("THERMAL_INVALID_TRANSITION:" + self.state.value + "->" + target.value)
        self.state = target

    def _abort(self, reason: str) -> None:
        if self.state is ThermalState.COMPLETE:
            raise ThermalGuardError("THERMAL_INVALID_TRANSITION:COMPLETE->ABORTED")
        self.abort_reason = reason
        self.state = ThermalState.ABORTED

    def _sample(self) -> ThermalSample:
        if self.state in (ThermalState.ABORTED, ThermalState.COMPLETE):
            raise ThermalGuardError("THERMAL_NOT_ACTIVE:" + self.state.value)
        try:
            raw = self.sample_provider()
            sample = validate_sample(raw, now=float(self.clock_provider()), previous_timestamp=self.previous_timestamp)
        except BaseException as exc:
            reason = str(exc) or type(exc).__name__
            self.sensor_failure_count += 1
            if "STALE" in reason or "TIMESTAMP_NOT_ADVANCING" in reason:
                self.stale_count += 1
            self._abort("THERMAL_SENSOR_FAILURE:" + reason)
            raise ThermalAbort(self.abort_reason) from exc
        self.previous_timestamp = sample.timestamp_monotonic
        self.sample_count += 1
        self.max_temperature = sample.gpu_temperature_c if self.max_temperature is None else max(self.max_temperature, sample.gpu_temperature_c)
        self.hardware_thermal_slowdown_seen |= sample.hardware_thermal_slowdown
        self.software_thermal_slowdown_seen |= sample.software_thermal_slowdown
        if sample.gpu_temperature_c >= ABORT_AT_C:
            self._abort("THERMAL_ABORT_TEMPERATURE")
            raise ThermalAbort(self.abort_reason)
        if sample.hardware_thermal_slowdown:
            self._abort("THERMAL_ABORT_HARDWARE_SLOWDOWN")
            raise ThermalAbort(self.abort_reason)
        return sample

    def precheck(self) -> bool:
        """One <=60C, fresh, no-HW-slowdown preflight check.  False means wait."""
        with self._lock:
            if self.state is not ThermalState.PRECHECK:
                raise ThermalGuardError("THERMAL_INVALID_TRANSITION:" + self.state.value + "->PRECHECK")
            sample = self._sample()
            self.precheck_temperature = sample.gpu_temperature_c
            if sample.gpu_temperature_c <= PRECHECK_MAX_C:
                self._transition(ThermalState.RUNNING, allowed=(ThermalState.PRECHECK,))
                return True
            return False

    def wait_for_precheck(self, *, timeout_seconds: float = COOLDOWN_TIMEOUT_SECONDS,
                          poll_seconds: float = 5.0) -> None:
        started = float(self.clock_provider())
        while not self.precheck():
            if float(self.clock_provider()) - started >= timeout_seconds:
                self._abort("THERMAL_PRECHECK_TIMEOUT")
                raise ThermalAbort(self.abort_reason)
            self.sleep_provider(poll_seconds)

    def observe_during_update(self) -> None:
        """Watchdog observation; it never starts an update and never consumes data."""
        with self._lock:
            if self.state is not ThermalState.RUNNING:
                return
            sample = self._sample()
            if sample.gpu_temperature_c >= PAUSE_AT_C:
                self._transition(ThermalState.PAUSE_PENDING, allowed=(ThermalState.RUNNING,))
                self.pause_events.append({"temperature_c": sample.gpu_temperature_c,
                                          "timestamp_monotonic": sample.timestamp_monotonic,
                                          "source": "watchdog", "state": "PAUSE_PENDING"})

    def before_update(self) -> bool:
        """Return permission for precisely the next update, otherwise pause/wait.

        The caller must invoke this before sampler/permutation/RNG consumption.
        """
        with self._lock:
            if self.state is ThermalState.ABORTED:
                raise ThermalAbort(self.abort_reason or "THERMAL_ABORTED")
            if self.state is ThermalState.COMPLETE:
                raise ThermalGuardError("THERMAL_INVALID_TRANSITION:COMPLETE->RUNNING")
            if self.state is ThermalState.PAUSE_PENDING:
                self._transition(ThermalState.PAUSED, allowed=(ThermalState.PAUSE_PENDING,))
            sample = self._sample()
            if self.state is ThermalState.RUNNING:
                if sample.gpu_temperature_c < PAUSE_AT_C:
                    return True
                self._transition(ThermalState.PAUSED, allowed=(ThermalState.RUNNING,))
                self.pause_events.append({"temperature_c": sample.gpu_temperature_c,
                                          "timestamp_monotonic": sample.timestamp_monotonic,
                                          "source": "boundary", "state": "PAUSED"})
                return False
            if self.state is ThermalState.PAUSED:
                if sample.gpu_temperature_c > RESUME_AT_OR_BELOW_C:
                    return False
                self._transition(ThermalState.RESUMING, allowed=(ThermalState.PAUSED,))
                self.resume_events.append({"temperature_c": sample.gpu_temperature_c,
                                           "timestamp_monotonic": sample.timestamp_monotonic,
                                           "state": "RESUMING"})
                self._transition(ThermalState.RUNNING, allowed=(ThermalState.RESUMING,))
                return True
            raise ThermalGuardError("THERMAL_INVALID_STATE:" + self.state.value)

    def wait_for_update_permission(self, *, poll_seconds: float = 1.0) -> None:
        while not self.before_update():
            self.sleep_provider(poll_seconds)

    def complete(self) -> None:
        with self._lock:
            # At the registered final update, a >=80C observation may have
            # already set PAUSE_PENDING.  There is no next update to permit,
            # so completing the fixed 122-update budget is safe without
            # consuming data or overriding a hard-abort condition.
            self._transition(ThermalState.COMPLETE,
                             allowed=(ThermalState.RUNNING, ThermalState.PAUSE_PENDING, ThermalState.PAUSED))

    def receipt(self) -> dict[str, Any]:
        with self._lock:
            return {"policy_version": POLICY_VERSION, "precheck_temperature": self.precheck_temperature,
                    "max_temperature": self.max_temperature, "sample_count": self.sample_count,
                    "pause_count": len(self.pause_events), "pause_events": list(self.pause_events),
                    "resume_events": list(self.resume_events), "sensor_failure_count": self.sensor_failure_count,
                    "stale_count": self.stale_count,
                    "hardware_thermal_slowdown_seen": self.hardware_thermal_slowdown_seen,
                    "software_thermal_slowdown_seen": self.software_thermal_slowdown_seen,
                    "abort_reason": self.abort_reason, "final_state": self.state.value}


class ThermalWatchdog:
    """Strict background observer; any provider error is propagated via guard state."""
    def __init__(self, guard: ThermalGuard, *, interval_seconds: float = WATCHDOG_INTERVAL_SECONDS) -> None:
        self.guard = guard
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self.stop_event.is_set() and self.guard.state not in (ThermalState.ABORTED, ThermalState.COMPLETE):
            try:
                self.guard.observe_during_update()
            except ThermalAbort:
                return
            except BaseException as exc:
                self.guard.sensor_failure_count += 1
                self.guard._abort("THERMAL_WATCHDOG_INTERNAL_FAILURE:" + (str(exc) or type(exc).__name__))
                return
            self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self.thread is not None:
            raise ThermalGuardError("THERMAL_WATCHDOG_ALREADY_STARTED")
        self.thread = threading.Thread(target=self._run, name="phase63-thermal-watchdog", daemon=True)
        self.thread.start()

    def finish(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                self.guard._abort("THERMAL_WATCHDOG_JOIN_TIMEOUT")
                raise ThermalAbort(self.guard.abort_reason)
        return self.guard.receipt()


def production_sample_provider() -> Mapping[str, Any]:
    """Future live adapter only.  K6 CPU validation never calls this function."""
    from training.run_foundation_v35_thermal_gate import query_gpu
    return {"timestamp_monotonic": time.monotonic(), **query_gpu()}
