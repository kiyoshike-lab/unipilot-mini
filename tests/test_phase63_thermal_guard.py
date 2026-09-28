"""CPU-only synthetic contract tests for the PHASE63 thermal runtime guard.

These tests use no CUDA tensors, model, nvidia-smi call, lock, worker launch,
optimizer, or checkpoint.  They exercise only injected samples and clocks.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from training.phase63_thermal_guard import (ABORT_AT_C, PAUSE_AT_C, PRECHECK_MAX_C,
                                             RESUME_AT_OR_BELOW_C, ThermalAbort,
                                             ThermalGuard, ThermalGuardError, ThermalState)


class Feed:
    def __init__(self, values, *, now=100.0):
        self.values = list(values)
        self.now = now
        self.sleeps = []

    def provider(self):
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, dict):
            self.now += 0.1
            return value
        sample = {"timestamp_monotonic": self.now, "gpu_temperature_c": value,
                  "hardware_thermal_slowdown": False, "software_thermal_slowdown": False}
        self.now += 0.1
        return sample

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def guard(self):
        return ThermalGuard(self.provider, clock_provider=self.clock, sleep_provider=self.sleep)


def abort(guard, operation):
    with pytest.raises(ThermalAbort):
        operation()
    assert guard.state is ThermalState.ABORTED


@pytest.mark.parametrize("temperature, expected", [(59.0, ThermalState.RUNNING),
                                                     (PRECHECK_MAX_C, ThermalState.RUNNING),
                                                     (61.0, ThermalState.PRECHECK)])
def test_abc_precheck_exact_boundary_and_cooldown_wait(temperature, expected):
    feed = Feed([temperature]); guard = feed.guard()
    assert guard.precheck() is (expected is ThermalState.RUNNING)
    assert guard.state is expected


@pytest.mark.parametrize("temperature, permitted, state", [(79.9, True, ThermalState.RUNNING),
                                                              (PAUSE_AT_C, False, ThermalState.PAUSED),
                                                              (84.9, False, ThermalState.PAUSED)])
def test_def_running_pause_boundaries(temperature, permitted, state):
    feed = Feed([59, temperature]); guard = feed.guard(); assert guard.precheck()
    assert guard.before_update() is permitted
    assert guard.state is state


def test_gh_abort_exact_boundary_and_hardware_slowdown():
    for value in (ABORT_AT_C, {"timestamp_monotonic": 100, "gpu_temperature_c": 50,
                               "hardware_thermal_slowdown": True, "software_thermal_slowdown": False}):
        feed = Feed([59, value]); guard = feed.guard(); assert guard.precheck(); abort(guard, guard.before_update)


@pytest.mark.parametrize("temperature, permitted", [(70.0, False), (RESUME_AT_OR_BELOW_C, True), (64.0, True)])
def test_ijk_paused_waits_then_resumes_at_exact_threshold(temperature, permitted):
    feed = Feed([59, 80, temperature]); guard = feed.guard(); assert guard.precheck(); assert not guard.before_update()
    assert guard.before_update() is permitted
    assert guard.state is (ThermalState.RUNNING if permitted else ThermalState.PAUSED)


@pytest.mark.parametrize("invalid", [OSError("provider"), TimeoutError("timeout"), math.nan, math.inf,
                                     {"timestamp_monotonic": 100, "hardware_thermal_slowdown": False,
                                      "software_thermal_slowdown": False},
                                     {"timestamp_monotonic": 100, "gpu_temperature_c": -1,
                                      "hardware_thermal_slowdown": False, "software_thermal_slowdown": False},
                                     {"timestamp_monotonic": 100, "gpu_temperature_c": 50,
                                      "hardware_thermal_slowdown": "false", "software_thermal_slowdown": False},
                                     {}])
def test_lmnop_and_malformed_sensor_values_abort_fail_closed(invalid):
    feed = Feed([invalid]); guard = feed.guard(); abort(guard, guard.precheck)
    receipt = guard.receipt(); assert receipt["sensor_failure_count"] == 1


def test_q_stale_sample_aborts_and_is_counted():
    feed = Feed([{"timestamp_monotonic": 90, "gpu_temperature_c": 50,
                  "hardware_thermal_slowdown": False, "software_thermal_slowdown": False}], now=100)
    guard = feed.guard(); abort(guard, guard.precheck)
    assert guard.receipt()["stale_count"] == 1


def test_r_timestamp_regression_aborts():
    feed = Feed([{"timestamp_monotonic": 100, "gpu_temperature_c": 59,
                  "hardware_thermal_slowdown": False, "software_thermal_slowdown": False},
                 {"timestamp_monotonic": 99, "gpu_temperature_c": 59,
                  "hardware_thermal_slowdown": False, "software_thermal_slowdown": False}])
    guard = feed.guard(); assert guard.precheck(); abort(guard, guard.before_update)


def test_repeated_timestamp_is_stale_fail_closed():
    same = {"timestamp_monotonic": 100, "gpu_temperature_c": 59,
            "hardware_thermal_slowdown": False, "software_thermal_slowdown": False}
    feed = Feed([same, same]); guard = feed.guard(); assert guard.precheck(); abort(guard, guard.before_update)
    assert guard.receipt()["stale_count"] == 1


def test_stu_watchdog_mid_update_pause_or_abort_request():
    feed = Feed([59, 80]); guard = feed.guard(); assert guard.precheck(); guard.observe_during_update()
    assert guard.state is ThermalState.PAUSE_PENDING
    feed = Feed([59, 85]); guard = feed.guard(); assert guard.precheck(); abort(guard, guard.observe_during_update)
    feed = Feed([59, OSError("lost")]); guard = feed.guard(); assert guard.precheck(); abort(guard, guard.observe_during_update)


def test_vwx_pause_does_not_consume_fake_optimizer_or_example_and_resume_keeps_identity():
    feed = Feed([59, 72, 80, 78, 68, 65, 70]); guard = feed.guard(); assert guard.precheck()
    steps = examples = 0; next_update_identity = "update-32001"
    permitted = []
    for _ in range(6):
        allowed = guard.before_update(); permitted.append(allowed)
        if allowed:
            assert next_update_identity == "update-32001" or next_update_identity == "update-32002"
            steps += 1; examples += 1; next_update_identity = "update-32002"
    assert permitted == [True, False, False, False, True, True]
    assert steps == examples == 3
    assert guard.receipt()["pause_count"] == 1
    assert len(guard.receipt()["resume_events"]) == 1


def test_yz_abort_and_complete_cannot_return_to_running():
    feed = Feed([59, 85]); guard = feed.guard(); assert guard.precheck(); abort(guard, guard.before_update)
    with pytest.raises(ThermalAbort): guard.before_update()
    feed = Feed([59]); guard = feed.guard(); assert guard.precheck(); guard.complete()
    with pytest.raises(ThermalGuardError, match="COMPLETE"):
        guard.before_update()


def test_final_pause_may_complete_without_permitting_another_update():
    feed = Feed([59, 80]); guard = feed.guard(); assert guard.precheck(); assert not guard.before_update()
    guard.complete()
    assert guard.receipt()["final_state"] == "COMPLETE"


def test_wait_for_precheck_uses_existing_300_second_timeout_without_data_consumption():
    feed = Feed([61] * 62); guard = feed.guard()
    with pytest.raises(ThermalAbort, match="THERMAL_PRECHECK_TIMEOUT"):
        guard.wait_for_precheck()
    # The safety timeout is measured by the injected monotonic clock; provider
    # calls themselves may consume time, so sleep alone is not the clock.
    assert 300 <= feed.now - 100 < 306
    assert guard.sample_count >= 60


def test_receipt_records_software_slowdown_without_changing_abort_policy():
    sample = {"timestamp_monotonic": 100, "gpu_temperature_c": 59,
              "hardware_thermal_slowdown": False, "software_thermal_slowdown": True}
    feed = Feed([sample, 70]); guard = feed.guard(); assert guard.precheck(); assert guard.before_update()
    receipt = guard.receipt(); assert receipt["software_thermal_slowdown_seen"] is True
    assert receipt["abort_reason"] is None


def test_worker_integration_is_boundary_ordered_and_legacy_monitor_is_not_live_path():
    source = (Path(__file__).parents[1] / "training" / "phase63_worker.py").read_text(encoding="utf-8")
    assert "from training.run_foundation_v35_thermal_gate import" not in source
    assert source.index("thermal.wait_for_precheck()") < source.index("torch.cuda.is_available()")
    assert source.index("thermal.wait_for_update_permission()") < source.index("macro_batch(train_data")
    disk_section = source[source.index("def disk_guard"):source.index("def thermal_guard")]
    assert "THERMAL" not in disk_section and "query_gpu" not in disk_section
