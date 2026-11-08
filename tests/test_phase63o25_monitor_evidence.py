"""CPU-only audit-evidence tests; monitor admission policy is intentionally unchanged."""
from __future__ import annotations

import json

import pytest

from training import gpu_execution_guard as guard
from training.gpu_execution_lock import atomic_json


NOW = 1_800_000_100.0
HASH = "a" * 64


def identity(pid: int, start: float) -> dict:
    return {"pid": pid, "process_start_time": start, "hostname": "test-host",
            "exe": rf"C:\\Program Files\\process-{pid}.exe", "exe_sha256": HASH}


def row(pid: int) -> dict:
    return {"pid": pid, "type": "C+G", "process_name": rf"C:\\Program Files\\process-{pid}.exe",
            "cuda_device": "GPU-test"}


def context(child: dict | None) -> dict:
    return {"monitor_phase": "POST_CHILD_RESUME", "owner": identity(10, 10.0), "child": child}


def test_child_self_remains_blocking_but_is_unambiguously_auditable():
    child = identity(71, 71.0); saved = []
    monitor = guard.ExternalComputeMonitor(lambda: [row(71)], lambda _: child, now=lambda: NOW,
                                           violation_recorder=saved.append)
    with pytest.raises(guard.GuardBlocked, match="PROCESS_PRESENT"):
        monitor.observe(context=context(child))
    record = saved[0]
    assert record["execution_child_identity"] == child
    assert guard.identities_match(record["violating_processes"][0]["identity"], child)
    assert record["monitor_phase"] == "POST_CHILD_RESUME"


def test_separate_external_process_is_auditable_as_not_the_child():
    child, external = identity(71, 71.0), identity(72, 72.0); saved = []
    monitor = guard.ExternalComputeMonitor(lambda: [row(72)], lambda _: external, now=lambda: NOW,
                                           violation_recorder=saved.append)
    with pytest.raises(guard.GuardBlocked, match="PROCESS_PRESENT"):
        monitor.observe(context=context(child))
    assert saved[0]["violating_processes"][0]["identity"] != saved[0]["execution_child_identity"]


def test_atomic_violation_evidence_write_and_collision_fail_closed(tmp_path):
    target = tmp_path / "external-monitor-violation.json"; current = identity(71, 71.0)
    monitor = guard.ExternalComputeMonitor(lambda: [row(71)], lambda _: current, now=lambda: NOW,
                                           violation_recorder=lambda value: atomic_json(target, value))
    with pytest.raises(guard.GuardBlocked, match="PROCESS_PRESENT"):
        monitor.observe(context=context(None))
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert guard.identities_match(stored["violating_processes"][0]["identity"], current)
    # A second write never replaces the first immutable record; it fails closed.
    with pytest.raises(guard.GuardBlocked, match="EVIDENCE_WRITE_FAILED"):
        monitor.observe(context=context(None))


def test_evidence_writer_failure_fails_closed_without_relaxing_monitor_policy():
    current = identity(71, 71.0)
    def broken(_):
        raise OSError("disk full")
    monitor = guard.ExternalComputeMonitor(lambda: [row(71)], lambda _: current, now=lambda: NOW,
                                           violation_recorder=broken)
    with pytest.raises(guard.GuardBlocked, match="EVIDENCE_WRITE_FAILED"):
        monitor.observe(context=context(None))
