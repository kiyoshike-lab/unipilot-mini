"""CPU-only tests for the PHASE63J guard revision.

No test in this module imports a model, calls nvidia-smi, acquires a real
runtime lock, starts a worker, or invokes torch.cuda.
"""
from __future__ import annotations

import copy
import json
import socket

import pytest

from training import gpu_execution_guard as guard
from training import phase63_worker as worker


NOW = 1_800_000_000.0
HASH_A = "a" * 64
HASH_B = "b" * 64


def process(pid=71, start=100.0, digest=HASH_A, exe=r"C:\\Program Files\\example.exe"):
    return {"pid": pid, "process_start_time": start, "hostname": "test-host", "exe": exe,
            "exe_sha256": digest}


def approval(current=None, **changes):
    value = {"schema": guard.APPROVAL_SCHEMA, "approval_id": "C63J-01-test", "purpose": guard.NO_TRAIN_ACTION,
             "process_identity": process() if current is None else current, "created_at": NOW - 10,
             "expires_at": NOW + 10, "allowed_phase": 63, "allowed_action": guard.NO_TRAIN_ACTION,
             "single_use": True, "evidence_reference": "evaluation/phase63/current-review.json",
             "gpu_uuid": "GPU-test"}
    value.update(changes)
    return value


def row(kind="C+G", pid=71):
    return {"pid": pid, "type": kind, "process_name": r"C:\\Program Files\\example.exe",
            "cuda_device": "GPU-test"}


def monitor(rows, identities, approvals=()):
    return guard.ExternalComputeMonitor(lambda: rows, lambda pid: identities.get(pid), approvals=approvals,
                                        now=lambda: NOW)


def test_exact_identity_requires_pid_start_host_exe_and_hash():
    original = process()
    assert guard.identities_match(original, process())
    for key, value in (("pid", 72), ("process_start_time", 101.0), ("hostname", "other"),
                       ("exe", r"C:\\other.exe"), ("exe_sha256", HASH_B)):
        changed = {**original, key: value}
        assert not guard.identities_match(original, changed)


def test_approval_is_exact_single_use_no_train_and_expires():
    assert guard.validate_approval(approval(), process(), now=NOW, gpu_uuid="GPU-test")["approval_id"] == "C63J-01-test"
    cases = [
        {"expires_at": NOW}, {"created_at": NOW + 1}, {"single_use": False}, {"consumed": True},
        {"revoked": True}, {"purpose": "training"}, {"allowed_action": "training"},
        {"allowed_phase": 64}, {"process_identity": process(digest=HASH_B)},
    ]
    for changes in cases:
        with pytest.raises(guard.GuardBlocked):
            guard.validate_approval(approval(**changes), process(), now=NOW, gpu_uuid="GPU-test")


def test_monitor_accepts_only_current_approved_c_plus_g_and_pure_graphics():
    assert monitor([row("G")], {}).observe()["approved"] == []
    result = monitor([row()], {71: process()}, [approval()]).observe()
    assert result["approved"][0]["approval"]["approval_id"] == "C63J-01-test"


@pytest.mark.parametrize("rows,identities,approvals", [
    ([row("C")], {71: process()}, [approval()]),  # pure compute never exception exempt
    ([row()], {71: process(digest=HASH_B)}, [approval()]),  # hash/PID-reuse identity mismatch
    ([row()], {71: process(start=101)}, [approval()]),  # process restart
    ([row()], {71: None}, [approval()]),  # disappears after inventory
    ([row()], {71: process()}, [approval(expires_at=NOW)]),  # approval expiry
    ([row()], {71: process()}, [approval(purpose="training")]),  # purpose mismatch
    ([row(), row(pid=72)], {71: process(), 72: process(pid=72)}, [approval()]),  # new C+G after approval
])
def test_monitor_fails_closed_for_unapproved_or_racy_compute(rows, identities, approvals):
    with pytest.raises(guard.GuardBlocked):
        monitor(rows, identities, approvals).observe()


def test_monitor_query_failure_and_ambiguous_row_fail_closed():
    broken = guard.ExternalComputeMonitor(lambda: (_ for _ in ()).throw(OSError("offline")), lambda pid: process())
    with pytest.raises(guard.GuardBlocked, match="QUERY_FAILED"):
        broken.observe()
    with pytest.raises(guard.GuardBlocked, match="AMBIGUOUS"):
        monitor([{"pid": 71, "type": "M"}], {71: process()}).observe()


def test_monitor_detects_toctou_new_process_after_initial_approval():
    snapshots = [[row()], [row(), row(pid=72)]]
    identities = {71: process(), 72: process(pid=72)}
    value = guard.ExternalComputeMonitor(lambda: snapshots.pop(0), lambda pid: identities[pid], approvals=[approval()], now=lambda: NOW)
    assert value.observe()["approved"]
    with pytest.raises(guard.GuardBlocked, match="PROCESS_PRESENT"):
        value.observe()


def owner():
    return {"pid": 10, "process_start_time": 10.0, "hostname": "test-host", "exe": r"C:\\runner.exe",
            "schema_version": "gpu-owner-v1", "phase": 63, "run_id": "phase63-no-train", "nonce": "nonce",
            "job_name": "Local\\UniPilot-test"}


def lock_assertion(tmp_path, value=None, **kwargs):
    path = tmp_path / "gpu.lock"
    path.write_text(json.dumps(owner() if value is None else value), encoding="utf-8")
    defaults = {"expected_nonce": "nonce", "expected_run_id": "phase63-no-train", "expected_phase": 63,
                "owner_identity_provider": lambda pid: {**owner(), "exe_sha256": HASH_A},
                "parent_identity_provider": lambda: {**owner(), "exe_sha256": HASH_A},
                "job_membership_provider": lambda job: True}
    defaults.update(kwargs)
    return guard.assert_live_lock_owner(path, **defaults)


def test_lock_assertion_requires_live_exact_owner_parent_nonce_and_job(tmp_path):
    assert lock_assertion(tmp_path)["nonce"] == "nonce"
    failures = [
        {"expected_nonce": "wrong"}, {"expected_run_id": "other"}, {"expected_phase": 64},
        {"owner_identity_provider": lambda pid: {**owner(), "pid": 99, "exe_sha256": HASH_A}},
        {"owner_identity_provider": lambda pid: {**owner(), "process_start_time": 99, "exe_sha256": HASH_A}},
        {"owner_identity_provider": lambda pid: {**owner(), "hostname": "other", "exe_sha256": HASH_A}},
        {"parent_identity_provider": lambda: {**owner(), "exe": r"C:\\other.exe", "exe_sha256": HASH_A}},
        {"job_membership_provider": lambda job: False},
    ]
    for changes in failures:
        with pytest.raises(guard.GuardBlocked):
            lock_assertion(tmp_path, **changes)


@pytest.mark.parametrize("content", [None, "not json", {"nonce": "nonce"}])
def test_lock_missing_stale_or_malformed_fails_closed(tmp_path, content):
    path = tmp_path / "missing.lock"
    if content is not None:
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    with pytest.raises(guard.GuardBlocked):
        guard.assert_live_lock_owner(path, expected_nonce="nonce", expected_run_id="run", expected_phase=63,
                                      owner_identity_provider=lambda pid: None,
                                      parent_identity_provider=lambda: None, job_membership_provider=lambda job: False)


def receipt():
    return {"schema": "phase63-no-train-receipt-v1", "kind": guard.NO_TRAIN_ACTION, "status": "COMPLETE",
            "states": list(guard.NO_TRAIN_STATES), "exit_code": 0, "optimizer_steps": 0, "backward_calls": 0,
            "checkpoint_count": 0, "training_examples_consumed": 0, "child_tree_exited": True,
            "stdout_closed": True, "stderr_closed": True, "gpu_context_released": True,
            "lock_lifecycle_complete": True, "approved_process_identities_valid": True,
            "no_new_unapproved_compute_process": True, "owner_nonce": "nonce", "completed_at": NOW,
            "lease_observed_at": NOW, "worker_sha256": HASH_A, "validator_sha256": HASH_A,
            "supervisor_sha256": HASH_A, "guard_sha256": HASH_A, "command_sha256": HASH_A,
            "parent_sha256": HASH_A, "journal_sha256": HASH_A, "device": "cuda:0",
            "streams": {"stdout_sha256": HASH_A, "stderr_sha256": HASH_A}}


def test_no_train_receipt_strict_zero_step_contract():
    assert guard.validate_no_train_receipt(receipt(), expected_owner_nonce="nonce")
    changes = [
        ("optimizer_steps", 1), ("backward_calls", 1), ("checkpoint_count", 1),
        ("training_examples_consumed", 1), ("exit_code", 1), ("states", ["PREPARED"]),
        ("owner_nonce", "other"), ("completed_at", float("nan")), ("worker_sha256", "short"),
        ("streams", {"stdout_sha256": HASH_A}), ("child_tree_exited", False),
    ]
    for key, changed in changes:
        data = receipt(); data[key] = changed
        assert not guard.validate_no_train_receipt(data, expected_owner_nonce="nonce")


def test_worker_guard_rejects_direct_or_replayed_environment_before_cuda(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"; path.write_text(json.dumps(owner()), encoding="utf-8")
    monkeypatch.setenv("UNIPILOT_GPU_LOCK_PATH", str(path))
    monkeypatch.setenv("UNIPILOT_GPU_OWNER_NONCE", "nonce")
    monkeypatch.setenv("UNIPILOT_GPU_RUN_ID", "phase63-no-train")
    assert worker.require_live_gpu_owner({}, owner_identity_provider=lambda pid: {**owner(), "exe_sha256": HASH_A},
                                         parent_identity_provider=lambda: {**owner(), "exe_sha256": HASH_A},
                                         job_membership_provider=lambda job: True)["run_id"] == "phase63-no-train"
    monkeypatch.setenv("UNIPILOT_GPU_OWNER_NONCE", "replayed")
    with pytest.raises(RuntimeError, match="LOCK_SCOPE"):
        worker.require_live_gpu_owner({}, owner_identity_provider=lambda pid: {**owner(), "exe_sha256": HASH_A},
                                      parent_identity_provider=lambda: {**owner(), "exe_sha256": HASH_A},
                                      job_membership_provider=lambda job: True)


def test_parent_disappearance_is_an_explicit_fail_closed_condition():
    # The supervisor maps this condition to HOST_PARENT_EXITED_LOCK_RETAINED;
    # a detached host may never be treated as an implicit live parent.
    assert not guard.identities_match(process(), None)
