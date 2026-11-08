"""CPU-only proof tests for the PHASE63 pre-child aborted-lock lifecycle."""
from __future__ import annotations

import copy
import hashlib
import json
import socket
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from training import exclusive_cuda_runner as runner
from training import gpu_execution_lock as lock
from training import checkpoint_paths


def put(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pre_child_fixture(tmp_path: Path, *, explicit: bool = True):
    runtime = tmp_path / "runtime"; lock_path = runtime / "gpu.lock"; run_id = "phase63-no_train-seed42-control"
    owner = {"pid": 101, "parent_pid": 1, "process_start_time": 10.0, "hostname": socket.gethostname(), "exe": "C:\\Python\\python.exe",
             "schema_version": "gpu-owner-v1", "phase": 63, "run_id": run_id, "cuda_device": "cuda:0", "repo_head": "h" * 40,
             "checkpoint_parent_sha256": "p" * 64, "created_timestamp": 11.0, "command_hash": "c" * 64, "nonce": "n", "job_name": "job"}
    put(lock_path, owner); run = runtime / "runs" / run_id
    reason = "ExecutionBlocked('CPU_HEAVY_JOB_PRESENT')"
    put(run / "01-LOCKED.json", {"state": "LOCKED", "at": 12.0, "owner": owner})
    put(run / "prelaunch.json", {"inventory": [], "temperature": {"temperature_c": 36, "hardware_thermal_slowdown": False},
                                  "cpu_heavy_warnings": [{"pid": 88, "command": "playwright"}], "cwd": "C:\\repo",
                                  "command": ["python", "worker.py"], "kind": "cuda-no-train"})
    if explicit:
        put(run / "pre-child-abort.json", {"schema": "phase63-pre-child-abort-v1",
            "child_creation_state": "CHILD_NOT_CREATED_PROVEN", "abort_boundary": "BEFORE_CHILD_PROCESS_CREATION",
            "reason": reason, "owner": owner, "cuda_state": "NOT_CREATED", "model_load": "NOT_STARTED",
            "forward_calls": 0, "backward_calls": 0, "optimizer_steps": 0, "checkpoint_writes": 0, "training": "NO"})
    put(run / "aborted.json", {"status": "INCOMPLETE", "ownership": "ABORTED", "reason": reason,
                                "owner": owner, "child": None, "lock_retained": True, "automatic_terminate": False})
    checkpoint_root = tmp_path / "checkpoints"
    expected = checkpoint_root / "experimental" / "phase63" / "continuation-stability" / "control" / "seed-42" / "checkpoint-tokens-16446464.pt"
    binding = tmp_path / "binding.json"; binding_value = {"checkpoint_root": str(checkpoint_root), "implementation_source_commit": "h" * 40,
        "source_manifest_sha256": "f" * 64, "source_manifest": [{"path": "training/phase63_worker.py", "sha256": "e" * 64}],
        "runs": [{"seed": 42, "arm": "control", "output_path": str(expected)}]}
    put(binding, binding_value)
    old = tmp_path / "old-auth.json"; old_value = {"authorization_id": "old", "binding_sha256": digest(binding), "phase": 63,
        "action": "no_train", "allowed_runs": [{"seed": 42, "arm": "control"}]}; put(old, old_value)
    put(old.with_name(old.name + ".phase63-claim.json"), {"claim": {"authorization_sha256": digest(old), "authorization_id": "old",
        "binding_sha256": digest(binding), "phase": 63, "action": "no_train", "allowed_runs": [{"seed": 42, "arm": "control"}],
        "execution_instance": run_id}})
    return lock_path, old, binding, {101: None}


def assess(lock_path, old, binding, identities, *, job=True):
    return lock.aborted_lock_recovery_proof(lock_path, execution_authorization_path=old, binding_path=binding,
                                             identity_provider=lambda pid: identities.get(pid), job_clear_provider=lambda _: job)


def recovery_authorization(tmp_path: Path, proof: dict) -> Path:
    now = datetime.now(UTC); path = tmp_path / "recovery-auth.json"
    execution = {key: proof[key] for key in ("run_id", "binding_path", "binding_sha256", "implementation_head",
        "source_manifest_sha256", "execution_authorization_path", "execution_authorization_sha256",
        "execution_claim_path", "execution_claim_sha256")}
    execution = {key.removeprefix("execution_"): value for key, value in execution.items()}
    if proof.get("recovery_lifecycle") == "PRE_CHILD_ABORTED":
        evidence = {key: proof[key] for key in ("aborted_path", "aborted_sha256", "child_created_path",
            "child_creation_state", "pre_child_evidence_path", "pre_child_evidence_sha256",
            "pre_child_evidence_kind", "failure_code")}
    else:
        evidence = {key: proof[key] for key in ("aborted_path", "aborted_sha256", "child_created_path",
            "child_created_sha256", "stderr_path", "stderr_sha256", "stdout_path", "stdout_sha256", "failure_code")}
    put(path, {"schema_version": lock.RECOVERY_AUTHORIZATION_SCHEMA, "authorization_id": "recover-1",
        "purpose": "lock_recovery", "scope": lock.RECOVERY_SCOPE,
        "valid_from": (now - timedelta(seconds=1)).isoformat(), "valid_until": (now + timedelta(minutes=5)).isoformat(),
        "target_lock": {"path": proof["lock_path"], "sha256": proof["lock_sha256"], "owner_sha256": proof["lock_owner_sha256"]},
        "execution": execution, "evidence": evidence})
    return path


def test_explicit_pre_child_abort_can_be_finalized_once(tmp_path):
    lock_path, old, binding, identities = pre_child_fixture(tmp_path); proof = assess(lock_path, old, binding, identities)
    assert proof["recovery_lifecycle"] == "PRE_CHILD_ABORTED"
    assert proof["child_creation_state"] == "CHILD_NOT_CREATED_PROVEN"
    auth = recovery_authorization(tmp_path, proof)
    result = lock.finalize_aborted_lock_recovery(lock_path, auth, execution_authorization_path=old, binding_path=binding,
                                                  identity_provider=lambda pid: identities.get(pid), job_clear_provider=lambda _: True)
    assert result["status"] == "FINALIZED_ABORTED" and not lock_path.exists()
    with pytest.raises(lock.ExecutionBlocked, match="ALREADY_CONSUMED"):
        lock.claim_recovery_authorization(auth, json.loads(auth.read_text()), digest(auth))


def test_legacy_pre_child_lifecycle_is_not_a_missing_child_shortcut(tmp_path):
    lock_path, old, binding, identities = pre_child_fixture(tmp_path, explicit=False)
    proof = assess(lock_path, old, binding, identities)
    assert proof["pre_child_evidence_kind"] == "LEGACY_LIFECYCLE_EQUIVALENT"
    (lock_path.parent / "runs" / proof["run_id"] / "prelaunch.json").unlink()
    with pytest.raises(lock.ExecutionBlocked, match="PRE_CHILD_LAUNCH_UNREADABLE"):
        assess(lock_path, old, binding, identities)


@pytest.mark.parametrize("field, value", [
    ("child_creation_state", "UNKNOWN"), ("cuda_state", "CREATED"), ("forward_calls", 1),
    ("optimizer_steps", 1), ("checkpoint_writes", 1), ("training", "YES"),
])
def test_explicit_pre_child_side_effect_or_unknown_state_is_rejected(tmp_path, field, value):
    lock_path, old, binding, identities = pre_child_fixture(tmp_path)
    path = lock_path.parent / "runs" / "phase63-no_train-seed42-control" / "pre-child-abort.json"
    value_json = json.loads(path.read_text()); value_json[field] = value; put(path, value_json)
    with pytest.raises(lock.ExecutionBlocked, match="PRE_CHILD_STATE_INVALID"):
        assess(lock_path, old, binding, identities)


def test_child_record_present_or_live_owner_or_nonclear_job_is_rejected(tmp_path):
    lock_path, old, binding, identities = pre_child_fixture(tmp_path)
    run = lock_path.parent / "runs" / "phase63-no_train-seed42-control"
    put(run / "child-created.json", {"child": {"pid": 102}, "suspended": True})
    with pytest.raises(lock.ExecutionBlocked, match="CHILD_RECORD_PRESENT"):
        assess(lock_path, old, binding, identities)
    (run / "child-created.json").unlink()
    owner = json.loads(lock_path.read_text()); identities[101] = copy.deepcopy(owner)
    with pytest.raises(lock.ExecutionBlocked, match="OWNER_STILL_LIVE"):
        assess(lock_path, old, binding, identities)
    identities[101] = None
    with pytest.raises(lock.ExecutionBlocked, match="JOB_NOT_CLEAR"):
        assess(lock_path, old, binding, identities, job=False)


def test_unconsumed_auth_and_mutation_guard_hold_reject(tmp_path, monkeypatch):
    lock_path, old, binding, identities = pre_child_fixture(tmp_path)
    old.with_name(old.name + ".phase63-claim.json").unlink()
    with pytest.raises(lock.ExecutionBlocked, match="EXECUTION_CLAIM_UNREADABLE"):
        assess(lock_path, old, binding, identities)
    lock_path, old, binding, identities = pre_child_fixture(tmp_path / "again")
    auth = recovery_authorization(tmp_path / "again", assess(lock_path, old, binding, identities))
    @contextmanager
    def held(_):
        raise lock.ExecutionBlocked("GPU_LOCK_HELD")
        yield
    monkeypatch.setattr(lock, "mutation_guard", held)
    with pytest.raises(lock.ExecutionBlocked, match="GPU_LOCK_HELD"):
        lock.finalize_aborted_lock_recovery(lock_path, auth, execution_authorization_path=old, binding_path=binding,
                                             identity_provider=lambda pid: identities.get(pid), job_clear_provider=lambda _: True)


def test_existing_child_exited_o5l_fixture_regression(tmp_path):
    from tests.test_phase63o5l_lock_recovery import fixture
    lock_path, old, binding, _auth, identities = fixture(tmp_path)
    proof = assess(lock_path, old, binding, identities)
    assert "recovery_lifecycle" not in proof and proof["failure_code"] == "BINDING_OUTPUT_MISMATCH"


def test_execute_records_explicit_pre_child_proof(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"; root.mkdir(); runtime = tmp_path / "runtime"
    monkeypatch.setenv("UNIPILOT_CHECKPOINT_ROOT", str(root))
    monkeypatch.setattr(checkpoint_paths, "checkpoint_root", lambda _: root)
    monkeypatch.setattr(runner, "heavy_jobs", lambda: [{"pid": 7, "command": "playwright"}])
    class DummyLock:
        def __init__(self, *_args, **_kwargs): self.owner = {"test": "owner"}; self.path = runtime / "gpu.lock"
        def acquire(self): pass
    monkeypatch.setattr(runner, "GPULock", DummyLock)
    class Monitor:
        def observe(self, *, context=None): return {"rows": []}
    with pytest.raises(lock.ExecutionBlocked, match="CPU_HEAVY_JOB_PRESENT"):
        runner.execute(["test"], cwd=tmp_path, root=root, runtime=runtime, run_id="test", phase=63,
                       kind="cuda-no-train", device="cuda:0", repo_head="h", parent_sha="p", verifier=lambda: {},
                       temperature=lambda: {"temperature_c": 36, "hardware_thermal_slowdown": False}, outputs=[root / "out"],
                       compute_monitor=Monitor())
    record = json.loads((runtime / "runs" / "test" / "pre-child-abort.json").read_text())
    assert record["child_creation_state"] == "CHILD_NOT_CREATED_PROVEN" and record["cuda_state"] == "NOT_CREATED"
