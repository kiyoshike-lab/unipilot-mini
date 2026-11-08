"""CPU-only proofs for post-child, compute-unknown aborted-lock recovery."""
from __future__ import annotations

import copy
import hashlib
import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from training import gpu_execution_lock as lock


def put(path: Path, value: dict | str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path):
    runtime = tmp_path / "runtime"; run_id = "phase63-no_train-seed42-control"; lock_path = runtime / "gpu.lock"
    parent = tmp_path / "parents" / "seed42.pt"; put(parent, b"known-parent"); parent_sha = digest(parent)
    owner = {"pid": 101, "parent_pid": 1, "process_start_time": 10.0, "hostname": socket.gethostname(),
             "exe": r"C:\\Python\\python.exe", "schema_version": "gpu-owner-v1", "phase": 63,
             "run_id": run_id, "cuda_device": "cuda:0", "repo_head": "h" * 40,
             "checkpoint_parent_sha256": parent_sha, "created_timestamp": 11.0,
             "command_hash": "c" * 64, "nonce": "n", "job_name": "job"}
    child = {"pid": 102, "parent_pid": 101, "process_start_time": 12.0, "hostname": socket.gethostname(),
             "exe": r"C:\\Python\\python.exe"}
    put(lock_path, owner); run = runtime / "runs" / run_id
    put(run / "child-created.json", {"child": child, "suspended": True})
    put(run / "02-RUNNING.json", {"state": "RUNNING", "at": 13.0, "child": child, "job_object_assigned": True})
    put(run / "aborted.json", {"status": "INCOMPLETE", "ownership": "ABORTED",
        "reason": "ExecutionBlocked('EXTERNAL_COMPUTE_PROCESS_PRESENT')", "owner": owner, "child": child,
        "lock_retained": True, "automatic_terminate": False})
    root = tmp_path / "checkpoints"
    output = root / "experimental" / "phase63" / "continuation-stability" / "control" / "seed-42" / "checkpoint-tokens-16446464.pt"
    binding = tmp_path / "binding.json"
    binding_value = {"checkpoint_root": str(root), "implementation_source_commit": "h" * 40,
        "source_manifest_sha256": "f" * 64, "source_manifest": [{"path": "training/phase63_worker.py", "sha256": "e" * 64}],
        "runs": [{"ordinal": 1, "seed": 42, "arm": "control", "runtime_lr": 5e-5, "optimizer_updates": 0,
                  "parent_path": str(parent), "parent_sha256": parent_sha, "output_path": str(output)}]}
    put(binding, binding_value)
    execution = tmp_path / "old-auth.json"
    auth = {"authorization_id": "old", "binding_sha256": digest(binding), "phase": 63, "action": "no_train",
            "allowed_runs": [{"seed": 42, "arm": "control"}]}
    put(execution, auth)
    put(execution.with_name(execution.name + ".phase63-claim.json"), {"claim": {
        "schema_version": "phase63-authorization-claim-v1", "authorization_sha256": digest(execution),
        "authorization_id": "old", "binding_sha256": digest(binding),
        "phase": 63, "action": "no_train", "allowed_runs": [{"seed": 42, "arm": "control"}],
        "execution_instance": run_id}, "claimed_at_utc": "2026-01-01T00:00:00+00:00"})
    return lock_path, execution, binding, {101: None, 102: None}, output, parent


def proof(lock_path, execution, binding, identities, *, job=True):
    return lock.aborted_lock_recovery_proof(lock_path, execution_authorization_path=execution, binding_path=binding,
                                             identity_provider=lambda pid: identities.get(pid), job_clear_provider=lambda _: job)


def recovery_authorization(tmp_path: Path, value: dict) -> Path:
    now = datetime.now(UTC); path = tmp_path / "recovery-auth.json"
    execution = {key.removeprefix("execution_"): value[key] for key in (
        "run_id", "binding_path", "binding_sha256", "implementation_head", "source_manifest_sha256",
        "execution_authorization_path", "execution_authorization_sha256", "execution_claim_path", "execution_claim_sha256")}
    evidence = {key: value[key] for key in (
        "aborted_path", "aborted_sha256", "child_created_path", "child_created_sha256", "running_path", "running_sha256",
        "failure_code", "canonical_output_path", "canonical_tmp_output_path", "declared_persistent_output_paths",
        "parent_checkpoint_path", "parent_checkpoint_sha256")}
    put(path, {"schema_version": lock.RECOVERY_AUTHORIZATION_SCHEMA, "authorization_id": "recover-unknown-1",
        "purpose": "lock_recovery", "scope": lock.RECOVERY_SCOPE,
        "valid_from": (now - timedelta(seconds=1)).isoformat(), "valid_until": (now + timedelta(minutes=5)).isoformat(),
        "target_lock": {"path": value["lock_path"], "sha256": value["lock_sha256"], "owner_sha256": value["lock_owner_sha256"]},
        "execution": execution, "evidence": evidence})
    return path


def test_post_child_unknown_compute_can_finalize_once_without_rewriting_history(tmp_path):
    lock_path, execution, binding, identities, _output, _parent = fixture(tmp_path)
    value = proof(lock_path, execution, binding, identities)
    assert value["recovery_lifecycle"] == "POST_CHILD_ABORTED_COMPUTE_UNKNOWN"
    assert all(value[key] == "UNKNOWN" for key in ("compute_state", "cuda_state", "model_load_state", "forward_calls"))
    auth = recovery_authorization(tmp_path, value)
    result = lock.finalize_aborted_lock_recovery(lock_path, auth, execution_authorization_path=execution,
                                                  binding_path=binding, identity_provider=lambda pid: identities.get(pid),
                                                  job_clear_provider=lambda _: True)
    receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
    assert result["status"] == "FINALIZED_ABORTED" and not lock_path.exists()
    assert receipt["proof"]["compute_state"] == "UNKNOWN"
    with pytest.raises(lock.ExecutionBlocked, match="ALREADY_CONSUMED"):
        lock.claim_recovery_authorization(auth, json.loads(auth.read_text()), digest(auth))


@pytest.mark.parametrize("mutation, expected", [
    ("output", "SIDE_EFFECT_UNPROVEN"), ("tmp", "SIDE_EFFECT_UNPROVEN"),
    ("parent", "PARENT_INTEGRITY_UNPROVEN"), ("live_child", "CHILD_STILL_LIVE"),
    ("live_owner", "OWNER_STILL_LIVE"), ("job", "JOB_NOT_CLEAR"),
    ("unconsumed", "EXECUTION_CLAIM_UNREADABLE"), ("wrong_purpose", "EXECUTION_AUTHORIZATION_NOT_TERMINAL"),
    ("wrong_lock", "TARGET_MISMATCH"), ("wrong_execution", "TARGET_MISMATCH"),
    ("wrong_binding", "TARGET_MISMATCH"), ("tampered_child", "CHILD_MISMATCH"),
])
def test_post_child_unknown_recovery_rejects_missing_required_evidence(tmp_path, mutation, expected):
    lock_path, execution, binding, identities, output, parent = fixture(tmp_path)
    job = True
    if mutation == "output": put(output, b"unexpected")
    elif mutation == "tmp": put(Path(str(output) + ".tmp"), b"unexpected")
    elif mutation == "parent": put(parent, b"changed-parent")
    elif mutation == "live_child":
        identities[102] = json.loads((lock_path.parent / "runs" / "phase63-no_train-seed42-control" / "child-created.json").read_text())["child"]
    elif mutation == "live_owner": identities[101] = json.loads(lock_path.read_text())
    elif mutation == "job": job = False
    elif mutation == "unconsumed": execution.with_name(execution.name + ".phase63-claim.json").unlink()
    elif mutation == "wrong_purpose":
        value = json.loads(execution.read_text()); value["action"] = "training"; put(execution, value)
    elif mutation == "tampered_child":
        p = lock_path.parent / "runs" / "phase63-no_train-seed42-control" / "child-created.json"
        value = json.loads(p.read_text()); value["child"]["process_start_time"] = 999.0; put(p, value)
    if mutation in {"wrong_lock", "wrong_execution", "wrong_binding"}:
        value = proof(lock_path, execution, binding, identities)
        auth = recovery_authorization(tmp_path, value)
        changed = json.loads(auth.read_text())
        if mutation == "wrong_lock": changed["target_lock"]["sha256"] = "0" * 64
        elif mutation == "wrong_execution": changed["execution"]["run_id"] = "other"
        else: changed["execution"]["binding_sha256"] = "0" * 64
        put(auth, changed)
        with pytest.raises(lock.ExecutionBlocked, match=expected):
            lock.finalize_aborted_lock_recovery(lock_path, auth, execution_authorization_path=execution,
                                                binding_path=binding, identity_provider=lambda pid: identities.get(pid),
                                                job_clear_provider=lambda _: True)
    else:
        with pytest.raises(lock.ExecutionBlocked, match=expected):
            proof(lock_path, execution, binding, identities, job=job)


@pytest.mark.parametrize("field, changed", [("forward_calls", 0), ("cuda_state", "NOT_CREATED")])
def test_unknown_compute_proof_cannot_be_downgraded_to_zero_or_not_created(tmp_path, field, changed):
    lock_path, execution, binding, identities, _output, _parent = fixture(tmp_path)
    value = proof(lock_path, execution, binding, identities); auth = recovery_authorization(tmp_path, value)
    altered = copy.deepcopy(value); altered[field] = changed
    with pytest.raises(lock.ExecutionBlocked, match="POST_CHILD_COMPUTE_STATE_INVALID"):
        lock.validate_recovery_authorization(auth, altered)


def test_existing_o5l_and_o20_paths_remain_distinct(tmp_path):
    from tests.test_phase63o5l_lock_recovery import fixture as o5l
    from tests.test_phase63o20_pre_child_lock_recovery import pre_child_fixture
    lock_path, execution, binding, _auth, identities = o5l(tmp_path / "o5l")
    assert "recovery_lifecycle" not in proof(lock_path, execution, binding, identities)
    lock_path, execution, binding, identities = pre_child_fixture(tmp_path / "o20")
    assert proof(lock_path, execution, binding, identities)["recovery_lifecycle"] == "PRE_CHILD_ABORTED"
