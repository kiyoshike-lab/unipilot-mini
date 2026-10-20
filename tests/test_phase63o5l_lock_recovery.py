"""CPU-only PHASE63O5L fixtures for exact aborted-lock recovery."""
from __future__ import annotations

import copy
import hashlib
import json
import socket
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from training import gpu_execution_lock as lock


def put(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path):
    runtime = tmp_path / "runtime"; lock_path = runtime / "gpu.lock"; run_id = "phase63-no-train-seed42-control"
    owner = {"pid": 101, "parent_pid": 1, "process_start_time": 10.0, "hostname": socket.gethostname(), "exe": "C:\\Python\\python.exe",
             "schema_version": "gpu-owner-v1", "phase": 63, "run_id": run_id, "cuda_device": "cuda:0", "repo_head": "h" * 40,
             "checkpoint_parent_sha256": "p" * 64, "created_timestamp": 11.0, "command_hash": "c" * 64, "nonce": "n", "job_name": "job"}
    child = {"pid": 102, "parent_pid": 101, "process_start_time": 12.0, "hostname": socket.gethostname(), "exe": "C:\\Python\\python.exe"}
    put(lock_path, owner); run = runtime / "runs" / run_id
    put(run / "aborted.json", {"status": "INCOMPLETE", "ownership": "ABORTED", "reason": "ExecutionBlocked('CHILD_EXIT_NONZERO:1')",
                                "owner": owner, "child": None, "lock_retained": True, "automatic_terminate": False})
    put(run / "child-created.json", {"child": child, "suspended": True})
    put(run / "stderr.log", "RuntimeError: BINDING_OUTPUT_MISMATCH\n"); put(run / "stdout.log", "")
    binding = tmp_path / "binding.json"; binding_value = {"checkpoint_root": "Z:\\AI\\unipilot-mini\\checkpoints", "implementation_source_commit": "h" * 40,
        "source_manifest_sha256": "f" * 64, "source_manifest": [{"path": "training/phase63_worker.py", "sha256": "e" * 64}],
        "runs": [{"seed": 42, "arm": "control", "output_path": "Z:\\AI\\unipilot-mini\\checkpoints\\experimental\\phase63\\continuation-stability\\control\\seed-42"}]}
    put(binding, binding_value)
    old = tmp_path / "old-auth.json"; old_value = {"authorization_id": "old", "binding_sha256": digest(binding), "phase": 63, "action": "no_train", "allowed_runs": [{"seed": 42, "arm": "control"}]}; put(old, old_value)
    put(lock.recovery_claim_path(old).with_name(old.name + ".phase63-claim.json"), {"claim": {"authorization_sha256": digest(old), "authorization_id": "old", "binding_sha256": digest(binding), "phase": 63, "action": "no_train", "allowed_runs": [{"seed": 42, "arm": "control"}], "execution_instance": run_id}})
    identities = {101: None, 102: None}
    proof = lock.aborted_lock_recovery_proof(lock_path, execution_authorization_path=old, binding_path=binding,
                                              identity_provider=lambda pid: identities.get(pid), job_clear_provider=lambda _: True)
    now = datetime.now(UTC); auth = tmp_path / "recovery-auth.json"
    value = {"schema_version": lock.RECOVERY_AUTHORIZATION_SCHEMA, "authorization_id": "recover-1", "purpose": "lock_recovery", "scope": lock.RECOVERY_SCOPE,
             "valid_from": (now - timedelta(seconds=1)).isoformat(), "valid_until": (now + timedelta(minutes=5)).isoformat(),
             "target_lock": {"path": proof["lock_path"], "sha256": proof["lock_sha256"], "owner_sha256": proof["lock_owner_sha256"]},
             "execution": {"run_id": proof["run_id"], "binding_path": proof["binding_path"], "binding_sha256": proof["binding_sha256"], "implementation_head": proof["implementation_head"], "source_manifest_sha256": proof["source_manifest_sha256"], "authorization_path": proof["execution_authorization_path"], "authorization_sha256": proof["execution_authorization_sha256"], "claim_path": proof["execution_claim_path"], "claim_sha256": proof["execution_claim_sha256"]},
             "evidence": {"aborted_path": proof["aborted_path"], "aborted_sha256": proof["aborted_sha256"], "child_created_path": proof["child_created_path"], "child_created_sha256": proof["child_created_sha256"], "stderr_path": proof["stderr_path"], "stderr_sha256": proof["stderr_sha256"], "stdout_path": proof["stdout_path"], "stdout_sha256": proof["stdout_sha256"], "failure_code": proof["failure_code"]}}
    put(auth, value)
    return lock_path, old, binding, auth, identities


def assess(lock_path, old, binding, identities):
    return lock.aborted_lock_recovery_proof(lock_path, execution_authorization_path=old, binding_path=binding,
                                             identity_provider=lambda pid: identities.get(pid), job_clear_provider=lambda _: True)


def test_current_block_shape_is_rejected_by_legacy_global_cleanup(tmp_path, monkeypatch):
    lock_path, old, binding, auth, identities = fixture(tmp_path)
    def legacy_identity(pid=None):
        if pid in (101, 102):
            return None
        return {"pid": pid, "parent_pid": 1, "process_start_time": float(pid), "hostname": socket.gethostname(), "exe": "C:\\Windows\\desktop.exe"}
    monkeypatch.setattr(lock, "identity", legacy_identity)
    monkeypatch.setattr(lock, "job_clear", lambda _: True)
    legacy = lock.stale_verification(lock_path, [{"pid": 900 + n, "type": "C+G", "process_name": "desktop.exe"} for n in range(20)],
                                     owned_children=[{"pid": 102}])
    assert legacy["stale_verified"] is False and legacy["checks"]["gpu_clear"] is False
    assert assess(lock_path, old, binding, identities)["failure_before_cuda"] is True


def test_exact_terminated_lock_can_be_finalized_with_bound_single_use_authority(tmp_path):
    lock_path, old, binding, auth, identities = fixture(tmp_path)
    result = lock.finalize_aborted_lock_recovery(lock_path, auth, execution_authorization_path=old, binding_path=binding,
                                                  identity_provider=lambda pid: identities.get(pid), job_clear_provider=lambda _: True)
    assert result["status"] == "FINALIZED_ABORTED" and not lock_path.exists()
    assert Path(result["archived_lock"]).exists() and Path(result["receipt"]).exists()
    with pytest.raises(lock.ExecutionBlocked, match="ALREADY_CONSUMED"):
        lock.claim_recovery_authorization(auth, json.loads(auth.read_text()), digest(auth))


@pytest.mark.parametrize("which", ("owner", "child"))
def test_live_exact_owner_or_child_blocks_recovery(tmp_path, which):
    lock_path, old, binding, auth, identities = fixture(tmp_path)
    live = {101: {"pid": 101, "parent_pid": 1, "process_start_time": 10.0, "hostname": socket.gethostname(), "exe": "C:\\Python\\python.exe"},
            102: {"pid": 102, "parent_pid": 101, "process_start_time": 12.0, "hostname": socket.gethostname(), "exe": "C:\\Python\\python.exe"}}
    identities[101 if which == "owner" else 102] = live[101 if which == "owner" else 102]
    with pytest.raises(lock.ExecutionBlocked, match="OWNER_STILL_LIVE|CHILD_STILL_LIVE"):
        assess(lock_path, old, binding, identities)


def test_pid_reuse_is_not_treated_as_old_instance(tmp_path):
    lock_path, old, binding, auth, identities = fixture(tmp_path)
    identities[102] = {"pid": 102, "parent_pid": 7, "process_start_time": 99.0, "hostname": "host", "exe": "C:\\Other\\new.exe"}
    assert assess(lock_path, old, binding, identities)["child_gone"] is True


@pytest.mark.parametrize("mutation", ("wrong_lock", "wrong_execution", "wrong_binding", "wrong_authorization", "tampered_lock"))
def test_authority_or_evidence_mismatch_fails_closed(tmp_path, mutation):
    lock_path, old, binding, auth, identities = fixture(tmp_path)
    value = json.loads(auth.read_text())
    if mutation == "wrong_lock": value["target_lock"]["sha256"] = "0" * 64
    elif mutation == "wrong_execution": value["execution"]["run_id"] = "other"
    elif mutation == "wrong_binding": value["execution"]["binding_sha256"] = "0" * 64
    elif mutation == "wrong_authorization": value["execution"]["authorization_sha256"] = "0" * 64
    else: put(lock_path, {**json.loads(lock_path.read_text()), "nonce": "tampered"})
    put(auth, value)
    with pytest.raises(lock.ExecutionBlocked):
        lock.finalize_aborted_lock_recovery(lock_path, auth, execution_authorization_path=old, binding_path=binding,
                                             identity_provider=lambda pid: identities.get(pid), job_clear_provider=lambda _: True)


@pytest.mark.parametrize("mutation", ("missing_claim", "malformed_lock"))
def test_unconsumed_or_malformed_execution_evidence_fails_closed(tmp_path, mutation):
    lock_path, old, binding, auth, identities = fixture(tmp_path)
    if mutation == "missing_claim":
        old.with_name(old.name + ".phase63-claim.json").unlink()
    else:
        put(lock_path, "not-json")
    with pytest.raises(lock.ExecutionBlocked):
        assess(lock_path, old, binding, identities)


def test_recovery_claim_is_single_winner_under_concurrency(tmp_path):
    lock_path, old, binding, auth, identities = fixture(tmp_path); proof = assess(lock_path, old, binding, identities)
    value = lock.validate_recovery_authorization(auth, proof); digest_value = digest(auth); results = []
    barrier = threading.Barrier(2)
    def claim_once():
        barrier.wait()
        try: lock.claim_recovery_authorization(auth, value, digest_value); results.append("ACCEPT")
        except lock.ExecutionBlocked as exc: results.append(str(exc))
    first = threading.Thread(target=claim_once); second = threading.Thread(target=claim_once); first.start(); second.start(); first.join(); second.join()
    assert results.count("ACCEPT") == 1 and results.count("LOCK_RECOVERY_AUTHORIZATION_ALREADY_CONSUMED") == 1
