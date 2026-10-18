"""Synthetic CPU-only tests for PHASE63M runtime binding enforcement."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from training import phase63_binding_contract as contract
from training import phase63_worker as worker
from training.phase63_authorization_claim import claim_path


ROOT = Path(__file__).parents[1]
PREREG = ROOT / "evaluation" / "phase62" / "phase63-clean-continuation-stability-preregistration.json"
K11 = ROOT / "evaluation" / "phase63" / "phase63k11" / "runtime-baseline-decision.json"
ZROOT = Path(r"Z:\AI\unipilot-mini\checkpoints")


def put(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def fixture_binding(source_root: Path = ROOT) -> dict:
    manifest = contract.current_source_manifest(source_root)
    rows = []
    for ordinal, (seed, arm, lr) in enumerate(((42, "control", 5e-5), (42, "half-lr", 2.5e-5),
                                                 (123, "control", 5e-5), (123, "half-lr", 2.5e-5),
                                                 (2026, "control", 5e-5), (2026, "half-lr", 2.5e-5)), 1):
        rows.append({"ordinal": ordinal, "seed": seed, "arm": arm, "runtime_lr": lr, "optimizer_updates": 122,
                     "parent_path": f"Z:\\parent-{seed}.pt", "parent_sha256": worker.PARENT_SHA[seed],
                     "output_path": str(worker.output(seed, arm))})
    return {"schema_version": contract.RUNTIME_SCHEMA, "phase": 63, "status": "UNARMED_CPU_VALIDATED",
            "branch": "foundation-research", "implementation_source_commit": contract.trusted_current_implementation_commit(),
            "source_manifest": manifest, "source_manifest_sha256": contract.canonical_sha256(manifest),
            "scientific_preregistration": {"path": contract.PREREG_RELATIVE_PATH, "sha256": contract.sha256_file(PREREG)},
            "interpreter": {"executable": str(Path(sys.executable).resolve()), "python_version": "test", "torch_version": "test", "torch_cuda_build": "test"},
            "checkpoint_root": str(ZROOT), "runs": rows,
            "contracts": {"guard_schema_version": "phase63-gpu-guard-contract-v2", "approval_schema_version": contract.AUTHORIZATION_SCHEMA,
                          "no_train_receipt_schema_version": "phase63-no-train-receipt-v1"},
            "gates": {"gpu_guard": "PHASE63_GPU_GUARD_CPU_VALIDATED", "cooling": "PHASE63_COOLING_RUNTIME_VERIFIED",
                      "process_approvals": "NONE", "generation_policy": "UNSAFE", "phase61": "EXPERIMENT_INVALID"},
            "cooling_evidence": {"path": contract.K11_EVIDENCE_RELATIVE_PATH, "sha256": contract.sha256_file(K11),
                                 "operational_gate": "PHASE63_COOLING_RUNTIME_VERIFIED", "runtime_baseline": "ACCEPTED",
                                 "historical_profile": "HISTORICAL_PROFILE_DRIFT_CAUSE_UNKNOWN"},
            "approved_processes": [], "execution_authorized": False, "no_train_gpu_authorized": False, "training_authorized": False}


def fixture_authorization(binding_path: Path, *, action="no_train", allowed_runs=None,
                          processes=None, exceptions=None) -> tuple[Path, str]:
    now = datetime.now(UTC)
    auth = {"schema_version": contract.AUTHORIZATION_SCHEMA, "authorization_id": "M-test", "binding_sha256": contract.sha256_file(binding_path),
            "phase": 63, "action": action, "allowed_runs": allowed_runs or [{"seed": 42, "arm": "control"}],
            "valid_from": (now - timedelta(minutes=1)).isoformat(), "valid_until": (now + timedelta(minutes=1)).isoformat(),
            "scope": "PHASE63_SINGLE_ACTION_SINGLE_RUN", "target_gpu": {"uuid": "GPU-test"},
            "approved_processes": processes or [], "external_process_exceptions": exceptions or []}
    path = binding_path.with_name("authorization.json"); put(path, auth)
    return path, contract.sha256_file(path)


def fixture_exception(binding_path: Path, *, identity=None, **changes) -> dict:
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    now = time.time()
    value = {"schema": "phase63-external-process-exception-v1", "exception_id": "fixture-exception",
             "authorization_id": "M-test", "binding_sha256": contract.sha256_file(binding_path),
             "implementation_head": binding["implementation_source_commit"],
             "source_manifest_sha256": binding["source_manifest_sha256"], "phase": 63,
             "action": "no-train-gpu-supervisor", "gpu_uuid": "GPU-test",
             "process_identity": identity or {"pid": 71, "process_start_time": 100.0,
                                                "hostname": "test-host", "exe": r"C:\\fixture.exe",
                                                "exe_sha256": "a" * 64},
             "created_at": now - 1, "expires_at": now + 30, "single_use": True}
    value.update(changes)
    return value


def validate(path: Path, auth: Path | None = None, auth_sha: str | None = None, *, action="no_train", provider=None) -> dict:
    return contract.validate_execution(path, auth, auth_sha, action=action, seed=42, arm="control", checkpoint_root=ZROOT,
                                       preregistration_path=PREREG, expected_interpreter=Path(sys.executable),
                                       process_identity_provider=provider)


def admit(path: Path, auth: Path, auth_sha: str, *, execution_instance="phase63-no-train-test") -> dict:
    return contract.admit_execution(path, auth, auth_sha, action="no_train", seed=42, arm="control",
                                    execution_instance=execution_instance, checkpoint_root=ZROOT,
                                    preregistration_path=PREREG, expected_interpreter=Path(sys.executable))


def test_valid_external_no_train_authorization_passes_cpu_only(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"; put(binding, fixture_binding())
    auth, digest = fixture_authorization(binding)
    assert validate(binding, auth, digest)["schema_version"] == contract.RUNTIME_SCHEMA


def test_exact_exception_is_bound_to_external_authorization_binding_head_and_no_train_only(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"; put(binding, fixture_binding())
    exception = fixture_exception(binding)
    auth, digest = fixture_authorization(binding, exceptions=[exception])
    _, exceptions = contract.prepare_execution(binding, auth, digest, action="no_train", seed=42,
                                               arm="control", checkpoint_root=ZROOT, preregistration_path=PREREG,
                                               expected_interpreter=Path(sys.executable))
    assert exceptions[0]["exception_id"] == "fixture-exception"
    assert exceptions[0]["process_identity"]["exe_sha256"] == "a" * 64
    wrong_binding = fixture_exception(binding, binding_sha256="b" * 64)
    auth, digest = fixture_authorization(binding, exceptions=[wrong_binding])
    with pytest.raises(contract.BindingContractError, match="PROCESS_EXCEPTION_SCOPE_DENIED"):
        validate(binding, auth, digest)
    training_exception = fixture_exception(binding)
    auth, digest = fixture_authorization(binding, action="training", exceptions=[training_exception])
    with pytest.raises(contract.BindingContractError, match="PROCESS_EXCEPTION_(SCOPE_DENIED|ACTION_DENIED)"):
        validate(binding, auth, digest, action="training")


def test_candidate_or_exception_without_separate_external_authorization_is_rejected(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"; put(binding, fixture_binding())
    with pytest.raises(contract.BindingContractError, match="EXTERNAL_EXECUTION_AUTHORIZATION_REQUIRED"):
        validate(binding)


def test_no_train_authorization_first_claim_passes_and_second_claim_is_rejected(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"; put(binding, fixture_binding())
    auth, digest = fixture_authorization(binding)
    assert admit(binding, auth, digest)["schema_version"] == contract.RUNTIME_SCHEMA
    with pytest.raises(contract.BindingContractError, match="AUTHORIZATION_ALREADY_CONSUMED"):
        admit(binding, auth, digest)


def test_consumed_no_train_authorization_rejects_after_new_validator_call(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"; put(binding, fixture_binding())
    auth, digest = fixture_authorization(binding)
    admit(binding, auth, digest, execution_instance="phase63-no-train-restart")
    # A second admission is a new function invocation, so acceptance cannot be
    # explained by an in-memory process-local set.
    with pytest.raises(contract.BindingContractError, match="AUTHORIZATION_ALREADY_CONSUMED"):
        admit(binding, auth, digest, execution_instance="phase63-no-train-restart")


def test_concurrent_no_train_claim_has_one_accept_and_one_reject(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"; put(binding, fixture_binding())
    auth, digest = fixture_authorization(binding)
    barrier = threading.Barrier(2)
    results: list[str] = []

    def attempt() -> None:
        barrier.wait()
        try:
            admit(binding, auth, digest, execution_instance="phase63-no-train-concurrent")
            results.append("ACCEPT")
        except contract.BindingContractError as exc:
            results.append(str(exc))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert results.count("ACCEPT") == 1
    assert results.count("AUTHORIZATION_ALREADY_CONSUMED") == 1


def test_claim_remains_consumed_after_simulated_failure(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"; put(binding, fixture_binding())
    auth, digest = fixture_authorization(binding)
    admit(binding, auth, digest, execution_instance="phase63-no-train-failure")
    # The simulated stage failure occurs after admission; no recovery path may
    # delete the durable claim.
    with pytest.raises(contract.BindingContractError, match="AUTHORIZATION_ALREADY_CONSUMED"):
        admit(binding, auth, digest, execution_instance="phase63-no-train-failure")


def test_malformed_durable_claim_state_fails_closed(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"; put(binding, fixture_binding())
    auth, digest = fixture_authorization(binding)
    claim_path(auth).write_text("not json", encoding="utf-8")
    with pytest.raises(contract.BindingContractError, match="AUTHORIZATION_CLAIM_STATE_CORRUPT"):
        admit(binding, auth, digest)


def test_worker_requires_the_supervisors_durable_claim_before_any_cuda_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding = tmp_path / "binding.json"; put(binding, fixture_binding())
    auth, digest = fixture_authorization(binding)
    monkeypatch.setenv("UNIPILOT_CHECKPOINT_ROOT", str(ZROOT))
    monkeypatch.setenv("UNIPILOT_GPU_RUN_ID", "phase63-no-train-worker")
    with pytest.raises(RuntimeError, match="AUTHORIZATION_NOT_CLAIMED"):
        worker.read_binding(binding, 42, "control", action="no_train", authorization_path=auth,
                            authorization_sha256=digest)
    admit(binding, auth, digest, execution_instance="phase63-no-train-worker")
    assert worker.read_binding(binding, 42, "control", action="no_train", authorization_path=auth,
                               authorization_sha256=digest)["phase"] == 63


def test_a_old_binding_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(contract.BindingContractError, match="INVALID_FIELDS|RUNTIME_BINDING_SCHEMA_OR_STATUS_MISMATCH"):
        contract.validate_unarmed_binding(ROOT / "evaluation" / "phase63" / "execution-binding.json", checkpoint_root=ZROOT,
                                          preregistration_path=PREREG)


@pytest.mark.parametrize("mutation", [
    lambda x: x["source_manifest"].pop(),
    lambda x: x["source_manifest"].__setitem__(0, {**x["source_manifest"][0], "sha256": "0" * 64}),
    lambda x: x["source_manifest"].append({"path": "README.md", "bytes": 1, "sha256": "1" * 64}),
])
def test_bcd_source_mismatches_are_rejected(tmp_path: Path, mutation) -> None:
    value = fixture_binding(); mutation(value); path = tmp_path / "binding.json"; put(path, value)
    with pytest.raises(contract.BindingContractError, match="RUNTIME_SOURCE_MANIFEST_MISMATCH"):
        contract.validate_unarmed_binding(path, checkpoint_root=ZROOT, preregistration_path=PREREG)


def test_e_binding_tamper_after_authorization_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"; value = fixture_binding(); put(path, value)
    auth, digest = fixture_authorization(path)
    value["runs"][0]["runtime_lr"] = 1e-4; put(path, value)
    with pytest.raises(contract.BindingContractError, match="AUTHORIZATION_BINDING_SHA_MISMATCH"):
        validate(path, auth, digest)


def test_f_unarmed_binding_without_separate_authorization_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"; put(path, fixture_binding())
    with pytest.raises(contract.BindingContractError, match="EXTERNAL_EXECUTION_AUTHORIZATION_REQUIRED"):
        validate(path)


def test_gh_authorization_action_cannot_be_missing_or_reused_for_training(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"; put(path, fixture_binding())
    auth, digest = fixture_authorization(path, action="no_train")
    with pytest.raises(contract.BindingContractError, match="AUTHORIZATION_ACTION_OR_SCHEMA_MISMATCH"):
        validate(path, auth, digest, action="training")


def test_unknown_authorization_purpose_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"; put(path, fixture_binding())
    auth, digest = fixture_authorization(path, action="not-a-phase63-purpose")
    with pytest.raises(contract.BindingContractError, match="AUTHORIZATION_ACTION_OR_SCHEMA_MISMATCH"):
        validate(path, auth, digest, action="not-a-phase63-purpose")


def test_i_legacy_process_approval_cannot_be_promoted_to_an_exception(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"; put(path, fixture_binding())
    approved = [{"pid": 99, "started": "2026-01-01T00:00:00Z", "exe_sha256": "a" * 64}]
    auth, digest = fixture_authorization(path, processes=approved)
    with pytest.raises(contract.BindingContractError, match="LEGACY_PROCESS_APPROVALS_FORBIDDEN"):
        validate(path, auth, digest, provider=lambda: [{"pid": 99, "started": "later", "exe_sha256": "a" * 64}])


def test_j_source_change_after_validation_is_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    for entry in contract.current_source_manifest(ROOT):
        relative = entry["path"]
        destination = source_root / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    value = fixture_binding(source_root)
    path = tmp_path / "binding.json"; put(path, value)
    contract.validate_unarmed_binding(path, checkpoint_root=ZROOT, preregistration_path=PREREG, source_root=source_root)
    (source_root / "training" / "phase63_worker.py").write_text("changed", encoding="utf-8")
    with pytest.raises(contract.BindingContractError, match="RUNTIME_SOURCE_CHANGED_AFTER_VALIDATION"):
        contract.assert_sources_unchanged(value, source_root=source_root)


@pytest.mark.parametrize("cooling", [
    "PHASE63_COOLING_REVIEW_REQUIRED",
    "PHASE63_COOLING_UNSAFE",
    "PHASE63_COOLING_RUNTIME_VERIFIED_FAKE",
    "COOLING_OK",
    "VERIFIED",
])
def test_k_cooling_literals_other_than_current_k11_gate_are_rejected(tmp_path: Path, cooling: str) -> None:
    path = tmp_path / "binding.json"; value = fixture_binding()
    value["gates"]["cooling"] = cooling
    put(path, value)
    with pytest.raises(contract.BindingContractError, match="RUNTIME_GATE_MISMATCH"):
        contract.validate_unarmed_binding(path, checkpoint_root=ZROOT, preregistration_path=PREREG)


def test_l_k11_evidence_tamper_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"; value = fixture_binding()
    value["cooling_evidence"]["historical_profile"] = "EXPLAINED"
    put(path, value)
    with pytest.raises(contract.BindingContractError, match="COOLING_PREREQUISITE_MISMATCH"):
        contract.validate_unarmed_binding(path, checkpoint_root=ZROOT, preregistration_path=PREREG)


@pytest.mark.parametrize("commit", [
    "0" * 40,
    "f" * 40,
    contract.trusted_current_implementation_commit()[:12],
    contract.trusted_current_implementation_commit().upper(),
    "HEAD",
    "foundation-research",
    " " + contract.trusted_current_implementation_commit(),
])
def test_m_implementation_source_commit_requires_exact_trusted_head(tmp_path: Path, commit: str) -> None:
    path = tmp_path / "binding.json"; value = fixture_binding()
    value["implementation_source_commit"] = commit
    put(path, value)
    with pytest.raises(contract.BindingContractError, match="IMPLEMENTATION_SOURCE_COMMIT_(MISMATCH|FORMAT_INVALID)"):
        contract.validate_unarmed_binding(path, checkpoint_root=ZROOT, preregistration_path=PREREG)


def test_n_parent_commit_is_rejected_when_available(tmp_path: Path) -> None:
    parent = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD^"],
                            check=True, capture_output=True, text=True).stdout.strip()
    path = tmp_path / "binding.json"; value = fixture_binding()
    value["implementation_source_commit"] = parent
    put(path, value)
    with pytest.raises(contract.BindingContractError, match="IMPLEMENTATION_SOURCE_COMMIT_MISMATCH"):
        contract.validate_unarmed_binding(path, checkpoint_root=ZROOT, preregistration_path=PREREG)


def test_o_head_resolution_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "binding.json"; put(path, fixture_binding())
    def fail() -> str:
        raise contract.BindingContractError("IMPLEMENTATION_SOURCE_COMMIT_UNRESOLVED")
    monkeypatch.setattr(contract, "trusted_current_implementation_commit", fail)
    with pytest.raises(contract.BindingContractError, match="IMPLEMENTATION_SOURCE_COMMIT_UNRESOLVED"):
        contract.validate_unarmed_binding(path, checkpoint_root=ZROOT, preregistration_path=PREREG)


def test_p_head_changes_between_admission_checks_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "binding.json"; put(path, fixture_binding())
    auth, digest = fixture_authorization(path)
    trusted = contract.trusted_current_implementation_commit()
    values = iter([trusted, "f" * 40])
    monkeypatch.setattr(contract, "trusted_current_implementation_commit", lambda: next(values))
    with pytest.raises(contract.BindingContractError, match="IMPLEMENTATION_SOURCE_COMMIT_MISMATCH"):
        validate(path, auth, digest)


def test_worker_and_supervisor_share_the_contract() -> None:
    worker_source = (ROOT / "training" / "phase63_worker.py").read_text(encoding="utf-8")
    supervisor_source = (ROOT / "scripts" / "run_phase63_study.py").read_text(encoding="utf-8")
    assert "validate_execution" in worker_source
    assert "admit_execution" in supervisor_source
    assert "run_no_train_once" in supervisor_source
    assert "--authorization-sha256" in worker_source
    assert "--authorization-sha256" in supervisor_source
