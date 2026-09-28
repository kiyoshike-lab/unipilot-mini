"""Synthetic CPU-only tests for PHASE63M runtime binding enforcement."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from training import phase63_binding_contract as contract
from training import phase63_worker as worker


ROOT = Path(__file__).parents[1]
PREREG = ROOT / "evaluation" / "phase62" / "phase63-clean-continuation-stability-preregistration.json"
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
                     "parent_path": f"Z:\\parent-{seed}.pt", "parent_sha256": "a" * 64,
                     "output_path": str(worker.output(seed, arm))})
    return {"schema_version": contract.RUNTIME_SCHEMA, "phase": 63, "status": "UNARMED_CPU_VALIDATED",
            "branch": "foundation-research", "implementation_source_commit": "b" * 40,
            "source_manifest": manifest, "source_manifest_sha256": contract.canonical_sha256(manifest),
            "scientific_preregistration": {"path": contract.PREREG_RELATIVE_PATH, "sha256": contract.sha256_file(PREREG)},
            "interpreter": {"executable": str(Path(sys.executable).resolve()), "python_version": "test", "torch_version": "test", "torch_cuda_build": "test"},
            "checkpoint_root": str(ZROOT), "runs": rows,
            "contracts": {"guard_schema_version": "phase63-gpu-guard-contract-v2", "approval_schema_version": contract.AUTHORIZATION_SCHEMA,
                          "no_train_receipt_schema_version": "phase63-no-train-receipt-v1"},
            "gates": {"gpu_guard": "PHASE63_GPU_GUARD_CPU_VALIDATED", "cooling": "PHASE63_COOLING_REVIEW_REQUIRED",
                      "process_approvals": "NONE", "generation_policy": "UNSAFE", "phase61": "EXPERIMENT_INVALID"},
            "approved_processes": [], "execution_authorized": False, "no_train_gpu_authorized": False, "training_authorized": False}


def fixture_authorization(binding_path: Path, *, action="no_train", allowed_runs=None, processes=None) -> tuple[Path, str]:
    now = datetime.now(UTC)
    auth = {"schema_version": contract.AUTHORIZATION_SCHEMA, "authorization_id": "M-test", "binding_sha256": contract.sha256_file(binding_path),
            "phase": 63, "action": action, "allowed_runs": allowed_runs or [{"seed": 42, "arm": "control"}],
            "valid_from": (now - timedelta(minutes=1)).isoformat(), "valid_until": (now + timedelta(minutes=1)).isoformat(),
            "scope": "PHASE63_SINGLE_ACTION_SINGLE_RUN", "target_gpu": {"pci": "test"}, "approved_processes": processes or []}
    path = binding_path.with_name("authorization.json"); put(path, auth)
    return path, contract.sha256_file(path)


def validate(path: Path, auth: Path | None = None, auth_sha: str | None = None, *, action="no_train", provider=None) -> dict:
    return contract.validate_execution(path, auth, auth_sha, action=action, seed=42, arm="control", checkpoint_root=ZROOT,
                                       preregistration_path=PREREG, expected_interpreter=Path(sys.executable),
                                       process_identity_provider=provider)


def test_valid_external_no_train_authorization_passes_cpu_only(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"; put(binding, fixture_binding())
    auth, digest = fixture_authorization(binding)
    assert validate(binding, auth, digest)["schema_version"] == contract.RUNTIME_SCHEMA


def test_a_old_binding_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(contract.BindingContractError, match="INVALID_FIELDS|RUNTIME_BINDING_SCHEMA_OR_STATUS_MISMATCH"):
        contract.validate_unarmed_binding(ROOT / "evaluation" / "phase63" / "execution-binding.json", checkpoint_root=ZROOT,
                                          preregistration_path=PREREG)


@pytest.mark.parametrize("mutation", [
    lambda x: x["source_manifest"].__setitem__("training/phase63_worker.py", "0" * 64),
    lambda x: x["source_manifest"].__setitem__("scripts/run_phase63_study.py", "1" * 64),
    lambda x: x["source_manifest"].__setitem__("training/gpu_execution_guard.py", "2" * 64),
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


def test_i_pid_reuse_identity_change_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"; put(path, fixture_binding())
    approved = [{"pid": 99, "started": "2026-01-01T00:00:00Z", "exe_sha256": "a" * 64}]
    auth, digest = fixture_authorization(path, processes=approved)
    with pytest.raises(contract.BindingContractError, match="APPROVED_PROCESS_IDENTITY_CHANGED"):
        validate(path, auth, digest, provider=lambda: [{"pid": 99, "started": "later", "exe_sha256": "a" * 64}])


def test_j_source_change_after_validation_is_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    for relative in contract.RUNTIME_SOURCE_FILES:
        destination = source_root / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    value = fixture_binding(source_root)
    path = tmp_path / "binding.json"; put(path, value)
    contract.validate_unarmed_binding(path, checkpoint_root=ZROOT, preregistration_path=PREREG, source_root=source_root)
    (source_root / "training" / "phase63_worker.py").write_text("changed", encoding="utf-8")
    with pytest.raises(contract.BindingContractError, match="RUNTIME_SOURCE_CHANGED_AFTER_VALIDATION"):
        contract.assert_sources_unchanged(value, source_root=source_root)


def test_worker_and_supervisor_share_the_contract() -> None:
    worker_source = (ROOT / "training" / "phase63_worker.py").read_text(encoding="utf-8")
    supervisor_source = (ROOT / "scripts" / "run_phase63_study.py").read_text(encoding="utf-8")
    assert "validate_execution" in worker_source
    assert "validate_execution" in supervisor_source
    assert "--authorization-sha256" in worker_source
    assert "--authorization-sha256" in supervisor_source
