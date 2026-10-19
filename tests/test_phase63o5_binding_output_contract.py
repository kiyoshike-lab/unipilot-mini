"""CPU-only regression for PHASE63O5 binding output admission."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from training import phase63_binding_contract as contract


ROOT = Path(__file__).parents[1]
ZROOT = Path(r"Z:\AI\unipilot-mini\checkpoints")
PREREG = ROOT / "evaluation" / "phase62" / "phase63-clean-continuation-stability-preregistration.json"


def put(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def binding() -> dict:
    manifest = contract.current_source_manifest(ROOT)
    rows = []
    for ordinal, (seed, arm, lr, parent_sha) in enumerate(((42, "control", 5e-5, "a55369c0e98779839750d727517ce6621bc40b5746f975a96bd9cbc0574db2e8"),
                                                             (42, "half-lr", 2.5e-5, "a55369c0e98779839750d727517ce6621bc40b5746f975a96bd9cbc0574db2e8"),
                                                             (123, "control", 5e-5, "78df96b70016dda180f4529833d904e2d448e11469a87278d4881ed09c93dc20"),
                                                             (123, "half-lr", 2.5e-5, "78df96b70016dda180f4529833d904e2d448e11469a87278d4881ed09c93dc20"),
                                                             (2026, "control", 5e-5, "7dcf5fa2da58777040cf9c03f1f513d707e6866f2eaf237bf203bd1b0135f069"),
                                                             (2026, "half-lr", 2.5e-5, "7dcf5fa2da58777040cf9c03f1f513d707e6866f2eaf237bf203bd1b0135f069")), 1):
        rows.append({"ordinal": ordinal, "seed": seed, "arm": arm, "runtime_lr": lr, "optimizer_updates": 122,
                     "parent_path": str(ZROOT / "experimental" / "phase48" / "arm-C" / f"seed-{seed}" / "checkpoint-tokens-16384000.pt"),
                     "parent_sha256": parent_sha, "output_path": str(contract.expected_output_path(ZROOT, seed, arm))})
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
            "cooling_evidence": {"path": contract.K11_EVIDENCE_RELATIVE_PATH, "sha256": contract.sha256_file(ROOT / contract.K11_EVIDENCE_RELATIVE_PATH),
                                 "operational_gate": "PHASE63_COOLING_RUNTIME_VERIFIED", "runtime_baseline": "ACCEPTED",
                                 "historical_profile": "HISTORICAL_PROFILE_DRIFT_CAUSE_UNKNOWN"},
            "approved_processes": [], "execution_authorized": False, "no_train_gpu_authorized": False, "training_authorized": False}


def validate(path: Path) -> dict:
    return contract.validate_unarmed_binding(path, checkpoint_root=ZROOT, preregistration_path=PREREG,
                                             expected_interpreter=Path(sys.executable), source_root=ROOT)


def test_output_filename_is_verified_before_any_authorization_or_gpu_stage(tmp_path: Path) -> None:
    valid = binding(); path = tmp_path / "binding.json"; put(path, valid)
    assert validate(path)["runs"][0]["output_path"] == str(contract.expected_output_path(ZROOT, 42, "control"))
    invalid = binding(); invalid["runs"][0]["output_path"] = str(contract.expected_output_path(ZROOT, 42, "control").parent)
    put(path, invalid)
    with pytest.raises(contract.BindingContractError, match="BINDING_OUTPUT_MISMATCH"):
        validate(path)
    assert not path.with_name(path.name + ".phase63-claim.json").exists()
