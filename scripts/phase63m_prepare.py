"""Create PHASE63M CPU-only runtime-binding evidence; never launches CUDA work."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.phase63_binding_contract import (AUTHORIZATION_SCHEMA, PREREG_RELATIVE_PATH, RUNTIME_SCHEMA,
                                               BindingContractError, canonical_sha256, current_source_manifest,
                                               sha256_file, validate_unarmed_binding)


OUT = ROOT / "evaluation" / "phase63" / "phase63m"
ZROOT = Path(r"Z:\AI\unipilot-mini\checkpoints")
PREREG = ROOT / PREREG_RELATIVE_PATH
PARENT_SHA = {42: "a55369c0e98779839750d727517ce6621bc40b5746f975a96bd9cbc0574db2e8",
              123: "78df96b70016dda180f4529833d904e2d448e11469a87278d4881ed09c93dc20",
              2026: "7dcf5fa2da58777040cf9c03f1f513d707e6866f2eaf237bf203bd1b0135f069"}


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def candidate(manifest: dict[str, str]) -> dict:
    import torch  # CPU metadata only; this script does not call torch.cuda.
    rows = []
    for ordinal, (seed, arm, lr) in enumerate(((42, "control", 5e-5), (42, "half-lr", 2.5e-5),
                                                 (123, "control", 5e-5), (123, "half-lr", 2.5e-5),
                                                 (2026, "control", 5e-5), (2026, "half-lr", 2.5e-5)), 1):
        rows.append({"ordinal": ordinal, "seed": seed, "arm": arm, "runtime_lr": lr, "optimizer_updates": 122,
                     "parent_path": str(ZROOT / "experimental" / "phase48" / "arm-C" / f"seed-{seed}" / "checkpoint-tokens-16384000.pt"),
                     "parent_sha256": PARENT_SHA[seed],
                     "output_path": str(ZROOT / "experimental" / "phase63" / "continuation-stability" / arm / f"seed-{seed}" / "checkpoint-tokens-16446464.pt")})
    return {"schema_version": RUNTIME_SCHEMA, "phase": 63, "status": "UNARMED_CPU_VALIDATED",
            "branch": git("branch", "--show-current"), "implementation_source_commit": git("rev-parse", "HEAD"),
            "source_manifest": manifest, "source_manifest_sha256": canonical_sha256(manifest),
            "scientific_preregistration": {"path": PREREG_RELATIVE_PATH, "sha256": sha256_file(PREREG)},
            "interpreter": {"executable": str(Path(sys.executable).resolve()), "python_version": sys.version,
                            "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda},
            "checkpoint_root": str(ZROOT), "runs": rows,
            "contracts": {"guard_schema_version": "phase63-gpu-guard-contract-v2", "approval_schema_version": AUTHORIZATION_SCHEMA,
                          "no_train_receipt_schema_version": "phase63-no-train-receipt-v1"},
            "gates": {"gpu_guard": "PHASE63_GPU_GUARD_CPU_VALIDATED", "cooling": "PHASE63_COOLING_REVIEW_REQUIRED",
                      "process_approvals": "NONE", "generation_policy": "UNSAFE", "phase61": "EXPERIMENT_INVALID"},
            "approved_processes": [], "execution_authorized": False, "no_train_gpu_authorized": False, "training_authorized": False}


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES_MUST_BE_MINUS_ONE")
    if OUT.exists():
        raise FileExistsError("PHASE63M_OUTPUT_EXISTS")
    manifest = current_source_manifest(ROOT)
    before = json.loads((ROOT / "evaluation" / "phase63" / "phase63l" / "source-hash-manifest.json").read_text(encoding="utf-8"))
    source = {"before_phase63l": before["source_manifest"], "after_phase63m": manifest,
              "after_manifest_sha256": canonical_sha256(manifest), "changed_paths": sorted(
                  set(before["source_manifest"]) | set(manifest))}
    value = candidate(manifest)
    OUT.mkdir(parents=True, exist_ok=False)
    binding_path = OUT / "execution-binding-candidate.json"; write_new(binding_path, value)
    verification = validate_unarmed_binding(binding_path, checkpoint_root=ZROOT, preregistration_path=PREREG)
    write_new(OUT / "source-hash-manifest.json", source)
    write_new(OUT / "runtime-binding-contract.json", {"runtime_schema": RUNTIME_SCHEMA, "authorization_schema": AUTHORIZATION_SCHEMA,
              "runtime_source_files": sorted(manifest), "binding_is_unarmed": True,
              "authorization_requirements": ["binding_sha256", "phase", "action", "allowed_runs", "valid_from", "valid_until", "scope", "target_gpu", "approved_processes"],
              "toctou_recheck": "source manifest is checked again immediately before CUDA initialization"})
    write_new(OUT / "binding-verification.json", {"candidate_sha256": sha256_file(binding_path), "candidate_self_hash": "NOT_EMBEDDED",
              "strict_unarmed_validation": {"pass": True, "schema": verification["schema_version"], "run_count": len(verification["runs"])}})
    stale = {}
    for name, path in {"legacy_execution_binding": ROOT / "evaluation" / "phase63" / "execution-binding.json",
                       "phase63l_candidate": ROOT / "evaluation" / "phase63" / "phase63l" / "execution-binding-candidate.json"}.items():
        try:
            validate_unarmed_binding(path, checkpoint_root=ZROOT, preregistration_path=PREREG)
            stale[name] = {"rejected": False}
        except BindingContractError as exc:
            stale[name] = {"rejected": True, "reason": str(exc), "sha256": sha256_file(path)}
    write_new(OUT / "stale-binding-rejection-results.json", stale)
    print(json.dumps({"candidate_sha256": sha256_file(binding_path), "manifest_sha256": canonical_sha256(manifest), "stale": stale}, ensure_ascii=False))


if __name__ == "__main__":
    main()
