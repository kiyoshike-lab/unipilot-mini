"""PHASE63L CPU-only, unarmed execution-binding preparation.

This module deliberately has no GPU inventory, lock, CUDA, worker, or training
entry point.  It reads immutable inputs, prepares an unarmed manifest, and
validates that manifest against the same CPU-derived expectations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "evaluation" / "phase63" / "phase63l"
PREREG_PATH = ROOT / "evaluation" / "phase62" / "phase63-clean-continuation-stability-preregistration.json"
OLD_BINDING_PATH = ROOT / "evaluation" / "phase63" / "execution-binding.json"
CHECKPOINT_ROOT = Path(r"Z:\AI\unipilot-mini\checkpoints")
CACHE_PATH = Path(r"Z:\AI\unipilot-mini\evaluation\phase57\cache-raw.json")
CACHE_SHA256 = "4978d51cb66326bccee71a966d6337555387ce74f5ac2d4415506076a69416dc"
EXECUTION_FILES = {
    "worker": "training/phase63_worker.py",
    "supervisor": "scripts/run_phase63_study.py",
    "validator": "scripts/run_phase63_study.py",
    "exclusive_cuda_runner": "training/exclusive_cuda_runner.py",
    "gpu_execution_lock": "training/gpu_execution_lock.py",
    "gpu_execution_guard": "training/gpu_execution_guard.py",
    "checkpoint_path_contract": "training/checkpoint_paths.py",
    "strict_checkpoint_validator": "training/run_foundation_v36_lr_review.py",
}
SCHEMA = "phase63-execution-binding-candidate-v1"


class ValidationError(ValueError):
    """Raised when an execution-binding candidate is not exactly unarmed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_cpu_only_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValidationError("CUDA_VISIBLE_DEVICES_MUST_BE_MINUS_ONE")


def source_manifest() -> dict[str, str]:
    return {role: sha256_file(ROOT / relative) for role, relative in EXECUTION_FILES.items()}


def scientific_integrity(prereg: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for relative, expected in prereg["source_sha256"].items():
        actual = sha256_file(ROOT / relative) if (ROOT / relative).is_file() else None
        checks.append({"kind": "scientific_source", "path": relative,
                       "expected_sha256": expected, "actual_sha256": actual,
                       "pass": actual == expected})
    data = prereg["data"]
    data_path = ROOT / data["path"]
    data_actual = sha256_file(data_path) if data_path.is_file() else None
    checks.append({"kind": "dataset", "path": data["path"],
                   "expected_sha256": data["sha256"], "actual_sha256": data_actual,
                   "pass": data_actual == data["sha256"]})
    tokenizer_path = ROOT / "tokenizer" / "foundation-v11-base-4096.json"
    tokenizer_expected = prereg["source_sha256"]["tokenizer/foundation-v11-base-4096.json"]
    tokenizer_actual = sha256_file(tokenizer_path) if tokenizer_path.is_file() else None
    checks.append({"kind": "tokenizer", "path": "tokenizer/foundation-v11-base-4096.json",
                   "expected_sha256": tokenizer_expected, "actual_sha256": tokenizer_actual,
                   "pass": tokenizer_actual == tokenizer_expected})
    cache_actual = sha256_file(CACHE_PATH) if CACHE_PATH.is_file() else None
    checks.append({"kind": "matching_cache", "path": str(CACHE_PATH),
                   "expected_sha256": CACHE_SHA256, "actual_sha256": cache_actual,
                   "pass": cache_actual == CACHE_SHA256})
    return {"preregistration_path": str(PREREG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "preregistration_sha256": sha256_file(PREREG_PATH), "checks": checks,
            "pass": all(row["pass"] for row in checks)}


def parent_integrity(prereg: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for parent in prereg["parents"]:
        path = Path(parent["path"])
        present = path.is_file()
        actual = sha256_file(path) if present else None
        rows.append({"seed": parent["seed"], "path": str(path), "present": present,
                     "file_size": path.stat().st_size if present else None,
                     "expected_sha256": parent["sha256"], "actual_sha256": actual,
                     "sha256_pass": actual == parent["sha256"],
                     "registered_tokens": parent["tokens"],
                     "cpu_strict_state_validation": "PENDING"})
    return {"checkpoint_root": str(CHECKPOINT_ROOT), "parents": rows,
            "pass": all(row["sha256_pass"] for row in rows)}


def _strict_parent_payload(path: Path, seed: int) -> dict[str, Any]:
    """Strictly reload a trusted parent on CPU, without restoring CUDA RNG."""
    import torch  # Importing torch is CPU-only here; no torch.cuda API is called.
    from training.run_foundation_v36_lr_review import verify_payload
    payload = torch.load(path, map_location="cpu", weights_only=False)
    report = verify_payload(payload, seed, 16_384_000, 5e-5)
    return {"pass": bool(report.get("pass")), "checks": report.get("checks", {})}


def add_cpu_strict_parent_validation(integrity: dict[str, Any]) -> dict[str, Any]:
    for row in integrity["parents"]:
        if not row["sha256_pass"]:
            row["cpu_strict_state_validation"] = "NOT_RUN_SHA_MISMATCH"
            continue
        result = _strict_parent_payload(Path(row["path"]), int(row["seed"]))
        row["cpu_strict_state_validation"] = result
    integrity["pass"] = bool(integrity["pass"]) and all(
        isinstance(row["cpu_strict_state_validation"], dict) and row["cpu_strict_state_validation"]["pass"]
        for row in integrity["parents"]
    )
    return integrity


def interpreter_metadata() -> dict[str, Any]:
    import torch  # Deliberately no torch.cuda call.
    return {"sys_executable": sys.executable, "python_version": sys.version,
            "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "cuda_api_calls": 0, "actual_gpu_availability": "NOT_RECHECKED"}


def run_rows(prereg: dict[str, Any]) -> list[dict[str, Any]]:
    parents = {int(row["seed"]): row for row in prereg["parents"]}
    rows: list[dict[str, Any]] = []
    for ordinal, item in enumerate(prereg["run_order"], start=1):
        seed, arm = int(item["seed"]), str(item["arm"])
        if arm not in {"control", "half-lr"}:
            raise ValidationError("UNKNOWN_REGISTERED_ARM")
        lr = 5e-5 if arm == "control" else 2.5e-5
        output = CHECKPOINT_ROOT / "experimental" / "phase63" / f"seed-{seed}" / arm / "checkpoint-tokens-16777216.pt"
        rows.append({"ordinal": ordinal, "seed": seed, "arm": arm, "runtime_lr": lr,
                     "optimizer_updates": 122, "parent_path": parents[seed]["path"],
                     "parent_sha256": parents[seed]["sha256"], "future_output_path": str(output)})
    return rows


def expected_contract(prereg: dict[str, Any], manifest: dict[str, str], metadata: dict[str, Any]) -> dict[str, Any]:
    return {"branch": git("branch", "--show-current"), "implementation_source_commit": git("rev-parse", "HEAD"),
            "source_manifest": manifest, "source_manifest_sha256": canonical_sha256(manifest),
            "scientific_preregistration_sha256": sha256_file(PREREG_PATH),
            "interpreter": metadata, "runs": run_rows(prereg),
            "checkpoint_root": str(CHECKPOINT_ROOT), "data_path": prereg["data"]["path"],
            "data_sha256": prereg["data"]["sha256"],
            "tokenizer_sha256": prereg["source_sha256"]["tokenizer/foundation-v11-base-4096.json"],
            "matching_cache_sha256": CACHE_SHA256}


def build_candidate(expected: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA, "phase": 63, "status": "CPU_PREPARATION_ONLY",
            "branch": expected["branch"], "implementation_source_commit": expected["implementation_source_commit"],
            "source_manifest": expected["source_manifest"], "source_manifest_sha256": expected["source_manifest_sha256"],
            "scientific_preregistration": {"path": "evaluation/phase62/phase63-clean-continuation-stability-preregistration.json",
                "sha256": expected["scientific_preregistration_sha256"]},
            "interpreter": expected["interpreter"], "checkpoint_root": expected["checkpoint_root"],
            "inputs": {"dataset_path": expected["data_path"], "dataset_sha256": expected["data_sha256"],
                       "tokenizer_sha256": expected["tokenizer_sha256"],
                       "matching_cache_sha256": expected["matching_cache_sha256"]},
            "runs": expected["runs"],
            "contracts": {"guard_schema_version": "phase63-gpu-guard-contract-v2",
                          "approval_schema_version": "phase63-gpu-exception-approval-v1",
                          "no_train_receipt_schema_version": "phase63-no-train-receipt-v1"},
            "gates": {"gpu_guard": "PHASE63_GPU_GUARD_CPU_VALIDATED",
                      "cooling": "PHASE63_COOLING_REVIEW_REQUIRED", "process_approvals": "NONE",
                      "old_execution_binding": "OLD_STALE", "generation_policy": "UNSAFE",
                      "phase61": "EXPERIMENT_INVALID"},
            "approved_processes": [], "execution_authorized": False,
            "no_train_gpu_authorized": False, "training_authorized": False}


def _sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValidationError(f"INVALID_SHA256_{label}")


def _finite_number(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"INVALID_NUMBER_{label}")


def validate_candidate(candidate: Any, expected: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValidationError("CANDIDATE_NOT_OBJECT")
    required = {"schema_version", "phase", "status", "branch", "implementation_source_commit", "source_manifest",
                "source_manifest_sha256", "scientific_preregistration", "interpreter", "checkpoint_root", "inputs", "runs",
                "contracts", "gates", "approved_processes", "execution_authorized", "no_train_gpu_authorized", "training_authorized"}
    if set(candidate) != required:
        raise ValidationError("TOP_LEVEL_FIELDS_MISMATCH")
    if candidate["schema_version"] != SCHEMA or candidate["phase"] != 63 or candidate["status"] != "CPU_PREPARATION_ONLY":
        raise ValidationError("SCHEMA_OR_PHASE_MISMATCH")
    for key in ("branch", "implementation_source_commit", "checkpoint_root"):
        if candidate[key] != expected[key]:
            raise ValidationError(f"EXPECTED_VALUE_MISMATCH_{key}")
    if candidate["source_manifest"] != expected["source_manifest"]:
        raise ValidationError("SOURCE_MANIFEST_MISMATCH")
    for role, digest in candidate["source_manifest"].items():
        _sha(digest, role)
    if candidate["source_manifest_sha256"] != expected["source_manifest_sha256"]:
        raise ValidationError("SOURCE_MANIFEST_SHA_MISMATCH")
    prereg = candidate["scientific_preregistration"]
    if not isinstance(prereg, dict) or prereg != {"path": "evaluation/phase62/phase63-clean-continuation-stability-preregistration.json",
                                                   "sha256": expected["scientific_preregistration_sha256"]}:
        raise ValidationError("PREREGISTRATION_MISMATCH")
    if candidate["interpreter"] != expected["interpreter"]:
        raise ValidationError("INTERPRETER_MISMATCH")
    inputs = candidate["inputs"]
    expected_inputs = {"dataset_path": expected["data_path"], "dataset_sha256": expected["data_sha256"],
                       "tokenizer_sha256": expected["tokenizer_sha256"], "matching_cache_sha256": expected["matching_cache_sha256"]}
    if inputs != expected_inputs:
        raise ValidationError("INPUTS_MISMATCH")
    for key, digest in inputs.items():
        if key.endswith("sha256"):
            _sha(digest, key)
    if candidate["runs"] != expected["runs"] or len(candidate["runs"]) != 6:
        raise ValidationError("RUN_CONTRACT_MISMATCH")
    seen: set[tuple[int, str]] = set()
    for run in candidate["runs"]:
        if not isinstance(run, dict):
            raise ValidationError("RUN_NOT_OBJECT")
        _finite_number(run["runtime_lr"], "runtime_lr")
        if run["runtime_lr"] not in (5e-5, 2.5e-5) or not isinstance(run["optimizer_updates"], int) or run["optimizer_updates"] != 122:
            raise ValidationError("RUN_LR_OR_UPDATES_MISMATCH")
        identity = (run["seed"], run["arm"])
        if identity in seen:
            raise ValidationError("DUPLICATE_RUN")
        seen.add(identity)
        _sha(run["parent_sha256"], "parent")
        output = run["future_output_path"]
        if not isinstance(output, str) or not output.startswith(expected["checkpoint_root"] + "\\experimental\\phase63\\"):
            raise ValidationError("INVALID_OUTPUT_PATH")
    if candidate["contracts"] != {"guard_schema_version": "phase63-gpu-guard-contract-v2",
                                  "approval_schema_version": "phase63-gpu-exception-approval-v1",
                                  "no_train_receipt_schema_version": "phase63-no-train-receipt-v1"}:
        raise ValidationError("CONTRACTS_MISMATCH")
    if candidate["gates"] != {"gpu_guard": "PHASE63_GPU_GUARD_CPU_VALIDATED",
                              "cooling": "PHASE63_COOLING_REVIEW_REQUIRED", "process_approvals": "NONE",
                              "old_execution_binding": "OLD_STALE", "generation_policy": "UNSAFE",
                              "phase61": "EXPERIMENT_INVALID"}:
        raise ValidationError("GATES_MISMATCH")
    if candidate["approved_processes"] != []:
        raise ValidationError("APPROVALS_NOT_EMPTY")
    if any(candidate[key] is not False for key in ("execution_authorized", "no_train_gpu_authorized", "training_authorized")):
        raise ValidationError("CANDIDATE_MUST_BE_UNARMED")
    return {"pass": True, "schema": SCHEMA, "run_count": 6, "unarmed": True}


def write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("xb") as handle:
        handle.write(canonical_bytes(value))


def generate() -> dict[str, Any]:
    require_cpu_only_environment()
    if OUT.exists():
        raise FileExistsError("PHASE63L_OUTPUT_ALREADY_EXISTS")
    prereg = read_json(PREREG_PATH)
    scientific = scientific_integrity(prereg)
    parents = parent_integrity(prereg)
    if not scientific["pass"] or not parents["pass"]:
        raise ValidationError("INPUT_INTEGRITY_BLOCKED")
    parents = add_cpu_strict_parent_validation(parents)
    if not parents["pass"]:
        raise ValidationError("CPU_STRICT_PARENT_VALIDATION_FAILED")
    metadata = interpreter_metadata()
    manifest = source_manifest()
    expected = expected_contract(prereg, manifest, metadata)
    candidate = build_candidate(expected)
    validation = validate_candidate(candidate, expected)
    write_new_json(OUT / "execution-binding-candidate.json", candidate)
    candidate_sha = sha256_file(OUT / "execution-binding-candidate.json")
    write_new_json(OUT / "binding-verification.json", {"candidate_path": "execution-binding-candidate.json",
        "candidate_sha256": candidate_sha, "canonical_utf8": True, "self_hash_embedded_in_candidate": False,
        "strict_validation": validation})
    write_new_json(OUT / "source-hash-manifest.json", {"source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest), "implementation_source_commit": expected["implementation_source_commit"],
        "old_binding_status": "STALE"})
    write_new_json(OUT / "scientific-integrity-check.json", scientific)
    write_new_json(OUT / "parent-checkpoint-integrity.json", parents)
    return {"candidate_sha256": candidate_sha, "source_manifest_sha256": canonical_sha256(manifest),
            "scientific_integrity": scientific, "parent_integrity": parents, "validation": validation}


def stale_binding_audit() -> dict[str, Any]:
    """Audit legacy acceptance without a worker process, lock, or CUDA call."""
    require_cpu_only_environment()
    old = read_json(OLD_BINDING_PATH)
    current_by_path = {relative: sha256_file(ROOT / relative) for relative in EXECUTION_FILES.values()}
    old_manifest = old.get("source_manifest", {})
    source_comparison = [{"path": path, "old_sha256": old_manifest.get(path),
                          "current_sha256": current, "match": old_manifest.get(path) == current}
                         for path, current in sorted(current_by_path.items())]
    worker_result: dict[str, Any]
    try:
        # This is a direct parser call only.  It intentionally does not call
        # require_live_gpu_owner, acquire a lock, or invoke a worker mode.
        from training.phase63_worker import read_binding
        read_binding(OLD_BINDING_PATH, 42, "control")
        worker_result = {"accepted": True, "cuda_context_created": False,
                         "lock_acquired": False, "worker_mode_invoked": False}
    except Exception as exc:  # The audit records the actual fail-closed result.
        worker_result = {"accepted": False, "exception_type": type(exc).__name__, "reason": str(exc),
                         "cuda_context_created": False, "lock_acquired": False, "worker_mode_invoked": False}
    result = {"old_binding_path": str(OLD_BINDING_PATH.relative_to(ROOT)).replace("\\", "/"),
              "old_binding_sha256": sha256_file(OLD_BINDING_PATH), "old_execution_authorized": old.get("execution_authorized"),
              "old_implementation_head": old.get("implementation_head"), "current_implementation_head": git("rev-parse", "HEAD"),
              "source_comparison": source_comparison,
              "source_mismatch_present": any(not row["match"] for row in source_comparison),
              "worker_read_binding_cpu_only": worker_result,
              "worker_source_manifest_enforced": False,
              "supervisor_require_preflight_source_manifest_check": True,
              "finding": "BINDING_RUNTIME_ENFORCEMENT_GAP" if worker_result["accepted"] else "WORKER_REJECTED_LEGACY_BINDING",
              "scope": "No CUDA API, GPU lock, inventory, worker CLI, NO_TRAIN, or training was invoked."}
    write_new_json(OUT / "stale-binding-rejection-audit.json", result)
    return result


def _junit_result(path: Path, exit_code: int, output: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    return {"exit_code": exit_code, "tests": counts["tests"],
            "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"],
            "failed": counts["failures"], "errors": counts["errors"], "skipped": counts["skipped"],
            "junit_path": str(path.relative_to(OUT)).replace("\\", "/"), "junit_sha256": sha256_file(path),
            "summary": output.strip().splitlines()[-1] if output.strip() else "NO_PYTEST_OUTPUT"}


def run_cpu_qa() -> dict[str, Any]:
    """Run self-contained CPU-only pytest and persist the actual exit codes."""
    require_cpu_only_environment()
    if (OUT / "cpu-binding-test-results.json").exists():
        raise FileExistsError("CPU_BINDING_QA_ALREADY_RECORDED")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
    results: dict[str, Any] = {"cuda_visible_devices": "-1", "gpu_execution": "NOT_RUN",
                                "cuda_context": "NOT_CREATED", "runs": {}}
    for name, args in (("targeted", ["tests/test_phase63l_binding.py"]), ("full", [])):
        xml = OUT / f"{name}-pytest-final.xml"
        command = [sys.executable, "-B", "-m", "pytest", "-q", *args, f"--junitxml={xml}"]
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        results["runs"][name] = _junit_result(xml, proc.returncode, proc.stdout)
        if proc.returncode != 0:
            break
    results["pass"] = all(row["exit_code"] == 0 for row in results["runs"].values()) and "full" in results["runs"]
    write_new_json(OUT / "cpu-binding-test-results.json", results)
    return results


def write_new_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(value, encoding="utf-8", newline="\n")


def preservation_check() -> dict[str, Any]:
    phase62 = read_json(ROOT / "evaluation" / "phase62" / "preflight.json")
    protected = phase62["protected"] + phase62["frozen"]
    protected_pass = all(sha256_file(ROOT / row["path"]) == row["sha256"] for row in protected)
    # The original 179 rows remain a subset; later PHASE63 evidence is already
    # user-owned dirty state and is deliberately not treated as ours.
    current = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True).splitlines()
    non_l = [line for line in current if not line[3:].replace("\\", "/").startswith(("evaluation/phase63/phase63l/", "scripts/phase63l_binding.py", "tests/test_phase63l_binding.py"))]
    original = {row["path"]: row["sha256"] for row in phase62["dirty"]}
    original_pass = all((ROOT / path).is_file() and sha256_file(ROOT / path) == digest
                        for path, digest in original.items() if digest is not None)
    return {"protected4_and_ready5_and_phase61_frozen": protected_pass,
            "preregistration_sha256": sha256_file(PREREG_PATH),
            "preregistration_pass": sha256_file(PREREG_PATH) == "856ed78873b469100f4c6c34df1c4d05c19b540969453fba8e41315a478bc48f",
            "original_dirty179_present_or_unchanged": original_pass,
            "later_user_owned_dirty_entries_excluding_phase63l": len(non_l)}


def emit_report() -> dict[str, Any]:
    require_cpu_only_environment()
    candidate = read_json(OUT / "execution-binding-candidate.json")
    prereg = read_json(PREREG_PATH)
    expected = expected_contract(prereg, source_manifest(), interpreter_metadata())
    validation = validate_candidate(candidate, expected)
    verification = read_json(OUT / "binding-verification.json")
    audit = read_json(OUT / "stale-binding-rejection-audit.json")
    qa = read_json(OUT / "cpu-binding-test-results.json")
    preservation = preservation_check()
    gate = "PHASE63L_BINDING_RUNTIME_ENFORCEMENT_GAP" if audit["finding"] == "BINDING_RUNTIME_ENFORCEMENT_GAP" else "PHASE63L_CPU_BINDING_PREPARED_UNARMED"
    report = "\n".join([
        "# PHASE63L CPU-only execution-binding preparation", "",
        f"Final gate: `{gate}`.", "",
        "## Result", "",
        "A fresh candidate was normalized as UTF-8 and strictly validated, but remains unarmed: `execution_authorized=false`, `no_train_gpu_authorized=false`, `training_authorized=false`, and `approved_processes=[]`.",
        "",
        "## Runtime enforcement audit", "",
        "The legacy binding has `execution_authorized=true` and stale worker/lock/runner source hashes. Its CPU-only `read_binding` parser accepted it despite those source mismatches. The supervisor has a source-manifest comparison, but the worker parser does not enforce it. This is a defense-in-depth `BINDING_RUNTIME_ENFORCEMENT_GAP`; no runtime code was changed in PHASE63L.",
        "",
        "## Immutable inputs", "",
        f"- Preregistration SHA256: `{sha256_file(PREREG_PATH)}`", f"- Execution source-manifest SHA256: `{canonical_sha256(source_manifest())}`",
        f"- Candidate SHA256: `{verification['candidate_sha256']}`", "- Three PHASE48 parents: SHA256 and CPU strict model/optimizer reload PASS.",
        "- Dataset, tokenizer, and matching cache SHA256: PASS.", "",
        "## QA", "",
        f"- Targeted: {qa['runs']['targeted']['summary']} (exit {qa['runs']['targeted']['exit_code']}).",
        f"- Full: {qa['runs']['full']['summary']} (exit {qa['runs']['full']['exit_code']}).",
        "- CUDA devices were hidden only within these test subprocesses; no CUDA API was called by the generator.", "",
        "## Safety state", "",
        "GPU execution: NOT_RUN. CUDA context: NOT_CREATED. NO_TRAIN: NOT_RUN. Training: NO. Optimizer steps: 0. PHASE63 checkpoints: 0.",
        "Cooling: PHASE63_COOLING_REVIEW_REQUIRED. GPU Guard: PHASE63_GPU_GUARD_CPU_VALIDATED. PHASE61: EXPERIMENT_INVALID. Generation Policy: UNSAFE.",
        "",
        "## Preservation", "",
        f"Protected4/READY5/PHASE61 frozen: {'PASS' if preservation['protected4_and_ready5_and_phase61_frozen'] else 'FAIL'}; original dirty179: {'PASS' if preservation['original_dirty179_present_or_unchanged'] else 'FAIL'}.",
        "",
        "No GPU lock, GPU process inventory, NO_TRAIN worker, training worker, checkpoint mutation, or execution authorization was used.", "",
    ])
    write_new_text(OUT / "binding-preparation-report.md", report)
    return {"gate": gate, "validation": validation, "preservation": preservation,
            "candidate_sha256": verification["candidate_sha256"], "qa_pass": qa["pass"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "audit-stale-binding", "run-cpu-qa", "emit-report"))
    args = parser.parse_args()
    if args.command == "generate":
        print(json.dumps(generate(), ensure_ascii=False, sort_keys=True))
    if args.command == "audit-stale-binding":
        print(json.dumps(stale_binding_audit(), ensure_ascii=False, sort_keys=True))
    if args.command == "run-cpu-qa":
        print(json.dumps(run_cpu_qa(), ensure_ascii=False, sort_keys=True))
    if args.command == "emit-report":
        print(json.dumps(emit_report(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
