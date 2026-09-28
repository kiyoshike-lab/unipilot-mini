"""Fail-closed PHASE63 runtime binding and external authorization contract.

This is deliberately a CPU-only parser/validator.  It never imports torch,
owns a GPU lock, inventories a GPU, or starts a child process.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCHEMA = "phase63-runtime-binding-v2"
AUTHORIZATION_SCHEMA = "phase63-execution-authorization-v1"
PREREG_RELATIVE_PATH = "evaluation/phase62/phase63-clean-continuation-stability-preregistration.json"
RUNTIME_SOURCE_FILES = (
    "training/phase63_worker.py",
    "scripts/run_phase63_study.py",
    "training/exclusive_cuda_runner.py",
    "training/gpu_execution_lock.py",
    "training/gpu_execution_guard.py",
    "training/checkpoint_paths.py",
    "training/run_foundation_v36_lr_review.py",
    "training/phase63_binding_contract.py",
)


class BindingContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BindingContractError("BINDING_OR_AUTHORIZATION_UNREADABLE") from exc
    if not isinstance(value, dict):
        raise BindingContractError("BINDING_OR_AUTHORIZATION_NOT_OBJECT")
    return value


def current_source_manifest(root: Path = ROOT) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in RUNTIME_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise BindingContractError("RUNTIME_SOURCE_MISSING:" + relative)
        manifest[relative] = sha256_file(path)
    return manifest


def _sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise BindingContractError("INVALID_SHA256:" + label)


def _exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise BindingContractError("INVALID_FIELDS:" + label)
    return value


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise BindingContractError("INVALID_TIME:" + label)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BindingContractError("INVALID_TIME:" + label) from exc
    if result.tzinfo is None:
        raise BindingContractError("INVALID_TIME:" + label)
    return result.astimezone(UTC)


def validate_unarmed_binding(path: Path, *, checkpoint_root: Path, preregistration_path: Path,
                             expected_interpreter: Path = Path(sys.executable), source_root: Path = ROOT) -> dict[str, Any]:
    binding = read_object(path)
    required = {"schema_version", "phase", "status", "branch", "implementation_source_commit", "source_manifest",
                "source_manifest_sha256", "scientific_preregistration", "interpreter", "checkpoint_root", "runs",
                "contracts", "gates", "approved_processes", "execution_authorized", "no_train_gpu_authorized",
                "training_authorized"}
    _exact_keys(binding, required, "binding")
    if binding["schema_version"] != RUNTIME_SCHEMA or binding["phase"] != 63 or binding["status"] != "UNARMED_CPU_VALIDATED":
        raise BindingContractError("RUNTIME_BINDING_SCHEMA_OR_STATUS_MISMATCH")
    if binding["checkpoint_root"] != str(checkpoint_root):
        raise BindingContractError("RUNTIME_BINDING_CHECKPOINT_ROOT_MISMATCH")
    if any(binding[key] is not False for key in ("execution_authorized", "no_train_gpu_authorized", "training_authorized")):
        raise BindingContractError("BINDING_MUST_REMAIN_UNARMED")
    if binding["approved_processes"] != []:
        raise BindingContractError("BINDING_MUST_NOT_EMBED_PROCESS_APPROVALS")
    expected_manifest = current_source_manifest(source_root)
    if binding["source_manifest"] != expected_manifest or binding["source_manifest_sha256"] != canonical_sha256(expected_manifest):
        raise BindingContractError("RUNTIME_SOURCE_MANIFEST_MISMATCH")
    prereg = _exact_keys(binding["scientific_preregistration"], {"path", "sha256"}, "scientific_preregistration")
    if prereg["path"] != PREREG_RELATIVE_PATH or prereg["sha256"] != sha256_file(preregistration_path):
        raise BindingContractError("PREREGISTRATION_SHA_MISMATCH")
    interpreter = _exact_keys(binding["interpreter"], {"executable", "python_version", "torch_version", "torch_cuda_build"}, "interpreter")
    if Path(interpreter["executable"]).resolve() != Path(expected_interpreter).resolve():
        raise BindingContractError("WORKER_INTERPRETER_BINDING_MISMATCH")
    contracts = _exact_keys(binding["contracts"], {"guard_schema_version", "approval_schema_version", "no_train_receipt_schema_version"}, "contracts")
    if contracts["approval_schema_version"] != AUTHORIZATION_SCHEMA:
        raise BindingContractError("AUTHORIZATION_SCHEMA_MISMATCH")
    gates = _exact_keys(binding["gates"], {"gpu_guard", "cooling", "process_approvals", "generation_policy", "phase61"}, "gates")
    if gates["gpu_guard"] != "PHASE63_GPU_GUARD_CPU_VALIDATED" or gates["cooling"] != "PHASE63_COOLING_REVIEW_REQUIRED" or gates["process_approvals"] != "NONE":
        raise BindingContractError("RUNTIME_GATE_MISMATCH")
    if not isinstance(binding["runs"], list) or len(binding["runs"]) != 6:
        raise BindingContractError("RUN_REGISTRY_MISMATCH")
    seen: set[tuple[Any, Any]] = set()
    for row in binding["runs"]:
        _exact_keys(row, {"ordinal", "seed", "arm", "runtime_lr", "optimizer_updates", "parent_path", "parent_sha256", "output_path"}, "run")
        if isinstance(row["runtime_lr"], bool) or not isinstance(row["runtime_lr"], (int, float)) or not math.isfinite(row["runtime_lr"]):
            raise BindingContractError("INVALID_RUNTIME_LR")
        _sha(row["parent_sha256"], "parent")
        key = (row["seed"], row["arm"])
        if key in seen:
            raise BindingContractError("DUPLICATE_RUN")
        seen.add(key)
    return binding


def assert_sources_unchanged(binding: dict[str, Any], *, source_root: Path = ROOT) -> None:
    current = current_source_manifest(source_root)
    if binding.get("source_manifest") != current or binding.get("source_manifest_sha256") != canonical_sha256(current):
        raise BindingContractError("RUNTIME_SOURCE_CHANGED_AFTER_VALIDATION")


def authorize_execution(binding_path: Path, binding: dict[str, Any], authorization_path: Path | None,
                        authorization_sha256: str | None, *, action: str, seed: int, arm: str,
                        now: datetime | None = None,
                        process_identity_provider: Callable[[], list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    if authorization_path is None or authorization_sha256 is None:
        raise BindingContractError("EXTERNAL_EXECUTION_AUTHORIZATION_REQUIRED")
    _sha(authorization_sha256, "authorization_argument")
    if sha256_file(authorization_path) != authorization_sha256:
        raise BindingContractError("AUTHORIZATION_FILE_SHA_MISMATCH")
    auth = read_object(authorization_path)
    required = {"schema_version", "authorization_id", "binding_sha256", "phase", "action", "allowed_runs",
                "valid_from", "valid_until", "scope", "target_gpu", "approved_processes"}
    _exact_keys(auth, required, "authorization")
    if auth["schema_version"] != AUTHORIZATION_SCHEMA or auth["phase"] != 63 or auth["action"] != action:
        raise BindingContractError("AUTHORIZATION_ACTION_OR_SCHEMA_MISMATCH")
    _sha(auth["binding_sha256"], "authorization_binding")
    if auth["binding_sha256"] != sha256_file(binding_path):
        raise BindingContractError("AUTHORIZATION_BINDING_SHA_MISMATCH")
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    if not (_utc(auth["valid_from"], "valid_from") <= moment <= _utc(auth["valid_until"], "valid_until")):
        raise BindingContractError("AUTHORIZATION_EXPIRED_OR_NOT_YET_VALID")
    if auth["scope"] != "PHASE63_SINGLE_ACTION_SINGLE_RUN" or not isinstance(auth["target_gpu"], dict):
        raise BindingContractError("AUTHORIZATION_SCOPE_OR_GPU_MISMATCH")
    if auth["allowed_runs"] != [{"seed": seed, "arm": arm}]:
        raise BindingContractError("AUTHORIZATION_RUN_SCOPE_MISMATCH")
    approvals = auth["approved_processes"]
    if not isinstance(approvals, list):
        raise BindingContractError("AUTHORIZATION_PROCESS_LIST_INVALID")
    if approvals:
        if process_identity_provider is None:
            raise BindingContractError("CURRENT_PROCESS_IDENTITY_PROVIDER_REQUIRED")
        actual = process_identity_provider()
        if approvals != actual:
            raise BindingContractError("APPROVED_PROCESS_IDENTITY_CHANGED")
    return auth


def validate_execution(binding_path: Path, authorization_path: Path | None, authorization_sha256: str | None, *,
                       action: str, seed: int, arm: str, checkpoint_root: Path, preregistration_path: Path,
                       expected_interpreter: Path = Path(sys.executable), source_root: Path = ROOT,
                       now: datetime | None = None,
                       process_identity_provider: Callable[[], list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    binding = validate_unarmed_binding(binding_path, checkpoint_root=checkpoint_root,
                                       preregistration_path=preregistration_path,
                                       expected_interpreter=expected_interpreter, source_root=source_root)
    authorize_execution(binding_path, binding, authorization_path, authorization_sha256, action=action, seed=seed,
                        arm=arm, now=now, process_identity_provider=process_identity_provider)
    # A second read closes the validation-to-launch source TOCTOU window.
    assert_sources_unchanged(binding, source_root=source_root)
    return binding
