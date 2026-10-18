"""Fail-closed PHASE63 runtime binding and external authorization contract.

This is deliberately a CPU-only parser/validator.  It never imports torch,
owns a GPU lock, inventories a GPU, or starts a child process.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from training.phase63_source_manifest import (
    MANDATORY_EXECUTION_ROOTS,
    SourceDependencyError,
    source_manifest,
)
from training.phase63_authorization_claim import AuthorizationClaimError, claim

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCHEMA = "phase63-runtime-binding-v3"
AUTHORIZATION_SCHEMA = "phase63-execution-authorization-v1"
AUTHORIZATION_ACTIONS = frozenset(("no_train", "training"))
PREREG_RELATIVE_PATH = "evaluation/phase62/phase63-clean-continuation-stability-preregistration.json"
K11_EVIDENCE_RELATIVE_PATH = "evaluation/phase63/phase63k11/runtime-baseline-decision.json"
RUNTIME_SOURCE_FILES = MANDATORY_EXECUTION_ROOTS


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


def trusted_current_implementation_commit() -> str:
    """Resolve the exact Git HEAD for this source tree, or fail closed."""
    try:
        top_level = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if Path(top_level).resolve() != ROOT.resolve():
            raise BindingContractError("IMPLEMENTATION_SOURCE_COMMIT_UNRESOLVED")
        value = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise BindingContractError("IMPLEMENTATION_SOURCE_COMMIT_UNRESOLVED") from exc
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise BindingContractError("IMPLEMENTATION_SOURCE_COMMIT_UNRESOLVED")
    return value


def _validate_implementation_source_commit(value: Any) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise BindingContractError("IMPLEMENTATION_SOURCE_COMMIT_FORMAT_INVALID")
    if value != trusted_current_implementation_commit():
        raise BindingContractError("IMPLEMENTATION_SOURCE_COMMIT_MISMATCH")


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BindingContractError("BINDING_OR_AUTHORIZATION_UNREADABLE") from exc
    if not isinstance(value, dict):
        raise BindingContractError("BINDING_OR_AUTHORIZATION_NOT_OBJECT")
    return value


def current_source_manifest(root: Path = ROOT) -> list[dict[str, int | str]]:
    try:
        return source_manifest(root, RUNTIME_SOURCE_FILES)
    except SourceDependencyError as exc:
        raise BindingContractError(str(exc)) from exc


def _sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise BindingContractError("INVALID_SHA256:" + label)


def _exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise BindingContractError("INVALID_FIELDS:" + label)
    return value


def _validate_cooling_evidence(value: Any, path: Path) -> None:
    evidence = _exact_keys(value, {"path", "sha256", "operational_gate", "runtime_baseline", "historical_profile"}, "cooling_evidence")
    if evidence["path"] != K11_EVIDENCE_RELATIVE_PATH or evidence["sha256"] != sha256_file(path):
        raise BindingContractError("COOLING_EVIDENCE_SHA_MISMATCH")
    if (evidence["operational_gate"] != "PHASE63_COOLING_RUNTIME_VERIFIED"
            or evidence["runtime_baseline"] != "ACCEPTED"
            or evidence["historical_profile"] != "HISTORICAL_PROFILE_DRIFT_CAUSE_UNKNOWN"):
        raise BindingContractError("COOLING_PREREQUISITE_MISMATCH")
    artifact = read_object(path)
    if (artifact.get("operational_cooling_gate") != evidence["operational_gate"]
            or artifact.get("current_runtime_baseline") != evidence["runtime_baseline"]
            or artifact.get("historical_profile_drift") != evidence["historical_profile"]):
        raise BindingContractError("COOLING_EVIDENCE_CONTENT_MISMATCH")


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
                             expected_interpreter: Path = Path(sys.executable), source_root: Path = ROOT,
                             cooling_evidence_path: Path = ROOT / K11_EVIDENCE_RELATIVE_PATH) -> dict[str, Any]:
    binding = read_object(path)
    required = {"schema_version", "phase", "status", "branch", "implementation_source_commit", "source_manifest",
                "source_manifest_sha256", "scientific_preregistration", "interpreter", "checkpoint_root", "runs",
                "contracts", "gates", "cooling_evidence", "approved_processes", "execution_authorized", "no_train_gpu_authorized",
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
    _validate_implementation_source_commit(binding["implementation_source_commit"])
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
    if (gates["gpu_guard"] != "PHASE63_GPU_GUARD_CPU_VALIDATED"
            or gates["cooling"] != "PHASE63_COOLING_RUNTIME_VERIFIED"
            or gates["process_approvals"] != "NONE"):
        raise BindingContractError("RUNTIME_GATE_MISMATCH")
    _validate_cooling_evidence(binding["cooling_evidence"], cooling_evidence_path)
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


def assert_implementation_commit_unchanged(binding: dict[str, Any]) -> None:
    _validate_implementation_source_commit(binding.get("implementation_source_commit"))


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
    if action not in AUTHORIZATION_ACTIONS or auth["schema_version"] != AUTHORIZATION_SCHEMA or auth["phase"] != 63 or auth["action"] != action:
        raise BindingContractError("AUTHORIZATION_ACTION_OR_SCHEMA_MISMATCH")
    if not isinstance(auth["authorization_id"], str) or not auth["authorization_id"]:
        raise BindingContractError("AUTHORIZATION_ID_INVALID")
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
                       cooling_evidence_path: Path = ROOT / K11_EVIDENCE_RELATIVE_PATH,
                       now: datetime | None = None,
                       process_identity_provider: Callable[[], list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    binding = validate_unarmed_binding(binding_path, checkpoint_root=checkpoint_root,
                                       preregistration_path=preregistration_path,
                                       expected_interpreter=expected_interpreter, source_root=source_root,
                                       cooling_evidence_path=cooling_evidence_path)
    authorize_execution(binding_path, binding, authorization_path, authorization_sha256, action=action, seed=seed,
                        arm=arm, now=now, process_identity_provider=process_identity_provider)
    # A second read closes the validation-to-launch source TOCTOU window.
    assert_sources_unchanged(binding, source_root=source_root)
    assert_implementation_commit_unchanged(binding)
    return binding


def admit_execution(binding_path: Path, authorization_path: Path | None, authorization_sha256: str | None, *,
                    action: str, seed: int, arm: str, execution_instance: str, checkpoint_root: Path,
                    preregistration_path: Path, expected_interpreter: Path = Path(sys.executable),
                    source_root: Path = ROOT, cooling_evidence_path: Path = ROOT / K11_EVIDENCE_RELATIVE_PATH,
                    now: datetime | None = None,
                    process_identity_provider: Callable[[], list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Validate and atomically consume one authorization before GPU lock admission."""
    binding = validate_execution(binding_path, authorization_path, authorization_sha256, action=action, seed=seed,
                                 arm=arm, checkpoint_root=checkpoint_root, preregistration_path=preregistration_path,
                                 expected_interpreter=expected_interpreter, source_root=source_root,
                                 cooling_evidence_path=cooling_evidence_path, now=now,
                                 process_identity_provider=process_identity_provider)
    assert authorization_path is not None and authorization_sha256 is not None
    authorization = read_object(authorization_path)
    try:
        claim(authorization_path, authorization, authorization_sha256, execution_instance=execution_instance)
    except AuthorizationClaimError as exc:
        raise BindingContractError(str(exc)) from exc
    # Recheck source and trusted HEAD after durable claim, before any lock/CUDA stage.
    assert_sources_unchanged(binding, source_root=source_root)
    assert_implementation_commit_unchanged(binding)
    return binding
