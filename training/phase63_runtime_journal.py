"""Fail-closed, pre-authorization runtime-journal reservations for PHASE63.

The journal identity is derived from the already validated authorization file,
not from a caller-supplied path.  Reserving its directory is the atomic
pre-claim collision check: a later authorization claim is therefore never
consumed merely because an old journal exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


SCHEMA = "phase63-runtime-journal-v1"
MARKER = "journal-reservation.json"


class RuntimeJournalError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _runtime_root(runtime: Path) -> Path:
    runtime = Path(runtime)
    root = runtime / "runs"
    if not root.is_dir() or root.is_symlink():
        raise RuntimeJournalError("RUNTIME_JOURNAL_ROOT_INVALID")
    resolved_runtime, resolved_root = runtime.resolve(), root.resolve()
    if resolved_root.parent != resolved_runtime:
        raise RuntimeJournalError("RUNTIME_JOURNAL_ROOT_ESCAPE")
    return resolved_root


def build_contract(authorization_path: Path, authorization_sha256: str, binding_sha256: str, *,
                   purpose: str, seed: int, arm: str, head: str,
                   source_manifest_sha256: str) -> dict[str, str | int]:
    """Derive a safe execution identity from immutable validated inputs only."""
    path = Path(authorization_path)
    if not re.fullmatch(r"[0-9a-f]{64}", authorization_sha256) or _sha(path) != authorization_sha256:
        raise RuntimeJournalError("JOURNAL_AUTHORIZATION_HASH_MISMATCH")
    try:
        auth = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeJournalError("JOURNAL_AUTHORIZATION_INVALID") from exc
    if (not isinstance(auth, dict) or auth.get("action") != purpose or auth.get("binding_sha256") != binding_sha256
            or auth.get("allowed_runs") != [{"seed": seed, "arm": arm}]
            or not isinstance(auth.get("authorization_id"), str)):
        raise RuntimeJournalError("JOURNAL_AUTHORIZATION_SCOPE_MISMATCH")
    identity_input = {"authorization_id": auth["authorization_id"], "authorization_sha256": authorization_sha256,
                      "binding_sha256": binding_sha256, "purpose": purpose, "seed": seed, "arm": arm}
    identity = hashlib.sha256(_canonical(identity_input)).hexdigest()
    run_id = f"phase63-{purpose}-seed{seed}-{arm}-{identity}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise RuntimeJournalError("JOURNAL_EXECUTION_ID_INVALID")
    return {"schema": SCHEMA, "execution_instance": run_id, "purpose": purpose, "authorization_id": auth["authorization_id"],
            "authorization_sha256": authorization_sha256, "binding_sha256": binding_sha256,
            "head": head, "source_manifest_sha256": source_manifest_sha256, "seed": seed, "arm": arm}


def journal_path(runtime: Path, contract: dict[str, Any]) -> Path:
    root = _runtime_root(runtime)
    run_id = contract.get("execution_instance")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise RuntimeJournalError("JOURNAL_EXECUTION_ID_INVALID")
    result = root / run_id
    if result.parent.resolve() != root or result.is_symlink():
        raise RuntimeJournalError("RUNTIME_JOURNAL_PATH_ESCAPE")
    return result


def reserve(runtime: Path, contract: dict[str, Any]) -> Path:
    """Atomically reserve a new journal directory before authorization claim."""
    result = journal_path(runtime, contract)
    try:
        result.mkdir()
    except FileExistsError as exc:
        raise RuntimeJournalError("RUNTIME_JOURNAL_COLLISION_PREAUTH") from exc
    try:
        marker = result / MARKER
        with marker.open("xb") as handle:
            handle.write(_canonical(contract)); handle.flush(); os.fsync(handle.fileno())
    except BaseException:
        # The directory remains auditable evidence; it is never removed or reused.
        raise RuntimeJournalError("RUNTIME_JOURNAL_RESERVATION_INCOMPLETE")
    validate_reserved(runtime, result, contract)
    return result


def validate_reserved(runtime: Path, path: Path, contract: dict[str, Any]) -> None:
    expected = journal_path(runtime, contract)
    actual = Path(path)
    if actual != expected or not actual.is_dir() or actual.is_symlink() or actual.resolve() != expected.resolve():
        raise RuntimeJournalError("RUNTIME_JOURNAL_CONTRACT_MISMATCH")
    marker = actual / MARKER
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeJournalError("RUNTIME_JOURNAL_RESERVATION_MISSING") from exc
    if value != contract:
        raise RuntimeJournalError("RUNTIME_JOURNAL_CONTRACT_MISMATCH")
