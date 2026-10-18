"""Durable, fail-closed one-time claims for PHASE63 external authorizations.

The claim file is intentionally adjacent to its authorization.  Exclusive
creation is the linearization point; an authorization is never reusable after
either a successful run or an aborted run that occurred after admission.
"""
from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CLAIM_SCHEMA = "phase63-authorization-claim-v1"


class AuthorizationClaimError(RuntimeError):
    pass


def claim_path(authorization_path: Path) -> Path:
    path = Path(authorization_path)
    return path.with_name(path.name + ".phase63-claim.json")


def _execution_instance(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise AuthorizationClaimError("AUTHORIZATION_EXECUTION_INSTANCE_INVALID")
    return value


def _expected(authorization: dict[str, Any], authorization_sha256: str, execution_instance: str) -> dict[str, Any]:
    return {
        "schema_version": CLAIM_SCHEMA,
        "authorization_sha256": authorization_sha256,
        "authorization_id": authorization["authorization_id"],
        "binding_sha256": authorization["binding_sha256"],
        "phase": authorization["phase"],
        "action": authorization["action"],
        "allowed_runs": authorization["allowed_runs"],
        "execution_instance": _execution_instance(execution_instance),
    }


def _read_claim(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorizationClaimError("AUTHORIZATION_CLAIM_STATE_CORRUPT") from exc
    if not isinstance(value, dict) or set(value) != {"claim", "claimed_at_utc"} or not isinstance(value["claim"], dict):
        raise AuthorizationClaimError("AUTHORIZATION_CLAIM_STATE_CORRUPT")
    try:
        datetime.fromisoformat(str(value["claimed_at_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuthorizationClaimError("AUTHORIZATION_CLAIM_STATE_CORRUPT") from exc
    return value


def claim(authorization_path: Path, authorization: dict[str, Any], authorization_sha256: str, *,
          execution_instance: str) -> dict[str, Any]:
    """Atomically consume a validated authorization, or fail closed.

    The caller must first validate immutable fields.  No deletion or release
    operation exists: a claim survives process restart and failed execution.
    """
    target = claim_path(authorization_path)
    expected = _expected(authorization, authorization_sha256, execution_instance)
    record = {"claim": expected, "claimed_at_utc": datetime.now(UTC).isoformat()}
    encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = _read_claim(target)
        if existing["claim"] == expected:
            raise AuthorizationClaimError("AUTHORIZATION_ALREADY_CONSUMED")
        raise AuthorizationClaimError("AUTHORIZATION_CLAIM_STATE_CONFLICT")
    except OSError as exc:
        raise AuthorizationClaimError("AUTHORIZATION_CLAIM_WRITE_FAILED") from exc
    return record


def assert_claimed(authorization_path: Path, authorization: dict[str, Any], authorization_sha256: str, *,
                   execution_instance: str) -> dict[str, Any]:
    """Require the immutable claim made by the supervisor before child launch."""
    expected = _expected(authorization, authorization_sha256, execution_instance)
    target = claim_path(authorization_path)
    if not target.exists():
        raise AuthorizationClaimError("AUTHORIZATION_NOT_CLAIMED")
    existing = _read_claim(target)
    if existing["claim"] != expected:
        raise AuthorizationClaimError("AUTHORIZATION_CLAIM_MISMATCH")
    return existing
