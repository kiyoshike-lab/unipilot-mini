"""Fail-closed, CPU-testable contracts for UniPilot GPU execution.

This module does not query CUDA, create a CUDA context, acquire a GPU lock, or
start a process.  The supervisor injects inventory/identity/lock providers.
Production callers must treat a provider error or any missing field as a block.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time


class GuardBlocked(RuntimeError):
    """The evidence is insufficient to safely continue."""


IDENTITY_FIELDS = ("pid", "process_start_time", "hostname", "exe", "exe_sha256")
APPROVAL_SCHEMA = "phase63-gpu-exception-approval-v1"
NO_TRAIN_ACTION = "no-train-gpu-supervisor"
NO_TRAIN_STATES = ("PREPARED", "LOCKED", "RUNNING", "PROCESS_EXITED", "RECEIPT_WRITTEN", "RELEASED")


def canonical_exe(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise GuardBlocked("PROCESS_IDENTITY_EXE_REQUIRED")
    return os.path.normcase(os.path.normpath(str(Path(value))))


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def sha256_file(path: object) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024**2), b""):
                digest.update(block)
    except OSError as exc:
        raise GuardBlocked("PROCESS_IDENTITY_HASH_UNAVAILABLE") from exc
    return digest.hexdigest()


def normalize_identity(value: object, *, require_hash: bool = True) -> dict:
    if not isinstance(value, dict):
        raise GuardBlocked("PROCESS_IDENTITY_MALFORMED")
    result = dict(value)
    if type(result.get("pid")) is not int or result["pid"] <= 0:
        raise GuardBlocked("PROCESS_IDENTITY_PID_REQUIRED")
    if not _finite_number(result.get("process_start_time")):
        raise GuardBlocked("PROCESS_IDENTITY_START_REQUIRED")
    if not isinstance(result.get("hostname"), str) or not result["hostname"]:
        raise GuardBlocked("PROCESS_IDENTITY_HOST_REQUIRED")
    result["exe"] = canonical_exe(result.get("exe"))
    digest = result.get("exe_sha256")
    if require_hash and (not isinstance(digest, str) or len(digest) != 64 or
                         any(c not in "0123456789abcdef" for c in digest.casefold())):
        raise GuardBlocked("PROCESS_IDENTITY_HASH_REQUIRED")
    if isinstance(digest, str):
        result["exe_sha256"] = digest.casefold()
    return result


def identities_match(expected: object, observed: object, *, require_hash: bool = True) -> bool:
    try:
        left = normalize_identity(expected, require_hash=require_hash)
        right = normalize_identity(observed, require_hash=require_hash)
    except GuardBlocked:
        return False
    return all(left[field] == right[field] for field in IDENTITY_FIELDS if require_hash or field != "exe_sha256")


def validate_approval(approval: object, observed: object, *, now: float | None = None,
                      phase: int = 63, action: str = NO_TRAIN_ACTION,
                      gpu_uuid: str | None = None) -> dict:
    """Validate one current-instance, one-use, NO_TRAIN-only exception.

    A valid approval only excludes that identified C+G instance from the
    *identity* block. It never proves that unrelated compute cannot occur.
    """
    if not isinstance(approval, dict) or approval.get("schema") != APPROVAL_SCHEMA:
        raise GuardBlocked("APPROVAL_SCHEMA_INVALID")
    required = ("approval_id", "purpose", "process_identity", "created_at", "expires_at",
                "allowed_phase", "allowed_action", "single_use", "evidence_reference")
    if any(key not in approval for key in required):
        raise GuardBlocked("APPROVAL_FIELD_MISSING")
    if (not isinstance(approval["approval_id"], str) or not approval["approval_id"] or
            not isinstance(approval["purpose"], str) or not approval["purpose"] or
            not isinstance(approval["evidence_reference"], str) or not approval["evidence_reference"]):
        raise GuardBlocked("APPROVAL_FIELD_INVALID")
    if (approval["purpose"] != action or approval["allowed_phase"] != phase or approval["allowed_action"] != action or
            approval["single_use"] is not True or approval.get("revoked") is True or
            approval.get("consumed") is True):
        raise GuardBlocked("APPROVAL_SCOPE_DENIED")
    if not _finite_number(approval["created_at"]) or not _finite_number(approval["expires_at"]):
        raise GuardBlocked("APPROVAL_TIME_INVALID")
    moment = time.time() if now is None else now
    if not _finite_number(moment) or approval["created_at"] > moment or approval["expires_at"] <= moment:
        raise GuardBlocked("APPROVAL_EXPIRED_OR_NOT_YET_VALID")
    if gpu_uuid is not None and approval.get("gpu_uuid") != gpu_uuid:
        raise GuardBlocked("APPROVAL_GPU_SCOPE_DENIED")
    expected = normalize_identity(approval["process_identity"])
    current = normalize_identity(observed)
    if not identities_match(expected, current):
        raise GuardBlocked("APPROVAL_IDENTITY_MISMATCH")
    return {"approval_id": approval["approval_id"], "expires_at": approval["expires_at"],
            "identity": current, "purpose": approval["purpose"]}


class ExternalComputeMonitor:
    """Re-query a supplied inventory and reject every unknown/unsafe observation."""
    def __init__(self, inventory_provider, identity_provider, *, approvals=(), now=time.time):
        self.inventory_provider = inventory_provider
        self.identity_provider = identity_provider
        self.approvals = tuple(approvals)
        self.now = now

    def _approval_for(self, observed: dict, gpu_uuid: str | None) -> dict | None:
        matches = []
        for candidate in self.approvals:
            try:
                matches.append(validate_approval(candidate, observed, now=self.now(), gpu_uuid=gpu_uuid))
            except GuardBlocked:
                continue
        if len(matches) != 1:
            return None
        return matches[0]

    def observe(self) -> dict:
        try:
            rows = self.inventory_provider()
        except BaseException as exc:
            raise GuardBlocked("EXTERNAL_COMPUTE_QUERY_FAILED") from exc
        if not isinstance(rows, list):
            raise GuardBlocked("EXTERNAL_COMPUTE_INVENTORY_MALFORMED")
        approved, blocking = [], []
        for row in rows:
            if not isinstance(row, dict) or row.get("type") not in ("G", "C", "C+G"):
                raise GuardBlocked("EXTERNAL_COMPUTE_INVENTORY_AMBIGUOUS")
            if row["type"] == "G":
                continue
            try:
                live = self.identity_provider(row["pid"])
            except BaseException as exc:
                raise GuardBlocked("EXTERNAL_COMPUTE_IDENTITY_QUERY_FAILED") from exc
            if live is None:
                raise GuardBlocked("EXTERNAL_COMPUTE_PROCESS_DISAPPEARED")
            try:
                live = normalize_identity(live)
            except GuardBlocked as exc:
                raise GuardBlocked("EXTERNAL_COMPUTE_IDENTITY_AMBIGUOUS") from exc
            # Pure C is never display/approval exempt. C+G requires a fresh,
            # exact, user-issued exception. Names and signatures are not used.
            match = self._approval_for(live, row.get("cuda_device")) if row["type"] == "C+G" else None
            if match is None:
                blocking.append({**row, "identity": live})
            else:
                approved.append({"row": row, "approval": match})
        if blocking:
            raise GuardBlocked("EXTERNAL_COMPUTE_PROCESS_PRESENT")
        return {"observed_at": self.now(), "approved": approved, "rows": rows}


def assert_live_lock_owner(lock_path: object, *, expected_nonce: object, expected_run_id: object,
                           expected_phase: int, owner_identity_provider, parent_identity_provider,
                           job_membership_provider) -> dict:
    """Fail closed before a worker initializes CUDA or restores CUDA RNG."""
    if not isinstance(expected_nonce, str) or not expected_nonce or not isinstance(expected_run_id, str) or not expected_run_id:
        raise GuardBlocked("WORKER_OWNER_ENV_MISSING")
    try:
        owner = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise GuardBlocked("WORKER_LOCK_MISSING_OR_MALFORMED") from exc
    if (owner.get("nonce") != expected_nonce or owner.get("run_id") != expected_run_id or
            owner.get("phase") != expected_phase):
        raise GuardBlocked("WORKER_LOCK_SCOPE_MISMATCH")
    # Owner identity itself has no file hash in the v1 lock schema. Its process
    # identity is still bound to PID/start/host/executable and the random nonce.
    owner_live = owner_identity_provider(owner.get("pid"))
    if not identities_match(owner, owner_live, require_hash=False):
        raise GuardBlocked("WORKER_LOCK_OWNER_NOT_LIVE")
    parent_live = parent_identity_provider()
    if not identities_match(owner, parent_live, require_hash=False):
        raise GuardBlocked("WORKER_PARENT_IDENTITY_MISMATCH")
    try:
        in_job = job_membership_provider(owner.get("job_name"))
    except BaseException as exc:
        raise GuardBlocked("WORKER_JOB_MEMBERSHIP_UNAVAILABLE") from exc
    if in_job is not True:
        raise GuardBlocked("WORKER_NOT_IN_OWNER_JOB")
    return owner


def windows_current_process_in_job(job_name: object) -> bool:
    if os.name != "nt" or not isinstance(job_name, str) or not job_name:
        raise GuardBlocked("WORKER_JOB_MEMBERSHIP_UNAVAILABLE")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.OpenJobObjectW.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenJobObjectW(4, False, job_name)  # JOB_OBJECT_QUERY
    if not handle:
        raise GuardBlocked("WORKER_JOB_MEMBERSHIP_UNAVAILABLE")
    try:
        result = wintypes.BOOL()
        if not kernel.IsProcessInJob(ctypes.c_void_p(-1), handle, ctypes.byref(result)):
            raise GuardBlocked("WORKER_JOB_MEMBERSHIP_UNAVAILABLE")
        return bool(result.value)
    finally:
        kernel.CloseHandle(handle)


def validate_no_train_receipt(receipt: object, *, expected_owner_nonce: str | None = None) -> bool:
    """Strictly validate a synthetic or future live NO_TRAIN receipt, CPU-only."""
    try:
        if not isinstance(receipt, dict) or receipt.get("schema") != "phase63-no-train-receipt-v1":
            return False
        if receipt.get("kind") != NO_TRAIN_ACTION or receipt.get("status") != "COMPLETE":
            return False
        if receipt.get("states") != list(NO_TRAIN_STATES) or receipt.get("exit_code") != 0:
            return False
        for key in ("optimizer_steps", "backward_calls", "checkpoint_count", "training_examples_consumed"):
            if type(receipt.get(key)) is not int or receipt[key] != 0:
                return False
        bool_fields = ("child_tree_exited", "stdout_closed", "stderr_closed", "gpu_context_released",
                       "lock_lifecycle_complete", "approved_process_identities_valid",
                       "no_new_unapproved_compute_process")
        if any(receipt.get(key) is not True for key in bool_fields):
            return False
        if expected_owner_nonce is not None and receipt.get("owner_nonce") != expected_owner_nonce:
            return False
        if not isinstance(receipt.get("owner_nonce"), str) or not receipt["owner_nonce"]:
            return False
        if not _finite_number(receipt.get("completed_at")) or not _finite_number(receipt.get("lease_observed_at")):
            return False
        for key in ("worker_sha256", "validator_sha256", "supervisor_sha256", "guard_sha256",
                    "command_sha256", "parent_sha256", "journal_sha256"):
            value = receipt.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.casefold()):
                return False
        if not isinstance(receipt.get("device"), str) or not receipt["device"]:
            return False
        streams = receipt.get("streams")
        if not isinstance(streams, dict) or any(not isinstance(streams.get(key), str) or len(streams[key]) != 64
                                               for key in ("stdout_sha256", "stderr_sha256")):
            return False
        return True
    except (TypeError, ValueError, KeyError):
        return False
