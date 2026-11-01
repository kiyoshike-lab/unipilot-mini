"""CPU-only O17 tests for canonical process-exception expiry production."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from training import gpu_execution_guard as guard
from training import phase63_binding_contract as contract


NOW = 1_800_000_000.0
BINDING = "a" * 64
HEAD = contract.trusted_current_implementation_commit()
MANIFEST = "b" * 64


def identity() -> dict:
    return {"pid": 71, "process_start_time": 100.0, "hostname": "test-host",
            "exe": r"C:\\fixture.exe", "exe_sha256": "c" * 64}


def authorization(*, requested: float | None = None, valid_until: datetime | str | None = None) -> dict:
    end = valid_until or datetime.fromtimestamp(NOW + 60, UTC)
    return contract.build_no_train_authorization(
        authorization_id="O17-test", binding_sha256=BINDING, implementation_head=HEAD,
        source_manifest_sha256=MANIFEST, target_gpu_uuid="GPU-test",
        allowed_runs=[{"seed": 42, "arm": "control"}],
        process_candidates=[{"exception_id": "O17-process", "process_identity": identity()}],
        valid_from=datetime.fromtimestamp(NOW - 10, UTC), valid_until=end,
        created_at=NOW, requested_exception_expiry=requested,
    )


def strict_check(auth: dict) -> dict:
    child = auth["external_process_exceptions"][0]
    _, parent_expiry = contract.canonical_authorization_time(auth["valid_until"], "valid_until")
    return guard.validate_external_process_exception(
        child, authorization_id=auth["authorization_id"], binding_sha256=BINDING,
        implementation_head=HEAD, source_manifest_sha256=MANIFEST,
        action=guard.NO_TRAIN_ACTION, authorization_expires_at=parent_expiry, now=NOW,
    )


def test_pre_fix_independent_high_precision_child_expiry_reproduces_strict_rejection() -> None:
    parent = datetime.fromtimestamp(NOW + 60.0000004, UTC)
    parent_text, parent_epoch = contract.canonical_authorization_time(parent, "valid_until")
    child = {"schema": guard.EXCEPTION_SCHEMA, "exception_id": "pre-fix", "authorization_id": "O17-test",
             "binding_sha256": BINDING, "implementation_head": HEAD, "source_manifest_sha256": MANIFEST,
             "phase": 63, "action": guard.NO_TRAIN_ACTION, "gpu_uuid": "GPU-test",
             "process_identity": identity(), "created_at": NOW, "expires_at": parent_epoch + 0.0000004,
             "single_use": True}
    assert parent_text.endswith("Z")
    with pytest.raises(guard.GuardBlocked, match="PROCESS_EXCEPTION_TIME_INVALID"):
        guard.validate_external_process_exception(child, authorization_id="O17-test", binding_sha256=BINDING,
                                                  implementation_head=HEAD, source_manifest_sha256=MANIFEST,
                                                  action=guard.NO_TRAIN_ACTION,
                                                  authorization_expires_at=parent_epoch, now=NOW)


def test_canonical_producer_clamps_sub_microsecond_drift_to_parent_expiry() -> None:
    raw_parent = datetime.fromtimestamp(NOW + 60.0000004, UTC)
    auth = authorization(valid_until=raw_parent, requested=NOW + 60.0000004)
    _, parent_expiry = contract.canonical_authorization_time(auth["valid_until"], "valid_until")
    parent_text, reparsed_expiry = contract.canonical_authorization_time(raw_parent, "valid_until")
    assert auth["valid_until"] == parent_text and parent_expiry == reparsed_expiry
    assert auth["external_process_exceptions"][0]["expires_at"] == parent_expiry
    assert strict_check(auth)["exception_id"] == "O17-process"


def test_equal_and_earlier_expiry_pass_but_later_expiry_is_rejected_by_strict_validator() -> None:
    auth = authorization()
    _, parent_expiry = contract.canonical_authorization_time(auth["valid_until"], "valid_until")
    assert strict_check(auth)["expires_at"] == parent_expiry
    early = authorization(requested=parent_expiry - 0.000001)
    assert strict_check(early)["expires_at"] == parent_expiry - 0.000001
    later = dict(auth["external_process_exceptions"][0])
    later["expires_at"] = parent_expiry + 0.000001
    with pytest.raises(guard.GuardBlocked, match="PROCESS_EXCEPTION_TIME_INVALID"):
        guard.validate_external_process_exception(later, authorization_id=auth["authorization_id"],
                                                  binding_sha256=BINDING, implementation_head=HEAD,
                                                  source_manifest_sha256=MANIFEST,
                                                  action=guard.NO_TRAIN_ACTION,
                                                  authorization_expires_at=parent_expiry, now=NOW)


def test_timezone_equivalence_and_invalid_timestamp_inputs_fail_closed() -> None:
    utc = datetime(2026, 11, 1, 3, 38, 22, 164492, tzinfo=UTC)
    offset = "2026-11-01T12:38:22.164492+09:00"
    assert contract.canonical_authorization_time(utc, "x") == contract.canonical_authorization_time(offset, "x")
    for value in (datetime(2026, 11, 1, 3, 38, 22), "malformed", 1.0):
        with pytest.raises(contract.BindingContractError, match="INVALID_TIME"):
            contract.canonical_authorization_time(value, "x")


def test_factory_does_not_weaken_authorization_or_exact_instance_binding() -> None:
    auth = authorization()
    different = dict(auth["external_process_exceptions"][0]); different["authorization_id"] = "other"
    _, parent_expiry = contract.canonical_authorization_time(auth["valid_until"], "valid_until")
    with pytest.raises(guard.GuardBlocked, match="PROCESS_EXCEPTION_SCOPE_DENIED"):
        guard.validate_external_process_exception(different, authorization_id=auth["authorization_id"],
                                                  binding_sha256=BINDING, implementation_head=HEAD,
                                                  source_manifest_sha256=MANIFEST,
                                                  action=guard.NO_TRAIN_ACTION,
                                                  authorization_expires_at=parent_expiry, now=NOW)
