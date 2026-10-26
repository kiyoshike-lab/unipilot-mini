"""CPU-only O13 regression tests for the single-reservation handoff."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import phase63_runtime_journal as journal
from training.phase63_authorization_claim import AuthorizationClaimError, claim


def put(path: Path, value: dict) -> str:
    import hashlib
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, dict]:
    runtime = tmp_path / "runtime"; (runtime / "runs").mkdir(parents=True)
    auth = tmp_path / "auth.json"
    digest = put(auth, {"authorization_id": "o13", "action": "no_train", "binding_sha256": "b" * 64,
                        "allowed_runs": [{"seed": 42, "arm": "control"}]})
    contract = journal.build_contract(auth, digest, "b" * 64, purpose="no_train", seed=42, arm="control",
                                      head="a" * 40, source_manifest_sha256="c" * 64)
    return runtime, contract


def test_valid_handoff_adopts_one_reservation_without_second_reserve(tmp_path: Path, monkeypatch):
    runtime, contract = fixture(tmp_path)
    calls = 0; original = journal.reserve
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(journal, "reserve", counted)
    directory, handoff = journal.reserve_handoff(runtime, contract)
    assert journal.adopt_handoff(runtime, contract, handoff) == directory
    assert calls == 1


def test_competing_runner_is_stopped_by_the_single_use_authorization_claim(tmp_path: Path):
    runtime, contract = fixture(tmp_path); _, handoff = journal.reserve_handoff(runtime, contract)
    assert journal.adopt_handoff(runtime, contract, handoff)
    authorization = {"authorization_id": "o13", "binding_sha256": "b" * 64, "phase": 63,
                     "action": "no_train", "allowed_runs": [{"seed": 42, "arm": "control"}]}
    auth_path = tmp_path / "claim-auth.json"; auth_path.write_text(json.dumps(authorization), encoding="utf-8")
    claim(auth_path, authorization, "a" * 64, execution_instance=contract["execution_instance"])
    with pytest.raises(AuthorizationClaimError, match="ALREADY_CONSUMED"):
        claim(auth_path, authorization, "a" * 64, execution_instance=contract["execution_instance"])


def test_pre_fix_second_reservation_still_fails_closed(tmp_path: Path):
    runtime, contract = fixture(tmp_path); journal.reserve(runtime, contract)
    with pytest.raises(journal.RuntimeJournalError, match="COLLISION_PREAUTH"):
        journal.reserve(runtime, contract)


@pytest.mark.parametrize("field,value", [("execution_instance", "other"), ("journal_path", "C:/wrong"),
                                           ("binding_sha256", "d" * 64), ("head", "e" * 40),
                                           ("source_manifest_sha256", "f" * 64), ("purpose", "training")])
def test_wrong_handoff_identity_rejects(tmp_path: Path, field: str, value: str):
    runtime, contract = fixture(tmp_path); _, handoff = journal.reserve_handoff(runtime, contract)
    value_json = json.loads(handoff.read_text(encoding="utf-8")); value_json[field] = value
    handoff.write_text(json.dumps(value_json), encoding="utf-8")
    with pytest.raises(journal.RuntimeJournalError, match="HANDOFF_MISMATCH"):
        journal.adopt_handoff(runtime, contract, handoff)


def test_tampered_marker_and_other_or_historical_handoff_reject(tmp_path: Path):
    runtime, contract = fixture(tmp_path); directory, handoff = journal.reserve_handoff(runtime, contract)
    (directory / journal.MARKER).write_text("{}", encoding="utf-8")
    with pytest.raises(journal.RuntimeJournalError): journal.adopt_handoff(runtime, contract, handoff)
    other = tmp_path / "other.json"; other.write_text(handoff.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(journal.RuntimeJournalError, match="HANDOFF_PATH_INVALID"):
        journal.adopt_handoff(runtime, contract, other)
