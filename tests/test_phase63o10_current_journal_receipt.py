"""CPU-only O10 receipt binding tests; no CUDA, lock, or model access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import exclusive_cuda_runner as runner
from training import phase63_runtime_journal as journal


def put(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_contract(tmp_path: Path, name: str) -> dict:
    auth = tmp_path / f"{name}.json"
    digest = put(auth, {"authorization_id": name, "action": "no_train", "binding_sha256": "b" * 64,
                       "allowed_runs": [{"seed": 42, "arm": "control"}]})
    return journal.build_contract(auth, digest, "b" * 64, purpose="no_train", seed=42, arm="control",
                                  head="a" * 40, source_manifest_sha256="c" * 64)


def sample(path: Path, contract: dict) -> dict:
    stdout = path / "stdout.log"
    return {"strict_reload": True, "updates_complete": True, "artifact_path": str(stdout),
            "artifact_sha256": hashlib.sha256(stdout.read_bytes()).hexdigest(),
            "execution_instance": contract["execution_instance"], "journal_contract": contract}


def fixture(tmp_path: Path):
    runtime = tmp_path / "runtime"; (runtime / "runs").mkdir(parents=True)
    contracts = [make_contract(tmp_path, name) for name in ("old", "current", "wrong")]
    old, current, wrong = [journal.reserve(runtime, value) for value in contracts]
    for path, content in ((old, "old-success"), (current, "current-success"), (wrong, "wrong-success")):
        (path / "stdout.log").write_text(content, encoding="utf-8")
    return runtime, old, current, wrong, contracts


def test_current_journal_is_authoritative_and_old_is_rejected(tmp_path: Path):
    runtime, old, current, wrong, (_, current_contract, _) = fixture(tmp_path)
    assert journal.validate_current_receipt(runtime, current, sample(current, current_contract)) == current / "stdout.log"
    old_sample = sample(old, current_contract)
    with pytest.raises(journal.RuntimeJournalError, match="ARTIFACT_INVALID"):
        journal.validate_current_receipt(runtime, current, old_sample)


def test_missing_malformed_and_wrong_journals_fail_closed_without_fallback(tmp_path: Path):
    runtime, old, current, wrong, (_, current_contract, _) = fixture(tmp_path)
    (current / "stdout.log").unlink()
    with pytest.raises(journal.RuntimeJournalError, match="ARTIFACT_INVALID"):
        journal.validate_current_receipt(runtime, current, sample(old, current_contract))
    (current / "stdout.log").write_text("current-success", encoding="utf-8")
    with pytest.raises(journal.RuntimeJournalError, match="CONTRACT_MISMATCH"):
        journal.validate_current_receipt(runtime, wrong, sample(wrong, current_contract))
    (current / journal.MARKER).write_text("{", encoding="utf-8")
    with pytest.raises(journal.RuntimeJournalError, match="RESERVATION_MISSING"):
        journal.validate_current_receipt(runtime, current, sample(current, current_contract))


@pytest.mark.parametrize("field,value", [("execution_instance", "other"), ("head", "d" * 40),
                                          ("source_manifest_sha256", "e" * 64), ("authorization_id", "other")])
def test_contract_toctou_mismatches_are_rejected(tmp_path: Path, field: str, value: str):
    runtime, old, current, wrong, (_, current_contract, _) = fixture(tmp_path)
    altered = {**current_contract, field: value}
    checked = sample(current, altered)
    with pytest.raises(journal.RuntimeJournalError, match="CONTRACT_MISMATCH"):
        journal.validate_current_receipt(runtime, current, checked)


def test_complete_receipt_requires_its_own_current_journal(tmp_path: Path):
    runtime, old, current, wrong, (_, current_contract, _) = fixture(tmp_path)
    for index, state in enumerate(runner.STATES):
        (current / f"{index:02d}-{state}.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    stderr = current / "stderr.log"; stderr.write_text("", encoding="utf-8")
    checked = sample(current, current_contract)
    receipt = {"status": "COMPLETE", "kind": "cuda-no-train", "exit_code": 0,
               "child_tree_exited": True, "gpu_context_released": True, "stdout_closed": True,
               "stderr_closed": True, "verification": checked,
               "stdout_sha256": checked["artifact_sha256"],
               "stderr_sha256": hashlib.sha256(stderr.read_bytes()).hexdigest()}
    receipt_path = current / "receipt.json"; receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    released = json.loads((current / "06-RELEASED.json").read_text(encoding="utf-8"))
    released["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    (current / "06-RELEASED.json").write_text(json.dumps(released), encoding="utf-8")
    assert runner.is_complete(current)
    receipt["verification"] = sample(old, current_contract)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    released["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    (current / "06-RELEASED.json").write_text(json.dumps(released), encoding="utf-8")
    assert not runner.is_complete(current)


def test_supervisor_uses_reserved_current_journal_without_fixed_run_lookup():
    root = Path(__file__).parents[1]
    source = (root / "scripts" / "run_phase63_study.py").read_text(encoding="utf-8")
    section = source[source.index("def run_no_train_once("):source.index("def main()")]
    assert section.index("reserve_journal") < section.index("admit_execution")
    assert 'verify(seed, arm, "no-train", journal_path=directory,' in section
    verifier = source[source.index("def verify("):source.index("def run(")]
    assert "validate_current_receipt(RUNTIME, Path(journal_path), verification)" in verifier
    assert 'receipt_path(seed, arm, kind) / "stdout.log"' not in verifier
