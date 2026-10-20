"""CPU-only regression tests for PHASE63 pre-authorization journal allocation."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

from training import phase63_runtime_journal as journal
from training.phase63_authorization_claim import claim_path


def put(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract(tmp_path: Path, name: str = "auth-a") -> dict:
    auth = tmp_path / f"{name}.json"
    digest = put(auth, {"authorization_id": name, "action": "no_train", "binding_sha256": "b" * 64,
                        "allowed_runs": [{"seed": 42, "arm": "control"}]})
    return journal.build_contract(auth, digest, "b" * 64, purpose="no_train", seed=42, arm="control",
                                  head="a" * 40, source_manifest_sha256="c" * 64)


def runtime(tmp_path: Path) -> Path:
    root = tmp_path / "runtime" / "runs"; root.mkdir(parents=True)
    return root.parent


def test_old_fixed_journal_is_preserved_and_new_authorization_gets_a_distinct_path(tmp_path: Path):
    root = runtime(tmp_path); old = root / "runs" / "phase63-no-train-seed42-control"; old.mkdir()
    (old / "old.json").write_text("historical", encoding="utf-8"); old_hash = hashlib.sha256((old / "old.json").read_bytes()).hexdigest()
    value = contract(tmp_path); fresh = journal.reserve(root, value)
    assert fresh != old and fresh.parent == old.parent
    assert hashlib.sha256((old / "old.json").read_bytes()).hexdigest() == old_hash
    journal.validate_reserved(root, fresh, value)


def test_same_execution_reuse_existing_destination_and_escape_are_rejected(tmp_path: Path):
    root = runtime(tmp_path); value = contract(tmp_path); fresh = journal.reserve(root, value)
    with pytest.raises(journal.RuntimeJournalError, match="COLLISION_PREAUTH"): journal.reserve(root, value)
    assert fresh.exists()
    with pytest.raises(journal.RuntimeJournalError, match="CONTRACT_MISMATCH"):
        journal.validate_reserved(root, root / "runs" / ".." / "elsewhere", value)


def test_collision_rejects_before_an_authorization_claim_can_exist(tmp_path: Path):
    root = runtime(tmp_path); value = contract(tmp_path); destination = journal.journal_path(root, value)
    destination.mkdir()
    authorization = tmp_path / "auth-a.json"
    with pytest.raises(journal.RuntimeJournalError, match="COLLISION_PREAUTH"):
        journal.reserve(root, value)
    assert not claim_path(authorization).exists()


def test_existing_file_destination_is_rejected_without_overwrite(tmp_path: Path):
    root = runtime(tmp_path); value = contract(tmp_path); destination = journal.journal_path(root, value)
    destination.write_text("do-not-overwrite", encoding="utf-8")
    with pytest.raises(journal.RuntimeJournalError, match="COLLISION_PREAUTH"):
        journal.reserve(root, value)
    assert destination.read_text(encoding="utf-8") == "do-not-overwrite"


def test_distinct_authorizations_have_distinct_paths_and_parent_child_mismatch_rejects(tmp_path: Path):
    root = runtime(tmp_path); first = contract(tmp_path, "a"); second = contract(tmp_path, "b")
    one, two = journal.reserve(root, first), journal.reserve(root, second)
    assert one != two
    with pytest.raises(journal.RuntimeJournalError, match="CONTRACT_MISMATCH"):
        journal.validate_reserved(root, one, second)


def test_concurrent_same_execution_has_one_reservation_winner(tmp_path: Path):
    root = runtime(tmp_path); value = contract(tmp_path); results = []
    def reserve():
        try: results.append(("winner", journal.reserve(root, value)))
        except journal.RuntimeJournalError as exc: results.append(("rejected", str(exc)))
    threads = [threading.Thread(target=reserve), threading.Thread(target=reserve)]
    [thread.start() for thread in threads]; [thread.join() for thread in threads]
    assert [kind for kind, _ in results].count("winner") == 1
    assert [kind for kind, _ in results].count("rejected") == 1


def test_no_train_supervisor_reserves_before_claim_and_worker_checks_the_same_contract():
    root = Path(__file__).parents[1]
    supervisor = (root / "scripts" / "run_phase63_study.py").read_text(encoding="utf-8")
    section = supervisor[supervisor.index("def run_no_train_once("):supervisor.index("def main()")]
    assert section.index("reserve_journal") < section.index("admit_execution")
    assert "journal_path=directory" in section and "journal_contract=journal_contract" in section
    worker = (root / "training" / "phase63_worker.py").read_text(encoding="utf-8")
    no_train = worker[worker.index("def no_train("):worker.index("def train(")]
    assert "validate_runtime_journal" in no_train and "UNIPILOT_RUNTIME_JOURNAL_PATH" in no_train
