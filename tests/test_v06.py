from __future__ import annotations

import json
from pathlib import Path

from evaluation.evaluate_v06 import rubric_scores
from retrieval.bm25 import LocalBM25
from scripts.prepare_dataset_v06 import SUBJECT_WORDS, normalized
from training.train_v06 import schedule_weights


def jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v06_dataset_required_fields_and_no_pair_duplicates():
    rows = sum((jsonl(f"data/v06/instruction/{split}.jsonl") for split in ("train", "validation", "test")), [])
    required = {"user", "assistant", "category", "difficulty", "source_type", "quality_score"}
    assert rows
    assert all(required.issubset(row) for row in rows)
    pairs = [normalized(row["user"] + row["assistant"]) for row in rows]
    assert len(pairs) == len(set(pairs))


def test_v06_replay_has_no_specific_subject_names():
    rows = jsonl("data/v06/stages/stage_a.jsonl")
    assert len(rows) >= 100
    assert all(not any(word in row["user"] + row["assistant"] for word in SUBJECT_WORDS) for row in rows)
    assert all(row["source_type"] == "subject_genericized_v04_replay" for row in rows)


def test_v06_has_300_held_out_prompts_and_corrections():
    prompts = json.loads(Path("evaluation/fixed_prompts_v06.json").read_text(encoding="utf-8"))
    corrections = jsonl("data/v06/stages/stage_e.jsonl")
    assert len(prompts) == 300
    assert len({item["id"] for item in prompts}) == 300
    assert len(corrections) >= 300


def test_v06_rubric_flags_unstated_subject_and_policy_claim():
    item = {"prompt": "試験勉強を手伝って", "category": "study", "length_type": "simple",
            "expected_keywords": ["試験", "確認"], "forbidden_keywords": ["経済学"], "requires_uncertainty": False}
    _, signals = rubric_scores(item, "経済学の試験は全国の大学で必ず合格します。", 30, True)
    assert signals["hallucination"]
    assert signals["policy_hallucination"]
    assert "経済学" in signals["extra_subjects"]


def test_local_bm25_prefers_matching_document():
    index = LocalBM25([
        {"id": "gpa", "title": "GPA", "text": "GPAは成績を数値化した平均指標です。"},
        {"id": "library", "title": "図書館", "text": "図書館では資料を検索できます。"},
    ])
    assert index.search("GPAと成績", top_k=1)[0]["id"] == "gpa"


def test_curriculum_reaches_accuracy_stage_by_step_500():
    settings = json.loads(Path("configs/unipilot-v06.json").read_text(encoding="utf-8"))
    assert "E" not in schedule_weights(settings, 0)
    final = schedule_weights(settings, 499)
    assert final["A"] >= 0.5
    assert final["E"] > 0
