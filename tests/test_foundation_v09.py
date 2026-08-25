from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer
from training.train_foundation_v09 import natural_text


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line]


def norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def test_campus_and_production_baseline_files_are_frozen():
    manifest = read_json("data/foundation_v09/manifest.json")
    for relative, expected in manifest["baseline"]["protected_file_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert manifest["baseline"]["answer_logic_frozen_after_baseline"] is True
    assert manifest["baseline"]["production_changed"] is False


def test_language_campus_and_instruction_stages_are_separate_and_licensed():
    manifest = read_json("data/foundation_v09/manifest.json")
    assert sum(manifest["base"].values()) == 1040
    assert sum(manifest["campus"].values()) == 330
    assert sum(manifest["instruction"].values()) == 11844
    assert manifest["human_approved"] == 4
    assert manifest["rag_only_documents"] == 114
    for split in ("train", "validation", "test"):
        assert all(row["kind"] == "text" and row["license"] == "CC BY-SA 4.0"
                   for row in read_jsonl(f"data/foundation_v09/base/{split}.jsonl"))
        assert all(row["kind"] == "text" and row["license"] == "CC0-1.0"
                   for row in read_jsonl(f"data/foundation_v09/campus/{split}.jsonl"))
        assert all(row["kind"] == "conversation"
                   for row in read_jsonl(f"data/foundation_v09/instruction/{split}.jsonl"))


def test_only_explicit_human_good_answers_are_approved():
    rows = read_jsonl("data/foundation_v09/human-approved.jsonl")
    assert len(rows) == 4
    assert all(row["human_rating"] == "good" and row["approval"] == "human_good" for row in rows)


def test_final_blind_1000_is_sealed_and_not_in_training():
    path = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = read_json("data/foundation_v09/manifest.json")
    assert payload["opened_for_this_phase"] is False
    assert payload["used_for_training"] is False
    assert len(payload["items"]) == 1000
    assert len({row["question"] for row in payload["items"]}) == 1000
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["evaluation"]["final_blind_sha256"]
    heldout = {norm(row["question"]) for row in payload["items"]}
    for split in ("train", "validation", "test"):
        training = read_jsonl(f"data/foundation_v09/instruction/{split}.jsonl")
        assert not heldout & {norm(row["user"]) for row in training}


def test_selected_standard_candidate_is_46m_vocab4096_context1024():
    settings = read_json("configs/unipilot-foundation-v09-sanity.json")
    config = ModelConfig(**settings["model"])
    tokenizer = BPETokenizer.load(ROOT / settings["tokenizer"])
    model = UniPilotTransformer(config)
    assert model.parameter_count() == 46_755_840
    assert tokenizer.vocab_size == config.vocab_size == 4096
    assert config.context_length == 1024
    assert [stage["name"] for stage in settings["stages"]] == ["BASE", "CAMPUS", "INSTRUCTION"]


def test_tokenizer_and_all_context_candidates_were_measured():
    tokenizer = read_json("evaluation/foundation-v09-tokenizer-benchmark.json")
    assert [row["actual_vocab"] for row in tokenizer["results"]] == [1024, 2048, 4096]
    assert tokenizer["selected_vocab"] == 4096
    assert all(row["exact_roundtrip_rate"] == 1.0 for row in tokenizer["results"])
    for context in (256, 512, 1024, 2048):
        row = read_json(f"evaluation/foundation-v09-context/context-{context}.json")
        assert row["context"] == context
        assert row["vocab"] == 4096


def test_v09_is_never_a_production_or_deployment_target():
    settings = read_json("configs/unipilot-foundation-v09-sanity.json")
    assert settings["production_enabled"] is False
    assert settings["render_free_target"] is False
    assert settings["external_pretrained_model"] is False


def test_generation_gate_accepts_japanese_and_rejects_empty_text():
    assert natural_text("まず条件を確認し、次に必要な手順を一つずつ進めます。")[0] is True
    assert natural_text("")[0] is False


def test_failed_100step_gate_prevents_all_longer_training():
    checkpoint = read_json("checkpoints/foundation-v09-sanity/checkpoint-step-100.manifest.json")
    summary = read_json("evaluation/foundation-v09-sanity/summary.json")
    comparison = read_json("evaluation/foundation-v09-sanity/validation-200-comparison.json")
    assert checkpoint["development_probe"]["text_generation_established"] is False
    assert checkpoint["continue_recommended"] is False
    assert summary["step_500"]["executed"] is False
    assert summary["standard_continue"] is False
    assert summary["next_training_step"] is None
    assert comparison["final_blind_opened"] is False
    assert not (ROOT / "checkpoints/foundation-v09-sanity/checkpoint-step-500.manifest.json").exists()
