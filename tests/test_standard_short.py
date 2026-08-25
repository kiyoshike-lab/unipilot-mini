from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_selected_standard_short_configuration_is_in_requested_range():
    settings = read_json("configs/unipilot-standard-50m-short.json")
    config = ModelConfig(**settings["model"])
    tokenizer = BPETokenizer.load(ROOT / settings["tokenizer"])
    model = UniPilotTransformer(config)
    assert 45_000_000 <= model.parameter_count() <= 60_000_000
    assert model.parameter_count() == 45_445_120
    assert tokenizer.vocab_size == config.vocab_size == 2048
    assert config.context_length == 512


def test_architecture_and_token_context_matrix_results_are_complete():
    output = ROOT / "evaluation/standard-50m-short"
    architectures = [read_json(f"evaluation/standard-50m-short/architecture-standard-v08-{name}.json")
                     for name in ("a-45m", "b-51m", "c-58m")]
    assert [row["parameters"] for row in architectures] == [44_920_832, 51_225_600, 56_729_088]
    for vocab in (1024, 2048):
        for context in (512, 1024):
            row = read_json(f"evaluation/standard-50m-short/vocab-{vocab}-context-{context}.json")
            assert row["vocab_size"] == vocab
            assert row["context_length"] == context
            assert row["exact_roundtrip_rate"] == 1.0
    assert output.is_dir()


def test_independent_blind_200_is_sealed_balanced_and_unique():
    path = ROOT / "data/standard_50m_short/blind-200.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = read_json("data/standard_50m_short/manifest.json")
    rows = payload["items"]
    assert payload["sealed_before_training"] is True
    assert payload["used_for_training"] is False
    assert len(rows) == len({row["id"] for row in rows}) == 200
    assert len({row["question"] for row in rows}) == 200
    counts = Counter(row["expected_category"] for row in rows)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert max(row["max_reference_similarity"] for row in rows) < .78
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["blind_sha256_at_seal"]


def test_short_curriculum_does_not_contain_blind_questions():
    blind = read_json("data/standard_50m_short/blind-200.json")["items"]
    blind_questions = {row["question"] for row in blind}
    for split, expected in (("train", 4000), ("validation", 400)):
        lines = (ROOT / f"data/standard_50m_short/curriculum/{split}.jsonl").read_text(
            encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line]
        assert len(rows) == expected
        assert not blind_questions & {row["user"] for row in rows}


def test_short_phase_never_enables_production():
    settings = read_json("configs/unipilot-standard-50m-short.json")
    assert settings["production_enabled"] is False
    assert settings["render_free_target"] is False
    assert settings["external_pretrained_model"] is False


def test_failed_step_100_gate_stops_before_step_500():
    manifest = read_json("checkpoints/standard-50m-short/checkpoint-step-100.manifest.json")
    summary = read_json("evaluation/standard-50m-short/summary.json")
    comparison = read_json("evaluation/standard-50m-short/blind-200-comparison.json")
    assert manifest["development_generation_probe"]["text_generation_established"] is False
    assert manifest["continue_recommended"] is False
    assert summary["step_500"]["executed"] is False
    assert summary["next_training_step"] is None
    assert comparison["standard_clearly_improved"] is False
    assert not (ROOT / "checkpoints/standard-50m-short/checkpoint-step-500.manifest.json").exists()
