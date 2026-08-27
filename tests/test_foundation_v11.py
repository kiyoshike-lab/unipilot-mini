from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_clean_corpus_quality_and_random_100_pass():
    audit = load("evaluation/foundation-v11-corpus-quality-audit.json")
    assert audit["documents"] == 10_315
    assert audit["corpus_quality"] == "PASS"
    assert not any(audit["full_corpus_residue_counts"].values())
    assert all(value == 1 for value in audit["random_sample_rates"].values())


def test_clean_splits_are_unique_and_contamination_safe_without_final_blind():
    audit = load("evaluation/foundation-v11-data-audit.json")
    assert audit["unique_documents"] == sum(audit["documents"].values())
    assert audit["excluded"]["semantic_duplicate"] > 0
    assert audit["holdout_audit"]["maximum_segment_question_similarity"] < .78
    assert audit["holdout_audit"]["final_blind_content_opened"] is False
    assert not any("final-blind-1000" in path for path in audit["holdout_audit"]["files"])


def test_clean_tokenizer_and_special_tokens_pass():
    benchmark = load("evaluation/foundation-v11-tokenizer-benchmark.json")
    special = load("evaluation/foundation-v11-special-token-audit.json")
    assert benchmark["selected_vocab"] == 4096
    assert all(row["exact_roundtrip_rate"] == 1 for row in benchmark["results"])
    assert all(row["campus_question_roundtrip_rate"] == 1 for row in benchmark["results"])
    assert special["tokenizer_gate"] == "PASS"
    assert special["packed_train_special_counts"]["<EOS>"] == 10_012


def test_checkpoint_v2_integrity_and_resume_reproducibility_pass():
    checkpoint = ROOT / "checkpoints/foundation-v11-clean-100/checkpoint-step-100.pt"
    manifest = load("checkpoints/foundation-v11-clean-100/checkpoint-step-100.manifest.json")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["checkpoint_format"] == "foundation-v11-v2"
    assert manifest["checkpoint_sha256"] == file_sha256(checkpoint)
    assert payload["global_step"] == payload["scheduler_state"]["global_step"] == 100
    assert {"python", "numpy", "torch_cpu"} <= payload["random_state"].keys()
    assert payload["sampler_state"]["position"] == 100
    resume = load("evaluation/foundation-v11-resume-reproducibility.json")
    assert resume["resume_integrity"] == "PASS"
    assert resume["maximum_loss_difference"] == 0
    assert resume["maximum_weight_difference"] == 0
    assert resume["maximum_optimizer_difference"] == 0


def test_dirty_checkpoint_is_deprecated_and_not_a_v11_resume_source():
    manifest = load("checkpoints/foundation-v10-sanity/20m/checkpoint-step-500.manifest.json")
    assert manifest["lifecycle_status"] == "DEPRECATED_DIRTY_CORPUS"
    assert manifest["resume_allowed_for_foundation_v11"] is False


def test_clean_100_loss_curve_is_healthy_and_bounded():
    manifest = load("checkpoints/foundation-v11-clean-100/checkpoint-step-100.manifest.json")
    history = manifest["history"]
    assert [row["step"] for row in history] == [0, 10, 25, 50, 75, 100]
    validation = [row["validation_loss"] for row in history]
    assert all(left > right for left, right in zip(validation, validation[1:]))
    assert history[-1]["tokens_processed"] == 51_200
    assert history[-1]["corpus_fraction"] < .01


def test_dirty_clean_generation_uses_fixed_nonblind_completion_prompts():
    report = load("evaluation/foundation-v11-dirty-clean-generation.json")
    dataset = load("data/foundation_v11/evaluation/base-completion-50.json")
    assert report["prompts"] == len(dataset["items"]) == 50
    assert report["final_blind_used"] is False
    assert dataset["final_blind_used"] is False
    assert {row["name"] for row in report["results"]} == {"dirty_v1.0", "clean_v1.1"}
    assert all(set(row["modes"]) == {"greedy", "sampling"} for row in report["results"])
