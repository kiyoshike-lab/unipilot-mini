from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation.investigate_foundation_v14 import language_proxy
from evaluation.investigate_foundation_v23_generation import (
    evaluator_reasons,
    loop_details,
    ngram_repetition,
)
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_training_inference_and_kv_cache_paths_match_directly() -> None:
    torch.manual_seed(34)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(
        model_name="phase34 parity unit",
        vocab_size=64,
        context_length=32,
        embedding_dim=16,
        n_layers=2,
        n_heads=2,
        ffn_dim=32,
        dropout=0.1,
    )).eval()
    prefix = torch.arange(1, 17)[None].remainder(64)
    targets = torch.arange(2, 18)[None].remainder(64)
    with torch.inference_mode():
        training_logits, _ = model(prefix, targets)
        inference_logits, _ = model(prefix)
        _, _, cache = model(prefix[:, :-1], use_cache=True)
        cached_logits, _, _ = model(prefix[:, -1:], past_key_values=cache, use_cache=True)
    assert torch.equal(training_logits, inference_logits)
    assert torch.allclose(training_logits[:, -1], cached_logits[:, -1], atol=1e-5, rtol=0)


def test_completed_full_checkpoint_parity_audit_passes_all_seeds() -> None:
    result = read("evaluation/foundation-v23-inference-parity.json")
    assert result["inference_parity"] == "PASS"
    assert result["kv_cache_parity"] == "PASS"
    assert len(result["rows"]) == 3
    assert all(row["pass"] for row in result["rows"])


def test_repetition_diagnostics_identify_loop_shape() -> None:
    ids = [10, 20, 21, 20, 21, 20, 21]
    details = loop_details(ids)
    assert details["loop_length"] == 2
    assert details["loop_onset_token"] == 2
    assert details["maximum_repeated_span"] == 6
    assert ngram_repetition([1, 1, 1, 1], 1) == 0.75


def test_natural_japanese_evaluator_thresholds_are_explainable() -> None:
    bad = "\n" * 20
    proxy = language_proxy(bad, eos_reached=False)
    reasons = evaluator_reasons(bad, proxy)
    assert proxy["natural_japanese_proxy"] is False
    assert reasons != ["all_natural_japanese_proxy_conditions_passed"]
    completed = read("evaluation/foundation-v23-generation-diagnostics-512000.json")
    audit = completed["natural_japanese_evaluator_audit"]
    assert audit["thresholds_changed"] is False
    assert len(audit["examples"]) == 50
    assert all(row["reasons"] for row in audit["examples"])


def test_prefix_completion_uses_fixed_held_out_set_and_minimum_200_examples() -> None:
    rows_by_tokens = {}
    for tokens in (256_000, 512_000, 640_000):
        payload = read(f"evaluation/foundation-v23-generation-diagnostics-{tokens}.json")
        rows = payload["validation_document_prefix"]["items"]
        rows_by_tokens[tokens] = rows
        assert len(rows) == 200
        assert {row["prefix_length"] for row in rows} == {16, 32, 64, 128}
        assert all(len(row["truth_ids"]) == 64 for row in rows)
        assert payload["validation_sentence_prefix"]["metrics"]["examples"] == 50
    identifiers = [[row["id"] for row in rows_by_tokens[tokens]] for tokens in rows_by_tokens]
    assert identifiers[0] == identifiers[1] == identifiers[2]


def test_teacher_horizon_and_top10_candidate_metrics_improve_to_pilot() -> None:
    old = read("evaluation/foundation-v23-generation-diagnostics-512000.json")
    new = read("evaluation/foundation-v23-generation-diagnostics-640000.json")
    old_teacher = old["validation_document_prefix"]["metrics"]["teacher_forced_horizon"]["32"]
    new_teacher = new["validation_document_prefix"]["metrics"]["teacher_forced_horizon"]["32"]
    assert new_teacher["loss"] < old_teacher["loss"]
    assert new_teacher["top_10_accuracy"] > old_teacher["top_10_accuracy"]
    assert new["validation_document_prefix"]["metrics"]["free_running"]["mean_divergence_position"] >= old["validation_document_prefix"]["metrics"]["free_running"]["mean_divergence_position"]


def test_boundary_eos_observation_and_pilot_checkpoint_integrity() -> None:
    diagnostic = read("evaluation/foundation-v23-generation-diagnostics-640000.json")
    assert diagnostic["corpus_boundary_counts"]["train"]["eos_count"] == 10_012
    assert diagnostic["corpus_boundary_counts"]["train"]["bos_count"] == 10_012
    assert diagnostic["training_exposure"][0]["supervised_eos_targets"] == 205
    assert set(diagnostic["boundary_diagnostics"]) == {"。", "！", "？", "newline", "<EOS>"}
    summary = read("evaluation/foundation-v23-summary.json")
    assert summary["checkpoint_verification"]["pass"] is True
    assert summary["pilot"]["status"] == "EXECUTED"
    assert summary["gate"] == "CONTINUE_1M_TOKEN_LIMITED"
    assert summary["one_million_next_phase"] == "YES"
    assert summary["foundation_base_complete"] is False
