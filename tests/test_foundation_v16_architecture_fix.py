from __future__ import annotations

import json
import math
from pathlib import Path
import random

import torch


ROOT = Path(__file__).resolve().parents[1]

from evaluation.measure_foundation_v16 import detailed_probe
from foundation.diagnostic_transformer_v15 import DiagnosticConfig, DiagnosticTransformer
from training.validate_foundation_v16_synthetic import (
    ANSWER,
    KEYS,
    conditioned_example,
    copy_example,
    example_hash,
    key_lookup_example,
    long_range_example,
    pattern_example,
)
from training.optimizer import create_optimizer


def load_result(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def build(scale: bool, dropout: float = 0.0) -> DiagnosticTransformer:
    torch.manual_seed(27)
    return DiagnosticTransformer(DiagnosticConfig(
        model_name="v1.6 scaling audit",
        vocab_size=64,
        context_length=32,
        embedding_dim=32,
        n_layers=2,
        n_heads=4,
        ffn_dim=64,
        dropout=dropout,
        scale_token_embedding=scale,
        weight_tying=True,
    ))


def test_formal_candidate_uses_exact_sqrt_d_model_scale():
    settings = json.loads(
        (ROOT / "configs/unipilot-foundation-v16.json").read_text(encoding="utf-8")
    )
    assert math.sqrt(settings["architecture"]["embedding_dim"]) == math.sqrt(384)
    assert math.isclose(math.sqrt(384), 19.595917942265423)
    scaled = build(True)
    assert scaled.embeddings.token_scale == math.sqrt(32)


def test_formula_a_scales_token_before_adding_unscaled_learned_position():
    model = build(True).eval()
    token_ids = torch.tensor([[7, 8, 9, 10]])
    positions = torch.arange(token_ids.size(1))
    token = model.embeddings.token(token_ids)
    position = model.embeddings.position(positions)[None, :, :]
    actual = model.embeddings(token_ids)
    formula_a = token * math.sqrt(model.config.embedding_dim) + position
    formula_b = (token + position) * math.sqrt(model.config.embedding_dim)
    assert torch.equal(actual, formula_a)
    assert not torch.equal(actual, formula_b)


def test_scaling_is_applied_once_and_never_in_tied_lm_head():
    model = build(True).eval()
    token_ids = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(1))
    measured = detailed_probe(model, token_ids)
    assert measured["manual_forward_max_absolute_error"] == 0.0
    assert measured["embedding"]["configured_token_scale"] == math.sqrt(32)
    assert math.isclose(
        measured["embedding"]["scaled_token"]["rms"]
        / measured["embedding"]["raw_token"]["rms"],
        math.sqrt(32), rel_tol=1e-6,
    )
    assert model.output.weight is model.embeddings.token.weight
    hidden = torch.randn(2, 3, 32)
    assert torch.equal(model.output(hidden), hidden @ model.embeddings.token.weight.T)


def test_checkpoint_load_preserves_scale_semantics_without_mutating_weights(tmp_path):
    model = build(True)
    path = tmp_path / "scaled.pt"
    torch.save({"config": model.config.to_dict(), "model_state": model.state_dict()}, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = DiagnosticTransformer(DiagnosticConfig(**payload["config"]))
    restored.load_state_dict(payload["model_state"], strict=True)
    assert restored.embeddings.token_scale == math.sqrt(32)
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), restored.state_dict().values())
    )
    assert restored.output.weight is restored.embeddings.token.weight


def test_unscaled_and_scaled_models_share_identical_parameter_initialization():
    current = build(False)
    scaled = build(True)
    assert current.parameter_count() == scaled.parameter_count()
    assert all(
        torch.equal(left, right)
        for left, right in zip(current.state_dict().values(), scaled.state_dict().values())
    )
    token_ids = torch.tensor([[4, 5, 6]])
    assert not torch.equal(current.embeddings(token_ids), scaled.embeddings(token_ids))


def test_synthetic_v2_difficulties_are_context_dependent_and_fit_context():
    makers = []
    for length in (4, 8, 16, 32, 64):
        makers.append(lambda rng, length=length: copy_example(rng, length))
    for pairs in (2, 4, 8, 16):
        for distance in ("short", "medium", "long"):
            makers.append(
                lambda rng, pairs=pairs, distance=distance:
                key_lookup_example(rng, pairs, distance)
            )
    makers.append(long_range_example)
    for pattern in ("abab", "abcabc", "numeric", "nested"):
        makers.append(lambda rng, pattern=pattern: pattern_example(rng, pattern))
    makers.append(conditioned_example)
    for index, maker in enumerate(makers):
        first = maker(random.Random(1000 + index))
        second = maker(random.Random(2000 + index))
        assert first[0][-1] == second[0][-1] == ANSWER
        assert len(first[0]) <= 80
        assert first[2]["required_context_distance"] > 1
        assert first[0][:-1] != second[0][:-1]


def test_synthetic_v2_seeded_train_and_test_examples_do_not_leak():
    train = {
        example_hash(*copy_example(random.Random(10 + index), 16)[:2])
        for index in range(100)
    }
    test = {
        example_hash(*copy_example(random.Random(10000 + index), 16)[:2])
        for index in range(100)
    }
    assert train.isdisjoint(test)


def test_context_controls_destroy_or_remove_the_only_condition():
    sequence, answer, metadata = conditioned_example(random.Random(27))
    assert sequence[metadata["condition_index"]] in KEYS[:4]
    shuffled = list(sequence)
    shuffled[metadata["condition_index"]] = KEYS[
        (KEYS.index(shuffled[metadata["condition_index"]]) + 1) % 4
    ]
    removed = list(sequence)
    removed[metadata["condition_index"]] = 226
    assert shuffled[-1] == removed[-1] == sequence[-1] == ANSWER
    assert shuffled[metadata["condition_index"]] != sequence[metadata["condition_index"]]
    assert removed[metadata["condition_index"]] not in KEYS[:4]
    assert answer in range(32, 36)


def _optimization_step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()


def test_scaled_checkpoint_resume_is_reproducible(tmp_path):
    generator = torch.Generator().manual_seed(2716)
    batches = [
        (
            torch.randint(0, 64, (2, 12), generator=generator),
            torch.randint(0, 64, (2, 12), generator=generator),
        )
        for _ in range(4)
    ]
    continuous = build(True)
    continuous_optimizer = create_optimizer(continuous, 3e-4, .01)
    for x, y in batches:
        _optimization_step(continuous, continuous_optimizer, x, y)

    staged = build(True)
    staged_optimizer = create_optimizer(staged, 3e-4, .01)
    for x, y in batches[:2]:
        _optimization_step(staged, staged_optimizer, x, y)
    checkpoint = tmp_path / "resume.pt"
    torch.save({
        "config": staged.config.to_dict(),
        "model_state": staged.state_dict(),
        "optimizer_state": staged_optimizer.state_dict(),
        "completed_updates": 2,
    }, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed = DiagnosticTransformer(DiagnosticConfig(**payload["config"]))
    resumed.load_state_dict(payload["model_state"], strict=True)
    resumed_optimizer = create_optimizer(resumed, 3e-4, .01)
    resumed_optimizer.load_state_dict(payload["optimizer_state"])
    for x, y in batches[payload["completed_updates"]:]:
        _optimization_step(resumed, resumed_optimizer, x, y)

    assert resumed.embeddings.token_scale == math.sqrt(32)
    assert all(
        torch.equal(left, right)
        for left, right in zip(continuous.state_dict().values(), resumed.state_dict().values())
    )


def test_three_seed_reproduction_artifacts_cover_activation_frequency_and_integrity():
    expected_milestones = {0, 10, 25, 50, 75, 100, 128}
    for variant in ("current_unscaled", "sqrt_scaled_a"):
        for seed in (42, 123, 2026):
            result = load_result(
                f"checkpoints/foundation-v16-reproduction/{variant}-seed-{seed}.json"
            )
            assert result["status"] == "COMPLETED"
            assert result["token_budget"] == 65536
            assert result["checkpoint"]["strict_reload"] is True
            assert result["sqrt_scaling_application_count"] == (
                1 if variant == "sqrt_scaled_a" else 0
            )
            assert result["lm_head_scaling_application_count"] == 0
            assert result["checkpoint_load_scaling_application_count"] == 0
            assert {row["update"] for row in result["history"]} == expected_milestones
            for row in result["history"]:
                assert row["probe"]["all_finite"] is True
                assert len(row["probe"]["layers"]) == 10
                for layer in row["probe"]["layers"]:
                    for component in ("input", "attention", "residual", "mlp", "output"):
                        assert layer[component]["finite"] is True
            assert set(result["final"]["frequency"]["buckets"]) == {
                "top_1_percent",
                "top_5_percent_excluding_top_1",
                "top_20_percent_excluding_top_5",
                "middle_20_to_80_percent",
                "rare_bottom_20_percent",
            }


def test_synthetic_v2_artifacts_are_leak_free_sufficient_and_gate_is_computed():
    for variant in ("current_unscaled", "sqrt_scaled_a"):
        result = load_result(f"checkpoints/foundation-v16-synthetic/{variant}.json")
        assert result["training"]["updates"] == 1600
        assert result["training"]["train_examples"] == 25600
        assert [row["training_percent"] for row in result["training"]["curve"]] == [
            10, 25, 50, 75, 100
        ]
        assert result["dataset_audit"]["train_test_exact_overlap"] == 0
        assert result["final"]["exact_train_test_overlap"] == 0
        assert result["final"]["test_hashes"] == 5888
        assert result["final"]["sequence_length"] == {"minimum": 8, "maximum": 68}
        assert result["checkpoint"]["strict_reload"] is True
        assert result["synthetic_gate_v2"] == (
            "PASS" if result["final"]["gate_pass"] else "FAIL"
        )
    assert load_result(
        "checkpoints/foundation-v16-synthetic/sqrt_scaled_a.json"
    )["synthetic_gate_v2"] == "FAIL"


def test_short_japanese_diagnostic_is_isolated_and_strictly_reloadable():
    results = {
        variant: load_result(f"checkpoints/foundation-v16-short-japanese/{variant}.json")
        for variant in ("current_unscaled", "sqrt_scaled_a")
    }
    for result in results.values():
        assert result["training"]["tokens_processed"] == 65536
        assert result["corpus"]["added_to_foundation_corpus"] is False
        assert result["checkpoint"]["strict_reload"] is True
        assert result["final_blind_used"] is False
        assert result["sentence_boundaries"]["。"]["targets"] > 0
        assert result["sentence_boundaries"]["<EOS>"]["targets"] > 0
    assert results["sqrt_scaled_a"]["final"]["loss"] < results["current_unscaled"]["final"]["loss"]
    assert results["sqrt_scaled_a"]["final"]["top_1_accuracy"] > results["current_unscaled"]["final"]["top_1_accuracy"]
    assert results["current_unscaled"]["baselines"] == results["sqrt_scaled_a"]["baselines"]


def test_architecture_fix_summary_stops_before_256k_when_any_gate_fails():
    summary = load_result("evaluation/foundation-v16-summary.json")
    assert summary["architecture_fix"] == "FAIL"
    assert summary["architecture_fix_gate"]["D_frequency_collapse_improved"] is False
    assert summary["architecture_fix_gate"]["F_synthetic_context_gate_v2"] is False
    assert summary["formal_architecture_change"] is False
    assert summary["controlled_256k"] == "NOT EXECUTED"
    assert summary["final_gate"] == "STOP"
    assert summary["final_blind"]["contents_opened"] is False
    assert summary["final_blind"]["hash_matches"] is True
