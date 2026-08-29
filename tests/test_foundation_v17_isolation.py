from __future__ import annotations

import json
import math
from pathlib import Path
import random

import torch


ROOT = Path(__file__).resolve().parents[1]

from evaluation.measure_foundation_v17 import architecture_probe
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.optimizer import create_optimizer
from training.validate_foundation_v16_synthetic import make_batch
from training.validate_foundation_v17_synthetic import (
    phase28_gate,
    position_example,
)


def build(
    token_scale: float = 1.0,
    position_scale: float = 1.0,
    residual_init_scale: float = 1.0,
) -> DiagnosticTransformerV17:
    torch.manual_seed(28)
    return DiagnosticTransformerV17(DiagnosticConfigV17(
        model_name="Foundation v1.7 unit isolation",
        vocab_size=256,
        context_length=32,
        embedding_dim=32,
        n_layers=2,
        n_heads=4,
        ffn_dim=64,
        dropout=0.0,
        token_embedding_scale=token_scale,
        position_embedding_scale=position_scale,
        residual_projection_init_scale=residual_init_scale,
    ))


def load_result(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_matrix_has_only_four_isolated_configurations():
    settings = json.loads(
        (ROOT / "configs/unipilot-foundation-v17.json").read_text(encoding="utf-8")
    )
    assert [row["name"] for row in settings["variants"]] == [
        "current_unscaled",
        "sqrt_scaled_a",
        "balanced_position_sqrt",
        "depth_scaled_residual_init",
    ]
    assert settings["full_256k_enabled"] is False
    assert settings["synthetic_v3"]["updates"] >= 2000


def test_balanced_position_scale_restores_effective_token_position_ratio():
    token_ids = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(7))
    current = architecture_probe(build(), token_ids)
    sqrt_a = architecture_probe(build(math.sqrt(32), 1.0), token_ids)
    balanced = architecture_probe(build(math.sqrt(32), math.sqrt(32)), token_ids)
    current_ratio = current["embedding"]["effective_token_to_position_rms_ratio"]
    sqrt_ratio = sqrt_a["embedding"]["effective_token_to_position_rms_ratio"]
    balanced_ratio = balanced["embedding"]["effective_token_to_position_rms_ratio"]
    assert .8 < current_ratio < 1.2
    assert sqrt_ratio > 4.0
    assert .8 < balanced_ratio < 1.2


def test_position_scale_is_applied_once_before_addition():
    model = build(math.sqrt(32), math.sqrt(32)).eval()
    token_ids = torch.tensor([[7, 8, 9]])
    positions = torch.arange(3)
    expected = (
        model.embeddings.token(token_ids) * math.sqrt(32)
        + model.embeddings.position(positions)[None] * math.sqrt(32)
    )
    assert torch.equal(model.embeddings(token_ids), expected)


def test_depth_scaled_initialization_changes_only_residual_output_projections():
    current = build()
    depth_scaled = build(residual_init_scale=1 / math.sqrt(4))
    changed = []
    for (name, left), (right_name, right) in zip(
        current.state_dict().items(), depth_scaled.state_dict().items()
    ):
        assert name == right_name
        if not torch.equal(left, right):
            changed.append(name)
    assert changed == [
        "blocks.0.attention.projection.weight",
        "blocks.0.feed_forward.network.2.weight",
        "blocks.1.attention.projection.weight",
        "blocks.1.feed_forward.network.2.weight",
    ]
    manifest = depth_scaled.initialization_manifest()
    assert manifest["attention_output_projection_std"] == .01
    assert manifest["mlp_output_projection_std"] == .01
    assert manifest["attention_qkv_std"] == .02
    assert manifest["mlp_input_projection_std"] == .02


def test_final_norm_is_present_between_blocks_and_lm_head():
    model = build().eval()
    token_ids = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(9))
    probe = architecture_probe(model, token_ids)
    assert probe["final_norm"]["present"] is True
    assert probe["final_norm"]["order"] == "Embedding -> Blocks -> Final Norm -> LM Head"
    assert probe["manual_forward_max_absolute_error"] == 0.0


def test_position_embedding_receives_gradient_and_changes():
    model = build(math.sqrt(32), math.sqrt(32))
    optimizer = create_optimizer(model, 3e-4, .01)
    before = model.embeddings.position.weight.detach().clone()
    examples = [position_example(random.Random(100 + index), 8) for index in range(8)]
    x, y = make_batch(examples)
    _, loss = model(x, y)
    loss.backward()
    gradient = model.embeddings.position.weight.grad
    assert gradient is not None
    assert float(gradient.square().mean().sqrt()) > 0
    optimizer.step()
    assert not torch.equal(before, model.embeddings.position.weight)


def test_position_task_answer_depends_on_queried_ordinal():
    sequence, answer, metadata = position_example(random.Random(28), 16)
    values = sequence[1:17]
    assert answer == values[metadata["queried_position"]]
    changed = list(sequence)
    changed[-2] = changed[-2] + 1 if metadata["queried_position"] < 15 else changed[-2] - 1
    assert values[metadata["queried_position"]] != values[
        metadata["queried_position"] + (1 if metadata["queried_position"] < 15 else -1)
    ]
    assert changed[:-2] == sequence[:-2]


def test_residual_contribution_ratios_are_recorded_for_every_layer():
    model = build()
    token_ids = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(3))
    probe = architecture_probe(model, token_ids)
    assert len(probe["layers"]) == 2
    for layer in probe["layers"]:
        assert layer["attention_to_residual_rms_ratio"] > 0
        assert layer["mlp_to_residual_rms_ratio"] > 0
        assert layer["post_mlp_residual"]["finite"] is True


def test_phase28_synthetic_gate_requires_every_threshold():
    base = {
        "copy": {"4": 1.0, "8": 1.0, "16": .9},
        "key_lookup": {
            "2": {"short": .95, "medium": .95, "long": .95},
            "4": {"short": .95, "medium": .95, "long": .95},
            "8": {"short": .90, "medium": .90, "long": .90},
        },
        "long_range": .95,
        "pattern": {"abab": .95, "abcabc": .95, "nested": .95, "numeric": .90},
        "context_conditioned": {"correct": .95, "shuffled": .2, "removed": .2},
    }
    position = {"minimum_accuracy": .95}
    assert phase28_gate(base, position)["pass"] is True
    base["copy"]["8"] = .949
    assert phase28_gate(base, position)["pass"] is False


def test_v17_checkpoint_resume_reproduces_continuous_training(tmp_path):
    generator = torch.Generator().manual_seed(2717)
    batches = [
        (
            torch.randint(0, 256, (2, 12), generator=generator),
            torch.randint(0, 256, (2, 12), generator=generator),
        )
        for _ in range(4)
    ]

    def step(model, optimizer, batch):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(*batch)
        loss.backward()
        optimizer.step()

    continuous = build(math.sqrt(32), math.sqrt(32))
    continuous_optimizer = create_optimizer(continuous, 3e-4, .01)
    for batch in batches:
        step(continuous, continuous_optimizer, batch)

    staged = build(math.sqrt(32), math.sqrt(32))
    staged_optimizer = create_optimizer(staged, 3e-4, .01)
    for batch in batches[:2]:
        step(staged, staged_optimizer, batch)
    checkpoint = tmp_path / "v17-resume.pt"
    torch.save({
        "config": staged.config.to_dict(),
        "model_state": staged.state_dict(),
        "optimizer_state": staged_optimizer.state_dict(),
        "update": 2,
    }, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    resumed.load_state_dict(payload["model_state"], strict=True)
    resumed_optimizer = create_optimizer(resumed, 3e-4, .01)
    resumed_optimizer.load_state_dict(payload["optimizer_state"])
    for batch in batches[payload["update"]:]:
        step(resumed, resumed_optimizer, batch)
    assert all(
        torch.equal(left, right)
        for left, right in zip(continuous.state_dict().values(), resumed.state_dict().values())
    )


def test_completed_reproduction_artifacts_have_position_residual_norm_and_integrity():
    paths = [
        "checkpoints/foundation-v17-reproduction/balanced_position_sqrt-seed-42.json",
        *[
            "checkpoints/foundation-v17-reproduction/"
            f"depth_scaled_residual_init-seed-{seed}.json"
            for seed in (42, 123, 2026)
        ],
    ]
    for path in paths:
        result = load_result(path)
        assert result["status"] == "COMPLETED"
        assert result["token_budget"] == 65536
        assert result["final_norm"] == "PRESENT"
        assert result["checkpoint"]["strict_reload"] is True
        assert result["checkpoint"]["optimizer_state_present"] is True
        assert {row["update"] for row in result["history"]} == {0, 10, 25, 50, 100, 128}
        for row in result["history"]:
            assert row["probe"]["all_finite"] is True
            assert row["probe"]["final_norm"]["present"] is True
            assert len(row["probe"]["layers"]) == 10
        assert result["final"]["position_learning"]["gradient"]["rms"] > 0
        assert result["final"]["position_learning"]["parameter_delta"]["rms"] > 0


def test_synthetic_v3_artifacts_are_leak_free_and_apply_strict_gate():
    for variant in (
        "current_unscaled",
        "sqrt_scaled_a",
        "balanced_position_sqrt",
        "depth_scaled_residual_init",
    ):
        result = load_result(
            f"checkpoints/foundation-v17-synthetic/{variant}-seed-42.json"
        )
        assert result["training"]["updates"] == 2000
        assert result["training"]["train_examples"] == 32000
        assert [row["training_percent"] for row in result["training"]["curve"]] == [
            10, 25, 50, 75, 100
        ]
        assert result["dataset_audit"]["base_test_overlap"] == 0
        assert result["dataset_audit"]["position_test_overlap"] == 0
        assert result["dataset_audit"]["input_contains_target_answer"] is False
        assert result["checkpoint"]["strict_reload"] is True
        assert result["checkpoint"]["optimizer_state_present"] is True
        assert result["final"]["phase28_gate"]["pass"] is False
        assert len(result["copy_failure_analysis"]["all_token_predictions"]) == 96
        assert result["numeric_failure_analysis"]["examples"] == 256


def test_depth_init_is_partial_fix_but_not_architecture_gate_pass():
    summary = load_result("evaluation/foundation-v17-summary.json")
    checks = summary["architecture_checks"]
    assert checks["three_seed_validation_loss_better_than_current"] is True
    assert checks["three_seed_top_1_better_than_current"] is True
    assert checks["layer_9_rms_better_than_current"] is True
    assert checks["clean_japanese_frequency_gate"] is True
    assert checks["full_corpus_non_top1_accuracy_above_zero"] is False
    assert checks["synthetic_gate"] is False
    assert summary["architecture_decision"] == "MULTI_COMPONENT_FIX_REQUIRED"
    assert summary["architecture_gate"] == "FAIL"
    assert summary["formal_architecture_change"] is False
    assert summary["combined_ablation_executed"] is False
    assert summary["full_256k"] == "NOT EXECUTED"
    assert summary["proceed_to_full_256k"] == "NO"


def test_final_blind_is_hash_checked_without_opening_and_production_is_untouched():
    summary = load_result("evaluation/foundation-v17-summary.json")
    assert summary["final_blind"]["contents_opened"] is False
    assert summary["final_blind"]["hash_matches"] is True
    assert summary["controls"] == {
        "production_changed": False,
        "campus_changed": False,
        "tokenizer_changed": False,
        "corpus_added": False,
        "standard_46m": False,
        "push_or_deploy": False,
        "external_ai_api": "OFF",
    }
