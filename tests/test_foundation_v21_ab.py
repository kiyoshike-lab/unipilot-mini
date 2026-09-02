from __future__ import annotations

import json
import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]

from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.optimizer import create_optimizer
from training.train_foundation_v21_ab import (
    CHECKPOINT_FORMAT,
    EXPECTED_PARAMETERS,
    activation_summary,
    build_paired_model,
    file_sha256,
    frequency_ranks,
    language_metrics,
    load_json,
    paired_initialization_audit,
    run_training,
    save_checkpoint,
)
from training.validate_foundation_v21_smoke import fixed_relation_example


def settings() -> dict:
    return load_json("configs/unipilot-foundation-v21.json")


def test_phase32_config_is_exact_fair_256k_ab() -> None:
    config = settings()
    assert config["seeds"] == [42, 123, 2026]
    assert config["training"] == {
        "token_budget": 256000,
        "effective_batch_tokens": 512,
        "optimizer": "AdamW",
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "peak_learning_rate": 1e-4,
        "warmup_updates": 20,
        "schedule_after_warmup": "constant",
        "milestone_tokens": [0, 64000, 128000, 192000, 256000],
    }
    assert config["architecture"]["context_length"] == 512
    assert config["parameter_target"] == EXPECTED_PARAMETERS
    assert config["maximum_allowed_tokens_per_run"] == 256000
    assert config["final_blind"]["content_opened"] is False


def test_current_and_depth_have_equal_parameters_and_only_paired_init_differs() -> None:
    config = settings()
    tokenizer = FoundationTokenizer.load(ROOT / config["tokenizer"])
    audit = paired_initialization_audit(config, tokenizer)
    assert audit["current_parameters"] == EXPECTED_PARAMETERS
    assert audit["depth_parameters"] == EXPECTED_PARAMETERS
    assert audit["parameter_equality"] is True
    assert audit["only_residual_output_projections_changed"] is True
    assert audit["paired_standardized_max_absolute_error"] == 0
    assert math.isclose(audit["depth_residual_std"], .02 / math.sqrt(20))


def test_paired_seed_shares_every_standardized_initial_draw() -> None:
    config = settings()
    tokenizer = FoundationTokenizer.load(ROOT / config["tokenizer"])
    current = build_paired_model(config, tokenizer, "current", 123)
    depth = build_paired_model(config, tokenizer, "depth_init", 123)
    scale = 1 / math.sqrt(20)
    for name, current_value in current.state_dict().items():
        depth_value = depth.state_dict()[name]
        if name.endswith("attention.projection.weight") or name.endswith(
            "feed_forward.network.2.weight"
        ):
            assert torch.equal(current_value * scale, depth_value)
        else:
            assert torch.equal(current_value, depth_value)


def test_optimizer_uses_required_betas_epsilon_and_decay_split() -> None:
    model = DiagnosticTransformerV17(DiagnosticConfigV17(
        model_name="phase32 optimizer unit",
        vocab_size=64,
        context_length=16,
        embedding_dim=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        dropout=0,
    ))
    optimizer = create_optimizer(model, 1e-4, .1)
    assert optimizer.defaults["betas"] == (.9, .95)
    assert optimizer.defaults["eps"] == 1e-8
    assert {group["weight_decay"] for group in optimizer.param_groups} == {.1, 0.0}


def test_language_metrics_have_frequency_punctuation_boundary_and_topk() -> None:
    config = settings()
    tokenizer = FoundationTokenizer.load(ROOT / config["tokenizer"])
    train = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/train.bin",
        dtype=np.uint16,
        mode="r",
    )
    validation = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/validation.bin",
        dtype=np.uint16,
        mode="r",
    )
    torch.manual_seed(32)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(
        model_name="phase32 metrics unit",
        vocab_size=tokenizer.vocab_size,
        context_length=32,
        embedding_dim=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        dropout=0,
    ))
    result = language_metrics(
        model, tokenizer, validation, frequency_ranks(train, tokenizer.vocab_size), 64
    )
    assert {"top_1_accuracy", "top_5_accuracy", "top_10_accuracy"} <= set(result)
    assert set(result["punctuation"]) == {"。", "、", "の", "に", "は", "を", "が", "と", "で"}
    assert set(result["sentence_boundaries"]) == {"。", "！", "？", "newline", "<EOS>"}
    assert len(result["frequency_buckets"]) == 5
    for bucket in result["frequency_buckets"].values():
        assert {"top_1_accuracy", "top_5_accuracy", "top_10_accuracy",
                "mean_correct_token_probability", "cross_entropy"} <= set(bucket)


def test_activation_summary_has_required_health_fields() -> None:
    config = settings()
    tokenizer = FoundationTokenizer.load(ROOT / config["tokenizer"])
    model = build_paired_model(config, tokenizer, "depth_init", 42).eval()
    from evaluation.measure_foundation_v17 import architecture_probe

    probe = architecture_probe(model, torch.arange(128).remainder(tokenizer.vocab_size)[None])
    result = activation_summary(probe)
    assert len(result["layers"]) == 10
    assert {"embedding_rms", "layer_9_rms", "final_residual_rms", "final_norm_rms",
            "logit_std", "all_finite", "nan", "inf", "explosion", "collapse"} <= set(result)
    assert result["all_finite"] is True


def test_checkpoint_strict_reload_integrity_and_resume_step_are_reproducible(tmp_path) -> None:
    torch.manual_seed(32)
    config = DiagnosticConfigV17(
        model_name="phase32 resume unit",
        vocab_size=64,
        context_length=16,
        embedding_dim=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        dropout=0,
    )
    model = DiagnosticTransformerV17(config)
    optimizer = create_optimizer(model, 1e-4, .1)
    permutation = torch.randperm(128, generator=torch.Generator().manual_seed(42))
    x = torch.arange(16).remainder(64)[None]
    y = torch.arange(1, 17).remainder(64)[None]
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "phase32.pt"
    metadata = save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        variant="current",
        seed=42,
        update=1,
        permutation=permutation,
        history=[],
        training_seconds=.1,
        settings={"maximum_allowed_tokens_per_run": 256000},
    )
    assert metadata["strict_reload"] is True
    assert metadata["integrity"] == "PASS"
    assert metadata["sha256"] == file_sha256(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["checkpoint_format"] == CHECKPOINT_FORMAT
    resumed = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    resumed_optimizer = create_optimizer(resumed, 1e-4, .1)
    resumed.load_state_dict(payload["model_state"], strict=True)
    resumed_optimizer.load_state_dict(payload["optimizer_state"])
    for candidate, candidate_optimizer in ((model, optimizer), (resumed, resumed_optimizer)):
        candidate_optimizer.zero_grad(set_to_none=True)
        _, next_loss = candidate(x, y)
        next_loss.backward()
        candidate_optimizer.step()
    for expected, actual in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(expected, actual)


def test_training_refuses_more_than_256k_before_any_run(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="forbids training beyond 256k"):
        run_training(
            settings=settings(),
            variant="current",
            seed=42,
            output_dir=tmp_path,
            token_budget=256512,
            validation_tokens=32,
            include_generation=False,
        )


def test_final_blind_hash_matches_without_parsing_content() -> None:
    config = settings()
    path = ROOT / config["final_blind"]["path"]
    assert file_sha256(path) == config["final_blind"]["expected_sha256"]


def test_fixed_relation_smoke_has_no_target_leak_and_completed_gate_passes() -> None:
    sequence, answer, metadata = fixed_relation_example(random.Random(32))
    assert answer not in sequence
    assert metadata["fixed_mapping"] is True
    result = json.loads(
        (ROOT / "evaluation/foundation-v21-synthetic-smoke.json").read_text(encoding="utf-8")
    )
    assert result["gate_pass"] is True
    assert result["novel_random_key_lookup_gate"] is False
    assert result["numeric_modular_addition_gate"] is False


def test_completed_ab_summary_obeys_generation_regression_selection_rule() -> None:
    summary = json.loads(
        (ROOT / "evaluation/foundation-v21-summary.json").read_text(encoding="utf-8")
    )
    assert summary["run_count"] == 6
    assert summary["all_runs_256k"] is True
    assert summary["parameters_equal"] is True
    assert summary["paired_data_order_equal"] is True
    assert summary["selection_gates"]["A_validation_loss_mean_improves"] is True
    assert summary["selection_gates"]["B_improvement_consistent_across_seeds"] is True
    assert summary["selection_gates"]["C_top_1_5_10_not_worse"] is True
    assert summary["selection_gates"]["H_generation_trend_not_worse"] is False
    assert summary["gate"] == "CURRENT_RETAIN"
    assert summary["formal_foundation_architecture"] == "Current"
    assert summary["next_token_budget"] == "512k"
    assert summary["foundation_base_complete"] is False
