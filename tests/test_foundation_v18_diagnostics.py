from __future__ import annotations

import copy
import io
import math
import random

import numpy as np
import pytest
import torch

from evaluation.audit_foundation_v18_numeric_tokenizer import audit
from evaluation.measure_foundation_v18_attention import attention_retrieval_metrics
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from foundation.reference_transformer_v18 import (
    ReferenceConfigV18,
    ReferenceTransformerV18,
)
from training.validate_foundation_v16_synthetic import make_batch
from training.validate_foundation_v18_synthetic import key_lookup_v4_example
from training.validate_foundation_v18_japanese import japanese_metrics


def build(kind: str):
    common = dict(
        model_name=f"v18-test-{kind}",
        vocab_size=256,
        context_length=80,
        embedding_dim=32,
        n_layers=2,
        n_heads=4,
        ffn_dim=64,
        dropout=0.0,
        bias=True,
        norm_epsilon=1e-5,
        residual_projection_init_scale=1 / math.sqrt(4),
        weight_tying=True,
    )
    if kind == "custom":
        return DiagnosticTransformerV17(DiagnosticConfigV17(
            token_embedding_scale=1.0,
            position_embedding_scale=1.0,
            norm="layernorm",
            activation="gelu",
            **common,
        ))
    return ReferenceTransformerV18(ReferenceConfigV18(**common))


@pytest.mark.parametrize("kind", ["custom", "reference"])
def test_key_lookup_can_overfit_and_attention_report_has_required_fields(kind):
    torch.manual_seed(7)
    model = build(kind)
    example = key_lookup_v4_example(random.Random(11), 2, "medium")
    examples = [example] * 16
    inputs, targets = make_batch(examples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
    assert (logits[:, -1].argmax(-1) == example[1]).float().mean() >= .90
    measured = attention_retrieval_metrics(model, kind, examples[:4])
    head = measured["layers"][0]["heads"][0]
    assert {
        "normalized_entropy", "max_attention_probability", "top_3_attention_mass",
        "correct_key_mass", "correct_value_mass", "correct_key_value_mass",
        "correct_position_mean_rank", "q_rms", "k_rms", "qk_dot_product_std",
        "scaled_attention_logit_std", "attention_margin",
    } <= set(head)
    assert 0 <= head["correct_key_value_mass"] <= 1


@pytest.mark.parametrize("kind", ["custom", "reference"])
def test_checkpoint_optimizer_resume_is_bit_reproducible(kind):
    torch.manual_seed(19)
    model = build(kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    examples = [key_lookup_v4_example(random.Random(seed), 2, "short") for seed in range(8)]
    inputs, targets = make_batch(examples)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    buffer = io.BytesIO()
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=False)
    resumed = build(kind)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=3e-4)
    resumed.load_state_dict(payload["model_state"], strict=True)
    resumed_optimizer.load_state_dict(payload["optimizer_state"])
    for candidate, candidate_optimizer in ((model, optimizer), (resumed, resumed_optimizer)):
        candidate_optimizer.zero_grad(set_to_none=True)
        _, next_loss = candidate(inputs, targets)
        next_loss.backward()
        candidate_optimizer.step()
    for expected, actual in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(expected, actual)


def test_numeric_tokenizer_audit_and_atomic_synthetic_control():
    tokenizer = FoundationTokenizer.load("tokenizer/foundation-v11-base-4096.json")
    result = audit(tokenizer, ("1", "12", "1 2 3 4"))
    assert result["round_trip_exact_rate"] == 1.0
    assert result["synthetic_v4_numeric_representation"]["uses_foundation_tokenizer"] is False
    sequence, answer, metadata = key_lookup_v4_example(random.Random(3), 4, "long")
    assert sequence[-1] != answer
    assert sequence[metadata["correct_value_position"]] == answer


def test_frequency_buckets_include_top_1_top_5_top_10_and_probability():
    tokenizer = FoundationTokenizer.load("tokenizer/foundation-v11-base-4096.json")
    torch.manual_seed(5)
    model = ReferenceTransformerV18(ReferenceConfigV18(
        model_name="v18-frequency-test",
        vocab_size=tokenizer.vocab_size,
        context_length=32,
        embedding_dim=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        dropout=0,
    ))
    train = np.memmap(
        "data/foundation_v15_diagnostic/train.bin", dtype=np.uint16, mode="r"
    )
    validation = np.memmap(
        "data/foundation_v15_diagnostic/validation.bin", dtype=np.uint16, mode="r"
    )
    measured = japanese_metrics(model, tokenizer, train, validation, 64)
    assert measured["tokens"] == 64
    assert set(measured["frequency_buckets"]) == {
        "top_1_percent",
        "top_5_percent_excluding_top_1",
        "top_20_percent_excluding_top_5",
        "middle_20_to_80_percent",
        "rare_bottom_20_percent",
    }
    for row in measured["frequency_buckets"].values():
        assert {"top_1_accuracy", "top_5_accuracy", "top_10_accuracy", "mean_correct_token_probability"} <= set(row)
