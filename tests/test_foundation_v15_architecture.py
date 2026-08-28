from __future__ import annotations

import json
import math
from pathlib import Path
import random
import copy

import torch


ROOT = Path(__file__).resolve().parents[1]


from foundation.diagnostic_transformer_v15 import (
    DiagnosticConfig,
    DiagnosticTransformer,
    RMSNorm,
)
from model.attention import CausalSelfAttention
from model.config import ModelConfig
from model.embeddings import TokenPositionEmbedding
from model.layers import TransformerBlock
from model.transformer import UniPilotTransformer
from training.run_foundation_v15_synthetic import (
    ANSWER_TOKEN,
    TASKS,
    make_batch,
    make_example,
)


def test_current_architecture_is_pre_layernorm_with_expected_dimensions():
    settings = json.loads(
        (ROOT / "configs/unipilot-foundation-v15.json").read_text(encoding="utf-8")
    )
    architecture = settings["architecture"]
    assert architecture == {
        "context_length": 512,
        "embedding_dim": 384,
        "n_layers": 10,
        "n_heads": 6,
        "ffn_dim": 1536,
        "dropout": 0.1,
        "bias": True,
        "norm": "layernorm",
        "norm_epsilon": 1e-5,
        "activation": "gelu",
        "scale_token_embedding": False,
        "weight_tying": True,
    }
    model = UniPilotTransformer(ModelConfig(
        model_name="audit", vocab_size=4096, **{
            key: architecture[key] for key in (
                "context_length", "embedding_dim", "n_layers", "n_heads",
                "ffn_dim", "dropout", "bias",
            )
        }
    ))
    assert model.parameter_count() == 19_514_880
    assert len(model.blocks) == 10
    assert model.blocks[0].attention.head_dim == 64
    assert isinstance(model.blocks[0].norm1, torch.nn.LayerNorm)
    assert isinstance(model.blocks[0].norm2, torch.nn.LayerNorm)
    assert model.blocks[0].norm1.eps == 1e-5
    assert model.output.weight is model.embeddings.token.weight
    assert model.output.bias is None


def test_attention_matches_qk_over_sqrt_head_dim_and_key_softmax():
    torch.manual_seed(26)
    attention = CausalSelfAttention(embedding_dim=16, n_heads=4,
                                    context_length=8, dropout=0.0, bias=True).eval()
    values = torch.randn(2, 5, 16)
    actual, _ = attention(values)
    q, k, v = attention.qkv(values).chunk(3, dim=-1)
    q = q.view(2, 5, 4, 4).transpose(1, 2)
    k = k.view(2, 5, 4, 4).transpose(1, 2)
    v = v.view(2, 5, 4, 4).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(4)
    mask = attention.causal_mask[:, :, :5, :5]
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    assert torch.equal(weights.masked_select(~mask), torch.zeros_like(weights.masked_select(~mask)))
    expected = weights @ v
    expected = expected.transpose(1, 2).contiguous().view(2, 5, 16)
    expected = attention.projection(expected)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_residual_paths_are_exactly_attention_then_mlp_preln():
    torch.manual_seed(27)
    block = TransformerBlock(16, 4, 32, 8, 0.0, True).eval()
    hidden = torch.randn(2, 6, 16)
    actual, _ = block(hidden)
    attended, _ = block.attention(block.norm1(hidden))
    after_attention = hidden + attended
    expected = after_attention + block.feed_forward(block.norm2(after_attention))
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert not torch.allclose(actual, attended)


def test_learned_absolute_positions_are_zero_based_and_batch_shared():
    torch.manual_seed(28)
    embedding = TokenPositionEmbedding(32, 16, 8, 0.0).eval()
    tokens = torch.tensor([[4, 5, 6], [4, 5, 6]])
    actual = embedding(tokens)
    positions = torch.arange(3)
    expected = embedding.token(tokens) + embedding.position(positions)[None, :, :]
    assert torch.equal(actual, expected)
    assert torch.equal(actual[0], actual[1])
    shifted = embedding(tokens, position_offset=4)
    assert not torch.equal(actual, shifted)


def test_isolated_current_diagnostic_model_is_exactly_production_equivalent():
    torch.manual_seed(29)
    production = UniPilotTransformer(ModelConfig(
        model_name="production-equivalent", vocab_size=64, context_length=16,
        embedding_dim=32, n_layers=2, n_heads=4, ffn_dim=64,
        dropout=0.0, bias=True,
    )).eval()
    diagnostic = DiagnosticTransformer(DiagnosticConfig(
        model_name="diagnostic", vocab_size=64, context_length=16,
        embedding_dim=32, n_layers=2, n_heads=4, ffn_dim=64,
        dropout=0.0, bias=True,
    )).eval()
    diagnostic.load_state_dict(production.state_dict())
    tokens = torch.randint(0, 64, (2, 16))
    production_logits, _ = production(tokens)
    diagnostic_logits, _ = diagnostic(tokens)
    assert torch.equal(production_logits, diagnostic_logits)


def test_rmsnorm_has_unit_rms_and_no_bias():
    torch.manual_seed(30)
    norm = RMSNorm(16, eps=1e-8)
    values = torch.randn(4, 7, 16)
    output = norm(values)
    rms = output.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)
    assert list(dict(norm.named_parameters())) == ["weight"]


def test_synthetic_tasks_require_context_beyond_the_last_token():
    for task_index, task in enumerate(TASKS):
        first, first_answer = make_example(task, random.Random(100 + task_index))
        second, second_answer = make_example(task, random.Random(200 + task_index))
        assert first[-1] == second[-1] == ANSWER_TOKEN
        assert first[:-1] != second[:-1]
        # Fixed seed choices are selected so at least one contextual answer changes.
        if task != "context_conditioned":
            assert first_answer != second_answer or first[:-3] != second[:-3]
    inputs, targets, names, answers = make_batch(random.Random(26), len(TASKS))
    assert inputs.shape == targets.shape == (len(TASKS), 32)
    assert names == TASKS
    assert torch.equal(inputs[:, -1], torch.full((len(TASKS),), ANSWER_TOKEN))
    assert torch.equal(targets[:, -1], torch.tensor(answers))
    assert int((targets[:, :-1] != -100).sum().item()) == 0


def test_phase26_synthetic_gate_is_computed_from_every_task():
    task_results = []
    for task in TASKS:
        result = json.loads(
            (ROOT / f"evaluation/foundation-v15-synthetic-{task}.json").read_text(
                encoding="utf-8"
            )
        )
        assert result["training_mode"] == "independent_task"
        assert result["final"]["last_input_token_is_constant"] == ANSWER_TOKEN
        task_results.append(result["final"]["by_task"][task]["accuracy"])
        expected = "PASS" if task_results[-1] > result["target_accuracy"] else "FAIL"
        assert result["context_learning"] == expected
    assert max(task_results) > 0.90
    assert min(task_results) < 0.90


def test_diagnostic_checkpoint_strict_integrity_and_bitwise_resume(tmp_path):
    config = DiagnosticConfig(
        model_name="resume-audit", vocab_size=32, context_length=8,
        embedding_dim=32, n_layers=2, n_heads=4, ffn_dim=64,
        dropout=0.1, bias=True,
    )
    batch_generator = torch.Generator().manual_seed(2600)
    batches = [
        (
            torch.randint(0, 32, (2, 8), generator=batch_generator),
            torch.randint(0, 32, (2, 8), generator=batch_generator),
        )
        for _ in range(6)
    ]

    def initialize():
        torch.manual_seed(2601)
        model = DiagnosticTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        return model, optimizer

    def update(model, optimizer, batch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(*batch)
        loss.backward()
        optimizer.step()

    uninterrupted, uninterrupted_optimizer = initialize()
    for batch in batches:
        update(uninterrupted, uninterrupted_optimizer, batch)

    partial, partial_optimizer = initialize()
    for batch in batches[:3]:
        update(partial, partial_optimizer, batch)
    checkpoint = {
        "model_state": copy.deepcopy(partial.state_dict()),
        "optimizer_state": copy.deepcopy(partial_optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state().clone(),
        "config": config.to_dict(),
        "update": 3,
    }
    path = tmp_path / "resume.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    resumed = DiagnosticTransformer(DiagnosticConfig(**loaded["config"]))
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    resumed.load_state_dict(loaded["model_state"], strict=True)
    resumed_optimizer.load_state_dict(loaded["optimizer_state"])
    torch.set_rng_state(loaded["torch_rng_state"])
    for batch in batches[3:]:
        update(resumed, resumed_optimizer, batch)
    assert all(
        torch.equal(left, right)
        for left, right in zip(uninterrupted.state_dict().values(), resumed.state_dict().values())
    )


def test_depth_width_ablations_stay_within_five_percent_parameter_budget():
    settings = json.loads(
        (ROOT / "configs/unipilot-foundation-v15.json").read_text(encoding="utf-8")
    )
    counts = {}
    for row in settings["ablations"]:
        architecture = dict(settings["architecture"])
        architecture.update(row["changes"])
        model = DiagnosticTransformer(DiagnosticConfig(
            model_name=row["name"], vocab_size=4096, **architecture
        ))
        counts[row["name"]] = model.parameter_count()
    baseline = counts["current_preln_gelu_tied"]
    for name in ("fewer_layers_wider", "more_layers_narrower"):
        assert abs(counts[name] / baseline - 1) <= 0.05


def test_measured_context_sensitivity_changes_logits_beyond_bigram():
    audit = json.loads(
        (ROOT / "evaluation/foundation-v15-architecture-audit.json").read_text(
            encoding="utf-8"
        )
    )
    sensitivity = audit["baseline_128k_context_sensitivity"]
    assert sensitivity["same_last_token_for_every_pair"]
    assert sensitivity["bigram_distribution_difference"] == 0.0
    assert sensitivity["context_sensitivity_score"] > 0.0
    assert sensitivity["top_1_changed_rate"] > 0.0
    ablation = audit["baseline_128k_context_ablation"]
    assert ablation["full_vs_last_1_loss_advantage"] > 0.0
    assert audit["checkpoint_integrity"]["strict_state_load"]
    assert audit["checkpoint_integrity"]["all_ablation_sha256_verified"]


def test_bigram_audit_has_no_train_validation_leakage():
    audit = json.loads(
        (ROOT / "evaluation/foundation-v15-architecture-audit.json").read_text(
            encoding="utf-8"
        )
    )
    bigram = audit["bigram_audit"]
    assert bigram["status"] == "PASS"
    assert bigram["document_overlap"] == 0
    assert bigram["train_sha256"] != bigram["validation_sha256"]
    assert all(bigram["checks"].values())


def test_controlled_corpus_is_isolated_short_clean_and_reloadable():
    manifest = json.loads(
        (ROOT / "data/foundation_v15_diagnostic/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["selection"]["segments"] == 5000
    assert 20 <= manifest["selection"]["minimum_tokens"]
    assert manifest["selection"]["maximum_tokens"] <= 200
    assert manifest["selection"]["sentence_boundaries_required"]
    assert manifest["selection"]["markup_rejected"]
    assert manifest["source_metadata_preserved"]
    assert not manifest["added_to_foundation_corpus"]
    assert manifest["artifacts"]["packed"]["train"]["sha256"] != manifest["artifacts"]["packed"]["validation"]["sha256"]
    experiment = json.loads(
        (ROOT / "evaluation/foundation-v15-controlled-corpus-experiment.json").read_text(
            encoding="utf-8"
        )
    )
    assert experiment["training"]["tokens_processed"] == 65536
    assert experiment["checkpoint"]["strict_reload"]
    assert not experiment["added_to_foundation_corpus"]
