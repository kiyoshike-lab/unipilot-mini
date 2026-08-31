from __future__ import annotations

import math

import torch

from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from foundation.reference_transformer_v18 import (
    ReferenceConfigV18,
    ReferenceTransformerV18,
)
from training.optimizer import create_optimizer


def reference(dropout: float = 0.0) -> ReferenceTransformerV18:
    torch.manual_seed(29)
    return ReferenceTransformerV18(ReferenceConfigV18(
        model_name="Foundation v1.8 reference unit",
        vocab_size=64,
        context_length=32,
        embedding_dim=32,
        n_layers=2,
        n_heads=4,
        ffn_dim=64,
        dropout=dropout,
        residual_projection_init_scale=1 / math.sqrt(4),
    ))


def test_reference_uses_independent_torch_multihead_attention():
    model = reference()
    assert all(isinstance(block.attention, torch.nn.MultiheadAttention) for block in model.blocks)
    manifest = model.architecture_manifest()
    assert manifest["custom_attention_code_shared"] is False
    assert manifest["custom_residual_block_code_shared"] is False
    assert manifest["final_norm"] == "PRESENT"


def test_reference_parameter_count_matches_custom_exactly():
    torch.manual_seed(29)
    custom = DiagnosticTransformerV17(DiagnosticConfigV17(
        model_name="custom parity",
        vocab_size=64,
        context_length=32,
        embedding_dim=32,
        n_layers=2,
        n_heads=4,
        ffn_dim=64,
        dropout=0.0,
        residual_projection_init_scale=1 / math.sqrt(4),
    ))
    assert reference().parameter_count() == custom.parameter_count()


def test_reference_causal_mask_prevents_future_influence():
    model = reference().eval()
    left = torch.tensor([[1, 2, 3, 4, 5, 6]])
    right = torch.tensor([[1, 2, 3, 17, 18, 19]])
    left_logits, _ = model(left)
    right_logits, _ = model(right)
    assert torch.equal(left_logits[:, :3], right_logits[:, :3])
    assert not torch.equal(left_logits[:, 3:], right_logits[:, 3:])


def test_reference_learned_positions_affect_representations():
    model = reference().eval()
    repeated = torch.full((1, 8), 7)
    embedded = model.embeddings(repeated)
    assert not torch.equal(embedded[:, 0], embedded[:, 1])
    logits, _ = model(repeated)
    assert not torch.equal(logits[:, 0], logits[:, 1])


def test_reference_gradients_reach_position_qkv_and_residual_outputs():
    model = reference()
    x = torch.randint(0, 64, (4, 12), generator=torch.Generator().manual_seed(11))
    y = torch.randint(0, 64, (4, 12), generator=torch.Generator().manual_seed(12))
    _, loss = model(x, y)
    loss.backward()
    assert model.embeddings.position.weight.grad is not None
    assert model.blocks[0].attention.in_proj_weight.grad is not None
    assert model.blocks[0].attention.out_proj.weight.grad is not None
    assert model.blocks[0].feed_forward[2].weight.grad is not None
    assert all(
        float(parameter.grad.square().mean().sqrt()) > 0
        for parameter in (
            model.embeddings.position.weight,
            model.blocks[0].attention.in_proj_weight,
            model.blocks[0].attention.out_proj.weight,
            model.blocks[0].feed_forward[2].weight,
        )
    )


def test_reference_tiny_overfit_and_eos_sanity():
    torch.manual_seed(29)
    model = ReferenceTransformerV18(ReferenceConfigV18(
        model_name="reference tiny overfit",
        vocab_size=16,
        context_length=8,
        embedding_dim=16,
        n_layers=1,
        n_heads=4,
        ffn_dim=32,
        dropout=0.0,
        residual_projection_init_scale=1 / math.sqrt(2),
    ))
    optimizer = create_optimizer(model, 1e-2, 0.0)
    x = torch.tensor([
        [1, 3, 4, 5],
        [1, 6, 7, 8],
        [1, 9, 10, 11],
        [1, 12, 13, 14],
    ])
    targets = torch.full_like(x, -100)
    targets[:, -1] = 2
    first_loss = None
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, targets)
        if first_loss is None:
            first_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
    logits, final_loss = model(x, targets)
    assert float(final_loss.detach()) < first_loss * .1
    assert torch.equal(logits[:, -1].argmax(-1), torch.full((4,), 2))
