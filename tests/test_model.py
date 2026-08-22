import torch
from model.config import ModelConfig
from model.transformer import UniPilotTransformer


def tiny_model(dropout=0.0):
    return UniPilotTransformer(ModelConfig(vocab_size=300, context_length=16, embedding_dim=32, n_layers=1, n_heads=4, ffn_dim=64, dropout=dropout))


def test_forward_shape_and_loss():
    model = tiny_model(); inputs = torch.randint(0, 300, (2, 12)); logits, loss = model(inputs, inputs)
    assert logits.shape == (2, 12, 300) and loss.ndim == 0


def test_causal_mask_prevents_future_leakage():
    model = tiny_model().eval(); first = torch.tensor([[1, 2, 3, 4]]); second = torch.tensor([[1, 2, 9, 8]])
    with torch.no_grad(): a, _ = model(first); b, _ = model(second)
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-6)


def test_kv_cache_matches_full_autoregressive_forward():
    model = tiny_model().eval(); tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.inference_mode():
        full, _ = model(tokens)
        first, _, cache = model(tokens[:, :4], use_cache=True)
        final, _, cache = model(tokens[:, 4:], past_key_values=cache, use_cache=True)
    assert first.shape == (1, 4, 300)
    assert len(cache) == model.config.n_layers and cache[0][0].shape[2] == 5
    assert torch.allclose(final[:, -1], full[:, -1], atol=1e-5)
