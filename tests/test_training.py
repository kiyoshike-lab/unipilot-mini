from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.optimizer import create_optimizer


def test_checkpoint_round_trip(tmp_path):
    config = ModelConfig(vocab_size=300, context_length=8, embedding_dim=16, n_layers=1, n_heads=4, ffn_dim=32)
    model = UniPilotTransformer(config); optimizer = create_optimizer(model, 1e-3)
    path = tmp_path / "model.pt"; save_checkpoint(path, model, optimizer, None, 0, 1, 2.0, config)
    payload = load_checkpoint(path, model, optimizer); assert payload["step"] == 1
