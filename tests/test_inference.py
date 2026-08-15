from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer
from inference.generate import generate_text


def test_generation_runs():
    tokenizer = BPETokenizer(); model = UniPilotTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, context_length=16, embedding_dim=16, n_layers=1, n_heads=4, ffn_dim=32))
    text, metrics = generate_text(model.eval(), tokenizer, "大学", max_new_tokens=3, temperature=0)
    assert isinstance(text, str) and metrics["tokens"] == 3
