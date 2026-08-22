import json

from fastapi.testclient import TestClient

import api.main as api_main
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer


def test_chat_stream_uses_loaded_model_and_keeps_chat_compatible(monkeypatch):
    tokenizer = BPETokenizer()
    model = UniPilotTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, context_length=32,
        embedding_dim=16, n_layers=1, n_heads=4, ffn_dim=32, dropout=0.0)).eval()
    monkeypatch.setattr(api_main, "load_runtime", lambda checkpoint=None: None)
    api_main.runtime.update(model=model, tokenizer=tokenizer, device="cpu", checkpoint="test", payload={})
    request = {"prompt": "GPAって何？", "max_new_tokens": 3, "temperature": 0,
               "top_k": 40, "top_p": .9, "repetition_penalty": 1.0}
    with TestClient(api_main.app) as client:
        streamed = client.post("/chat/stream", json=request)
        regular = client.post("/chat", json=request)
    snapshots = [json.loads(line) for line in streamed.text.splitlines()]
    assert streamed.status_code == 200 and streamed.headers["content-type"].startswith("application/x-ndjson")
    assert snapshots and snapshots[-1]["tokens"] <= 3 and snapshots[-1]["kv_cache"]
    assert all(snapshots[index]["tokens"] == index + 1 for index in range(len(snapshots)))
    assert regular.status_code == 200 and regular.json()["local"] is True
