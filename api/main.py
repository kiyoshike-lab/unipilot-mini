from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChatRequest, GenerateRequest


app = FastAPI(title="UniPilot Mini Local API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
runtime = {"model": None, "tokenizer": None, "device": "not loaded", "checkpoint": None, "payload": {}}


def load_runtime(checkpoint: str | None = None):
    from inference.generate import load_model
    checkpoint = checkpoint or os.getenv("UNIPILOT_CHECKPOINT", "checkpoints/sanity-100/checkpoint-step-100.pt")
    tokenizer = os.getenv("UNIPILOT_TOKENIZER", "tokenizer/vocab.json")
    if not Path(checkpoint).exists():
        return
    model, token, device, payload = load_model(checkpoint, tokenizer)
    runtime.update(model=model, tokenizer=token, device=device, checkpoint=checkpoint, payload=payload)


@app.on_event("startup")
def startup():
    load_runtime()


@app.get("/health")
def health():
    return {"status": "ok", "model": "UniPilot Mini", "local": True, "external_ai_api": "OFF", "loaded": runtime["model"] is not None}


@app.get("/model-info")
def model_info():
    model = runtime["model"]
    if model is None:
        return {"model": "UniPilot Mini", "loaded": False, "checkpoint": runtime["checkpoint"], "external_ai_api": "OFF"}
    config = model.config
    return {"model": config.model_name, "loaded": True, "parameters": model.parameter_count(), "checkpoint": runtime["checkpoint"],
            "tokenizer": "unipilot-byte-bpe-v1", "vocab_size": runtime["tokenizer"].vocab_size,
            "context_length": config.context_length, "layers": config.n_layers, "heads": config.n_heads,
            "step": runtime["payload"].get("step", 0), "validation_loss": runtime["payload"].get("loss"),
            "device": runtime["device"], "external_ai_api": "OFF"}


def run_generation(request: GenerateRequest, chat: bool):
    if runtime["model"] is None:
        raise HTTPException(503, "checkpoint not loaded; set UNIPILOT_CHECKPOINT")
    from inference.generate import generate_text
    prompt = f"<BOS><USER>\n{request.prompt}\n<ASSISTANT>\n" if chat else request.prompt
    text, metrics = generate_text(runtime["model"], runtime["tokenizer"], prompt, request.max_new_tokens,
                                  request.temperature, request.top_k, request.top_p, request.repetition_penalty)
    return {"text": text, "model": "UniPilot Mini", "local": True, "metrics": metrics}


@app.post("/generate")
def generate(request: GenerateRequest): return run_generation(request, False)


@app.post("/chat")
def chat(request: ChatRequest): return run_generation(request, True)
