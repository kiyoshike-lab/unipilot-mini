from __future__ import annotations

import os
import csv
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChatRequest, GenerateRequest, ModelLoadRequest


app = FastAPI(title="UniPilot Mini Local API", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
runtime = {"model": None, "tokenizer": None, "device": "not loaded", "checkpoint": None, "payload": {}}


def load_runtime(checkpoint: str | None = None):
    from inference.generate import load_model
    checkpoint = checkpoint or os.getenv("UNIPILOT_CHECKPOINT", "checkpoints/v03-scratch-001/stage-c/checkpoint-step-5000.pt")
    tokenizer = os.getenv("UNIPILOT_TOKENIZER", "tokenizer/vocab-v02-512.json")
    if not Path(checkpoint).exists():
        return
    model, token, device, payload = load_model(checkpoint, tokenizer)
    runtime.update(model=model, tokenizer=token, device=device, checkpoint=checkpoint, payload=payload)


@app.on_event("startup")
def startup():
    load_runtime()


@app.get("/health")
def health():
    return {"status": "ok", "model": "UniPilot Mini", "local": True, "external_ai_api": "OFF", "loaded": runtime["model"] is not None,
            "developer_mode": os.getenv("UNIPILOT_DEV_MODE") == "1"}


@app.get("/model-info")
def model_info():
    model = runtime["model"]
    if model is None:
        return {"model": "UniPilot Mini", "loaded": False, "checkpoint": runtime["checkpoint"], "external_ai_api": "OFF"}
    config = model.config
    manifest = runtime["payload"].get("v03_manifest", {})
    return {"model": config.model_name, "loaded": True, "parameters": model.parameter_count(), "checkpoint": runtime["checkpoint"],
            "tokenizer": manifest.get("tokenizer_version", "unipilot-byte-bpe-v02-512"), "vocab_size": runtime["tokenizer"].vocab_size,
            "context_length": config.context_length, "layers": config.n_layers, "heads": config.n_heads,
            "step": runtime["payload"].get("step", 0), "validation_loss": runtime["payload"].get("loss"),
            "stage": manifest.get("stage", "legacy"), "experiment_id": manifest.get("experiment_id"),
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


@app.get("/checkpoints")
def checkpoints():
    root = Path("checkpoints").resolve()
    return {"checkpoints": [{"path": str(path.relative_to(Path.cwd())).replace("\\", "/"), "size_bytes": path.stat().st_size,
                              "modified": path.stat().st_mtime} for path in sorted(root.rglob("*.pt"))]}


@app.post("/model/load")
def model_load(request: ModelLoadRequest):
    if os.getenv("UNIPILOT_DEV_MODE") != "1":
        raise HTTPException(403, "model switching is disabled; set UNIPILOT_DEV_MODE=1 for local development")
    root = Path("checkpoints").resolve(); candidate = Path(request.checkpoint).resolve()
    try: candidate.relative_to(root)
    except ValueError: raise HTTPException(400, "checkpoint must be inside the local checkpoints directory")
    if not candidate.is_file(): raise HTTPException(404, "checkpoint not found")
    tokenizer_path = Path(request.tokenizer).resolve()
    if not tokenizer_path.is_file(): raise HTTPException(404, "tokenizer not found")
    from inference.generate import load_model
    model, token, device, payload = load_model(str(candidate), str(tokenizer_path))
    runtime.update(model=model, tokenizer=token, device=device, checkpoint=str(candidate.relative_to(Path.cwd())), payload=payload)
    return model_info()


@app.get("/evaluation/latest")
def evaluation_latest():
    preferred = Path("evaluation/results-v03-5000.json")
    files = [path for path in Path("evaluation").glob("*results*.json") if "human" not in path.name]
    if not files: raise HTTPException(404, "no evaluation result found")
    latest = preferred if preferred.exists() else max(files, key=lambda path: path.stat().st_mtime)
    return {"file": str(latest).replace("\\", "/"), "result": json.loads(latest.read_text(encoding="utf-8"))}


@app.get("/evaluation/comparison")
def evaluation_comparison():
    path = Path("evaluation/v02-v03-generations.json")
    if not path.exists(): raise HTTPException(404, "comparison result not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/training/latest")
def training_latest():
    path = Path("checkpoints/v03-scratch-001/training_log.csv")
    if not path.exists(): raise HTTPException(404, "v0.3 training log not found")
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows: raise HTTPException(404, "v0.3 training log is empty")
    row = rows[-1]
    numeric = {"step", "train_loss", "stage_validation_loss", "tokens_per_second", "eta_seconds"}
    return {key: float(value) if key in numeric else value for key, value in row.items() if key in numeric | {"stage"}}
