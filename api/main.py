from __future__ import annotations

import os
import csv
import json
from pathlib import Path
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import (CampusHumanScoreRequest, CampusV2HumanScoreRequest, ChatRequest, GenerateRequest,
                         HumanScoreRequest, ModelLoadRequest)


app = FastAPI(title="UniPilot Mini Local API", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://unipilot-mini-pjgy.vercel.app"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
runtime = {"model": None, "tokenizer": None, "device": "not loaded", "checkpoint": None, "payload": {}, "pipeline": None}


def process_memory() -> dict:
    current = peak = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        current = resident_pages * page_size / 1024**2
    except (AttributeError, IndexError, OSError, ValueError):
        pass
    try:
        import resource
        raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = raw_peak / (1024**2 if sys.platform == "darwin" else 1024)
    except (ImportError, OSError, ValueError):
        pass
    return {"rss_mb": current, "peak_rss_mb": peak}


def load_runtime(checkpoint: str | None = None):
    from inference.generate import load_model
    inference_checkpoint = Path("checkpoints/v04-eos15/unipilot-mini-v04-inference.pt")
    default_checkpoint = inference_checkpoint if inference_checkpoint.exists() else Path("checkpoints/v04-eos15/checkpoint-step-2000.pt")
    configured_checkpoint = checkpoint or os.getenv("UNIPILOT_CHECKPOINT")
    if os.getenv("RENDER"):
        from scripts.download_production_checkpoint import ensure_checkpoint
        ensure_checkpoint(inference_checkpoint)
        if configured_checkpoint is None or Path(configured_checkpoint).name == "checkpoint-step-2000.pt":
            configured_checkpoint = str(inference_checkpoint)
    checkpoint = configured_checkpoint or str(default_checkpoint)
    tokenizer = os.getenv("UNIPILOT_TOKENIZER", "tokenizer/vocab-v02-512.json")
    if not Path(checkpoint).exists():
        return
    model, token, device, payload = load_model(checkpoint, tokenizer)
    expected_version = os.getenv("UNIPILOT_EXPECT_MODEL_VERSION", "v0.4" if os.getenv("RENDER") else "")
    if expected_version and expected_version not in model.config.model_name:
        raise RuntimeError(f"production checkpoint must be {expected_version}, got {model.config.model_name}")
    pipeline = None
    pipeline_version = os.getenv("UNIPILOT_PIPELINE_VERSION")
    if pipeline_version == "v0.7":
        from pipeline.v07 import V07Pipeline
        pipeline = V07Pipeline(model, token, top_k=max(1, int(os.getenv("UNIPILOT_RAG_TOP_K", "1"))))
    elif pipeline_version == "v0.8":
        from pipeline.v08 import V08Pipeline
        pipeline = V08Pipeline(model, token, retrieval_method=os.getenv("UNIPILOT_RETRIEVAL_METHOD", "tfidf"),
                               top_k=max(1, int(os.getenv("UNIPILOT_RAG_TOP_K", "3"))))
    elif pipeline_version == "campus-v1":
        from pipeline.campus_v1 import UniPilotCampusV1
        pipeline = UniPilotCampusV1(model, token)
    elif pipeline_version == "campus-v2":
        from pipeline.campus_v2 import UniPilotCampusV2
        pipeline = UniPilotCampusV2(model, token)
    elif pipeline_version == "campus-v2.1":
        from pipeline.campus_v21 import UniPilotCampusV21
        pipeline = UniPilotCampusV21(model, token)
    runtime.update(model=model, tokenizer=token, device=device, checkpoint=checkpoint, payload=payload, pipeline=pipeline)


@app.on_event("startup")
def startup():
    load_runtime()


@app.get("/health")
def health():
    return {"status": "ok", "model": "UniPilot Mini", "local": True, "external_ai_api": "OFF", "loaded": runtime["model"] is not None,
            "developer_mode": os.getenv("UNIPILOT_DEV_MODE") == "1", **process_memory()}


@app.get("/model-info")
def model_info():
    model = runtime["model"]
    if model is None:
        return {"model": "UniPilot Mini", "loaded": False, "checkpoint": runtime["checkpoint"], "external_ai_api": "OFF"}
    config = model.config
    manifest = (runtime["payload"].get("v08_manifest") or runtime["payload"].get("v07_manifest") or runtime["payload"].get("v06_manifest") or
                runtime["payload"].get("v04_manifest") or runtime["payload"].get("v03_manifest", {}))
    return {"model": config.model_name, "loaded": True, "parameters": model.parameter_count(), "checkpoint": runtime["checkpoint"],
            "tokenizer": manifest.get("tokenizer_version", "unipilot-byte-bpe-v02-512"), "vocab_size": runtime["tokenizer"].vocab_size,
            "context_length": config.context_length, "layers": config.n_layers, "heads": config.n_heads,
            "step": runtime["payload"].get("step", 0), "validation_loss": runtime["payload"].get("loss"),
            "stage": manifest.get("stage", "Clean C" if runtime["payload"].get("v04_manifest") else "legacy"), "experiment_id": manifest.get("experiment_id"),
            "device": runtime["device"], "pipeline": runtime["pipeline"].version if runtime["pipeline"] else "legacy",
            "external_ai_api": "OFF"}


def chat_prompt(prompt: str) -> str:
    system_text = "あなたは大学生活を支援する完全ローカルのUniPilot Miniです。情報がない場合は推測せず、確認方法を案内します。"
    return f"<BOS><SYSTEM>\n{system_text}\n<USER>\n{prompt}\n<ASSISTANT>\n"


def run_generation(request: GenerateRequest, chat: bool):
    if runtime["model"] is None:
        raise HTTPException(503, "checkpoint not loaded; set UNIPILOT_CHECKPOINT")
    if chat and runtime["pipeline"] is not None:
        if runtime["pipeline"].version in ("campus-v1", "campus-v2", "campus-v2.1"):
            result = runtime["pipeline"].answer(
                request.prompt, request.max_new_tokens, request.temperature, request.top_k,
                request.top_p, request.repetition_penalty, request.response_mode,
                request.session_id, request.tool_inputs,
            )
        elif runtime["pipeline"].version == "v0.8":
            result = runtime["pipeline"].answer(request.prompt, request.max_new_tokens, request.temperature, request.top_k,
                                                request.top_p, request.repetition_penalty, request.response_mode)
        else:
            result = runtime["pipeline"].answer(request.prompt, request.max_new_tokens, request.temperature, request.top_k,
                                                request.top_p, request.repetition_penalty,
                                                candidates=max(1, int(os.getenv("UNIPILOT_V07_CANDIDATES", "1"))))
        model_label = ({"campus-v1": "UniPilot Campus v1", "campus-v2": "UniPilot Campus v2",
                        "campus-v2.1": "UniPilot Campus v2.1"}[runtime["pipeline"].version]
                       if runtime["pipeline"].version in ("campus-v1", "campus-v2", "campus-v2.1") else
            runtime["model"].config.model_name if runtime["pipeline"].version == "v0.8" else "UniPilot Mini")
        return {**result, "model": model_label, "local": True, "metrics": result["generation_metrics"]}
    from inference.generate import generate_text
    prompt = chat_prompt(request.prompt) if chat else request.prompt
    text, metrics = generate_text(runtime["model"], runtime["tokenizer"], prompt, request.max_new_tokens,
                                  request.temperature, request.top_k, request.top_p, request.repetition_penalty)
    return {"text": text, "model": "UniPilot Mini", "local": True, "metrics": metrics}


@app.post("/generate")
def generate(request: GenerateRequest): return run_generation(request, False)


@app.post("/chat")
def chat(request: ChatRequest): return run_generation(request, True)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    if runtime["model"] is None:
        raise HTTPException(503, "checkpoint not loaded; set UNIPILOT_CHECKPOINT")
    if runtime["pipeline"] is not None:
        if runtime["pipeline"].version in ("campus-v1", "campus-v2", "campus-v2.1"):
            def campus_events():
                for snapshot in runtime["pipeline"].iter_answer(
                        request.prompt, request.max_new_tokens, request.temperature, request.top_k,
                        request.top_p, request.repetition_penalty, request.response_mode,
                        request.session_id, request.tool_inputs):
                    yield json.dumps(snapshot, ensure_ascii=False) + "\n"
            return StreamingResponse(campus_events(), media_type="application/x-ndjson",
                                     headers={"Cache-Control": "no-cache"})
        if runtime["pipeline"].version == "v0.8":
            def standard_events():
                for snapshot in runtime["pipeline"].iter_answer(
                        request.prompt, request.max_new_tokens, request.temperature, request.top_k,
                        request.top_p, request.repetition_penalty, request.response_mode):
                    yield json.dumps(snapshot, ensure_ascii=False) + "\n"
            return StreamingResponse(standard_events(), media_type="application/x-ndjson",
                                     headers={"Cache-Control": "no-cache"})
        def grounded_events():
            result = runtime["pipeline"].answer(request.prompt, request.max_new_tokens, request.temperature, request.top_k,
                                                request.top_p, request.repetition_penalty,
                                                candidates=max(1, int(os.getenv("UNIPILOT_V07_CANDIDATES", "1"))))
            ids = runtime["tokenizer"].encode(result["text"])
            for index in range(1, len(ids) + 1):
                yield json.dumps({"text": runtime["tokenizer"].decode(ids[:index], skip_special=True),
                                  "tokens": index, "eos_reached": index == len(ids), "kv_cache": True,
                                  "pipeline": "v0.7", "fallback_used": result["fallback_used"],
                                  "category": result["category"]}, ensure_ascii=False) + "\n"
        return StreamingResponse(grounded_events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})
    from inference.generate import iter_generate_text

    def events():
        for snapshot in iter_generate_text(
            runtime["model"], runtime["tokenizer"], chat_prompt(request.prompt), request.max_new_tokens,
            request.temperature, request.top_k, request.top_p, request.repetition_penalty,
        ):
            yield json.dumps(snapshot, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


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
    pipeline = None
    pipeline_version = os.getenv("UNIPILOT_PIPELINE_VERSION")
    if pipeline_version == "v0.7":
        from pipeline.v07 import V07Pipeline
        pipeline = V07Pipeline(model, token, top_k=max(1, int(os.getenv("UNIPILOT_RAG_TOP_K", "1"))))
    elif pipeline_version == "v0.8":
        from pipeline.v08 import V08Pipeline
        pipeline = V08Pipeline(model, token, retrieval_method=os.getenv("UNIPILOT_RETRIEVAL_METHOD", "tfidf"),
                               top_k=max(1, int(os.getenv("UNIPILOT_RAG_TOP_K", "3"))))
    elif pipeline_version == "campus-v1":
        from pipeline.campus_v1 import UniPilotCampusV1
        pipeline = UniPilotCampusV1(model, token)
    elif pipeline_version == "campus-v2":
        from pipeline.campus_v2 import UniPilotCampusV2
        pipeline = UniPilotCampusV2(model, token)
    elif pipeline_version == "campus-v2.1":
        from pipeline.campus_v21 import UniPilotCampusV21
        pipeline = UniPilotCampusV21(model, token)
    runtime.update(model=model, tokenizer=token, device=device, checkpoint=str(candidate.relative_to(Path.cwd())),
                   payload=payload, pipeline=pipeline)
    return model_info()


@app.get("/evaluation/latest")
def evaluation_latest():
    preferred = Path("evaluation/results-v04-best-2000.json")
    files = [path for path in Path("evaluation").glob("*results*.json") if "human" not in path.name]
    if not files: raise HTTPException(404, "no evaluation result found")
    latest = preferred if preferred.exists() else max(files, key=lambda path: path.stat().st_mtime)
    return {"file": str(latest).replace("\\", "/"), "result": json.loads(latest.read_text(encoding="utf-8"))}


@app.get("/evaluation/comparison")
def evaluation_comparison():
    path = Path("evaluation/v02-v03-generations.json")
    if not path.exists(): raise HTTPException(404, "comparison result not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/evaluation/v03-v04")
def evaluation_v03_v04():
    path = Path("evaluation/v03-v04-generations.json")
    if not path.exists(): raise HTTPException(404, "v0.3/v0.4 comparison not found")
    return json.loads(path.read_text(encoding="utf-8"))


HUMAN_V04 = Path("evaluation/results-v04-best-2000-human.json")


@app.get("/human-eval/v04")
def human_eval_v04():
    if not HUMAN_V04.exists(): raise HTTPException(404, "v0.4 human evaluation file not found")
    return {"status": "PENDING", "items": json.loads(HUMAN_V04.read_text(encoding="utf-8"))}


@app.post("/human-eval/v04")
def human_eval_v04_score(request: HumanScoreRequest):
    if not HUMAN_V04.exists(): raise HTTPException(404, "v0.4 human evaluation file not found")
    rows = json.loads(HUMAN_V04.read_text(encoding="utf-8")); found = False
    for row in rows:
        if row["id"] == request.item_id:
            row["score"] = request.score; row["notes"] = request.notes; found = True; break
    if not found: raise HTTPException(404, "human evaluation item not found")
    temporary = HUMAN_V04.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"); temporary.replace(HUMAN_V04)
    return {"saved": True, "item_id": request.item_id, "score": request.score}


HUMAN_CAMPUS = Path("evaluation/human-comparison-campus-v1.json")
HUMAN_CAMPUS_V2 = Path("evaluation/human-comparison-campus-v2.json")
HUMAN_CAMPUS_V21 = Path("evaluation/human-comparison-campus-v21.json")


@app.get("/human-eval/campus")
def human_eval_campus():
    if not HUMAN_CAMPUS.exists():
        raise HTTPException(404, "Campus human comparison file not found")
    rows = json.loads(HUMAN_CAMPUS.read_text(encoding="utf-8"))
    completed = sum(row.get("scores", {}).get("campus") is not None for row in rows)
    return {"status": "COMPLETE" if completed == len(rows) else "PENDING", "completed": completed,
            "total": len(rows), "items": rows}


@app.post("/human-eval/campus")
def human_eval_campus_score(request: CampusHumanScoreRequest):
    if not HUMAN_CAMPUS.exists():
        raise HTTPException(404, "Campus human comparison file not found")
    rows = json.loads(HUMAN_CAMPUS.read_text(encoding="utf-8")); found = False
    for row in rows:
        if row["id"] == request.item_id:
            row["scores"] = {"campus": request.campus_score, "chatgpt": request.chatgpt_score,
                             "gemini": request.gemini_score}
            row["winners"] = {"correct": request.correct_winner, "specific": request.specific_winner,
                              "usable": request.usable_winner, "fast": request.fast_winner,
                              "student_preference": request.student_preference}
            row["chatgpt_answer"] = request.chatgpt_answer
            row["gemini_answer"] = request.gemini_answer
            row["notes"] = request.notes
            found = True
            break
    if not found:
        raise HTTPException(404, "Campus human evaluation item not found")
    temporary = HUMAN_CAMPUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(HUMAN_CAMPUS)
    return {"saved": True, "item_id": request.item_id, "score": request.campus_score}


@app.get("/human-eval/campus-v2")
def human_eval_campus_v2():
    if not HUMAN_CAMPUS_V2.exists():
        raise HTTPException(404, "Campus v2 human comparison file not found")
    rows = json.loads(HUMAN_CAMPUS_V2.read_text(encoding="utf-8"))
    completed = sum(row.get("scores", {}).get("correctness") is not None for row in rows)
    return {"status": "COMPLETE" if completed == len(rows) else "PENDING", "completed": completed,
            "total": len(rows), "items": rows, "external_ai_api": "OFF"}


@app.post("/human-eval/campus-v2")
def human_eval_campus_v2_score(request: CampusV2HumanScoreRequest):
    if not HUMAN_CAMPUS_V2.exists():
        raise HTTPException(404, "Campus v2 human comparison file not found")
    rows = json.loads(HUMAN_CAMPUS_V2.read_text(encoding="utf-8")); found = False
    for row in rows:
        if row["id"] == request.item_id:
            row["scores"] = {"correctness": request.correctness, "relevance": request.relevance,
                             "actionable": request.actionable, "naturalness": request.naturalness,
                             "would_use_again": request.would_use_again}
            row["competitor_scores"] = {"chatgpt": request.chatgpt_score, "gemini": request.gemini_score}
            row["chatgpt_answer"] = request.chatgpt_answer
            row["gemini_answer"] = request.gemini_answer
            row["notes"] = request.notes
            row["evaluation_status"] = "SCORED_MANUALLY"
            found = True
            break
    if not found:
        raise HTTPException(404, "Campus v2 human evaluation item not found")
    temporary = HUMAN_CAMPUS_V2.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(HUMAN_CAMPUS_V2)
    return {"saved": True, "item_id": request.item_id, "scores": rows[next(
        index for index, row in enumerate(rows) if row["id"] == request.item_id)]["scores"]}


@app.get("/human-eval/campus-v21")
def human_eval_campus_v21():
    if not HUMAN_CAMPUS_V21.exists():
        raise HTTPException(404, "Campus v2.1 human comparison file not found")
    rows = json.loads(HUMAN_CAMPUS_V21.read_text(encoding="utf-8"))
    completed = sum(row.get("scores", {}).get("correctness") is not None for row in rows)
    return {"status": "COMPLETE" if completed == len(rows) else "PENDING", "completed": completed,
            "total": len(rows), "items": rows, "external_ai_api": "OFF"}


@app.post("/human-eval/campus-v21")
def human_eval_campus_v21_score(request: CampusV2HumanScoreRequest):
    if not HUMAN_CAMPUS_V21.exists():
        raise HTTPException(404, "Campus v2.1 human comparison file not found")
    rows = json.loads(HUMAN_CAMPUS_V21.read_text(encoding="utf-8")); found = False
    for row in rows:
        if row["id"] == request.item_id:
            row["scores"] = {"correctness": request.correctness, "relevance": request.relevance,
                             "actionable": request.actionable, "naturalness": request.naturalness,
                             "would_use_again": request.would_use_again}
            row["competitor_scores"] = {"chatgpt": request.chatgpt_score, "gemini": request.gemini_score}
            row["chatgpt_answer"] = request.chatgpt_answer
            row["gemini_answer"] = request.gemini_answer
            row["notes"] = request.notes
            row["evaluation_status"] = "SCORED_MANUALLY"
            found = True
            break
    if not found:
        raise HTTPException(404, "Campus v2.1 human evaluation item not found")
    temporary = HUMAN_CAMPUS_V21.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(HUMAN_CAMPUS_V21)
    saved = next(row for row in rows if row["id"] == request.item_id)
    return {"saved": True, "item_id": request.item_id, "scores": saved["scores"]}


@app.get("/campus/session/{session_id}")
def campus_session(session_id: str):
    pipeline = runtime.get("pipeline")
    if pipeline is None or getattr(pipeline, "version", None) not in ("campus-v1", "campus-v2", "campus-v2.1"):
        raise HTTPException(404, "Campus mode is not enabled")
    return {"session_id": session_id, "state": pipeline.sessions.get(session_id)}


@app.delete("/campus/session/{session_id}")
def campus_session_delete(session_id: str):
    pipeline = runtime.get("pipeline")
    if pipeline is None or getattr(pipeline, "version", None) not in ("campus-v1", "campus-v2", "campus-v2.1"):
        raise HTTPException(404, "Campus mode is not enabled")
    return {"deleted": pipeline.sessions.delete(session_id)}


@app.get("/campus/benchmark")
def campus_benchmark():
    path = Path("evaluation/campus-v1-benchmark.json")
    if not path.exists():
        raise HTTPException(404, "Campus benchmark not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/training/latest")
def training_latest():
    path = Path("checkpoints/v04-eos15/training_log.csv")
    v04 = path.exists()
    if not v04: path = Path("checkpoints/v03-scratch-001/training_log.csv")
    if not path.exists(): raise HTTPException(404, "v0.3 training log not found")
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows: raise HTTPException(404, "v0.3 training log is empty")
    row = rows[-1]
    if v04:
        return {"step": float(row["step"]), "stage": "Clean C", "train_loss": float(row["loss"]),
                "stage_validation_loss": float(row["validation_loss"]), "tokens_per_second": float(row["tokens_per_second"]), "eta_seconds": 0.0}
    numeric = {"step", "train_loss", "stage_validation_loss", "tokens_per_second", "eta_seconds"}
    return {key: float(value) if key in numeric else value for key, value in row.items() if key in numeric | {"stage"}}
