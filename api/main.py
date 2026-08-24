from __future__ import annotations

import os
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import (CampusHumanScoreRequest, CampusV2HumanScoreRequest, CampusV21HumanScoreRequest,
                         CampusV21QuickScoreRequest, CampusAIReviewRequest,
                         CampusV22HumanScoreRequest,
                         CampusV21KnownIssueReviewRequest, ChatRequest, GenerateRequest, HumanScoreRequest,
                         ModelLoadRequest)


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
    elif pipeline_version == "campus-v2.2":
        from pipeline.campus_v22 import UniPilotCampusV22
        pipeline = UniPilotCampusV22(model, token)
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
        if runtime["pipeline"].version in ("campus-v1", "campus-v2", "campus-v2.1", "campus-v2.2"):
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
        if runtime["pipeline"].version == "campus-v2.1" and getattr(request, "quality_mode", "off") == "improve":
            from quality.campus_answer_improver import CampusAnswerImprover
            quality = CampusAnswerImprover().improve(request.prompt, result["text"], result)
            result = {**result, "text": quality["improved_answer"], "original_text": quality["original"],
                      "quality_mode": "improve", "ai_judge": quality["after_judge"],
                      "self_critique": quality["critique"], "rewrite_count": quality["rewrite_count"],
                      "automatic_training": False}
        model_label = ({"campus-v1": "UniPilot Campus v1", "campus-v2": "UniPilot Campus v2",
                        "campus-v2.1": "UniPilot Campus v2.1", "campus-v2.2": "UniPilot Campus v2.2"}[runtime["pipeline"].version]
                       if runtime["pipeline"].version in ("campus-v1", "campus-v2", "campus-v2.1", "campus-v2.2") else
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
        if runtime["pipeline"].version in ("campus-v1", "campus-v2", "campus-v2.1", "campus-v2.2"):
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
    elif pipeline_version == "campus-v2.2":
        from pipeline.campus_v22 import UniPilotCampusV22
        pipeline = UniPilotCampusV22(model, token)
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
HUMAN_CAMPUS_V21_MANIFEST = Path("evaluation/campus-v21-rc-manifest.json")
HUMAN_CAMPUS_V21_AUDIT = Path("evaluation/campus-v21-human-audit.json")
HUMAN_CAMPUS_V21_KNOWN_ISSUES = Path("evaluation/campus-v21-rc-known-issues.json")
HUMAN_CAMPUS_V21_RESULTS = Path("evaluation/campus-v21-human-results.json")
HUMAN_CAMPUS_V21_REPORT = Path("evaluation/campus-v21-human-report.md")
HUMAN_CAMPUS_V21_QUICK_SELECTION = Path("evaluation/campus-v21-quick-selection.json")
HUMAN_CAMPUS_V21_QUICK_RESULTS = Path("evaluation/campus-v21-quick-human-results.json")
HUMAN_CAMPUS_V21_QUICK_REPORT = Path("evaluation/campus-v21-quick-human-report.md")
HUMAN_CAMPUS_V22 = Path("data/campus_v22/benchmarks/human-knowledge-100.jsonl")
CAMPUS_AI_REVIEW_QUEUE = Path("evaluation/campus-ai-review-queue.json")
CAMPUS_AI_REVIEW_DECISIONS = Path("evaluation/campus-ai-review-decisions.json")
CAMPUS_AI_APPROVED_ANSWERS = Path("data/curated/human-approved-answers.jsonl")

V21_HUMAN_AXES = ("correctness", "relevance", "actionable", "naturalness", "would_use_again")
V21_HUMAN_THRESHOLDS = {"correctness": 4.2, "relevance": 4.2, "actionable": 4.2,
                        "naturalness": 4.2, "would_use_again": 4.0}
V21_AUTOMATED_CORRECTNESS_PERCENT = 99.2


def _campus_v21_item_complete(row: dict) -> bool:
    scores = row.get("scores", {})
    return bool(row.get("issues_reviewed")) and all(scores.get(axis) is not None for axis in V21_HUMAN_AXES)


def _pairwise_counts(rows: list[dict], competitor: str) -> dict:
    counts = {"win": 0, "tie": 0, "loss": 0, "unscored": 0}
    by_axis = {axis: dict(counts) for axis in ("correctness", "specificity", "actionability", "readability", "would_use")}
    for row in rows:
        values = row.get("pairwise", {}).get(competitor, {})
        for axis in by_axis:
            choice = values.get(axis, "unscored")
            result = {"unipilot": "win", "competitor": "loss", "tie": "tie"}.get(choice, "unscored")
            counts[result] += 1
            by_axis[axis][result] += 1
    return {**counts, "by_axis": by_axis}


def _campus_v21_error_categories(rows: list[dict]) -> list[dict]:
    mappings = {
        "ROUTER": ("router_error",),
        "RETRIEVAL": ("retrieval_error",),
        "TOOL": ("tool_error",),
        "MODEL": ("model_error",),
        "KNOWLEDGE": ("factual_error", "university_policy_assertion", "faq_error"),
        "UX": ("unnecessary_information", "unusable_answer", "too_long", "too_short"),
        "OTHER": ("unanswered", "other_error"),
    }
    recommendations = {
        "ROUTER": "Routerの誤分類例を優先して境界条件を見直す",
        "RETRIEVAL": "検索失敗・誤取得例を優先して検索条件を見直す",
        "TOOL": "Tool選択または計算結果の誤りを優先して直す",
        "MODEL": "Model回答品質の失敗例を優先して評価・改善する",
        "KNOWLEDGE": "事実誤りと大学固有制度の誤断定を優先して直す",
        "UX": "冗長さ・不足・利用しづらさを優先して直す",
        "OTHER": "その他の人手メモを確認して原因を再分類する",
    }
    result = []
    for category, keys in mappings.items():
        item_ids = [row.get("id") for row in rows if any(row.get("issue_flags", {}).get(key) for key in keys)]
        result.append({"category": category, "count": len(item_ids), "item_ids": item_ids,
                       "v2_2_recommendation": recommendations[category]})
    return sorted(result, key=lambda value: (-value["count"], value["category"]))


def build_campus_v21_human_summary(rows: list[dict]) -> dict:
    total = len(rows)
    completed_rows = [row for row in rows if _campus_v21_item_complete(row)]
    completed = len(completed_rows)
    averages = {
        axis: (round(sum(row["scores"][axis] for row in completed_rows) / completed, 3) if completed else None)
        for axis in V21_HUMAN_AXES
    }
    issue_count = lambda key: sum(bool(row.get("issue_flags", {}).get(key)) for row in completed_rows)
    critical_errors = issue_count("critical_error")
    university_assertions = issue_count("university_policy_assertion")
    fully_complete = total == 100 and completed == 100
    threshold_checks = {axis: fully_complete and averages[axis] >= threshold
                        for axis, threshold in V21_HUMAN_THRESHOLDS.items()}
    critical_rate = round(critical_errors / completed, 4) if completed else None
    university_rate = round(university_assertions / completed, 4) if completed else None
    rate_checks = {
        "critical_error_rate": fully_complete and critical_rate <= .01,
        "university_policy_assertion_rate": fully_complete and university_rate <= .01,
    }
    gate_status = "PASS" if fully_complete and all(threshold_checks.values()) and all(rate_checks.values()) else (
        "FAIL" if fully_complete else "PENDING")
    automated_comparison = None
    if fully_complete:
        human_percent = round(averages["correctness"] / 5 * 100, 2)
        gap = round(V21_AUTOMATED_CORRECTNESS_PERCENT - human_percent, 2)
        if gap >= 3:
            analysis = "自動評価が人手評価を3ポイント以上上回り、過大評価している可能性があります。"
        elif gap > 0:
            analysis = "自動評価が人手評価をわずかに上回ります。誤回答例を確認してください。"
        else:
            analysis = "自動評価が人手評価を上回る傾向は確認されませんでした。"
        automated_comparison = {"automated_correctness_percent": V21_AUTOMATED_CORRECTNESS_PERCENT,
                                "human_correctness_percent": human_percent,
                                "gap_percentage_points": gap, "analysis": analysis}
    errors = _campus_v21_error_categories(completed_rows)
    return {
        "status": "COMPLETE" if fully_complete else "PENDING",
        "completed": completed,
        "pending": max(total - completed, 0),
        "total": total,
        "averages_0_to_5": averages,
        "issue_counts": {
            "critical_error": critical_errors,
            "university_policy_assertion": university_assertions,
            "router_error": issue_count("router_error"),
            "retrieval_error": issue_count("retrieval_error"),
            "tool_error": issue_count("tool_error"),
            "model_error": issue_count("model_error"),
        },
        "pairwise": {"chatgpt": _pairwise_counts(completed_rows, "chatgpt"),
                     "gemini": _pairwise_counts(completed_rows, "gemini")},
        "human_gate": {"status": gate_status, "evaluated_only_when_complete": True,
                       "thresholds": V21_HUMAN_THRESHOLDS, "threshold_checks": threshold_checks,
                       "critical_error_rate": critical_rate, "university_policy_assertion_rate": university_rate,
                       "rate_checks": rate_checks},
        "automated_comparison": automated_comparison,
        "error_categories": errors,
        "v2_2_priorities": errors if fully_complete else [],
    }


def _campus_v21_export_paths(source_path: Path) -> tuple[Path, Path]:
    if source_path.resolve() == HUMAN_CAMPUS_V21.resolve():
        return HUMAN_CAMPUS_V21_RESULTS, HUMAN_CAMPUS_V21_REPORT
    return (source_path.with_name("campus-v21-human-results.json"),
            source_path.with_name("campus-v21-human-report.md"))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _campus_ai_review_decisions() -> dict:
    if not CAMPUS_AI_REVIEW_DECISIONS.exists():
        return {"schema_version": "campus-ai-review-decisions-v1", "items": {}}
    payload = json.loads(CAMPUS_AI_REVIEW_DECISIONS.read_text(encoding="utf-8"))
    payload.setdefault("items", {})
    return payload


def _campus_ai_approved_rows() -> list[dict]:
    if not CAMPUS_AI_APPROVED_ANSWERS.exists():
        return []
    return [json.loads(line) for line in CAMPUS_AI_APPROVED_ANSWERS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def export_campus_v21_human_results(rows: list[dict], source_path: Path = HUMAN_CAMPUS_V21) -> dict:
    summary = build_campus_v21_human_summary(rows)
    results_path, report_path = _campus_v21_export_paths(source_path)
    payload = {"schema_version": "campus-v21-human-eval-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
               "rc_source_commit": "0dc18789be28613a8c651cfefde63fb659ee2019",
               "answer_logic_changed": False, "production_changed": False, "external_ai_api": "OFF",
               **summary, "items": rows}
    _atomic_write(results_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    averages = summary["averages_0_to_5"]
    issues = summary["issue_counts"]
    lines = ["# Campus v2.1 Human Evaluation Report", "", f"- Status: {summary['status']}",
             f"- Human Gate: {summary['human_gate']['status']}",
             f"- Progress: {summary['completed']} / {summary['total']}", f"- Pending: {summary['pending']}",
             "- RC answer logic changed: NO", "- Production changed: NO", "- External AI API: OFF", "",
             "## 5-axis averages (0-5)", ""]
    lines.extend(f"- {axis}: {averages[axis] if averages[axis] is not None else 'N/A'}" for axis in V21_HUMAN_AXES)
    lines.extend(["", "## Error counts", "", f"- Critical: {issues['critical_error']}",
                  f"- University policy assertion: {issues['university_policy_assertion']}",
                  f"- Router: {issues['router_error']}", f"- Retrieval: {issues['retrieval_error']}",
                  f"- Tool: {issues['tool_error']}", f"- Model: {issues['model_error']}", "",
                  "## Pairwise totals", ""])
    for competitor in ("chatgpt", "gemini"):
        counts = summary["pairwise"][competitor]
        lines.append(f"- {competitor.title()}: W {counts['win']} / T {counts['tie']} / L {counts['loss']}")
    if summary["automated_comparison"]:
        comparison = summary["automated_comparison"]
        lines.extend(["", "## Automated vs human correctness", "",
                      f"- Automated: {comparison['automated_correctness_percent']}%",
                      f"- Human: {comparison['human_correctness_percent']}%",
                      f"- Gap: {comparison['gap_percentage_points']} percentage points",
                      f"- Analysis: {comparison['analysis']}", "", "## Campus v2.2 priorities", ""])
        lines.extend(f"- {item['category']}: {item['count']} — {item['v2_2_recommendation']}"
                     for item in summary["v2_2_priorities"])
    else:
        lines.extend(["", "Automated correctness comparison and Campus v2.2 priorities are withheld until 100/100 completion."])
    _atomic_write(report_path, "\n".join(lines) + "\n")
    return {"results_path": str(results_path).replace("\\", "/"),
            "report_path": str(report_path).replace("\\", "/"), "summary": summary}


def build_campus_v21_quick_summary(items: list[dict]) -> dict:
    ratings = [item.get("quick_rating") for item in items]
    counts = {rating: ratings.count(rating) for rating in ("good", "close", "bad")}
    completed = sum(counts.values())
    total = len(items)
    rates = {rating: (round(count / completed * 100, 2) if completed else 0.0)
             for rating, count in counts.items()}
    if total != 20 or completed != 20:
        gate_status, gate_label = "PENDING", "評価中"
    elif rates["good"] >= 80 and rates["bad"] <= 5:
        gate_status, gate_label = "PASS_CANDIDATE", "PASS候補"
    elif rates["good"] < 65 or rates["bad"] >= 10:
        gate_status, gate_label = "FAIL", "FAIL"
    else:
        gate_status, gate_label = "NEEDS_IMPROVEMENT", "要改善"
    return {"status": "COMPLETE" if total == completed == 20 else "PENDING",
            "completed": completed, "pending": max(total - completed, 0), "total": total,
            "counts": counts, "rates_percent": rates,
            "quick_human_gate": {"status": gate_status, "label": gate_label, "is_simplified": True,
                                 "rules": {"pass_candidate": "good >= 80% and bad <= 5%",
                                           "needs_improvement": "good 65-79% unless FAIL condition applies",
                                           "fail": "good < 65% or bad >= 10%"}}}


def _campus_v21_quick_items() -> list[dict]:
    if not HUMAN_CAMPUS_V21.exists() or not HUMAN_CAMPUS_V21_QUICK_SELECTION.exists():
        raise HTTPException(404, "Campus v2.1 quick evaluation data not found")
    source_rows = json.loads(HUMAN_CAMPUS_V21.read_text(encoding="utf-8"))
    source_by_id = {row["id"]: row for row in source_rows}
    selection = json.loads(HUMAN_CAMPUS_V21_QUICK_SELECTION.read_text(encoding="utf-8"))["items"]
    if len(selection) != 20 or len({entry["item_id"] for entry in selection}) != 20:
        raise HTTPException(500, "Campus v2.1 quick selection must contain 20 unique items")
    saved_by_id = {}
    if HUMAN_CAMPUS_V21_QUICK_RESULTS.exists():
        saved = json.loads(HUMAN_CAMPUS_V21_QUICK_RESULTS.read_text(encoding="utf-8"))
        saved_by_id = {item["item_id"]: item for item in saved.get("items", [])}
    items = []
    for entry in selection:
        item_id = entry["item_id"]
        if item_id not in source_by_id:
            raise HTTPException(500, f"quick evaluation item not found: {item_id}")
        saved = saved_by_id.get(item_id, {})
        items.append({**source_by_id[item_id], "focus": entry["focus"],
                      "quick_rating": saved.get("rating"), "quick_reason": saved.get("reason"),
                      "quick_scored_at": saved.get("scored_at")})
    return items


def export_campus_v21_quick_results(items: list[dict], results_path: Path = HUMAN_CAMPUS_V21_QUICK_RESULTS,
                                    report_path: Path = HUMAN_CAMPUS_V21_QUICK_REPORT) -> dict:
    summary = build_campus_v21_quick_summary(items)
    result_items = [{"item_id": item["id"], "source_id": item.get("source_id"),
                     "question": item["question"], "category": item["category"],
                     "difficulty": item.get("difficulty"), "evaluation_bucket": item.get("evaluation_bucket"),
                     "focus": item["focus"], "campus_answer": item["campus_answer"],
                     "campus_metadata": item.get("campus_metadata", {}), "rating": item.get("quick_rating"),
                     "reason": item.get("quick_reason"), "scored_at": item.get("quick_scored_at")}
                    for item in items]
    payload = {"schema_version": "campus-v21-quick-human-eval-v1",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "rc_source_commit": "0dc18789be28613a8c651cfefde63fb659ee2019",
               "evaluation_type": "simplified_20_question_human_gate", "answer_logic_changed": False,
               "production_changed": False, "external_ai_api": "OFF", **summary, "items": result_items}
    _atomic_write(results_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    counts, rates = summary["counts"], summary["rates_percent"]
    lines = ["# Campus v2.1 Quick Human Evaluation", "", "This is a simplified Human Gate based on 20 fixed questions.", "",
             f"- Status: {summary['status']}", f"- Progress: {summary['completed']} / {summary['total']}",
             f"- Quick Human Gate: {summary['quick_human_gate']['label']}",
             f"- ◎ Good: {counts['good']} ({rates['good']:.2f}%)",
             f"- △ Close: {counts['close']} ({rates['close']:.2f}%)",
             f"- × Bad: {counts['bad']} ({rates['bad']:.2f}%)", "",
             "## Gate rules", "", "- PASS candidate: ◎ >= 80% and × <= 5%",
             "- Needs improvement: ◎ 65-79% unless the FAIL condition applies",
             "- FAIL: ◎ < 65% or × >= 10%", "", "- RC answer logic changed: NO",
             "- Production changed: NO", "- External AI API: OFF"]
    _atomic_write(report_path, "\n".join(lines) + "\n")
    return {"results_path": str(results_path).replace("\\", "/"),
            "report_path": str(report_path).replace("\\", "/"), "summary": summary}


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
    summary = build_campus_v21_human_summary(rows)
    issues_reviewed = sum(bool(row.get("issues_reviewed")) for row in rows)
    manifest = json.loads(HUMAN_CAMPUS_V21_MANIFEST.read_text(encoding="utf-8")) if HUMAN_CAMPUS_V21_MANIFEST.exists() else None
    audit = json.loads(HUMAN_CAMPUS_V21_AUDIT.read_text(encoding="utf-8")) if HUMAN_CAMPUS_V21_AUDIT.exists() else None
    return {**summary, "issues_reviewed": issues_reviewed, "items": rows,
            "manifest": manifest, "audit": audit, "external_ai_api": "OFF"}


@app.post("/human-eval/campus-v21")
def human_eval_campus_v21_score(request: CampusV21HumanScoreRequest):
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
            row["issue_flags"] = request.issue_flags.model_dump()
            row["issues_reviewed"] = request.issues_reviewed
            row["pairwise"] = request.pairwise.model_dump()
            row["ux"] = request.ux.model_dump()
            row["other_issue"] = request.other_issue
            row["notes"] = request.notes
            row["evaluation_status"] = "SCORED_MANUALLY" if request.issues_reviewed else "PENDING_ISSUE_REVIEW"
            found = True
            break
    if not found:
        raise HTTPException(404, "Campus v2.1 human evaluation item not found")
    temporary = HUMAN_CAMPUS_V21.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(HUMAN_CAMPUS_V21)
    saved = next(row for row in rows if row["id"] == request.item_id)
    exported = export_campus_v21_human_results(rows, HUMAN_CAMPUS_V21)
    return {"saved": True, "item_id": request.item_id, "scores": saved["scores"],
            "summary": exported["summary"], "exports": {"results_path": exported["results_path"],
                                                            "report_path": exported["report_path"]}}


@app.post("/human-eval/campus-v21/export")
def human_eval_campus_v21_export():
    if not HUMAN_CAMPUS_V21.exists():
        raise HTTPException(404, "Campus v2.1 human comparison file not found")
    rows = json.loads(HUMAN_CAMPUS_V21.read_text(encoding="utf-8"))
    return export_campus_v21_human_results(rows, HUMAN_CAMPUS_V21)


@app.get("/human-eval/campus-v21/quick")
def human_eval_campus_v21_quick():
    items = _campus_v21_quick_items()
    return {**build_campus_v21_quick_summary(items), "items": items, "external_ai_api": "OFF"}


@app.post("/human-eval/campus-v21/quick")
def human_eval_campus_v21_quick_score(request: CampusV21QuickScoreRequest):
    items = _campus_v21_quick_items()
    found = False
    for item in items:
        if item["id"] == request.item_id:
            item["quick_rating"] = request.rating
            item["quick_reason"] = request.reason if request.rating == "bad" else None
            item["quick_scored_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break
    if not found:
        raise HTTPException(404, "Campus v2.1 quick evaluation item not found")
    exported = export_campus_v21_quick_results(items, HUMAN_CAMPUS_V21_QUICK_RESULTS,
                                               HUMAN_CAMPUS_V21_QUICK_REPORT)
    return {"saved": True, "item_id": request.item_id, "rating": request.rating,
            "reason": request.reason if request.rating == "bad" else None,
            "summary": exported["summary"]}


@app.post("/human-eval/campus-v21/quick/export")
def human_eval_campus_v21_quick_export():
    items = _campus_v21_quick_items()
    return export_campus_v21_quick_results(items, HUMAN_CAMPUS_V21_QUICK_RESULTS,
                                           HUMAN_CAMPUS_V21_QUICK_REPORT)


@app.get("/ai-review/campus")
def campus_ai_review():
    if not CAMPUS_AI_REVIEW_QUEUE.exists():
        raise HTTPException(404, "Campus AI review queue not found; run the local quality evaluation")
    queue = json.loads(CAMPUS_AI_REVIEW_QUEUE.read_text(encoding="utf-8"))
    decisions = _campus_ai_review_decisions()["items"]
    items = [{**item, **decisions.get(item["item_id"], {})} for item in queue.get("items", [])]
    counts = {decision: sum(item.get("decision") == decision for item in items)
              for decision in ("adopt", "revise", "reject")}
    reviewed = sum(item.get("decision") in counts for item in items)
    return {**queue, "reviewed": reviewed, "pending": len(items) - reviewed,
            "decision_counts": counts, "items": items, "external_ai_api": "OFF",
            "automatic_training": False}


@app.post("/ai-review/campus")
def campus_ai_review_save(request: CampusAIReviewRequest):
    if not CAMPUS_AI_REVIEW_QUEUE.exists():
        raise HTTPException(404, "Campus AI review queue not found; run the local quality evaluation")
    queue = json.loads(CAMPUS_AI_REVIEW_QUEUE.read_text(encoding="utf-8"))
    item = next((row for row in queue.get("items", []) if row["item_id"] == request.item_id), None)
    if item is None:
        raise HTTPException(404, "Campus AI review item not found")
    now = datetime.now(timezone.utc).isoformat()
    decisions = _campus_ai_review_decisions()
    decisions["updated_at"] = now
    decisions["items"][request.item_id] = {
        "decision": request.decision,
        "edited_answer": request.edited_answer,
        "notes": request.notes,
        "reviewed_at": now,
    }
    _atomic_write(CAMPUS_AI_REVIEW_DECISIONS,
                  json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")

    approved = [row for row in _campus_ai_approved_rows() if row.get("item_id") != request.item_id]
    if request.decision == "adopt":
        approved_answer = request.edited_answer.strip() or item["improved_answer"]
        approved.append({
            "schema_version": "campus-human-approved-answer-v1",
            "id": f"approved-{request.item_id}",
            "item_id": request.item_id,
            "question": item["question"],
            "original_answer": item["original_answer"],
            "approved_answer": approved_answer,
            "category": item.get("category"),
            "route": item.get("route"),
            "source_ids": item.get("source_ids", []),
            "ai_judge_score": item.get("ai_judge_score"),
            "improved_score": item.get("improved_score"),
            "human_notes": request.notes,
            "approved_at": now,
            "external_ai_api": "OFF",
            "automatic_training": False,
            "requires_training_review": True,
        })
    approved_content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in approved)
    _atomic_write(CAMPUS_AI_APPROVED_ANSWERS, approved_content)
    return {"saved": True, "item_id": request.item_id, "decision": request.decision,
            "approved_memory_count": len(approved), "automatic_training": False}


@app.get("/human-eval/campus-v21/known-issues")
def human_eval_campus_v21_known_issues():
    if not HUMAN_CAMPUS_V21_KNOWN_ISSUES.exists():
        raise HTTPException(404, "Campus v2.1 known-issue review file not found")
    payload = json.loads(HUMAN_CAMPUS_V21_KNOWN_ISSUES.read_text(encoding="utf-8"))
    items = [item for group in payload["groups"].values() for item in group]
    reviewed = sum(item.get("human_review", {}).get("status") != "pending" for item in items)
    return {**payload, "status": "COMPLETE" if reviewed == len(items) else "PENDING",
            "reviewed": reviewed, "total": len(items)}


@app.post("/human-eval/campus-v21/known-issues")
def human_eval_campus_v21_known_issue_score(request: CampusV21KnownIssueReviewRequest):
    if not HUMAN_CAMPUS_V21_KNOWN_ISSUES.exists():
        raise HTTPException(404, "Campus v2.1 known-issue review file not found")
    payload = json.loads(HUMAN_CAMPUS_V21_KNOWN_ISSUES.read_text(encoding="utf-8")); found = False
    for item in payload["groups"].get(request.group, []):
        if item["id"] == request.item_id:
            item["human_review"] = {"status": request.status, "severity": request.severity,
                                    "blocks_production": request.blocks_production, "notes": request.notes}
            found = True
            break
    if not found:
        raise HTTPException(404, "Campus v2.1 known issue not found")
    temporary = HUMAN_CAMPUS_V21_KNOWN_ISSUES.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(HUMAN_CAMPUS_V21_KNOWN_ISSUES)
    return {"saved": True, "item_id": request.item_id}


@app.get("/human-eval/campus-v22")
def human_eval_campus_v22():
    if not HUMAN_CAMPUS_V22.exists():
        raise HTTPException(404, "Campus v2.2 human knowledge evaluation file not found")
    rows = [json.loads(line) for line in HUMAN_CAMPUS_V22.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = sum(row.get("scores", {}).get("correctness") is not None for row in rows)
    return {"status": "COMPLETE" if completed == len(rows) else "PENDING", "completed": completed,
            "total": len(rows), "items": rows, "external_ai_api": "OFF"}


@app.post("/human-eval/campus-v22")
def human_eval_campus_v22_score(request: CampusV22HumanScoreRequest):
    if not HUMAN_CAMPUS_V22.exists():
        raise HTTPException(404, "Campus v2.2 human knowledge evaluation file not found")
    rows = [json.loads(line) for line in HUMAN_CAMPUS_V22.read_text(encoding="utf-8").splitlines() if line.strip()]
    found = False
    for row in rows:
        if row["id"] == request.item_id:
            row["scores"] = {key: getattr(request, key) for key in (
                "correctness", "depth", "grounding", "usefulness", "naturalness", "would_use_again"
            )}
            row["notes"] = request.notes
            row["evaluation_status"] = "SCORED_MANUALLY"
            found = True
            break
    if not found:
        raise HTTPException(404, "Campus v2.2 human evaluation item not found")
    temporary = HUMAN_CAMPUS_V22.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(HUMAN_CAMPUS_V22)
    return {"saved": True, "item_id": request.item_id}


@app.get("/campus/session/{session_id}")
def campus_session(session_id: str):
    pipeline = runtime.get("pipeline")
    if pipeline is None or getattr(pipeline, "version", None) not in ("campus-v1", "campus-v2", "campus-v2.1", "campus-v2.2"):
        raise HTTPException(404, "Campus mode is not enabled")
    return {"session_id": session_id, "state": pipeline.sessions.get(session_id)}


@app.delete("/campus/session/{session_id}")
def campus_session_delete(session_id: str):
    pipeline = runtime.get("pipeline")
    if pipeline is None or getattr(pipeline, "version", None) not in ("campus-v1", "campus-v2", "campus-v2.1", "campus-v2.2"):
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
