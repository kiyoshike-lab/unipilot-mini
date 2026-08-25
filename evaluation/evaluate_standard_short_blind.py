from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

import psutil
import torch

from evaluation.evaluate_campus_v22_generalization import aggregate, judge_result
from inference.generate import iter_generate_text, load_model
from pipeline.campus_v23 import UniPilotCampusV23
from pipeline.v08 import MODE_LIMITS, V08Pipeline
from quality.campus_ai_judge import CampusAIJudge
from tokenizer.tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parents[1]
BLIND = ROOT / "data/standard_50m_short/blind-200.json"
MANIFEST = ROOT / "data/standard_50m_short/manifest.json"
OUT = ROOT / "evaluation/standard-50m-short"
AXES = ("correctness", "relevance", "completeness", "specificity", "naturalness", "actionable")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_seal() -> str:
    expected = read_json(MANIFEST)["blind_sha256_at_seal"]
    actual = hashlib.sha256(BLIND.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("independent Blind 200 changed after sealing")
    return actual


def performance_summary(records: list[dict], peak_rss_mb: float, system: str) -> dict:
    base = aggregate(records)
    if any("fallback_used" in row for row in records):
        base["fallback_rate"] = round(mean(bool(row.get("fallback_used")) for row in records), 4)
    raw_records = [row for row in records if "raw_answer" in row]
    return {
        **base,
        "axis_percent": {axis: base["axis_percent"][axis] for axis in AXES},
        "average_first_token_seconds": round(mean(row["first_token_seconds"] for row in records), 6),
        "average_total_seconds": round(mean(row["total_seconds"] for row in records), 6),
        "average_tokens_per_second": round(mean(row["tokens_per_second"] for row in records), 3),
        "peak_rss_mb": round(peak_rss_mb, 3),
        "raw_generation_success_rate": (
            round(mean(bool(row["raw_answer"].strip()) for row in raw_records), 4)
            if raw_records else None
        ),
        "average_raw_characters": (
            round(mean(len(row["raw_answer"].strip()) for row in raw_records), 2)
            if raw_records else None
        ),
        "latency_semantics": (
            "autoregressive first token and generated-token throughput"
            if system == "standard"
            else "deterministic completed-response latency and effective output-token throughput"
        ),
    }


def campus_records(rows: list[dict], tokenizer, judge: CampusAIJudge) -> tuple[list[dict], float]:
    pipeline = UniPilotCampusV23()
    process = psutil.Process()
    peak = process.memory_info().rss / 1024**2
    records = []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        result = pipeline.answer(row["question"], response_mode="auto",
                                 session_id=f"standard-short-campus-{index:03d}")
        elapsed = time.perf_counter() - started
        judged = judge_result(judge, row["question"], row["expected_category"], result)
        tokens = len(tokenizer.encode(result.get("text", "")))
        records.append({
            "id": row["id"], "question": row["question"], "category": row["expected_category"],
            "predicted_category": result.get("category"), "route": result.get("route"),
            "answer": result.get("text", ""), "judge": judged, "score": judged["overall_score"],
            "first_token_seconds": elapsed, "total_seconds": elapsed,
            "tokens_per_second": tokens / max(elapsed, 1e-9),
            "performance_note": "Campus v2.3 is deterministic; first token equals completed-response latency.",
            "retrieval": result.get("retrieval", []), "sources": result.get("sources", []),
            "cards": result.get("cards", []), "validator": result.get("validator", {}),
        })
        peak = max(peak, process.memory_info().rss / 1024**2)
    return records, peak


def standard_answer(pipeline: V08Pipeline, question: str, max_new_tokens: int) -> dict:
    prepared = pipeline.prepare(question, "auto")
    cap = min(max_new_tokens, MODE_LIMITS[prepared["mode"]], prepared["max_answer_tokens"])
    first = None
    last = None
    for snapshot in iter_generate_text(pipeline.model, pipeline.tokenizer, prepared["prompt"], cap,
                                       temperature=0.0, top_k=40, top_p=0.9,
                                       repetition_penalty=1.1):
        if first is None:
            first = float(snapshot["seconds"])
        last = snapshot
    raw = last["text"] if last else ""
    validation = pipeline.validator.validate(question, raw, prepared["context"])
    fallback = not validation.valid
    answer = pipeline.validator.fallback(prepared["category"]) if fallback else raw
    generation_seconds = float(last["seconds"]) if last else 0.0
    return {
        "text": answer, "raw_text": raw, "category": prepared["category"], "route": "model",
        "route_action": "MODEL", "documents": prepared["documents"], "fallback_used": fallback,
        "validator": validation.to_dict(), "first_token_seconds": first or 0.0,
        "total_seconds": prepared["prepare_seconds"] + generation_seconds,
        "tokens_per_second": float(last["tokens_per_sec"]) if last else 0.0,
        "generated_tokens": int(last["tokens"]) if last else 0,
        "eos_reached": bool(last["eos_reached"]) if last else False,
    }


def standard_records(rows: list[dict], checkpoint: str, tokenizer_path: str,
                     judge: CampusAIJudge, max_new_tokens: int) -> tuple[list[dict], float, dict]:
    model, tokenizer, _, payload = load_model(checkpoint, tokenizer_path, "cpu")
    pipeline = V08Pipeline(model, tokenizer, retrieval_method="tfidf", top_k=3)
    process = psutil.Process()
    peak = process.memory_info().rss / 1024**2
    records = []
    for row in rows:
        result = standard_answer(pipeline, row["question"], max_new_tokens)
        sources = [document.get("text", "") for document in result["documents"]]
        metadata = {
            "category": row["expected_category"], "predicted_category": result["category"],
            "route": "model", "action": "MODEL", "cards": [],
            "sources": [document.get("id") for document in result["documents"]],
        }
        judged = judge.evaluate(row["question"], result["text"], metadata, sources)
        records.append({
            "id": row["id"], "question": row["question"], "category": row["expected_category"],
            "predicted_category": result["category"], "route": "model", "answer": result["text"],
            "raw_answer": result["raw_text"], "judge": judged, "score": judged["overall_score"],
            "first_token_seconds": result["first_token_seconds"],
            "total_seconds": result["total_seconds"], "tokens_per_second": result["tokens_per_second"],
            "generated_tokens": result["generated_tokens"], "eos_reached": result["eos_reached"],
            "fallback_used": result["fallback_used"], "validator": result["validator"],
            "retrieved_ids": [document.get("id") for document in result["documents"]],
        })
        peak = max(peak, process.memory_info().rss / 1024**2)
    model_meta = {
        "model": model.config.model_name, "parameters": model.parameter_count(),
        "vocab": tokenizer.vocab_size, "context": model.config.context_length,
        "step": payload.get("step"), "checkpoint": checkpoint,
    }
    return records, peak, model_meta


def combine() -> dict:
    campus = read_json(OUT / "blind-200-campus-v23.json")
    standard = read_json(OUT / "blind-200-standard.json")
    campus_metrics = performance_summary(campus["items"], campus["summary"]["peak_rss_mb"], "campus")
    standard_metrics = performance_summary(standard["items"], standard["summary"]["peak_rss_mb"], "standard")
    campus["summary"] = campus_metrics
    standard["summary"] = standard_metrics
    write_json(OUT / "blind-200-campus-v23.json", campus)
    write_json(OUT / "blind-200-standard.json", standard)
    deltas = {axis: round(standard_metrics["axis_percent"][axis] - campus_metrics["axis_percent"][axis], 2)
              for axis in AXES}
    improved_axes = [axis for axis, delta in deltas.items() if delta >= 2.0]
    clearly_improved = (
        len(improved_axes) >= 4
        and standard_metrics["critical_errors"] <= campus_metrics["critical_errors"]
        and standard_metrics["hallucination_rate"] <= campus_metrics["hallucination_rate"]
        and standard_metrics["unsupported_claim_rate"] <= campus_metrics["unsupported_claim_rate"]
    )
    comparison = {
        "schema_version": "unipilot-standard-50m-short-comparison-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blind_sha256": campus["blind_sha256"],
        "questions": 200,
        "campus_v23": campus_metrics,
        "standard": standard_metrics,
        "axis_delta_standard_minus_campus": deltas,
        "improved_axes_gte_2_points": improved_axes,
        "clear_improvement_rule": "At least 4/6 axes improve >=2 points with no critical, hallucination, or unsupported-claim regression.",
        "standard_clearly_improved": clearly_improved,
        "continue_beyond_500": clearly_improved,
        "next_step": 1000 if clearly_improved else None,
        "external_ai_api": "OFF",
        "production_changed": False,
        "push_or_deploy_performed": False,
    }
    write_json(OUT / "blind-200-comparison.json", comparison)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("campus", "standard"))
    parser.add_argument("--checkpoint", default="checkpoints/standard-50m-short/checkpoint-step-500-inference.pt")
    parser.add_argument("--tokenizer", default="tokenizer/vocab-standard-v08-2048.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    if args.combine:
        print(json.dumps(combine(), ensure_ascii=False, indent=2))
        return 0
    if not args.system:
        parser.error("--system is required unless --combine is used")
    torch.set_num_threads(max(1, args.cpu_threads))
    blind_sha256 = verify_seal()
    rows = read_json(BLIND)["items"]
    judge = CampusAIJudge()
    tokenizer = BPETokenizer.load(args.tokenizer)
    if args.system == "campus":
        records, peak = campus_records(rows, tokenizer, judge)
        meta = {"model": "Campus v2.3", "parameters": 19_814_784, "step": None}
        output = OUT / "blind-200-campus-v23.json"
    else:
        records, peak, meta = standard_records(rows, args.checkpoint, args.tokenizer,
                                               judge, args.max_new_tokens)
        output = OUT / "blind-200-standard.json"
    report = {
        "schema_version": "unipilot-standard-50m-short-blind-result-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blind_sha256": blind_sha256,
        "holdout": True,
        "used_for_improvement": False,
        "system": args.system,
        "model": meta,
        "summary": performance_summary(records, peak, args.system),
        "items": records,
        "automatic_evaluation_limit": "Local deterministic rubric; no external model or API and no human score claimed.",
    }
    write_json(output, report)
    print(json.dumps({"output": str(output), "model": meta, "summary": report["summary"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
