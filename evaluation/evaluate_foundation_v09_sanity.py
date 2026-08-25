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
from inference.generate import load_model
from pipeline.campus_v23 import UniPilotCampusV23
from quality.campus_ai_judge import CampusAIJudge
from tokenizer.tokenizer import BPETokenizer
from training.train_foundation_v09 import SYSTEM, natural_text, safe_generate


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "data/foundation_v09/evaluation/validation-200.json"
FINAL_BLIND = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
MANIFEST = ROOT / "data/foundation_v09/manifest.json"
OUT = ROOT / "evaluation/foundation-v09-sanity"
AXES = ("correctness", "relevance", "completeness", "specificity", "naturalness", "actionable")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_final_blind_seal() -> str:
    expected = read(MANIFEST)["evaluation"]["final_blind_sha256"]
    actual = hashlib.sha256(FINAL_BLIND.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError("final Blind 1000 seal changed")
    return actual


def generation_failure_penalty(judged: dict, answer: str) -> dict:
    natural, repetition = natural_text(answer)
    judged = {**judged, "scores_0_to_5": dict(judged["scores_0_to_5"]),
              "issues": list(judged["issues"]), "checks": dict(judged["checks"])}
    judged["checks"]["v09_generation_natural"] = natural
    judged["checks"]["v09_repetition_rate"] = repetition
    if natural:
        return judged
    caps = {"correctness": 1.0, "relevance": 1.0, "completeness": .5,
            "specificity": .5, "naturalness": 0.0, "actionable": .5,
            "grounding": 1.0, "conciseness": 1.0, "helpfulness": .5}
    for axis, cap in caps.items():
        judged["scores_0_to_5"][axis] = min(judged["scores_0_to_5"][axis], cap)
    judged["issues"].append("MODEL_GENERATION_FAILURE")
    judged["primary_issue"] = "MODEL_GENERATION_FAILURE"
    judged["quality_label"] = "bad"
    judged["overall_score"] = round(mean(judged["scores_0_to_5"].values()) / 5 * 100, 2)
    return judged


def summarize(records: list[dict], peak: float, semantics: str) -> dict:
    base = aggregate(records)
    return {
        **base,
        "axis_percent": {axis: base["axis_percent"][axis] for axis in AXES},
        "average_first_token_seconds": round(mean(row["first_token_seconds"] for row in records), 6),
        "average_total_seconds": round(mean(row["total_seconds"] for row in records), 6),
        "average_tokens_per_second": round(mean(row["tokens_per_second"] for row in records), 3),
        "natural_generation_rate": round(mean(row["natural_generation"] for row in records), 4),
        "peak_rss_mb": round(peak, 3), "latency_semantics": semantics,
    }


def campus(rows: list[dict], tokenizer, judge: CampusAIJudge) -> tuple[list[dict], float]:
    pipeline = UniPilotCampusV23()
    process = psutil.Process()
    peak = process.memory_info().rss / 1024**2
    records = []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        result = pipeline.answer(row["question"], response_mode="auto", session_id=f"v09-campus-{index:03d}")
        elapsed = time.perf_counter() - started
        judged = judge_result(judge, row["question"], row["expected_category"], result)
        tokens = len(tokenizer.encode(result["text"]))
        records.append({
            "id": row["id"], "question": row["question"], "category": row["expected_category"],
            "predicted_category": result.get("category"), "route": result.get("route"),
            "answer": result["text"], "judge": judged, "score": judged["overall_score"],
            "natural_generation": True, "first_token_seconds": elapsed, "total_seconds": elapsed,
            "tokens_per_second": tokens / max(elapsed, 1e-9), "sources": result.get("sources", []),
        })
        peak = max(peak, process.memory_info().rss / 1024**2)
    return records, peak


def standard(rows: list[dict], checkpoint: str, tokenizer_path: str,
             judge: CampusAIJudge) -> tuple[list[dict], float, dict]:
    model, tokenizer, _, payload = load_model(checkpoint, tokenizer_path, "cpu")
    process = psutil.Process()
    peak = process.memory_info().rss / 1024**2
    records = []
    for row in rows:
        prompt = f"<BOS><SYSTEM>\n{SYSTEM}\n<USER>\n{row['question']}\n<ASSISTANT>\n"
        answer, metrics = safe_generate(model, tokenizer, prompt, 64)
        natural, repetition = natural_text(answer)
        metadata = {"category": row["expected_category"], "predicted_category": row["expected_category"],
                    "route": "model", "action": "MODEL", "cards": [], "sources": []}
        judged = generation_failure_penalty(judge.evaluate(row["question"], answer, metadata, []), answer)
        records.append({
            "id": row["id"], "question": row["question"], "category": row["expected_category"],
            "predicted_category": row["expected_category"], "route": "model", "answer": answer,
            "judge": judged, "score": judged["overall_score"], "natural_generation": natural,
            "repetition_rate": repetition, "first_token_seconds": metrics["first_token_seconds"],
            "total_seconds": metrics["seconds"], "tokens_per_second": metrics["tokens_per_second"],
            "generated_tokens": metrics["tokens"], "eos_reached": metrics["eos_reached"],
        })
        peak = max(peak, process.memory_info().rss / 1024**2)
    return records, peak, {
        "model": model.config.model_name, "parameters": model.parameter_count(),
        "vocab": tokenizer.vocab_size, "context": model.config.context_length,
        "step": payload.get("step"), "checkpoint": checkpoint,
    }


def combine() -> dict:
    campus_report = read(OUT / "validation-200-campus-v23.json")
    standard_report = read(OUT / "validation-200-standard.json")
    left, right = campus_report["summary"], standard_report["summary"]
    delta = {axis: round(right["axis_percent"][axis] - left["axis_percent"][axis], 2) for axis in AXES}
    improved = [axis for axis, value in delta.items() if value >= 2]
    clear = (len(improved) >= 4 and right["critical_errors"] <= left["critical_errors"]
             and right["hallucination_rate"] <= left["hallucination_rate"]
             and right["unsupported_claim_rate"] <= left["unsupported_claim_rate"])
    payload = {
        "schema_version": "foundation-v09-validation-comparison-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(), "questions": 200,
        "evaluation_split": "validation-not-final-blind", "final_blind_opened": False,
        "final_blind_sha256": verify_final_blind_seal(), "campus_v23": left, "standard": right,
        "axis_delta_standard_minus_campus": delta, "improved_axes_gte_2_points": improved,
        "standard_clearly_improved": clear, "standard_continue": clear,
        "external_ai_api": "OFF", "production_changed": False,
    }
    write(OUT / "validation-200-comparison.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("campus", "standard"))
    parser.add_argument("--checkpoint", default="checkpoints/foundation-v09-sanity/checkpoint-step-100-inference.pt")
    parser.add_argument("--tokenizer", default="tokenizer/vocab-standard-v09-4096.json")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    if args.combine:
        print(json.dumps(combine(), ensure_ascii=False, indent=2))
        return 0
    if not args.system:
        parser.error("--system is required unless --combine is used")
    torch.set_num_threads(max(1, args.cpu_threads))
    verify_final_blind_seal()
    rows = read(VALIDATION)["items"]
    judge = CampusAIJudge()
    tokenizer = BPETokenizer.load(args.tokenizer)
    if args.system == "campus":
        records, peak = campus(rows, tokenizer, judge)
        meta = {"model": "Campus v2.3", "parameters": 19_814_784}
        semantics = "deterministic completed-response latency and effective output throughput"
        output = OUT / "validation-200-campus-v23.json"
    else:
        records, peak, meta = standard(rows, args.checkpoint, args.tokenizer, judge)
        semantics = "autoregressive first token and generated-token throughput"
        output = OUT / "validation-200-standard.json"
    report = {
        "schema_version": "foundation-v09-validation-result-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(), "evaluation_split": "validation",
        "final_blind_opened": False, "model": meta, "summary": summarize(records, peak, semantics),
        "items": records,
        "automatic_evaluation_limit": "Local deterministic rubric with explicit generation-failure cap; no human score claimed.",
    }
    write(output, report)
    print(json.dumps({"output": str(output), "model": meta, "summary": report["summary"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
