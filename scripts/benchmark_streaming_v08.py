from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import psutil

from inference.generate import load_model
from pipeline.v08 import V08Pipeline


PROMPTS = (
    "GPAとは何ですか？",
    "教授へ欠席メールを作ってください。",
    "明日試験なのに課題も終わっていません。優先順位を教えてください。",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/standard-v08-scratch/unipilot-standard-v08-a100-inference.pt")
    parser.add_argument("--tokenizer", default="tokenizer/vocab-standard-v08-1024.json")
    parser.add_argument("--output", default="evaluation/streaming-benchmark-standard-v08.json")
    args = parser.parse_args()
    model, tokenizer, _, _ = load_model(args.checkpoint, args.tokenizer, "cpu")
    pipeline = V08Pipeline(model, tokenizer, retrieval_method="tfidf", top_k=3)
    rows = []
    process = psutil.Process()
    peak = process.memory_info().rss / 1024**2
    for prompt in PROMPTS:
        started = time.perf_counter()
        first = None
        snapshots = []
        for snapshot in pipeline.iter_answer(prompt, 64, temperature=0.0, top_k=40, top_p=0.9,
                                             repetition_penalty=1.1, response_mode="auto"):
            if first is None:
                first = time.perf_counter() - started
            snapshots.append(snapshot)
            peak = max(peak, process.memory_info().rss / 1024**2)
        total = time.perf_counter() - started
        rows.append({
            "prompt": prompt, "first_event_seconds": first, "total_seconds": total,
            "events": len(snapshots), "last_phase": snapshots[-1].get("phase") if snapshots else None,
            "fallback_used": snapshots[-1].get("fallback_used", False) if snapshots else False,
            "final_text": snapshots[-1].get("text", "") if snapshots else "",
        })
    report = {
        "checkpoint": args.checkpoint, "questions": len(rows),
        "mean_first_event_seconds": statistics.fmean(row["first_event_seconds"] for row in rows),
        "max_first_event_seconds": max(row["first_event_seconds"] for row in rows),
        "mean_total_seconds": statistics.fmean(row["total_seconds"] for row in rows),
        "peak_rss_mb": peak, "incremental_events_before_final_validation": True,
        "rows": rows, "external_ai_api": "OFF",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
