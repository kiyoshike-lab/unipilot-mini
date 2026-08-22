from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import psutil
import torch

from inference.generate import load_model
from pipeline.v07 import V07Pipeline
from training.dataset_v03 import SYSTEM_TEXT


def main() -> None:
    torch.set_num_threads(1)
    checkpoint = "checkpoints/v07-grounded/unipilot-mini-v07-inference.pt"
    model, tokenizer, _, _ = load_model(checkpoint, "tokenizer/vocab-v02-512.json", "cpu")
    pipeline = V07Pipeline(model, tokenizer)
    prompts = json.loads(Path("evaluation/fixed_prompts_v07.json").read_text(encoding="utf-8"))
    diverse, seen = [], set()
    for item in prompts:
        if item["category"] not in seen:
            diverse.append(item)
            seen.add(item["category"])
    process = psutil.Process()
    candidate_results = []
    for candidates in (1, 2, 3):
        rows = []
        peak = process.memory_info().rss / 1024**2
        started = time.perf_counter()
        for item in diverse:
            result = pipeline.answer(item["prompt"], max_new_tokens=80, candidates=candidates, force_model=True)
            peak = max(peak, process.memory_info().rss / 1024**2)
            rows.append(result)
        elapsed = time.perf_counter() - started
        candidate_results.append({
            "candidates": candidates, "questions": len(rows), "total_seconds": elapsed,
            "mean_seconds": elapsed / len(rows), "mean_raw_tokens_per_second": statistics.fmean(
                row["generation_metrics"].get("tokens_per_sec", 0) for row in rows),
            "grounded_selection_rate": sum(row["grounded_selected"] for row in rows) / len(rows),
            "fallback_rate": sum(row["fallback_used"] for row in rows) / len(rows), "peak_rss_mb": peak,
        })
    lengths = []
    for item in prompts:
        category = pipeline.classifier.predict(item["prompt"])[0]
        document = pipeline.retriever.retrieve(item["prompt"], category, 1)[0]
        text = document.get("answer") or document["text"]
        full = f"<BOS><SYSTEM>\n{SYSTEM_TEXT}\n<CONTEXT>\n[{document['title']}]\n{text}\n<USER>\n{item['prompt']}\n<ASSISTANT>\n"
        lengths.append(len(tokenizer.encode(full)))
    context_results = []
    for context in (256, 512, 1024):
        added = (context - 256) * model.config.embedding_dim
        context_results.append({"context": context, "full_prompt_fit_rate": sum(length <= context for length in lengths) / len(lengths),
                                "mean_full_prompt_tokens": statistics.fmean(lengths),
                                "p95_full_prompt_tokens": sorted(lengths)[int(.95 * (len(lengths) - 1))],
                                "position_parameter_delta_mb": added * 4 / 1024**2,
                                "attention_memory_relative_to_256": (context / 256) ** 2})
    report = {
        "checkpoint": checkpoint, "parameters": model.parameter_count(), "candidate_generation": candidate_results,
        "context_candidates": context_results, "selected_candidates": 1,
        "selected_context": 256,
        "decision": "Multiple model candidates have identical grounded-selection quality and linear latency cost. FAQ fast path avoids generation; non-FAQ path keeps one candidate.",
    }
    Path("evaluation/pipeline-benchmark-v07.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
