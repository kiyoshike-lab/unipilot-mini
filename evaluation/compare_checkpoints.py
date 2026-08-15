from __future__ import annotations

import argparse
import json
from pathlib import Path


def comparison(results: list[dict]) -> str:
    lines = ["# UniPilot Mini v0.2 checkpoint comparison", "",
             "| Model | Step | Val Loss | PPL | Repetition | JP Ratio | Non-empty | Keyword relevance |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        metrics = result["metrics"]
        lines.append(f"| {result['model']} | {result.get('step', 0)} | {result['validation_loss']:.4f} | {result['perplexity']:.2f} | {metrics['repetition_rate']:.3f} | {metrics['japanese_character_ratio']:.3f} | {metrics['response_not_empty']:.3f} | {metrics['keyword_relevance']:.3f} |")
    lines.extend(["", "## Same-prompt generations", ""])
    prompts = {}
    for result in results:
        for row in result.get("generations", []): prompts.setdefault(row["id"], {"prompt": row["prompt"]})[result["model"]] = row["answer"]
    lines.append("```json")
    lines.append(json.dumps(list(prompts.values()), ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", default="evaluation/comparison-v02.md")
    args = parser.parse_args()
    loaded = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.results]
    Path(args.output).write_text(comparison(loaded), encoding="utf-8")
    print(f"wrote {args.output} for {len(loaded)} checkpoints")


if __name__ == "__main__": main()
