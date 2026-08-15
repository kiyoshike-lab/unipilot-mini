from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from evaluation.evaluate_v02 import balanced_prompts
from evaluation.metrics_v02 import japanese_character_ratio, repetition_rate
from evaluation.metrics_v03 import aggregate, broken_text_metrics, semantic_score
from inference.generate import generate_text, load_model
from training.dataset_v03 import SYSTEM_TEXT


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/vocab-v02-512.json"); parser.add_argument("--prompt-count", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64); parser.add_argument("--output", default="evaluation/generation-search-v03.json")
    args = parser.parse_args(); model, tokenizer, _, _ = load_model(args.checkpoint, args.tokenizer)
    prompts = balanced_prompts(json.loads(Path("evaluation/fixed_prompts_v03.json").read_text(encoding="utf-8")), args.prompt_count)
    configs = [
        {"temperature": .5, "top_k": 40, "top_p": .9, "repetition_penalty": 1.1},
        {"temperature": .7, "top_k": 40, "top_p": .9, "repetition_penalty": 1.1},
        {"temperature": .9, "top_k": 40, "top_p": .9, "repetition_penalty": 1.1},
        {"temperature": .7, "top_k": 20, "top_p": .9, "repetition_penalty": 1.1},
        {"temperature": .7, "top_k": 60, "top_p": .9, "repetition_penalty": 1.1},
        {"temperature": .7, "top_k": 40, "top_p": .85, "repetition_penalty": 1.1},
        {"temperature": .7, "top_k": 40, "top_p": .95, "repetition_penalty": 1.1},
        {"temperature": .7, "top_k": 40, "top_p": .9, "repetition_penalty": 1.0},
        {"temperature": .7, "top_k": 40, "top_p": .9, "repetition_penalty": 1.05},
        {"temperature": .7, "top_k": 40, "top_p": .9, "repetition_penalty": 1.15},
    ]
    results = []
    for config_index, config in enumerate(configs):
        torch.manual_seed(42); rows = []
        for item in prompts:
            prompt = f"<BOS><SYSTEM>\n{SYSTEM_TEXT}\n<USER>\n{item['prompt']}\n<ASSISTANT>\n"
            answer, speed = generate_text(model, tokenizer, prompt, args.max_new_tokens, **config)
            rows.append({**item, "answer": answer, **speed, "repetition_rate": repetition_rate(answer),
                         "japanese_character_ratio": japanese_character_ratio(answer), "broken": broken_text_metrics(answer),
                         **semantic_score(answer, item)})
        results.append({"config": config, "metrics": aggregate(rows, args.max_new_tokens)})
    results.sort(key=lambda row: (row["metrics"]["relevance_score"], row["metrics"]["meaningful_response_rate"],
                                  row["metrics"]["category_accuracy"], -row["metrics"]["repetition_rate"]), reverse=True)
    payload = {"checkpoint": args.checkpoint, "prompt_count": len(prompts), "recommended": results[0], "all_results": results}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__": main()
