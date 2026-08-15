from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

from evaluation.metrics_v02 import aggregate_generation_metrics, japanese_character_ratio, keyword_relevance, repetition_rate
from inference.generate import generate_text, load_model
from training.dataset import V02LanguageModelDataset


def balanced_prompts(prompts: list[dict], count: int) -> list[dict]:
    groups = defaultdict(list)
    for prompt in prompts: groups[prompt["category"]].append(prompt)
    selected = []; index = 0
    while len(selected) < min(count, len(prompts)):
        added = False
        for category in sorted(groups):
            if index < len(groups[category]) and len(selected) < count:
                selected.append(groups[category][index]); added = True
        if not added: break
        index += 1
    return selected


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/vocab-v02-512.json")
    parser.add_argument("--dataset", default="data/test/v02.jsonl")
    parser.add_argument("--prompts", default="evaluation/fixed_prompts_v02.json")
    parser.add_argument("--prompt-count", type=int, default=20)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer)
    dataset = V02LanguageModelDataset(args.dataset, tokenizer, model.config.context_length, True, max_records=max(100, args.validation_batches))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    losses = []
    for index, (inputs, targets, _) in enumerate(loader):
        if index >= args.validation_batches: break
        _, loss = model(inputs.to(device), targets.to(device)); losses.append(loss.item())
    prompts = balanced_prompts(json.loads(Path(args.prompts).read_text(encoding="utf-8")), args.prompt_count)
    generations = []
    for item in prompts:
        formatted = f"<BOS><USER>\n{item['prompt']}\n<ASSISTANT>\n"
        answer, speed = generate_text(model, tokenizer, formatted, args.max_new_tokens, args.temperature, args.top_k, args.top_p, 1.1)
        generations.append({**item, "answer": answer, **speed, "repetition_rate": repetition_rate(answer),
                            "japanese_character_ratio": japanese_character_ratio(answer), "keyword_relevance": keyword_relevance(answer, item["keywords"])})
    loss = statistics.mean(losses)
    result = {"model": model.config.model_name, "checkpoint": args.checkpoint, "step": payload.get("step"),
              "validation_loss": loss, "perplexity": math.exp(min(loss, 20)),
              "generation_settings": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                                      "repetition_penalty": 1.1, "max_new_tokens": args.max_new_tokens},
              "metrics": aggregate_generation_metrics(generations), "generations": generations}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "generations"}, ensure_ascii=True, indent=2))


if __name__ == "__main__": main()
