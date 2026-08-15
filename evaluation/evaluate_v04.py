from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

import torch
from torch.utils.data import DataLoader

from evaluation.evaluate_v02 import balanced_prompts
from evaluation.metrics_v02 import japanese_character_ratio, repetition_rate
from evaluation.metrics_v03 import broken_text_metrics, semantic_score
from evaluation.metrics_v04 import aggregate, broken_generation_metrics, ngram_repetition
from inference.generate import generate_text, load_model
from training.dataset_v03 import CurriculumDataset, SYSTEM_TEXT, dynamic_collate


@torch.inference_mode()
def clean_loss(model, tokenizer, device, batches=30):
    dataset = CurriculumDataset("data/v04/stage_c/test.jsonl", tokenizer, model.config.context_length, True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dynamic_collate); values = []
    for index, (inputs, targets, _, _) in enumerate(loader):
        if index >= batches: break
        _, loss = model(inputs.to(device), targets.to(device)); values.append(loss.item())
    return statistics.mean(values)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/vocab-v02-512.json"); parser.add_argument("--prompt-count", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=64); parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-k", type=int, default=40); parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer); loss = clean_loss(model, tokenizer, device)
    prompts = balanced_prompts(json.loads(Path("evaluation/fixed_prompts_v03.json").read_text(encoding="utf-8")), args.prompt_count)
    rows = []
    for item in prompts:
        formatted = f"<BOS><SYSTEM>\n{SYSTEM_TEXT}\n<USER>\n{item['prompt']}\n<ASSISTANT>\n"
        answer, speed = generate_text(model, tokenizer, formatted, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty)
        rows.append({**item, "answer": answer, **speed, "repetition_rate": repetition_rate(answer),
                     "japanese_character_ratio": japanese_character_ratio(answer), "broken": broken_text_metrics(answer),
                     "ngram_repetition": {str(n): ngram_repetition(answer, n) for n in [2, 3, 4]},
                     "broken_generation": broken_generation_metrics(answer), **semantic_score(answer, item)})
    manifest = payload.get("v04_manifest", {})
    result = {"model": model.config.model_name, "checkpoint": args.checkpoint, "step": payload.get("step"),
              "experiment_id": manifest.get("experiment_id"), "eos_weight": manifest.get("eos_weight"),
              "dataset_version": "unipilot-clean-conversation-v04", "evaluation_version": "unipilot-eval-v03-300",
              "validation_loss": loss, "perplexity": math.exp(min(loss, 20)),
              "generation_settings": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                                      "repetition_penalty": args.repetition_penalty, "max_new_tokens": args.max_new_tokens, "seed": args.seed},
              "metrics": aggregate(rows, args.max_new_tokens), "generations": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    base_human = json.loads(Path("evaluation/human-eval-v03.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in rows}
    human = [{**item, "model_answer": by_id.get(item["id"], {}).get("answer", item.get("model_answer", ""))} for item in base_human]
    output.with_name(output.stem + "-human.json").write_text(json.dumps(human, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "generations"}, ensure_ascii=True, indent=2))


if __name__ == "__main__": main()
