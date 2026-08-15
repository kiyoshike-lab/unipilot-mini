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
from evaluation.metrics_v03 import aggregate, broken_text_metrics, semantic_score
from inference.generate import generate_text, load_model
from training.dataset_v03 import CurriculumDataset, SYSTEM_TEXT, dynamic_collate


@torch.inference_mode()
def stage_loss(model, tokenizer, path, device, assistant_only, batches):
    dataset = CurriculumDataset(path, tokenizer, model.config.context_length, assistant_only, max_records=max(100, batches))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dynamic_collate); losses = []
    for index, (inputs, targets, _, _) in enumerate(loader):
        if index >= batches: break
        _, loss = model(inputs.to(device), targets.to(device)); losses.append(loss.item())
    return statistics.mean(losses)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--tokenizer", default="tokenizer/vocab-v02-512.json")
    parser.add_argument("--prompts", default="evaluation/fixed_prompts_v03.json"); parser.add_argument("--prompt-count", type=int, default=300)
    parser.add_argument("--loss-batches", type=int, default=30); parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7); parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9); parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output", required=True)
    args = parser.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer)
    losses = {
        "general": stage_loss(model, tokenizer, "data/v03/stage_a/test.jsonl", device, False, args.loss_batches),
        "university": stage_loss(model, tokenizer, "data/v03/stage_b/test.jsonl", device, False, args.loss_batches),
        "conversation": stage_loss(model, tokenizer, "data/v03/stage_c/test.jsonl", device, True, args.loss_batches),
    }
    prompts = balanced_prompts(json.loads(Path(args.prompts).read_text(encoding="utf-8")), args.prompt_count)
    rows = []
    for item in prompts:
        prompt = f"<BOS><SYSTEM>\n{SYSTEM_TEXT}\n<USER>\n{item['prompt']}\n<ASSISTANT>\n"
        answer, speed = generate_text(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty)
        semantic = semantic_score(answer, item)
        rows.append({**item, "answer": answer, **speed, "repetition_rate": repetition_rate(answer),
                     "japanese_character_ratio": japanese_character_ratio(answer), "broken": broken_text_metrics(answer), **semantic})
    conversation_loss = losses["conversation"]
    result = {"model": model.config.model_name, "checkpoint": args.checkpoint, "step": payload.get("step"),
              "stage": payload.get("v03_manifest", {}).get("stage", "legacy"), "loss_by_stage": losses,
              "validation_loss": conversation_loss, "perplexity": math.exp(min(conversation_loss, 20)),
              "generation_settings": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                                      "repetition_penalty": args.repetition_penalty, "max_new_tokens": args.max_new_tokens, "seed": args.seed},
              "metrics": aggregate(rows, args.max_new_tokens), "generations": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    human_path = output.with_name(output.stem + "-human.json")
    human_path.write_text(json.dumps([{"id": row["id"], "prompt": row["prompt"], "category": row["category"],
                                      "model_answer": row["answer"], "score": None, "notes": ""} for row in rows[:50]], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "generations"}, ensure_ascii=True, indent=2))


if __name__ == "__main__": main()
