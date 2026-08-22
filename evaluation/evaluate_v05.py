from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re

import torch
from torch.utils.data import DataLoader

from inference.generate import generate_text
from inference.generate import load_model
from training.dataset_v03 import CurriculumDataset, dynamic_collate


SYSTEM = "あなたは大学生活を支援する完全ローカルのUniPilot Miniです。情報がない場合は推測せず、確認方法を案内します。"


def japanese_ratio(text: str) -> float:
    meaningful = [char for char in text if not char.isspace()]
    if not meaningful: return 0.0
    return sum("\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff" for char in meaningful) / len(meaningful)


def repetition_rate(text: str) -> float:
    chunks = [text[index:index + 3] for index in range(max(0, len(text) - 2))]
    return 0.0 if not chunks else 1 - len(set(chunks)) / len(chunks)


@torch.inference_mode()
def validation_loss(model, tokenizer, device) -> float:
    dataset = CurriculumDataset("data/v05/conversation/validation.jsonl", tokenizer, model.config.context_length, True)
    loader = DataLoader(dataset, batch_size=1, collate_fn=dynamic_collate); values = []
    for inputs, targets, _, _ in loader:
        _, loss = model(inputs.to(device), targets.to(device)); values.append(loss.item())
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/vocab-v02-512.json")
    parser.add_argument("--prompts", default="evaluation/fixed_prompts_v05.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(); random.seed(5052026); torch.manual_seed(5052026)
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer, "cpu")
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8")); rows = []
    for item in prompts:
        formatted = f"<BOS><SYSTEM>\n{SYSTEM}\n<USER>\n{item['prompt']}\n<ASSISTANT>\n"
        answer, timing = generate_text(model, tokenizer, formatted, args.max_new_tokens, .7, 40, .9, 1.1)
        expected = item.get("expected_keywords", []); forbidden = item.get("forbidden_keywords", [])
        rows.append({**item, "answer": answer, **timing,
                     "keyword_hit": not expected or any(word in answer for word in expected),
                     "forbidden_hit": any(word in answer for word in forbidden),
                     "complete": bool(answer) and answer.rstrip().endswith(("。", "！", "？", "ます", "です")),
                     "japanese_ratio": japanese_ratio(answer), "repetition_rate": repetition_rate(answer),
                     "broken": "�" in answer or bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", answer))})
    count = len(rows); val = validation_loss(model, tokenizer, device)
    metrics = {"questions": count, "nonempty_rate": sum(bool(row["answer"].strip()) for row in rows) / count,
               "eos_rate": sum(row["eos_reached"] for row in rows) / count,
               "completion_rate": sum(row["complete"] for row in rows) / count,
               "keyword_rate": sum(row["keyword_hit"] for row in rows) / count,
               "forbidden_rate": sum(row["forbidden_hit"] for row in rows) / count,
               "broken_rate": sum(row["broken"] for row in rows) / count,
               "mean_japanese_ratio": sum(row["japanese_ratio"] for row in rows) / count,
               "mean_repetition_rate": sum(row["repetition_rate"] for row in rows) / count,
               "mean_tokens_per_second": sum(row["tokens_per_sec"] for row in rows) / count}
    result = {"checkpoint": args.checkpoint, "model": model.config.model_name, "parameters": model.parameter_count(),
              "step": payload.get("step"), "validation_loss_v05": val, "perplexity_v05": math.exp(min(val, 20)),
              "metrics": metrics, "generations": rows}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "generations"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
