from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from evaluation.perplexity import perplexity
from evaluation.test_prompts import PROMPTS
from inference.generate import generate_text, load_model
from training.dataset import LanguageModelDataset, load_documents


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/vocab.json")
    parser.add_argument("--dataset", default="data/conversations")
    parser.add_argument("--output", default="evaluation/results.json")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer)
    loader = DataLoader(LanguageModelDataset(load_documents(args.dataset), tokenizer, model.config.context_length), batch_size=4)
    losses = []
    for inputs, targets in loader:
        _, loss = model(inputs.to(device), targets.to(device)); losses.append(loss.item())
    average = sum(losses) / len(losses)
    generations = []
    for prompt in PROMPTS:
        formatted = f"<BOS><USER>\n{prompt}\n<ASSISTANT>\n"
        answer, metrics = generate_text(model, tokenizer, formatted, args.max_new_tokens)
        generations.append({"prompt": prompt, "answer": answer, **metrics})
    result = {"checkpoint": args.checkpoint, "step": payload.get("step"), "validation_loss": average,
              "perplexity": perplexity(average), "generations": generations}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # ASCII escaping keeps arbitrary early-training Unicode printable in Windows consoles.
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__": main()
