from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.investigate_foundation_v14 import frequency_baselines
from evaluation.measure_foundation_v16 import frequency_metrics, validation_metrics
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v15 import DiagnosticConfig, DiagnosticTransformer
from training.optimizer import create_optimizer
from training.train_foundation_v15_controlled import generation_probe, macro_batch


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


@torch.inference_mode()
def sentence_boundary_metrics(
    model: DiagnosticTransformer,
    tokenizer: FoundationTokenizer,
    validation: np.memmap,
    probe_tokens: int = 8192,
) -> dict:
    patterns = {
        text: tokenizer.encode(text) if text != "<EOS>" else [tokenizer.eos_id]
        for text in ("。", "！", "？", "<EOS>")
    }
    totals = {text: 0 for text in patterns}
    top1_hits = {text: 0 for text in patterns}
    top5_hits = {text: 0 for text in patterns}
    probability_sums = {text: 0.0 for text in patterns}
    context = model.config.context_length
    model.eval()
    for start in range(0, probe_tokens, context):
        values = np.asarray(validation[start:start + context + 1], dtype=np.int64).copy()
        x = torch.from_numpy(values[:-1]).unsqueeze(0)
        targets = torch.from_numpy(values[1:])
        logits, _ = model(x)
        logits = logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        top5 = logits.topk(5, dim=-1).indices
        for text, token_ids in patterns.items():
            width = len(token_ids)
            for offset in range(len(targets) - width + 1):
                expected = torch.tensor(token_ids, dtype=targets.dtype)
                if not torch.equal(targets[offset:offset + width], expected):
                    continue
                predicted = top5[offset:offset + width]
                totals[text] += 1
                top1_hits[text] += int(all(
                    int(predicted[index, 0]) == token_id
                    for index, token_id in enumerate(token_ids)
                ))
                top5_hits[text] += int(all(
                    bool((predicted[index] == token_id).any())
                    for index, token_id in enumerate(token_ids)
                ))
                log_probability = sum(
                    math.log(max(float(probabilities[offset + index, token_id]), 1e-30))
                    for index, token_id in enumerate(token_ids)
                )
                probability_sums[text] += math.exp(log_probability / width)
    return {
        text: {
            "token_ids": patterns[text],
            "targets": totals[text],
            "top_1_accuracy": top1_hits[text] / totals[text] if totals[text] else None,
            "top_5_accuracy": top5_hits[text] / totals[text] if totals[text] else None,
            "mean_probability_when_target": (
                probability_sums[text] / totals[text] if totals[text] else None
            ),
        }
        for text in patterns
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v16.json")
    parser.add_argument("--variant", choices=["current_unscaled", "sqrt_scaled_a"], required=True)
    parser.add_argument("--output-dir", default="checkpoints/foundation-v16-short-japanese")
    args = parser.parse_args()
    settings = load_json(args.config)
    short = settings["short_japanese"]
    manifest = load_json(settings["diagnostic_corpus_manifest"])
    variant = next(row for row in settings["variants"] if row["name"] == args.variant)
    train_meta = manifest["artifacts"]["packed"]["train"]
    validation_meta = manifest["artifacts"]["packed"]["validation"]
    train = np.memmap(ROOT / train_meta["path"], dtype=np.uint16, mode="r")
    validation = np.memmap(ROOT / validation_meta["path"], dtype=np.uint16, mode="r")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.variant}.json"
    checkpoint_path = output_dir / f"{args.variant}.pt"
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite short Japanese run: {args.variant}")

    seed = int(short["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(settings["cpu_threads"]))
    architecture = dict(settings["architecture"])
    architecture.update({
        "context_length": int(short["context_length"]),
        "scale_token_embedding": variant["scale_token_embedding"],
    })
    model = DiagnosticTransformer(DiagnosticConfig(
        model_name=f"Foundation v1.6 short Japanese {args.variant}",
        vocab_size=tokenizer.vocab_size,
        **architecture,
    ))
    optimizer = create_optimizer(model, 1e-4, .1)
    updates = int(short["token_budget"]) // int(short["effective_batch_tokens"])
    macro_count = (len(train) - 1) // 512
    permutation = torch.randperm(macro_count, generator=torch.Generator().manual_seed(seed))
    history = [{
        "update": 0,
        "tokens_processed": 0,
        "validation": validation_metrics(model, validation, 8192),
    }]
    recent = []
    started = time.perf_counter()
    for update in range(1, updates + 1):
        x, y = macro_batch(train, int(permutation[update - 1]), int(short["context_length"]))
        lr = 1e-4 * min(1.0, update / 20)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite short Japanese loss")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        recent.append(float(loss.item()))
        if update in {10, 64, 100, 128}:
            history.append({
                "update": update,
                "tokens_processed": update * 512,
                "train_loss": sum(recent) / len(recent),
                "gradient_norm": gradient_norm,
                "validation": validation_metrics(model, validation, 8192),
            })
            recent.clear()
    baselines = frequency_baselines(
        train, validation, tokenizer.vocab_size, 8192, alpha=.1
    )
    final_frequency = frequency_metrics(
        model, tokenizer, train, validation, probe_tokens=8192
    )
    boundaries = sentence_boundary_metrics(model, tokenizer, validation, 8192)
    generation = generation_probe(model, tokenizer, validation, seed)
    payload = {
        "checkpoint_format": "foundation-v16-short-japanese-v1",
        "model_state": model.state_dict(),
        "config": model.config.to_dict(),
        "variant": variant,
        "tokens_processed": int(short["token_budget"]),
        "diagnostic_only": True,
    }
    torch.save(payload, checkpoint_path)
    restored_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = DiagnosticTransformer(DiagnosticConfig(**restored_payload["config"]))
    restored.load_state_dict(restored_payload["model_state"], strict=True)
    strict_reload = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), restored.state_dict().values())
    )
    report = {
        "schema_version": "foundation-v16-short-japanese-diagnostic-v1",
        "variant": variant,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "corpus": {
            "manifest": settings["diagnostic_corpus_manifest"],
            "segments": manifest["selection"]["segments"],
            "added_to_foundation_corpus": False,
            "train_sha256": train_meta["sha256"],
            "validation_sha256": validation_meta["sha256"],
        },
        "training": {
            "seed": seed,
            "tokens_processed": int(short["token_budget"]),
            "effective_batch_tokens": short["effective_batch_tokens"],
            "history": history,
            "wall_seconds": time.perf_counter() - started,
        },
        "final": history[-1]["validation"],
        "frequency_buckets": final_frequency["buckets"],
        "selected_token_frequency": final_frequency["tokens"],
        "sentence_boundaries": boundaries,
        "generation": generation,
        "baselines": {
            "unigram": baselines["unigram"],
            "bigram": baselines["bigram"],
            "method": baselines["method"],
        },
        "checkpoint": {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "strict_reload": strict_reload,
        },
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "variant": args.variant,
        "final": report["final"],
        "period": report["selected_token_frequency"]["。"],
        "sentence_boundaries": boundaries,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
