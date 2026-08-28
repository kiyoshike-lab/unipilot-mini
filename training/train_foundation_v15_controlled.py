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

from evaluation.evaluate_foundation_v13 import PRIMARY_MODES
from evaluation.investigate_foundation_v14 import aggregate_generation, generate_ids
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v15 import DiagnosticConfig, DiagnosticTransformer
from training.optimizer import create_optimizer


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def validate(model: DiagnosticTransformer, tokens: np.memmap, count: int = 8192) -> dict:
    model.eval()
    context = model.config.context_length
    count = min(count, len(tokens) - 1)
    count -= count % context
    loss_sum = 0.0
    correct = {1: 0, 5: 0, 10: 0}
    for start in range(0, count, context):
        values = np.asarray(tokens[start:start + context + 1], dtype=np.int64).copy()
        x = torch.from_numpy(values[:-1]).unsqueeze(0)
        y = torch.from_numpy(values[1:]).unsqueeze(0)
        logits, loss = model(x, y)
        loss_sum += float(loss) * context
        top = logits.topk(10, dim=-1).indices
        for k in correct:
            correct[k] += int((top[..., :k] == y[..., None]).any(-1).sum())
    loss = loss_sum / count
    return {
        "tokens": count,
        "loss": loss,
        "perplexity": math.exp(min(loss, 50)),
        "top_1_accuracy": correct[1] / count,
        "top_5_accuracy": correct[5] / count,
        "top_10_accuracy": correct[10] / count,
    }


def macro_batch(tokens: np.memmap, index: int, context: int = 128):
    start = index * 512
    values = np.asarray(tokens[start:start + 513], dtype=np.int64).copy()
    if len(values) != 513:
        raise RuntimeError("short controlled-corpus macroblock")
    inputs = []
    targets = []
    for offset in range(0, 512, context):
        inputs.append(values[offset:offset + context])
        targets.append(values[offset + 1:offset + context + 1])
    return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets))


@torch.inference_mode()
def generation_probe(
    model: DiagnosticTransformer,
    tokenizer: FoundationTokenizer,
    validation: np.memmap,
    seed: int,
) -> dict:
    rows = []
    eos = tokenizer.eos_id
    bos_positions = np.flatnonzero(validation == tokenizer.bos_id)
    for index, position in enumerate(bos_positions[:20]):
        start = int(position)
        end_candidates = np.flatnonzero(validation[start + 1:] == eos)
        if len(end_candidates) == 0:
            continue
        end = start + 1 + int(end_candidates[0])
        document = np.asarray(validation[start:end + 1], dtype=np.int64).tolist()
        if len(document) < 24:
            continue
        prompt = document[:min(32, max(16, len(document) // 2))]
        generated = generate_ids(
            model, tokenizer, prompt, PRIMARY_MODES[0], seed + index, max_new_tokens=64
        )
        rows.append({
            "prompt": tokenizer.decode(prompt, skip_special=True),
            "reference": tokenizer.decode(document[len(prompt):], skip_special=True),
            **generated,
        })
        if len(rows) == 10:
            break
    return {"metrics": aggregate_generation(rows), "items": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v15.json")
    parser.add_argument("--variant", default="tied_embedding_sqrt_scale")
    parser.add_argument("--output", default="evaluation/foundation-v15-controlled-corpus-experiment.json")
    parser.add_argument("--checkpoint", default="checkpoints/foundation-v15-controlled-corpus.pt")
    args = parser.parse_args()
    settings = load_json(args.config)
    manifest = load_json("data/foundation_v15_diagnostic/manifest.json")
    variant = next(row for row in settings["ablations"] if row["name"] == args.variant)
    train_meta = manifest["artifacts"]["packed"]["train"]
    validation_meta = manifest["artifacts"]["packed"]["validation"]
    train = np.memmap(ROOT / train_meta["path"], dtype=np.uint16, mode="r")
    validation = np.memmap(ROOT / validation_meta["path"], dtype=np.uint16, mode="r")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(settings["cpu_threads"]))
    architecture = dict(settings["architecture"])
    architecture.update(variant["changes"])
    architecture["context_length"] = 128
    model = DiagnosticTransformer(DiagnosticConfig(
        model_name=f"Foundation v1.5 controlled corpus {args.variant}",
        vocab_size=tokenizer.vocab_size,
        **architecture,
    ))
    optimizer = create_optimizer(model, float(settings["learning_rate"]), float(settings["weight_decay"]))
    macro_count = (len(train) - 1) // 512
    permutation = torch.randperm(macro_count, generator=torch.Generator().manual_seed(seed))
    updates = int(settings["controlled_corpus"]["training_token_budget"]) // 512
    if updates > len(permutation):
        raise RuntimeError("controlled corpus does not contain enough unique macroblocks")
    history = [{"update": 0, "tokens_processed": 0, "validation": validate(model, validation)}]
    started = time.perf_counter()
    losses = []
    for update in range(1, updates + 1):
        x, y = macro_batch(train, int(permutation[update - 1]))
        lr = float(settings["learning_rate"]) * min(1.0, update / int(settings["warmup_updates"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite controlled-corpus loss")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"])))
        optimizer.step()
        losses.append(float(loss.item()))
        if update in {10, 64, 100, updates}:
            history.append({
                "update": update,
                "tokens_processed": update * 512,
                "train_loss_since_last_probe": sum(losses) / len(losses),
                "gradient_norm": gradient_norm,
                "learning_rate": lr,
                "validation": validate(model, validation),
            })
            losses.clear()
            print(json.dumps(history[-1]), flush=True)
    checkpoint = ROOT / args.checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_format": "foundation-v15-controlled-diagnostic-v1",
        "model_state": model.state_dict(),
        "config": model.config.to_dict(),
        "update": updates,
        "tokens_processed": updates * 512,
        "variant": variant,
        "diagnostic_only": True,
        "production_changed": False,
    }, checkpoint)
    report = {
        "schema_version": "foundation-v15-controlled-corpus-experiment-v1",
        "purpose": "isolated short-sentence syntax diagnostic; not Foundation corpus training",
        "corpus_manifest": manifest,
        "variant": variant,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "training": {
            "scratch": True,
            "updates": updates,
            "tokens_processed": updates * 512,
            "effective_batch_tokens": 512,
            "history": history,
            "wall_seconds": time.perf_counter() - started,
        },
        "generation": generation_probe(model, tokenizer, validation, seed),
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "strict_reload": False,
        },
        "added_to_foundation_corpus": False,
        "external_ai_api": "OFF",
        "production_changed": False,
        "final_blind_used": False,
    }
    reloaded, payload = DiagnosticTransformer(DiagnosticConfig(**model.config.to_dict())), torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    reloaded.load_state_dict(payload["model_state"], strict=True)
    report["checkpoint"]["strict_reload"] = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), reloaded.state_dict().values())
    )
    (ROOT / args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "validation": history[-1]["validation"],
        "generation": report["generation"]["metrics"],
        "checkpoint": report["checkpoint"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
