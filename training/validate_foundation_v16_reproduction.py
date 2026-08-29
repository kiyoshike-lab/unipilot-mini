from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.measure_foundation_v16 import complete_metrics
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v15 import DiagnosticConfig, DiagnosticTransformer
from training.investigate_foundation_v14 import macro_batch, macro_permutation
from training.optimizer import create_optimizer


CHECKPOINT_FORMAT = "foundation-v16-architecture-fix-reproduction-v1"


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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def variant_by_name(settings: dict, name: str) -> dict:
    return next(row for row in settings["variants"] if row["name"] == name)


def build_model(settings: dict, variant: dict, vocab_size: int, seed: int):
    seed_all(seed)
    return DiagnosticTransformer(DiagnosticConfig(
        model_name=f"UniPilot Foundation v1.6 {variant['name']} seed {seed}",
        vocab_size=vocab_size,
        scale_token_embedding=variant["scale_token_embedding"],
        **settings["architecture"],
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v16.json")
    parser.add_argument("--variant", choices=["current_unscaled", "sqrt_scaled_a"], required=True)
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026], required=True)
    parser.add_argument("--output-dir", default="checkpoints/foundation-v16-reproduction")
    args = parser.parse_args()
    settings = load_json(args.config)
    reproduction = settings["reproduction"]
    variant = variant_by_name(settings, args.variant)
    corpus = load_json(settings["corpus_manifest"])
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    train = np.memmap(ROOT / corpus["splits"]["train"]["path"], dtype=np.uint16, mode="r")
    validation = np.memmap(
        ROOT / corpus["splits"]["validation"]["path"], dtype=np.uint16, mode="r"
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}-seed-{args.seed}"
    result_path = output_dir / f"{stem}.json"
    checkpoint_path = output_dir / f"{stem}.pt"
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite reproduction run: {stem}")

    torch.set_num_threads(int(settings["cpu_threads"]))
    model = build_model(settings, variant, int(corpus["vocab"]), args.seed)
    optimizer = create_optimizer(
        model, float(reproduction["learning_rate"]), float(reproduction["weight_decay"])
    )
    token_budget = int(reproduction["token_budget"])
    effective_batch = int(reproduction["effective_batch_tokens"])
    updates = token_budget // effective_batch
    permutation = macro_permutation((len(train) - 1) // 512, args.seed)
    milestones = set(int(value) for value in reproduction["milestones"])
    audit_tokens = torch.from_numpy(
        np.asarray(validation[8192:8320], dtype=np.int64).copy()
    ).unsqueeze(0)
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss / 1024**2
    history = []
    started = time.perf_counter()

    def measure(update: int, gradient_norm: float | None, recent_loss: float | None):
        nonlocal peak_ram
        model.eval()
        row = {
            "update": update,
            "tokens_processed": update * effective_batch,
            "learning_rate": (
                float(reproduction["learning_rate"])
                * min(1.0, max(1, update) / int(reproduction["warmup_updates"]))
            ),
            "recent_train_loss": recent_loss,
            "gradient_norm": gradient_norm,
            **complete_metrics(
                model, tokenizer, train, validation, audit_tokens,
                int(reproduction["validation_tokens"]),
                include_frequency=update == updates,
            ),
        }
        peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
        row["peak_ram_mb"] = peak_ram
        history.append(row)
        print(json.dumps({
            "variant": args.variant,
            "seed": args.seed,
            "update": update,
            "validation": row["validation"],
            "layer9_output_rms": row["probe"]["layers"][-1]["output"]["rms"],
            "logit_entropy": row["probe"]["logits"]["mean_softmax_entropy"],
        }), flush=True)

    measure(0, None, None)
    recent_losses = []
    recent_gradient = None
    for update_index in range(updates):
        x, y = macro_batch(train, [int(permutation[update_index])], 512)
        completed = update_index + 1
        lr = float(reproduction["learning_rate"]) * min(
            1.0, completed / int(reproduction["warmup_updates"])
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite reproduction loss: {stem}")
        loss.backward()
        recent_gradient = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(reproduction["gradient_clip"])
        ))
        if not math.isfinite(recent_gradient):
            raise RuntimeError(f"non-finite reproduction gradient: {stem}")
        optimizer.step()
        recent_losses.append(float(loss.item()))
        if completed in milestones:
            measure(
                completed, recent_gradient,
                sum(recent_losses) / len(recent_losses),
            )
            recent_losses.clear()

    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "architecture_version": "foundation-v1.6-diagnostic-candidate",
        "architecture_formula": variant["formula"],
        "model_state": model.state_dict(),
        "config": model.config.to_dict(),
        "variant": variant,
        "seed": args.seed,
        "update": updates,
        "tokens_processed": token_budget,
        "scratch": True,
        "semantic_compatibility_with_unscaled_checkpoint": (
            variant["name"] == "current_unscaled"
        ),
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    reloaded_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded = DiagnosticTransformer(DiagnosticConfig(**reloaded_payload["config"]))
    reloaded.load_state_dict(reloaded_payload["model_state"], strict=True)
    strict_equal = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), reloaded.state_dict().values())
    )
    result = {
        "schema_version": "foundation-v16-reproduction-result-v1",
        "status": "COMPLETED",
        "variant": variant,
        "seed": args.seed,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "token_budget": token_budget,
        "effective_batch_tokens": effective_batch,
        "updates": updates,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": reproduction["learning_rate"],
            "betas": [0.9, 0.95],
            "weight_decay": reproduction["weight_decay"],
            "gradient_clip": reproduction["gradient_clip"],
        },
        "same_seed_controls_model_initialization_and_macro_order": True,
        "history": history,
        "final": history[-1],
        "checkpoint": {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
            "strict_reload": strict_equal,
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_ram_mb": peak_ram,
        "sqrt_scaling_application_count": 1 if variant["scale_token_embedding"] else 0,
        "lm_head_scaling_application_count": 0,
        "checkpoint_load_scaling_application_count": 0,
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
