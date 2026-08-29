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

from evaluation.audit_foundation_v15_architecture import context_sensitivity
from evaluation.measure_foundation_v16 import frequency_metrics
from evaluation.measure_foundation_v17 import (
    architecture_probe,
    hidden_token_similarity,
    stats,
    validation_metrics,
)
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.investigate_foundation_v14 import macro_batch, macro_permutation
from training.optimizer import create_optimizer


CHECKPOINT_FORMAT = "foundation-v17-isolation-reproduction-v1"


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
    return DiagnosticTransformerV17(DiagnosticConfigV17(
        model_name=f"UniPilot Foundation v1.7 {variant['name']} seed {seed}",
        vocab_size=vocab_size,
        token_embedding_scale=variant["token_embedding_scale"],
        position_embedding_scale=variant["position_embedding_scale"],
        residual_projection_init_scale=variant["residual_projection_init_scale"],
        **settings["architecture"],
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v17.json")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026], required=True)
    parser.add_argument("--output-dir", default="checkpoints/foundation-v17-reproduction")
    args = parser.parse_args()
    settings = load_json(args.config)
    real = settings["real_corpus"]
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
        raise RuntimeError(f"refusing to overwrite v1.7 reproduction: {stem}")

    torch.set_num_threads(int(settings["cpu_threads"]))
    model = build_model(settings, variant, int(corpus["vocab"]), args.seed)
    optimizer = create_optimizer(
        model, float(real["learning_rate"]), float(real["weight_decay"])
    )
    token_budget = int(real["token_budget"])
    effective_batch = int(real["effective_batch_tokens"])
    updates = token_budget // effective_batch
    permutation = macro_permutation((len(train) - 1) // 512, args.seed)
    milestones = set(int(value) for value in real["milestones"])
    audit_tokens = torch.from_numpy(
        np.asarray(validation[8192:8320], dtype=np.int64).copy()
    ).unsqueeze(0)
    initial_position = model.embeddings.position.weight.detach().clone()
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss / 1024**2
    history = []
    started = time.perf_counter()
    latest_position_gradient = None

    def position_learning() -> dict:
        delta = model.embeddings.position.weight.detach() - initial_position
        gradient = latest_position_gradient
        return {
            "gradient": None if gradient is None else stats(gradient),
            "parameter_delta": stats(delta),
            "parameter_delta_relative_norm": (
                float(torch.linalg.vector_norm(delta))
                / max(float(torch.linalg.vector_norm(initial_position)), 1e-12)
            ),
        }

    def measure(update: int, gradient_norm: float | None, recent_loss: float | None):
        nonlocal peak_ram
        row = {
            "update": update,
            "tokens_processed": update * effective_batch,
            "recent_train_loss": recent_loss,
            "gradient_norm": gradient_norm,
            "validation": validation_metrics(
                model, validation, int(real["validation_tokens"])
            ),
            "probe": architecture_probe(model, audit_tokens),
            "position_learning": position_learning(),
        }
        peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
        row["peak_ram_mb"] = peak_ram
        history.append(row)
        print(json.dumps({
            "variant": args.variant,
            "seed": args.seed,
            "update": update,
            "validation": row["validation"],
            "position_ratio": row["probe"]["embedding"][
                "effective_token_to_position_rms_ratio"
            ],
            "layer9_rms": row["probe"]["layers"][-1]["post_mlp_residual"]["rms"],
        }), flush=True)

    measure(0, None, None)
    recent_losses = []
    recent_gradient_norm = None
    for update_index in range(updates):
        x, y = macro_batch(train, [int(permutation[update_index])], 512)
        completed = update_index + 1
        lr = float(real["learning_rate"]) * min(
            1.0, completed / int(real["warmup_updates"])
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite v1.7 loss: {stem}")
        loss.backward()
        if model.embeddings.position.weight.grad is None:
            raise RuntimeError("position embedding received no gradient")
        latest_position_gradient = model.embeddings.position.weight.grad.detach().clone()
        recent_gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(real["gradient_clip"])
        ))
        if not math.isfinite(recent_gradient_norm):
            raise RuntimeError(f"non-finite v1.7 gradient: {stem}")
        optimizer.step()
        recent_losses.append(float(loss.item()))
        if completed in milestones:
            measure(
                completed,
                recent_gradient_norm,
                sum(recent_losses) / len(recent_losses),
            )
            recent_losses.clear()

    frequency = frequency_metrics(
        model, tokenizer, train, validation, int(real["validation_tokens"])
    )
    context = context_sensitivity(model, validation)
    similarity = hidden_token_similarity(model, tokenizer, validation)
    final = history[-1]
    final["frequency"] = frequency
    final["context_sensitivity"] = context
    final["hidden_token_similarity"] = similarity
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config.to_dict(),
        "variant": variant,
        "seed": args.seed,
        "update": updates,
        "tokens_processed": token_budget,
        "macro_permutation_seed": args.seed,
        "scratch": True,
        "diagnostic_only": True,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    reloaded_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = DiagnosticTransformerV17(DiagnosticConfigV17(**reloaded_payload["config"]))
    restored.load_state_dict(reloaded_payload["model_state"], strict=True)
    strict_equal = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), restored.state_dict().values())
    )
    report = {
        "schema_version": "foundation-v17-reproduction-result-v1",
        "status": "COMPLETED",
        "variant": variant,
        "seed": args.seed,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "initialization": model.initialization_manifest(),
        "token_budget": token_budget,
        "effective_batch_tokens": effective_batch,
        "updates": updates,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": real["learning_rate"],
            "betas": [0.9, 0.95],
            "weight_decay": real["weight_decay"],
            "gradient_clip": real["gradient_clip"],
        },
        "history": history,
        "final": final,
        "checkpoint": {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
            "strict_reload": strict_equal,
            "optimizer_state_present": "optimizer_state" in reloaded_payload,
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_ram_mb": peak_ram,
        "final_norm": "PRESENT",
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
