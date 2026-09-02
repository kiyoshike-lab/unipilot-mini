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

from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.optimizer import create_optimizer
from training.validate_foundation_v16_synthetic import (
    ANSWER,
    FILLERS,
    KEYS,
    QUERY,
    VALUES,
    accuracy_for_examples,
    conditioned_example,
    copy_example,
    fixed_examples,
    long_range_example,
    make_batch,
)


VARIANTS = {
    "current": 1.0,
    "depth_init": 1 / math.sqrt(20),
}
PRIOR_SMOKE = {
    "current": "checkpoints/foundation-v17-synthetic/current_unscaled-seed-42.json",
    "depth_init": "checkpoints/foundation-v17-synthetic/depth_scaled_residual_init-seed-42.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_smoke_model(variant: str, seed: int) -> DiagnosticTransformerV17:
    torch.manual_seed(seed)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(
        model_name=f"Foundation v2.1 {variant} synthetic smoke",
        vocab_size=256,
        context_length=80,
        embedding_dim=384,
        n_layers=10,
        n_heads=6,
        ffn_dim=1536,
        dropout=0,
        residual_projection_init_scale=1.0,
    ))
    scale = VARIANTS[variant]
    if variant == "depth_init":
        model.config.residual_projection_init_scale = scale
        with torch.no_grad():
            for block in model.blocks:
                block.attention.projection.weight.mul_(scale)
                block.feed_forward.network[2].weight.mul_(scale)
    return model


def fixed_relation_example(rng: random.Random) -> tuple[list[int], int, dict]:
    mapping = dict(zip(KEYS[:4], VALUES[:4]))
    keys = list(mapping)
    selected = rng.choice(keys)
    sequence = [2, selected, *[rng.choice(FILLERS) for _ in range(8)], QUERY, ANSWER]
    return sequence, mapping[selected], {
        "difficulty": "fixed_relation_4",
        "required_context_distance": len(sequence) - 2,
        "fixed_mapping": True,
        "relation_key_position": 1,
    }


def _train_updates(
    model: DiagnosticTransformerV17,
    makers: list[tuple],
    updates: int,
    seed: int,
    batch_size: int = 16,
) -> dict:
    optimizer = create_optimizer(model, 3e-3, 0.0)
    rng = random.Random(seed)
    losses = []
    started = time.perf_counter()
    for update in range(updates):
        maker, args = makers[update % len(makers)]
        examples = [maker(rng, *args) for _ in range(batch_size)]
        inputs, targets = make_batch(examples)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite PHASE 32 synthetic smoke loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "updates": updates,
        "final_20_mean_loss": sum(losses[-20:]) / min(20, len(losses)),
        "wall_seconds": time.perf_counter() - started,
    }


def _accuracy(model, maker, seed: int, *args) -> float:
    return accuracy_for_examples(model, fixed_examples(maker, 128, seed, *args))


def prior_architecture_smoke(variant: str) -> dict:
    path = ROOT / PRIOR_SMOKE[variant]
    payload = json.loads(path.read_text(encoding="utf-8"))
    final = payload["final"]
    base = final["base"]
    position = final["position"]
    expected_scale = VARIANTS[variant]
    config = payload["config"]
    architecture_match = all((
        config["embedding_dim"] == 384,
        config["n_layers"] == 10,
        config["n_heads"] == 6,
        config["ffn_dim"] == 1536,
        config["residual_projection_init_scale"] == expected_scale,
    ))
    return {
        "source": PRIOR_SMOKE[variant],
        "source_sha256": file_sha256(path),
        "source_phase": "PHASE 28 full 20M architecture synthetic validation",
        "architecture_match": architecture_match,
        "copy": {key: base["copy"][key] for key in ("4", "8", "16")},
        "position": {key: position["by_length"][key] for key in ("4", "8", "16")},
        "long_range": base["long_range"],
        "context_conditioned": base["context_conditioned"],
    }


def run_variant(variant: str, seed: int) -> dict:
    # Tiny overfit is intentionally the exact same example.
    tiny_model = build_smoke_model(variant, seed)
    tiny_example = copy_example(random.Random(seed + 1), 4)
    tiny_train = _train_updates(
        tiny_model,
        [(lambda _rng: tiny_example, ())],
        40,
        seed + 10,
    )
    tiny_accuracy = accuracy_for_examples(tiny_model, [tiny_example] * 32)
    prior = prior_architecture_smoke(variant)
    copy = prior["copy"]
    position = prior["position"]
    long_range = prior["long_range"]
    context = prior["context_conditioned"]

    relation_model = build_smoke_model(variant, seed + 100)
    relation_train = _train_updates(
        relation_model, [(fixed_relation_example, ())], 400, seed + 110
    )
    fixed_relation = _accuracy(relation_model, fixed_relation_example, seed + 120)
    checks = {
        "prior_architecture_match": prior["architecture_match"],
        "tiny_overfit": tiny_accuracy >= .99,
        **{f"copy_{length}": value >= .90 for length, value in copy.items()},
        **{f"position_{length}": value >= .90 for length, value in position.items()},
        "long_range": long_range >= .95,
        "context_conditioned": context["correct"] >= .95,
        "context_control": context["correct"] - max(context["shuffled"], context["removed"]) >= .40,
        "fixed_relation_lookup": fixed_relation >= .95,
    }
    return {
        "variant": variant,
        "tiny_overfit": {"accuracy": tiny_accuracy, "training": tiny_train},
        "prior_full_architecture_evidence": prior,
        "copy": {"accuracy": copy, "evidence": "prior_full_architecture_evidence"},
        "position": {"accuracy": position, "evidence": "prior_full_architecture_evidence"},
        "long_range": {"accuracy": long_range, "evidence": "prior_full_architecture_evidence"},
        "context_conditioned": {"accuracy": context, "evidence": "prior_full_architecture_evidence"},
        "fixed_relation_lookup": {"accuracy": fixed_relation, "training": relation_train},
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/foundation-v21-synthetic-smoke.json")
    parser.add_argument("--seed", type=int, default=32021)
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    results = {variant: run_variant(variant, args.seed) for variant in VARIANTS}
    payload = {
        "schema_version": "foundation-v21-synthetic-smoke-v1",
        "purpose": "fatal-regression smoke only; novel mapping and numeric modular addition are not gates",
        "seed": args.seed,
        "results": results,
        "gate_pass": all(row["pass"] for row in results.values()),
        "novel_random_key_lookup_gate": False,
        "numeric_modular_addition_gate": False,
        "production_changed": False,
        "final_blind_used": False,
    }
    path = ROOT / args.output
    if path.exists():
        raise RuntimeError(f"refusing to overwrite PHASE 32 smoke result: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if payload["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
