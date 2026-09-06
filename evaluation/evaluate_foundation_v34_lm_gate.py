"""PHASE 45 deterministic LM and rare-bucket diagnostics (read-only checkpoints)."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.diagnose_foundation_v29_generation import build_prefixes, document_ranges
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import file_sha256, frequency_ranks


SEEDS = (42, 123, 2026)
STAGES = ("baseline", "gate1", "gate2")
PROBE_TOKENS = 8192
RECIPE_PATHS = {
    "standard_ce_eos_1_0": ROOT
    / "checkpoints/foundation-v30-eos-experimental/control/seed-42/checkpoint-tokens-15616000.pt",
    "eos_weight_1_5": ROOT
    / "checkpoints/foundation-v30-eos-experimental/eos_weight_1.5/seed-42/checkpoint-tokens-15616000.pt",
}


def checkpoint(stage: str, seed: int) -> Path:
    if stage == "baseline":
        return ROOT / f"checkpoints/foundation-v28-current/current/seed-{seed}/checkpoint-tokens-15360000.pt"
    gate = 1 if stage == "gate1" else 2
    tokens = 15_616_000 if gate == 1 else 15_872_000
    return ROOT / f"checkpoints/foundation-v33-context-gate/gate-{gate}/seed-{seed}/checkpoint-tokens-{tokens}.pt"


def load_model(path: Path) -> tuple[dict, DiagnosticTransformerV17]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return payload, model


def distribution(values: torch.Tensor) -> dict:
    probabilities = values.double().numpy()
    logs = np.log(np.clip(probabilities, 1e-30, None))
    return {
        "mean_correct_token_probability": float(np.mean(probabilities)),
        "median_correct_token_probability": float(np.median(probabilities)),
        "geometric_mean_correct_token_probability": float(np.exp(np.mean(logs))),
        "cross_entropy": float(-np.mean(logs)),
        "probability_quantiles": {
            key: float(np.quantile(probabilities, quantile))
            for key, quantile in (("p00", 0.0), ("p10", 0.1), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p100", 1.0))
        },
    }


@torch.inference_mode()
def detailed_metrics(
    model: DiagnosticTransformerV17,
    validation: np.memmap,
    ranks: np.ndarray,
) -> dict:
    model.eval()
    target_rows: list[torch.Tensor] = []
    top_rows: list[torch.Tensor] = []
    assigned_rows: list[torch.Tensor] = []
    weighted_loss = 0.0
    total = 0
    context = model.config.context_length
    for start in range(0, PROBE_TOKENS, context):
        size = min(context, PROBE_TOKENS - start)
        values = np.asarray(validation[start : start + size + 1], dtype=np.int64).copy()
        inputs = torch.from_numpy(values[:-1]).unsqueeze(0)
        targets = torch.from_numpy(values[1:])
        logits, loss = model(inputs, targets.unsqueeze(0))
        logits = logits[0].float()
        top = logits.topk(10, dim=-1).indices
        assigned = torch.softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
        target_rows.append(targets)
        top_rows.append(top)
        assigned_rows.append(assigned)
        weighted_loss += float(loss) * size
        total += size
    targets = torch.cat(target_rows)
    top = torch.cat(top_rows)
    assigned = torch.cat(assigned_rows)
    target_ranks = ranks[targets.numpy()]
    vocab = model.config.vocab_size
    lower = math.ceil(vocab * 0.20)
    rare_lower = math.ceil(vocab * 0.80)
    buckets = {}
    for name, low, high in (
        ("middle_20_to_80_percent", lower, rare_lower),
        ("rare_bottom_20_percent", rare_lower, vocab),
    ):
        mask = torch.from_numpy((target_ranks >= low) & (target_ranks < high))
        bucket_targets = targets[mask]
        count = int(mask.sum())
        buckets[name] = {
            "rank_range": [low, high - 1],
            "targets": count,
            "top_1_accuracy": float((top[mask, 0] == bucket_targets).float().mean()),
            "top_5_accuracy": float(
                (top[mask, :5] == bucket_targets[:, None]).any(-1).float().mean()
            ),
            "top_10_accuracy": float(
                (top[mask] == bucket_targets[:, None]).any(-1).float().mean()
            ),
            **distribution(assigned[mask]),
        }
    return {
        "tokens": total,
        "loss": weighted_loss / total,
        "perplexity": math.exp(weighted_loss / total),
        "top_1_accuracy": float((top[:, 0] == targets).float().mean()),
        "top_5_accuracy": float((top[:, :5] == targets[:, None]).any(-1).float().mean()),
        "top_10_accuracy": float((top == targets[:, None]).any(-1).float().mean()),
        "frequency_buckets": buckets,
    }


def signature(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in ("loss", "perplexity", "top_1_accuracy", "top_5_accuracy", "top_10_accuracy")
    } | {
        "middle": metrics["frequency_buckets"]["middle_20_to_80_percent"],
        "rare": metrics["frequency_buckets"]["rare_bottom_20_percent"],
    }


def stored_match(stage: str, seed: int, metrics: dict) -> dict:
    stored = json.loads(
        (ROOT / f"evaluation/phase44/{stage}/seed-{seed}.json").read_text(encoding="utf-8")
    )["validation"]
    keys = ("loss", "top_1_accuracy", "top_5_accuracy", "top_10_accuracy")
    deltas = {key: metrics[key] - stored[key] for key in keys}
    return {"deltas": deltas, "pass": all(abs(value) <= 1e-6 for value in deltas.values())}


def main() -> None:
    torch.set_num_threads(4)
    tokenizer = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v11-base-4096.json")
    train = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/train.bin", dtype=np.uint16, mode="r"
    )
    validation = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/validation.bin", dtype=np.uint16, mode="r"
    )
    ranks = frequency_ranks(train, tokenizer.vocab_size)
    paths = [checkpoint(stage, seed) for stage in STAGES for seed in SEEDS] + list(
        RECIPE_PATHS.values()
    )
    hashes_before = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in paths}
    trajectory = {}
    determinism = {}
    for stage in STAGES:
        trajectory[stage] = {}
        for seed in SEEDS:
            path = checkpoint(stage, seed)
            payload, model = load_model(path)
            first = detailed_metrics(model, validation, ranks)
            trajectory[stage][str(seed)] = {
                "tokens_processed": int(payload["tokens_processed"]),
                "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                "metrics": first,
                "matches_phase44": stored_match(stage, seed, first),
            }
            if stage == "gate2":
                second = detailed_metrics(model, validation, ranks)
                determinism[str(seed)] = {
                    "runs": [signature(first), signature(second)],
                    "exact_match": signature(first) == signature(second),
                }
    ranges = document_ranges(validation, tokenizer.bos_id, tokenizer.eos_id)
    prefixes = build_prefixes(validation, ranges, tokenizer)
    prefix_payload = json.dumps(
        [row["prefix_ids"] for row in prefixes], separators=(",", ":")
    ).encode("utf-8")
    sampling = {
        "fixed_prefix_count": len(prefixes),
        "fixed_prefix_ids_sha256": hashlib.sha256(prefix_payload).hexdigest(),
        "sampling_seeds": "44000 + prefix_index",
        "temperature": 0.7,
        "max_new_tokens": 64,
        "all_phase44_rows_use_same_evaluator": True,
    }
    recipe_control = {}
    for name, path in RECIPE_PATHS.items():
        payload, model = load_model(path)
        recipe_control[name] = {
            "tokens_processed": int(payload["tokens_processed"]),
            "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": hashes_before[str(path.relative_to(ROOT)).replace("\\", "/")],
            "metrics": detailed_metrics(model, validation, ranks),
        }
    hashes_after = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in paths}
    blind_path = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    result = {
        "schema": "foundation-v34-lm-diagnostics-v1",
        "phase": 45,
        "large_scale_training_executed": False,
        "recipe_pilot_executed": False,
        "trajectory": trajectory,
        "deterministic_evaluation": {
            "repeat_count_per_gate2_checkpoint": 2,
            "by_seed": determinism,
            "pass": all(row["exact_match"] for row in determinism.values()),
        },
        "sampling_protocol": sampling,
        "existing_recipe_control_detailed": recipe_control,
        "checkpoint_integrity": {
            "before": hashes_before,
            "after": hashes_after,
            "unchanged": hashes_before == hashes_after,
        },
        "final_blind": {
            "opened": False,
            "sha256": file_sha256(blind_path),
        },
    }
    target = ROOT / "evaluation/foundation-v34-determinism-and-rare.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "deterministic": result["deterministic_evaluation"]["pass"],
                "checkpoint_integrity": result["checkpoint_integrity"]["unchanged"],
            }
        )
    )


if __name__ == "__main__":
    main()
