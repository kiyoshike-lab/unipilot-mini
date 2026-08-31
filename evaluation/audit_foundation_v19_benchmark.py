from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from foundation.synthetic_context_v3 import (
    LEVEL_PAIRS,
    ambiguity_audit,
    causal_learnability,
    key_lookup_example_v3,
    make_batch_v3,
    scan_dataset,
    token_label,
    weighted_relation_loss,
)
from training.validate_foundation_v16_synthetic import VALUES, make_batch, pattern_example
from training.validate_foundation_v18_synthetic import build_model, key_lookup_v4_example, load_json, model_spec


def labels(sequence: list[int]) -> list[str]:
    return [token_label(token) for token in sequence]


def old_example_row(example: tuple[list[int], int, dict]) -> dict:
    sequence, answer, metadata = example
    inputs, targets = make_batch([example])
    key_positions = list(range(1, 1 + metadata["pairs"] * 2, 2))
    value_positions = [position + 1 for position in key_positions]
    query_key_position = len(sequence) - 2
    selected = sequence[query_key_position]
    selected_index = [sequence[position] for position in key_positions].index(selected)
    return {
        "input_token_sequence": labels(sequence),
        "token_ids": sequence,
        "target_sequence": targets[0].tolist(),
        "loss_mask": (targets[0] != -100).int().tolist(),
        "query_position": query_key_position,
        "key_position": key_positions[selected_index],
        "value_position": value_positions[selected_index],
        "answer_prediction_position": len(sequence) - 1,
        "answer_id": answer,
        "human_readable": " ".join(labels(sequence)) + f" -> {token_label(answer)}",
    }


def old_dataset_audit(seed: int) -> dict:
    examples = []
    for pairs in (1, 2, 4, 8, 16):
        for distance in ("short", "medium", "long"):
            rng = random.Random(seed + pairs * 100 + len(distance))
            examples.extend(key_lookup_v4_example(rng, pairs, distance) for _ in range(1000))
    ambiguity_count = 0
    causal_failures = 0
    answer_leakage = 0
    for sequence, answer, metadata in examples:
        keys = sequence[1:1 + metadata["pairs"] * 2:2]
        values = sequence[2:1 + metadata["pairs"] * 2:2]
        query_position = len(sequence) - 2
        matches = [value for key, value in zip(keys, values) if key == sequence[query_position]]
        ambiguity_count += int(
            len(keys) != len(set(keys))
            or len(values) != len(set(values))
            or len(matches) != 1
            or matches[0] != answer
        )
        selected = keys.index(sequence[query_position])
        causal_failures += int(not all(
            position < len(sequence) - 1
            for position in (1 + selected * 2, 2 + selected * 2, query_position)
        ))
        answer_leakage += int(sequence[-1] == answer or sequence[-2] == answer)
    return {
        "examples_scanned": len(examples),
        "causal_failures": causal_failures,
        "ambiguity_count": ambiguity_count,
        "answer_leakage_in_query_suffix": answer_leakage,
        "chance_by_pair_count": {
            str(pairs): {
                "actual_unique_candidate_values": pairs,
                "candidate_choice_accuracy": 1 / pairs,
                "full_value_vocabulary_accuracy": 1 / len(VALUES),
            }
            for pairs in (1, 2, 4, 8, 16)
        },
    }


def gradient_norm(model, inputs, targets, weights) -> float:
    model.zero_grad(set_to_none=True)
    _, loss = weighted_relation_loss(model, inputs, targets, weights)
    loss.backward()
    squared = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in model.parameters() if parameter.grad is not None
    )
    return math.sqrt(squared), float(loss.detach())


def loss_supervision_audit(settings: dict, seed: int) -> dict:
    spec = model_spec(settings, "reference_mha")
    examples = [
        key_lookup_example_v3(random.Random(seed + index), 2, split="train")
        for index in range(16)
    ]
    rows = {}
    for supervision, answer_weight in (
        ("answer_only", 1), ("all_token", 1), ("all_token", 4), ("all_token", 16)
    ):
        model = build_model(settings, spec, 256, 128, seed)
        model.eval()
        inputs, targets, weights = make_batch_v3(examples, supervision, answer_weight)
        total_gradient, total_loss = gradient_norm(model, inputs, targets, weights)
        answer_targets = torch.full_like(targets, -100)
        answer_targets[:, -1] = targets[:, -1]
        answer_weights = torch.zeros_like(weights)
        answer_weights[:, -1] = 1
        answer_gradient, answer_loss = gradient_norm(model, inputs, answer_targets, answer_weights)
        if supervision == "all_token":
            non_targets = targets.clone()
            non_targets[:, -1] = -100
            non_weights = weights.clone()
            non_weights[:, -1] = 0
            non_gradient, non_loss = gradient_norm(model, inputs, non_targets, non_weights)
        else:
            non_gradient, non_loss = None, None
        key = f"{supervision}-{answer_weight}x"
        rows[key] = {
            "active_target_tokens": int((targets != -100).sum()),
            "answer_target_tokens": len(examples),
            "total_loss": total_loss,
            "answer_token_loss": answer_loss,
            "non_answer_loss": non_loss,
            "total_gradient_l2": total_gradient,
            "answer_gradient_l2": answer_gradient,
            "non_answer_gradient_l2": non_gradient,
        }
    return {
        "old_v2_v4_actual_training_supervision": "answer-only; only final answer position is active and all other targets are -100",
        "structural_loss_dominated_old_run": False,
        "initial_reference_gradient_comparison": rows,
        "interpretation": "all-token is a requested counterfactual; it was not the cause of the old run",
    }


def old_training_distribution() -> dict:
    rows = {}
    for name in ("custom_current", "custom_depth_init", "reference_mha"):
        path = ROOT / "checkpoints/foundation-v18-synthetic" / f"{name}-lr-3em04.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        by_difficulty = report["training"]["examples_by_difficulty"]
        cells = {key: value for key, value in by_difficulty.items() if key.startswith("key_lookup:")}
        batch_size = int(report["training"]["batch_size"])
        key_examples = sum(cells.values())
        rows[name] = {
            "total_updates": report["training"]["updates"],
            "total_examples": report["training"]["train_examples"],
            "key_lookup_examples": key_examples,
            "key_lookup_update_equivalents": key_examples // batch_size,
            "key_lookup_percent": 100 * key_examples / report["training"]["train_examples"],
            "cell_update_equivalents": {key: value // batch_size for key, value in cells.items()},
            "minimum_updates_in_one_cell": min(cells.values()) // batch_size,
            "maximum_updates_in_one_cell": max(cells.values()) // batch_size,
        }
    return rows


def numeric_audit(tokenizer: FoundationTokenizer) -> dict:
    standalone = [str(value) for value in range(10)] + [
        "10", "11", "12", "20", "32", "64", "99", "100", "123", "2026",
        "12,345", "3.14", "-7", "GPA 3.5",
    ]
    standalone_rows = [{
        "raw_text": text,
        "token_ids": tokenizer.encode(text),
        "token_count": len(tokenizer.encode(text)),
        "decoded": tokenizer.decode(tokenizer.encode(text)),
    } for text in standalone]
    patterns = []
    for index in range(20):
        start = index % 10
        step = index % 4 + 1
        source = [start + offset * step for offset in range(5)]
        target = start + 5 * step
        raw = " ".join(map(str, source))
        patterns.append({
            "raw_text": raw,
            "source_numbers": source,
            "token_ids": tokenizer.encode(raw),
            "target_text": str(target),
            "target_ids": tokenizer.encode(str(target)),
            "requires_arithmetic": True,
        })
    old_synthetic = []
    rng = random.Random(919)
    for _ in range(20):
        sequence, answer, metadata = pattern_example(rng, "numeric")
        old_synthetic.append({
            "raw_text": None,
            "token_ids": sequence,
            "target_ids": [answer],
            "atomic_synthetic_ids": True,
            "difficulty": metadata["difficulty"],
        })
    return {
        "tokenizer": "tokenizer/foundation-v11-base-4096.json",
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "standalone_samples": standalone_rows,
        "actual_numeric_pattern_samples": patterns,
        "old_synthetic_numeric_samples": old_synthetic,
        "single_digit_atomic_rate": sum(row["token_count"] == 1 for row in standalone_rows[:10]) / 10,
        "tasks_separated": {
            "tokenizer_independent_symbolic": "finite repeating-symbol continuation; no arithmetic; architecture gate >=95%",
            "actual_numeric_token_pattern": "Foundation tokenizer IDs from raw decimal text; diagnostic only",
            "deprecated_v4_numeric": "atomic synthetic IDs but modular arithmetic progression; not a pure pattern-copy test",
        },
        "arithmetic_audit": {
            "deprecated_v4_numeric_requires_modular_addition": True,
            "actual_numeric_examples_require_addition": True,
            "symbolic_repetition_requires_arithmetic": False,
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/foundation-v19-benchmark-audit.json")
    args = parser.parse_args()
    seed = 43
    old_rng = random.Random(seed)
    old_examples = [
        old_example_row(key_lookup_v4_example(old_rng, pairs, distance))
        for pairs, distance in (
            (1, "short"), (2, "short"), (2, "medium"), (2, "long"),
            (4, "short"), (4, "medium"), (4, "long"),
            (8, "short"), (8, "medium"), (16, "long"),
        )
    ]
    v3_examples = [
        key_lookup_example_v3(random.Random(seed + index), index % 6)
        for index in range(12)
    ]
    v3_rows = []
    for example in v3_examples:
        inputs, targets, _ = make_batch_v3([example])
        v3_rows.append({
            "input_token_sequence": labels(example[0]),
            "token_ids": example[0],
            "target_sequence": targets[0].tolist(),
            "loss_mask": (targets[0] != -100).int().tolist(),
            "answer_id": example[1],
            "metadata": example[2],
            "causal": causal_learnability(example),
            "ambiguity": ambiguity_audit(example),
        })
    scan_examples = [
        key_lookup_example_v3(random.Random(seed + index), index % 6)
        for index in range(10_000)
    ]
    settings = load_json("configs/unipilot-foundation-v18.json")
    tokenizer_path = ROOT / "tokenizer/foundation-v11-base-4096.json"
    result = {
        "schema_version": "foundation-v19-benchmark-audit-v1",
        "old_benchmark": {
            "versions": ["Synthetic Context Benchmark v1", "v2", "v4/Phase29"],
            "status": "DEPRECATED_RETAINED",
            "examples": old_examples,
            "full_scan": old_dataset_audit(seed),
            "training_distribution": old_training_distribution(),
        },
        "loss_supervision": loss_supervision_audit(settings, seed),
        "benchmark_v3": {
            "examples": v3_rows,
            "scan": scan_dataset(scan_examples),
            "markers": ["<PAIR>", "<KEY>", "<VALUE>", "<QUERY>", "<ANSWER>"],
            "curriculum_pairs": LEVEL_PAIRS,
        },
        "numeric": numeric_audit(FoundationTokenizer.load(tokenizer_path)),
        "source_sha256": {
            "synthetic_context_v3.py": sha256(ROOT / "foundation/synthetic_context_v3.py"),
            "foundation_tokenizer": sha256(tokenizer_path),
        },
        "final_blind_used": False,
        "production_changed": False,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "old_causal_failures": result["old_benchmark"]["full_scan"]["causal_failures"],
        "old_ambiguities": result["old_benchmark"]["full_scan"]["ambiguity_count"],
        "v3_causal_failures": result["benchmark_v3"]["scan"]["causal_failures"],
        "v3_ambiguities": result["benchmark_v3"]["scan"]["ambiguity_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
