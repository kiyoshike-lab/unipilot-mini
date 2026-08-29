from __future__ import annotations

import argparse
from collections import Counter
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

from foundation.diagnostic_transformer_v15 import DiagnosticConfig, DiagnosticTransformer
from training.optimizer import create_optimizer


TASKS = ("copy", "key_lookup", "long_range", "pattern", "context_conditioned")
TASK_TOKEN = {
    "copy": 1,
    "key_lookup": 2,
    "long_range": 3,
    "pattern_abab": 4,
    "pattern_abcabc": 5,
    "pattern_numeric": 6,
    "pattern_nested": 7,
    "context_conditioned": 8,
}
VALUES = list(range(32, 64))
KEYS = list(range(64, 80))
INDEX_TOKENS = list(range(128, 192))
FILLERS = list(range(192, 224))
QUERY = 224
ANSWER = 225
REMOVED_CONTEXT = 226


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def example_hash(sequence: list[int], answer: int) -> str:
    payload = bytes(sequence) + bytes([answer])
    return hashlib.sha256(payload).hexdigest()


def filler(rng: random.Random, count: int) -> list[int]:
    return [rng.choice(FILLERS) for _ in range(count)]


def copy_example(rng: random.Random, length: int) -> tuple[list[int], int, dict]:
    values = (
        rng.sample(VALUES, length) if length <= len(VALUES)
        else [rng.choice(VALUES) for _ in range(length)]
    )
    query_index = rng.randrange(length)
    sequence = [TASK_TOKEN["copy"], *values, QUERY, INDEX_TOKENS[query_index], ANSWER]
    return sequence, values[query_index], {
        "difficulty": f"length_{length}",
        "source_items": length,
        "required_context_distance": len(sequence) - 1 - (1 + query_index),
    }


def key_lookup_example(
    rng: random.Random, pairs: int, distance: str
) -> tuple[list[int], int, dict]:
    keys = rng.sample(KEYS, pairs)
    values = rng.sample(VALUES, pairs)
    selected = rng.randrange(pairs)
    gap = {"short": 0, "medium": 16, "long": 32}[distance]
    sequence = [TASK_TOKEN["key_lookup"]]
    value_positions = []
    for key, value in zip(keys, values):
        sequence.extend((key, value))
        value_positions.append(len(sequence) - 1)
    sequence.extend(filler(rng, gap))
    sequence.extend((QUERY, keys[selected], ANSWER))
    return sequence, values[selected], {
        "difficulty": f"pairs_{pairs}_{distance}",
        "pairs": pairs,
        "distance": distance,
        "required_context_distance": len(sequence) - 1 - value_positions[selected],
    }


def long_range_example(rng: random.Random) -> tuple[list[int], int, dict]:
    answer = rng.choice(VALUES)
    sequence = [TASK_TOKEN["long_range"], answer, *filler(rng, 60), QUERY, ANSWER]
    return sequence, answer, {
        "difficulty": "distance_62",
        "required_context_distance": len(sequence) - 2,
    }


def pattern_example(rng: random.Random, pattern: str) -> tuple[list[int], int, dict]:
    if pattern == "abab":
        a, b = rng.sample(VALUES, 2)
        base = [a, b]
        source = (base * 8)[:15]
        answer = base[len(source) % len(base)]
    elif pattern == "abcabc":
        base = rng.sample(VALUES, 3)
        source = (base * 6)[:17]
        answer = base[len(source) % len(base)]
    elif pattern == "numeric":
        start = rng.randrange(32)
        step = rng.randrange(1, 5)
        source = [VALUES[(start + index * step) % 32] for index in range(15)]
        answer = VALUES[(start + len(source) * step) % 32]
    elif pattern == "nested":
        a, b = rng.sample(VALUES, 2)
        base = [a, b, b, a]
        source = (base * 5)[:19]
        answer = base[len(source) % len(base)]
    else:
        raise KeyError(pattern)
    # A task-irrelevant nonce makes exact train/test examples disjoint even for
    # finite pattern families such as 32 starts x 4 steps. The answer never
    # depends on these nonce tokens.
    sequence = [
        TASK_TOKEN[f"pattern_{pattern}"], *filler(rng, 3), *source, QUERY, ANSWER
    ]
    return sequence, answer, {
        "difficulty": pattern,
        "required_context_distance": len(source),
    }


def conditioned_example(rng: random.Random) -> tuple[list[int], int, dict]:
    condition_index = rng.randrange(4)
    condition = KEYS[condition_index]
    answer = VALUES[condition_index]
    sequence = [
        TASK_TOKEN["context_conditioned"], condition,
        *filler(rng, 28), QUERY, ANSWER,
    ]
    return sequence, answer, {
        "difficulty": "correct_context",
        "required_context_distance": len(sequence) - 2,
        "condition_index": 1,
    }


def curriculum(progress: float, task: str, rng: random.Random):
    if task == "copy":
        lengths = [4]
        if progress >= .10:
            lengths.append(8)
        if progress >= .25:
            lengths.append(16)
        if progress >= .50:
            lengths.append(32)
        if progress >= .75:
            lengths.append(64)
        length = rng.choice(lengths)
        return copy_example(rng, length)
    if task == "key_lookup":
        pairs = [2]
        if progress >= .10:
            pairs.append(4)
        if progress >= .25:
            pairs.append(8)
        if progress >= .75:
            pairs.append(16)
        distances = ["short"]
        if progress >= .25:
            distances.append("medium")
        if progress >= .50:
            distances.append("long")
        return key_lookup_example(rng, rng.choice(pairs), rng.choice(distances))
    if task == "long_range":
        return long_range_example(rng)
    if task == "pattern":
        patterns = ["abab"]
        if progress >= .10:
            patterns.append("abcabc")
        if progress >= .25:
            patterns.append("numeric")
        if progress >= .50:
            patterns.append("nested")
        return pattern_example(rng, rng.choice(patterns))
    if task == "context_conditioned":
        return conditioned_example(rng)
    raise KeyError(task)


def make_batch(examples: list[tuple[list[int], int, dict]]):
    lengths = {len(sequence) for sequence, _, _ in examples}
    if len(lengths) != 1:
        raise RuntimeError("synthetic update must use one fixed sequence length")
    inputs = torch.tensor([sequence for sequence, _, _ in examples], dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    targets[:, -1] = torch.tensor([answer for _, answer, _ in examples])
    return inputs, targets


@torch.inference_mode()
def accuracy_for_examples(
    model: DiagnosticTransformer, examples: list[tuple[list[int], int, dict]]
) -> float:
    model.eval()
    correct = 0
    batch_size = 32
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        inputs, _ = make_batch(batch)
        logits, _ = model(inputs)
        predictions = logits[:, -1].argmax(-1).tolist()
        correct += sum(
            prediction == answer
            for prediction, (_, answer, _) in zip(predictions, batch)
        )
    return correct / len(examples)


def fixed_examples(
    maker, count: int, seed: int, *args, excluded_hashes: set[str] | None = None
) -> list[tuple[list[int], int, dict]]:
    rng = random.Random(seed)
    excluded = excluded_hashes or set()
    examples = []
    hashes = set()
    attempts = 0
    while len(examples) < count:
        candidate = maker(rng, *args)
        digest = example_hash(candidate[0], candidate[1])
        attempts += 1
        if digest in excluded or digest in hashes:
            if attempts > count * 1000:
                raise RuntimeError("unable to construct leakage-free synthetic test split")
            continue
        hashes.add(digest)
        examples.append(candidate)
    return examples


@torch.inference_mode()
def evaluate_suite(
    model: DiagnosticTransformer,
    examples_per_cell: int,
    seed: int,
    training_hashes: set[str],
    full: bool,
) -> dict:
    copy_lengths = [4, 8, 16] + ([32, 64] if full else [])
    key_pairs = [2, 4, 8] + ([16] if full else [])
    patterns = ["abab", "abcabc", "numeric"] + (["nested"] if full else [])
    test_hashes = set()
    sequence_lengths = []
    context_distances = []

    def track(examples):
        for sequence, answer, metadata in examples:
            test_hashes.add(example_hash(sequence, answer))
            sequence_lengths.append(len(sequence))
            context_distances.append(metadata["required_context_distance"])
        return examples

    copy = {}
    for index, length in enumerate(copy_lengths):
        examples = track(fixed_examples(
            copy_example, examples_per_cell, seed + 1000 + index, length,
            excluded_hashes=training_hashes,
        ))
        copy[str(length)] = accuracy_for_examples(model, examples)
    lookup = {}
    for pair_index, pairs in enumerate(key_pairs):
        lookup[str(pairs)] = {}
        for distance_index, distance in enumerate(("short", "medium", "long")):
            examples = track(fixed_examples(
                key_lookup_example, examples_per_cell,
                seed + 2000 + pair_index * 10 + distance_index,
                pairs, distance,
                excluded_hashes=training_hashes,
            ))
            lookup[str(pairs)][distance] = accuracy_for_examples(model, examples)
    long_examples = track(fixed_examples(
        long_range_example, examples_per_cell, seed + 3000
        , excluded_hashes=training_hashes
    ))
    long_range = accuracy_for_examples(model, long_examples)
    pattern = {}
    for index, name in enumerate(patterns):
        examples = track(fixed_examples(
            pattern_example, examples_per_cell, seed + 4000 + index, name
            , excluded_hashes=training_hashes
        ))
        pattern[name] = accuracy_for_examples(model, examples)

    correct_examples = track(fixed_examples(
        conditioned_example, examples_per_cell, seed + 5000
        , excluded_hashes=training_hashes
    ))
    shuffled_examples = []
    removed_examples = []
    for index, (sequence, answer, metadata) in enumerate(correct_examples):
        shuffled = list(sequence)
        original = shuffled[metadata["condition_index"]]
        shuffled[metadata["condition_index"]] = KEYS[
            (KEYS.index(original) + 1 + index % 3) % 4
        ]
        shuffled_examples.append((shuffled, answer, dict(metadata, difficulty="shuffled_context")))
        removed = list(sequence)
        removed[metadata["condition_index"]] = REMOVED_CONTEXT
        removed_examples.append((removed, answer, dict(metadata, difficulty="removed_context")))
    conditioned = {
        "correct": accuracy_for_examples(model, correct_examples),
        "shuffled": accuracy_for_examples(model, shuffled_examples),
        "removed": accuracy_for_examples(model, removed_examples),
    }
    gate_cells = [copy[str(length)] for length in (4, 8, 16)]
    gate_cells += [
        lookup[str(pairs)][distance]
        for pairs in (2, 4, 8)
        for distance in ("short", "medium", "long")
    ]
    gate_cells += [long_range]
    gate_cells += [pattern[name] for name in ("abab", "abcabc", "numeric")]
    gate_cells += [conditioned["correct"]]
    gate_accuracy = min(gate_cells)
    control_advantage = conditioned["correct"] - max(
        conditioned["shuffled"], conditioned["removed"]
    )
    return {
        "examples_per_cell": examples_per_cell,
        "copy": copy,
        "key_lookup": lookup,
        "long_range": long_range,
        "pattern": pattern,
        "context_conditioned": conditioned,
        "minimum_gate_cell_accuracy": gate_accuracy,
        "context_control_advantage": control_advantage,
        "gate_pass": gate_accuracy >= .90 and control_advantage >= .40,
        "test_hashes": len(test_hashes),
        "exact_train_test_overlap": len(test_hashes & training_hashes),
        "sequence_length": {
            "minimum": min(sequence_lengths),
            "maximum": max(sequence_lengths),
        },
        "required_context_distance": {
            "minimum": min(context_distances),
            "maximum": max(context_distances),
            "mean": sum(context_distances) / len(context_distances),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v16.json")
    parser.add_argument("--variant", choices=["current_unscaled", "sqrt_scaled_a"], required=True)
    parser.add_argument("--output-dir", default="checkpoints/foundation-v16-synthetic")
    args = parser.parse_args()
    settings = load_json(args.config)
    synthetic = settings["synthetic_v2"]
    variant = next(row for row in settings["variants"] if row["name"] == args.variant)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.variant}.json"
    checkpoint_path = output_dir / f"{args.variant}.pt"
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite synthetic v2: {args.variant}")

    seed = 2716
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(settings["cpu_threads"]))
    architecture = dict(settings["architecture"])
    architecture.update({
        "context_length": int(synthetic["context_length"]),
        "scale_token_embedding": variant["scale_token_embedding"],
    })
    model = DiagnosticTransformer(DiagnosticConfig(
        model_name=f"Foundation v1.6 Synthetic v2 {args.variant}",
        vocab_size=int(synthetic["vocab_size"]),
        **architecture,
    ))
    optimizer = create_optimizer(
        model, float(synthetic["learning_rate"]), float(synthetic["weight_decay"])
    )
    updates = int(synthetic["updates"])
    batch_size = int(synthetic["batch_size"])
    curve_updates = {
        max(1, round(updates * percentage / 100)): percentage
        for percentage in synthetic["curve_percentages"]
    }
    training_rng = random.Random(seed + 1)
    training_hashes: set[str] = set()
    training_examples = Counter()
    training_lengths = []
    training_distances = []
    curve = []
    losses = []
    started = time.perf_counter()
    for update in range(1, updates + 1):
        task = TASKS[(update - 1) % len(TASKS)]
        progress = update / updates
        prototype = curriculum(progress, task, training_rng)
        difficulty = prototype[2]["difficulty"]
        examples = [prototype]
        while len(examples) < batch_size:
            candidate = curriculum(progress, task, training_rng)
            if candidate[2]["difficulty"] == difficulty:
                examples.append(candidate)
        for sequence, answer, metadata in examples:
            training_hashes.add(example_hash(sequence, answer))
            training_examples[f"{task}:{metadata['difficulty']}"] += 1
            training_lengths.append(len(sequence))
            training_distances.append(metadata["required_context_distance"])
        inputs, targets = make_batch(examples)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite synthetic v2 loss: {args.variant}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        losses.append(float(loss.item()))
        if update in curve_updates:
            evaluated = evaluate_suite(
                model, 64, seed + 100_000, training_hashes, full=False
            )
            row = {
                "update": update,
                "training_percent": curve_updates[update],
                "recent_loss": sum(losses[-max(1, updates // 10):])
                / min(len(losses), max(1, updates // 10)),
                "gradient_norm": gradient_norm,
                "evaluation": evaluated,
            }
            curve.append(row)
            print(json.dumps({
                "variant": args.variant,
                "update": update,
                "percent": curve_updates[update],
                "minimum_gate_accuracy": evaluated["minimum_gate_cell_accuracy"],
                "context_advantage": evaluated["context_control_advantage"],
                "gate_pass": evaluated["gate_pass"],
            }), flush=True)
    final = evaluate_suite(
        model, int(synthetic["evaluation_examples"]), seed + 200_000,
        training_hashes, full=True,
    )
    if final["exact_train_test_overlap"]:
        raise RuntimeError("synthetic v2 train/test leakage detected")
    payload = {
        "checkpoint_format": "foundation-v16-synthetic-v2-v1",
        "model_state": model.state_dict(),
        "config": model.config.to_dict(),
        "variant": variant,
        "updates": updates,
        "seed": seed,
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
        "schema_version": "foundation-v16-synthetic-context-v2",
        "variant": variant,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "training": {
            "seed": seed,
            "updates": updates,
            "batch_size": batch_size,
            "train_examples": updates * batch_size,
            "unique_train_hashes": len(training_hashes),
            "examples_by_difficulty": dict(sorted(training_examples.items())),
            "sequence_length": {
                "minimum": min(training_lengths),
                "maximum": max(training_lengths),
            },
            "required_context_distance": {
                "minimum": min(training_distances),
                "maximum": max(training_distances),
                "mean": sum(training_distances) / len(training_distances),
            },
            "learning_rate": synthetic["learning_rate"],
            "weight_decay": synthetic["weight_decay"],
            "dropout_unchanged": settings["architecture"]["dropout"],
            "curve": curve,
            "wall_seconds": time.perf_counter() - started,
        },
        "dataset_audit": {
            "vocabulary_size": synthetic["vocab_size"],
            "value_tokens": VALUES,
            "key_tokens": KEYS,
            "index_tokens": INDEX_TOKENS,
            "filler_tokens": FILLERS,
            "last_input_token_constant": ANSWER,
            "chance_baselines": {
                "value_vocabulary": 1 / len(VALUES),
                "context_conditioned_four_way": .25,
                "copy_query_ignored": {
                    str(length): 1 / length for length in synthetic["copy_lengths"]
                },
                "key_query_ignored": {
                    str(pairs): 1 / pairs for pairs in synthetic["key_pairs"]
                },
            },
            "train_generation_seed": seed + 1,
            "test_generation_seed": seed + 200_000,
            "train_test_exact_overlap": final["exact_train_test_overlap"],
            "external_data": False,
            "external_ai": False,
        },
        "final": final,
        "synthetic_gate_v2": "PASS" if final["gate_pass"] else "FAIL",
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
        "synthetic_gate_v2": report["synthetic_gate_v2"],
        "minimum_gate_accuracy": final["minimum_gate_cell_accuracy"],
        "context_conditioned": final["context_conditioned"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
