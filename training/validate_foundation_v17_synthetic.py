from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
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

from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.optimizer import create_optimizer
from training.validate_foundation_v16_synthetic import (
    ANSWER,
    INDEX_TOKENS,
    QUERY,
    TASKS,
    TASK_TOKEN,
    VALUES,
    copy_example,
    curriculum,
    evaluate_suite,
    example_hash,
    fixed_examples,
    make_batch,
    pattern_example,
)


POSITION_TASK = "position"
POSITION_TASK_TOKEN = 9
POSITION_ORDINALS = list(range(80, 112))
TASKS_V17 = (*TASKS, POSITION_TASK)


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def variant_by_name(settings: dict, name: str) -> dict:
    return next(row for row in settings["variants"] if row["name"] == name)


def position_example(rng: random.Random, length: int) -> tuple[list[int], int, dict]:
    values = rng.sample(VALUES, length)
    queried_position = rng.randrange(length)
    sequence = [
        POSITION_TASK_TOKEN,
        *values,
        QUERY,
        POSITION_ORDINALS[queried_position],
        ANSWER,
    ]
    return sequence, values[queried_position], {
        "difficulty": f"length_{length}",
        "queried_position": queried_position,
        "required_context_distance": len(sequence) - 1 - (1 + queried_position),
    }


def position_curriculum(progress: float, rng: random.Random):
    lengths = [4]
    if progress >= .10:
        lengths.append(8)
    if progress >= .25:
        lengths.append(16)
    return position_example(rng, rng.choice(lengths))


@torch.inference_mode()
def position_evaluation(
    model: DiagnosticTransformerV17,
    examples_per_length: int,
    seed: int,
    training_hashes: set[str],
) -> dict:
    by_length = {}
    test_hashes = set()
    for index, length in enumerate((4, 8, 16)):
        examples = fixed_examples(
            position_example,
            examples_per_length,
            seed + index,
            length,
            excluded_hashes=training_hashes,
        )
        inputs, _ = make_batch(examples)
        correct = 0
        for start in range(0, len(examples), 32):
            batch = examples[start:start + 32]
            x, _ = make_batch(batch)
            logits, _ = model(x)
            predictions = logits[:, -1].argmax(-1).tolist()
            correct += sum(
                predicted == answer
                for predicted, (_, answer, _) in zip(predictions, batch)
            )
        for sequence, answer, _ in examples:
            test_hashes.add(example_hash(sequence, answer))
        by_length[str(length)] = correct / len(examples)
    return {
        "by_length": by_length,
        "macro_accuracy": sum(by_length.values()) / len(by_length),
        "minimum_accuracy": min(by_length.values()),
        "test_hashes": len(test_hashes),
        "exact_train_test_overlap": len(test_hashes & training_hashes),
        "chance_baseline": {str(length): 1 / length for length in (4, 8, 16)},
    }


def phase28_gate(base: dict, position: dict) -> dict:
    checks = {
        "copy_4": base["copy"]["4"] >= .95,
        "copy_8": base["copy"]["8"] >= .95,
        "copy_16": base["copy"]["16"] >= .90,
        "key_lookup_2": min(base["key_lookup"]["2"].values()) >= .95,
        "key_lookup_4": min(base["key_lookup"]["4"].values()) >= .95,
        "key_lookup_8": min(base["key_lookup"]["8"].values()) >= .90,
        "long_range": base["long_range"] >= .95,
        "basic_pattern": min(
            base["pattern"][name] for name in ("abab", "abcabc", "nested")
        ) >= .95,
        "numeric_pattern": base["pattern"]["numeric"] >= .90,
        "context_conditioned": base["context_conditioned"]["correct"] >= .95,
        "context_control_advantage": (
            base["context_conditioned"]["correct"]
            - max(
                base["context_conditioned"]["shuffled"],
                base["context_conditioned"]["removed"],
            )
        ) >= .50,
        "position_task": position["minimum_accuracy"] >= .95,
    }
    return {"checks": checks, "pass": all(checks.values())}


@torch.inference_mode()
def copy_failure_analysis(model: DiagnosticTransformerV17, seed: int) -> dict:
    rows = []
    classes = Counter()
    offsets = Counter()
    for length_index, length in enumerate((4, 8, 16)):
        examples = fixed_examples(copy_example, 32, seed + length_index, length)
        for sequence, expected, _ in examples:
            inputs = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
            logits, _ = model(inputs)
            top5 = logits[0, -1].topk(5).indices.tolist()
            predicted = top5[0]
            values = sequence[1:1 + length]
            query_position = INDEX_TOKENS.index(sequence[-2])
            if predicted == expected:
                classification = "correct"
            elif predicted in values:
                predicted_position = values.index(predicted)
                offset = predicted_position - query_position
                offsets[str(offset)] += 1
                classification = "position_shift"
            elif predicted == sequence[-2]:
                classification = "previous_token_repetition"
            elif predicted in VALUES:
                classification = "value_frequency_substitution"
            else:
                classification = "non_value_fixed_substitution"
            classes[classification] += 1
            rows.append({
                "length": length,
                "position": query_position,
                "expected": expected,
                "predicted": predicted,
                "top_5": top5,
                "classification": classification,
            })
    return {
        "examples": len(rows),
        "all_token_predictions": rows,
        "classification_counts": dict(sorted(classes.items())),
        "position_shift_offsets": dict(sorted(offsets.items(), key=lambda row: int(row[0]))),
    }


@torch.inference_mode()
def numeric_failure_analysis(model: DiagnosticTransformerV17, seed: int) -> dict:
    examples = fixed_examples(pattern_example, 256, seed, "numeric")
    rows = []
    classes = Counter()
    for sequence, expected, _ in examples:
        inputs = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
        logits, _ = model(inputs)
        top5 = logits[0, -1].topk(5).indices.tolist()
        predicted = top5[0]
        source = sequence[4:-2]
        if predicted == expected:
            classification = "correct"
        elif predicted in source:
            classification = "seen_pattern_token_wrong_phase"
        elif predicted in VALUES:
            classification = "value_token_outside_pattern"
        else:
            classification = "non_value_token"
        classes[classification] += 1
        rows.append({
            "expected": expected,
            "predicted": predicted,
            "top_5": top5,
            "classification": classification,
        })
    return {
        "examples": len(rows),
        "accuracy": classes["correct"] / len(rows),
        "classification_counts": dict(sorted(classes.items())),
        "sample_predictions": rows[:64],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v17.json")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="checkpoints/foundation-v17-synthetic")
    args = parser.parse_args()
    settings = load_json(args.config)
    synthetic = settings["synthetic_v3"]
    variant = variant_by_name(settings, args.variant)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}-seed-{args.seed}"
    result_path = output_dir / f"{stem}.json"
    checkpoint_path = output_dir / f"{stem}.pt"
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite v1.7 synthetic: {stem}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(settings["cpu_threads"]))
    architecture = dict(settings["architecture"])
    architecture["context_length"] = int(synthetic["context_length"])
    model = DiagnosticTransformerV17(DiagnosticConfigV17(
        model_name=f"Foundation v1.7 Synthetic v3 {args.variant} seed {args.seed}",
        vocab_size=int(synthetic["vocab_size"]),
        token_embedding_scale=variant["token_embedding_scale"],
        position_embedding_scale=variant["position_embedding_scale"],
        residual_projection_init_scale=variant["residual_projection_init_scale"],
        **architecture,
    ))
    optimizer = create_optimizer(
        model, float(synthetic["learning_rate"]), float(synthetic["weight_decay"])
    )
    updates = int(synthetic["updates"])
    batch_size = int(synthetic["batch_size"])
    curve_updates = {
        round(updates * percentage / 100): percentage
        for percentage in synthetic["curve_percentages"]
    }
    rng = random.Random(args.seed + 1)
    training_hashes = set()
    examples_by_difficulty = Counter()
    training_lengths = []
    curve = []
    losses = []
    input_tokens = 0
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss / 1024**2
    started = time.perf_counter()
    for update in range(1, updates + 1):
        task = TASKS_V17[(update - 1) % len(TASKS_V17)]
        progress = update / updates
        prototype = (
            position_curriculum(progress, rng)
            if task == POSITION_TASK
            else curriculum(progress, task, rng)
        )
        difficulty = prototype[2]["difficulty"]
        examples = [prototype]
        while len(examples) < batch_size:
            candidate = (
                position_curriculum(progress, rng)
                if task == POSITION_TASK
                else curriculum(progress, task, rng)
            )
            if candidate[2]["difficulty"] == difficulty:
                examples.append(candidate)
        for sequence, answer, metadata in examples:
            training_hashes.add(example_hash(sequence, answer))
            examples_by_difficulty[f"{task}:{metadata['difficulty']}"] += 1
            training_lengths.append(len(sequence))
            input_tokens += len(sequence)
        inputs, targets = make_batch(examples)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite v1.7 synthetic loss: {stem}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        losses.append(float(loss.item()))
        peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
        if update in curve_updates:
            base = evaluate_suite(
                model, 64, args.seed + 100_000, training_hashes, full=True
            )
            position = position_evaluation(
                model, 64, args.seed + 110_000, training_hashes
            )
            gate = phase28_gate(base, position)
            curve.append({
                "update": update,
                "training_percent": curve_updates[update],
                "recent_loss": sum(losses[-200:]) / min(len(losses), 200),
                "gradient_norm": gradient_norm,
                "base": base,
                "position": position,
                "phase28_gate": gate,
            })
            print(json.dumps({
                "variant": args.variant,
                "seed": args.seed,
                "update": update,
                "copy": base["copy"],
                "numeric": base["pattern"]["numeric"],
                "position": position["by_length"],
                "gate": gate["pass"],
            }), flush=True)

    base = evaluate_suite(
        model,
        int(synthetic["evaluation_examples"]),
        args.seed + 200_000,
        training_hashes,
        full=True,
    )
    position = position_evaluation(
        model,
        int(synthetic["evaluation_examples"]),
        args.seed + 210_000,
        training_hashes,
    )
    if base["exact_train_test_overlap"] or position["exact_train_test_overlap"]:
        raise RuntimeError("v1.7 synthetic train/test leakage detected")
    gate = phase28_gate(base, position)
    copy_failures = copy_failure_analysis(model, args.seed + 300_000)
    numeric_failures = numeric_failure_analysis(model, args.seed + 310_000)
    elapsed = time.perf_counter() - started
    payload = {
        "checkpoint_format": "foundation-v17-synthetic-v3-v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config.to_dict(),
        "variant": variant,
        "updates": updates,
        "seed": args.seed,
        "diagnostic_only": True,
    }
    torch.save(payload, checkpoint_path)
    restored_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = DiagnosticTransformerV17(DiagnosticConfigV17(**restored_payload["config"]))
    restored.load_state_dict(restored_payload["model_state"], strict=True)
    strict_reload = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), restored.state_dict().values())
    )
    report = {
        "schema_version": "foundation-v17-synthetic-context-v3",
        "variant": variant,
        "seed": args.seed,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "initialization": model.initialization_manifest(),
        "training": {
            "updates": updates,
            "batch_size": batch_size,
            "train_examples": updates * batch_size,
            "unique_train_hashes": len(training_hashes),
            "examples_by_difficulty": dict(sorted(examples_by_difficulty.items())),
            "sequence_length": {
                "minimum": min(training_lengths),
                "maximum": max(training_lengths),
            },
            "input_tokens": input_tokens,
            "wall_seconds": elapsed,
            "input_tokens_per_second": input_tokens / elapsed,
            "examples_per_second": updates * batch_size / elapsed,
            "peak_ram_mb": peak_ram,
            "curve": curve,
        },
        "dataset_audit": {
            "teacher_forcing": "single supervised answer at final position",
            "target_construction": "all positions -100 except final queried answer",
            "query_input_sentinel": ANSWER,
            "eos_used": False,
            "sequence_packing": False,
            "input_contains_target_answer": False,
            "copy_requires_explicit_position_index": True,
            "position_task_ordinal_tokens": POSITION_ORDINALS,
            "training_hashes": len(training_hashes),
            "base_test_overlap": base["exact_train_test_overlap"],
            "position_test_overlap": position["exact_train_test_overlap"],
        },
        "final": {
            "base": base,
            "position": position,
            "phase28_gate": gate,
        },
        "copy_failure_analysis": copy_failures,
        "numeric_failure_analysis": numeric_failures,
        "checkpoint": {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "strict_reload": strict_reload,
            "optimizer_state_present": True,
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
        "seed": args.seed,
        "gate": gate,
        "copy": base["copy"],
        "key_lookup": base["key_lookup"],
        "pattern": base["pattern"],
        "position": position,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
