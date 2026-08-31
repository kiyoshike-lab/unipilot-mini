from __future__ import annotations

import argparse
from collections import Counter
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

from evaluation.measure_foundation_v18_attention import attention_retrieval_metrics
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from foundation.reference_transformer_v18 import (
    ReferenceConfigV18,
    ReferenceTransformerV18,
)
from training.optimizer import create_optimizer
from training.validate_foundation_v16_synthetic import (
    ANSWER,
    FILLERS,
    KEYS,
    QUERY,
    REMOVED_CONTEXT,
    TASK_TOKEN,
    VALUES,
    conditioned_example,
    copy_example,
    example_hash,
    filler,
    fixed_examples,
    long_range_example,
    make_batch,
    pattern_example,
)
from training.validate_foundation_v17_synthetic import position_example


SYMBOLIC_TASK_TOKEN = 10
TASKS = ("copy", "key_lookup", "long_range", "pattern", "context_conditioned", "position")


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def model_spec(settings: dict, name: str) -> dict:
    return next(row for row in settings["models"] if row["name"] == name)


def build_model(settings: dict, spec: dict, vocab_size: int, context_length: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    architecture = dict(settings["architecture"])
    architecture["context_length"] = context_length
    common = {
        "model_name": f"Foundation v1.8 Synthetic {spec['name']}",
        "vocab_size": vocab_size,
        "residual_projection_init_scale": spec["residual_projection_init_scale"],
        **architecture,
    }
    if spec["implementation"] == "custom":
        return DiagnosticTransformerV17(DiagnosticConfigV17(
            token_embedding_scale=1.0,
            position_embedding_scale=1.0,
            norm="layernorm",
            activation="gelu",
            **common,
        ))
    reference_common = dict(common)
    reference_common.pop("weight_tying")
    reference_common.pop("bias")
    reference_common.pop("dropout")
    return ReferenceTransformerV18(ReferenceConfigV18(
        dropout=architecture["dropout"],
        bias=architecture["bias"],
        weight_tying=architecture["weight_tying"],
        **reference_common,
    ))


def key_lookup_v4_example(
    rng: random.Random, pairs: int, distance: str
) -> tuple[list[int], int, dict]:
    keys = rng.sample(KEYS, pairs)
    values = rng.sample(VALUES, pairs)
    selected = rng.randrange(pairs)
    gap = {"short": 0, "medium": 16, "long": 32}[distance]
    sequence = [TASK_TOKEN["key_lookup"]]
    key_positions = []
    value_positions = []
    for key, value in zip(keys, values):
        key_positions.append(len(sequence))
        sequence.append(key)
        value_positions.append(len(sequence))
        sequence.append(value)
    sequence.extend(filler(rng, gap))
    sequence.extend((QUERY, keys[selected], ANSWER))
    return sequence, values[selected], {
        "difficulty": f"pairs_{pairs}_{distance}",
        "pairs": pairs,
        "distance": distance,
        "correct_key_position": key_positions[selected],
        "correct_value_position": value_positions[selected],
        "all_key_positions": key_positions,
        "all_value_positions": value_positions,
        "query_position": len(sequence) - 1,
        "required_context_distance": len(sequence) - 1 - value_positions[selected],
    }


def symbolic_example(rng: random.Random) -> tuple[list[int], int, dict]:
    symbols = rng.sample(VALUES, 4)
    source = (symbols * 5)[:19]
    answer = symbols[len(source) % 4]
    sequence = [SYMBOLIC_TASK_TOKEN, *filler(rng, 3), *source, QUERY, ANSWER]
    return sequence, answer, {
        "difficulty": "symbolic",
        "required_context_distance": len(source),
        "atomic_symbol_tokens": True,
    }


def pattern_v4_example(rng: random.Random, pattern: str):
    return symbolic_example(rng) if pattern == "symbolic" else pattern_example(rng, pattern)


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
        return copy_example(rng, rng.choice(lengths))
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
        return key_lookup_v4_example(rng, rng.choice(pairs), rng.choice(distances))
    if task == "long_range":
        return long_range_example(rng)
    if task == "pattern":
        patterns = ["abab"]
        if progress >= .10:
            patterns.append("abcabc")
        if progress >= .25:
            patterns.extend(("numeric", "symbolic"))
        if progress >= .50:
            patterns.append("nested")
        return pattern_v4_example(rng, rng.choice(patterns))
    if task == "context_conditioned":
        return conditioned_example(rng)
    if task == "position":
        lengths = [4]
        if progress >= .10:
            lengths.append(8)
        if progress >= .25:
            lengths.append(16)
        return position_example(rng, rng.choice(lengths))
    raise KeyError(task)


@torch.inference_mode()
def accuracy_loss(model, examples: list[tuple]) -> dict:
    model.eval()
    correct = 0
    loss_sum = 0.0
    for start in range(0, len(examples), 32):
        batch = examples[start:start + 32]
        inputs, targets = make_batch(batch)
        logits, loss = model(inputs, targets)
        correct += sum(
            predicted == answer
            for predicted, (_, answer, _) in zip(
                logits[:, -1].argmax(-1).tolist(), batch
            )
        )
        loss_sum += float(loss) * len(batch)
    return {"accuracy": correct / len(examples), "loss": loss_sum / len(examples)}


def _fixed(maker, count, seed, *args, training_hashes):
    return fixed_examples(
        maker, count, seed, *args, excluded_hashes=training_hashes
    )


@torch.inference_mode()
def evaluate_suite(model, count: int, seed: int, training_hashes: set[str]) -> dict:
    all_examples = []
    copy = {}
    for index, length in enumerate((4, 8, 16, 32, 64)):
        examples = _fixed(
            copy_example, count, seed + 1000 + index, length,
            training_hashes=training_hashes,
        )
        all_examples.extend(examples)
        copy[str(length)] = accuracy_loss(model, examples)
    lookup = {}
    lookup_examples = {}
    for pair_index, pairs in enumerate((2, 4, 8, 16)):
        lookup[str(pairs)] = {}
        lookup_examples[str(pairs)] = {}
        for distance_index, distance in enumerate(("short", "medium", "long")):
            examples = _fixed(
                key_lookup_v4_example,
                count,
                seed + 2000 + pair_index * 10 + distance_index,
                pairs,
                distance,
                training_hashes=training_hashes,
            )
            all_examples.extend(examples)
            lookup_examples[str(pairs)][distance] = examples
            lookup[str(pairs)][distance] = accuracy_loss(model, examples)
    long_examples = _fixed(
        long_range_example, count, seed + 3000, training_hashes=training_hashes
    )
    all_examples.extend(long_examples)
    patterns = {}
    for index, name in enumerate(("abab", "abcabc", "symbolic", "numeric", "nested")):
        examples = _fixed(
            pattern_v4_example,
            count,
            seed + 4000 + index,
            name,
            training_hashes=training_hashes,
        )
        all_examples.extend(examples)
        patterns[name] = accuracy_loss(model, examples)
    correct_examples = _fixed(
        conditioned_example, count, seed + 5000, training_hashes=training_hashes
    )
    all_examples.extend(correct_examples)
    shuffled_examples = []
    removed_examples = []
    for index, (sequence, answer, metadata) in enumerate(correct_examples):
        shuffled = list(sequence)
        original = shuffled[metadata["condition_index"]]
        shuffled[metadata["condition_index"]] = KEYS[
            (KEYS.index(original) + 1 + index % 3) % 4
        ]
        shuffled_examples.append((shuffled, answer, metadata))
        removed = list(sequence)
        removed[metadata["condition_index"]] = REMOVED_CONTEXT
        removed_examples.append((removed, answer, metadata))
    context = {
        "correct": accuracy_loss(model, correct_examples),
        "shuffled": accuracy_loss(model, shuffled_examples),
        "removed": accuracy_loss(model, removed_examples),
    }
    position = {}
    for index, length in enumerate((4, 8, 16)):
        examples = _fixed(
            position_example,
            count,
            seed + 6000 + index,
            length,
            training_hashes=training_hashes,
        )
        all_examples.extend(examples)
        position[str(length)] = accuracy_loss(model, examples)
    test_hashes = {example_hash(row[0], row[1]) for row in all_examples}
    answer_leakage = sum(
        int(sequence[-1] == answer or sequence[-2] == answer)
        for sequence, answer, _ in all_examples
    )
    return {
        "examples_per_cell": count,
        "copy": copy,
        "key_lookup": lookup,
        "long_range": accuracy_loss(model, long_examples),
        "pattern": patterns,
        "context_conditioned": context,
        "position": position,
        "test_hashes": len(test_hashes),
        "exact_train_test_overlap": len(test_hashes & training_hashes),
        "answer_leakage_in_query_suffix": answer_leakage,
        "lookup_examples": lookup_examples,
    }


def gate(settings: dict, evaluated: dict) -> dict:
    thresholds = settings["gates"]
    checks = {}
    for length, minimum in thresholds["copy"].items():
        checks[f"copy_{length}"] = evaluated["copy"][length]["accuracy"] >= minimum
    for pairs, minimum in thresholds["key_lookup"].items():
        checks[f"key_{pairs}"] = min(
            row["accuracy"] for row in evaluated["key_lookup"][pairs].values()
        ) >= minimum
    checks["long_range"] = evaluated["long_range"]["accuracy"] >= thresholds["long_range"]
    for pattern, minimum in thresholds["patterns"].items():
        checks[f"pattern_{pattern}"] = evaluated["pattern"][pattern]["accuracy"] >= minimum
    checks["context"] = (
        evaluated["context_conditioned"]["correct"]["accuracy"]
        >= thresholds["context_conditioned"]
    )
    checks["context_control"] = (
        evaluated["context_conditioned"]["correct"]["accuracy"]
        - max(
            evaluated["context_conditioned"]["shuffled"]["accuracy"],
            evaluated["context_conditioned"]["removed"]["accuracy"],
        ) >= .50
    )
    checks["position"] = min(
        row["accuracy"] for row in evaluated["position"].values()
    ) >= thresholds["position"]
    return {"checks": checks, "pass": all(checks.values())}


def attention_curve(model, implementation: str, evaluated: dict) -> dict:
    representative = evaluated["lookup_examples"]["4"]["medium"][:16]
    return attention_retrieval_metrics(model, implementation, representative)


def strip_examples(evaluated: dict) -> dict:
    return {key: value for key, value in evaluated.items() if key != "lookup_examples"}


def pilot_task(rng: random.Random, update: int):
    task = ("key_lookup", "numeric", "symbolic")[(update - 1) % 3]
    if task == "key_lookup":
        pairs = (2, 4, 8)[((update - 1) // 3) % 3]
        distance = ("short", "medium", "long")[((update - 1) // 9) % 3]
        return task, key_lookup_v4_example(rng, pairs, distance)
    return task, pattern_v4_example(rng, task)


@torch.inference_mode()
def evaluate_pilot(model, seed: int, training_hashes: set[str]) -> dict:
    key = {}
    for pairs in (2, 4, 8):
        key[str(pairs)] = {}
        for index, distance in enumerate(("short", "medium", "long")):
            examples = _fixed(
                key_lookup_v4_example,
                64,
                seed + pairs * 10 + index,
                pairs,
                distance,
                training_hashes=training_hashes,
            )
            key[str(pairs)][distance] = accuracy_loss(model, examples)
    numeric = accuracy_loss(model, _fixed(
        pattern_v4_example, 128, seed + 1000, "numeric",
        training_hashes=training_hashes,
    ))
    symbolic = accuracy_loss(model, _fixed(
        pattern_v4_example, 128, seed + 2000, "symbolic",
        training_hashes=training_hashes,
    ))
    chance_normalized = []
    for pairs in (2, 4, 8):
        chance = 1 / pairs
        for row in key[str(pairs)].values():
            chance_normalized.append(max(0.0, (row["accuracy"] - chance) / (1 - chance)))
    chance_normalized.extend((numeric["accuracy"], symbolic["accuracy"]))
    return {
        "key_lookup": key,
        "numeric": numeric,
        "symbolic": symbolic,
        "normalized_score": sum(chance_normalized) / len(chance_normalized),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v18.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    settings = load_json(args.config)
    spec = model_spec(settings, args.model)
    synthetic = settings["synthetic_v4"]
    pilot = settings["synthetic_lr_pilot"]
    updates = int(pilot["updates"] if args.pilot else synthetic["maximum_updates"])
    batch_size = int(pilot["batch_size"] if args.pilot else synthetic["batch_size"])
    default_dir = (
        "checkpoints/foundation-v18-lr-pilot"
        if args.pilot else "checkpoints/foundation-v18-synthetic"
    )
    output_dir = ROOT / (args.output_dir or default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_name = f"{args.learning_rate:.0e}".replace("-", "m")
    stem = f"{args.model}-lr-{lr_name}"
    result_path = output_dir / f"{stem}.json"
    checkpoint_path = output_dir / f"{stem}.pt"
    resume_path = output_dir / f"{stem}.resume.pt"
    if result_path.exists() or (not args.pilot and checkpoint_path.exists()):
        raise RuntimeError(f"refusing to overwrite v1.8 synthetic run: {stem}")
    if args.resume and args.pilot:
        raise RuntimeError("LR pilots are intentionally too short to resume")
    if args.resume and not resume_path.exists():
        raise RuntimeError(f"resume checkpoint does not exist: {resume_path}")
    if not args.resume and resume_path.exists():
        raise RuntimeError(f"resume checkpoint already exists; pass --resume: {resume_path}")
    seed = int(settings["seed"])
    torch.set_num_threads(int(settings["cpu_threads"]))
    model = build_model(
        settings,
        spec,
        int(synthetic["vocab_size"]),
        int(synthetic["context_length"]),
        seed,
    )
    optimizer = create_optimizer(model, args.learning_rate, float(synthetic["weight_decay"]))
    rng = random.Random(seed + 1)
    training_hashes = set()
    training_templates = Counter()
    training_examples = Counter()
    curve = []
    losses = []
    processed_tokens = 0
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss / 1024**2
    started = time.perf_counter()
    elapsed_before_resume = 0.0

    if args.resume:
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume["model_name"] != args.model or resume["learning_rate"] != args.learning_rate:
            raise RuntimeError("resume model/LR does not match command")
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        rng.setstate(resume["python_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        training_hashes = set(resume["training_hashes"])
        training_templates = Counter(resume["training_templates"])
        training_examples = Counter(resume["training_examples"])
        curve = resume["curve"]
        losses = resume["losses"]
        processed_tokens = int(resume["processed_tokens"])
        peak_ram = max(peak_ram, float(resume["peak_ram_mb"]))
        elapsed_before_resume = float(resume["elapsed_seconds"])
        completed_updates = int(resume["update"])
        start_update = completed_updates + 1
    else:
        completed_updates = 0
        start_update = 1

    if args.pilot:
        milestones = {updates}
    elif not args.resume:
        milestones = set(int(value) for value in synthetic["milestones"])
        initial = evaluate_suite(model, 32, seed + 100_000, training_hashes)
        curve.append({
            "update": 0,
            "percent_of_phase28_budget": 0,
            "processed_input_tokens": 0,
            "evaluation": strip_examples(initial),
            "gate": gate(settings, initial),
            "attention": attention_curve(model, spec["implementation"], initial),
        })
    else:
        milestones = set(int(value) for value in synthetic["milestones"])

    stopped_early = False
    for update in range(start_update, updates + 1):
        if args.pilot:
            task, prototype = pilot_task(rng, update)
        else:
            task = TASKS[(update - 1) % len(TASKS)]
            progress = min(1.0, update / int(synthetic["base_updates"]))
            prototype = curriculum(progress, task, rng)
        difficulty = prototype[2]["difficulty"]
        examples = [prototype]
        while len(examples) < batch_size:
            if args.pilot:
                candidate_task, candidate = pilot_task(rng, update)
                if candidate_task != task:
                    raise RuntimeError("pilot task schedule changed within update")
            else:
                candidate = curriculum(progress, task, rng)
            if candidate[2]["difficulty"] == difficulty:
                examples.append(candidate)
        for sequence, answer, metadata in examples:
            training_hashes.add(example_hash(sequence, answer))
            training_examples[f"{task}:{difficulty}"] += 1
            training_templates[
                f"{task}:{difficulty}:length_{len(sequence)}"
            ] += 1
            processed_tokens += len(sequence)
        inputs, targets = make_batch(examples)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite v1.8 synthetic loss: {stem}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(synthetic["gradient_clip"])
        ))
        optimizer.step()
        losses.append(float(loss.detach()))
        completed_updates = update
        peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
        if update in milestones:
            if args.pilot:
                evaluated_pilot = evaluate_pilot(model, seed + 200_000, training_hashes)
                curve.append({
                    "update": update,
                    "recent_loss": sum(losses[-100:]) / min(len(losses), 100),
                    "gradient_norm": gradient_norm,
                    "evaluation": evaluated_pilot,
                })
                print(json.dumps({
                    "model": args.model,
                    "learning_rate": args.learning_rate,
                    "pilot": evaluated_pilot,
                }), flush=True)
            else:
                evaluated = evaluate_suite(model, 64, seed + 300_000, training_hashes)
                checked = gate(settings, evaluated)
                percentage = round(100 * update / int(synthetic["base_updates"]))
                curve.append({
                    "update": update,
                    "percent_of_phase28_budget": percentage,
                    "processed_input_tokens": processed_tokens,
                    "recent_loss": sum(losses[-200:]) / min(len(losses), 200),
                    "gradient_norm": gradient_norm,
                    "evaluation": strip_examples(evaluated),
                    "gate": checked,
                    "attention": attention_curve(model, spec["implementation"], evaluated),
                })
                print(json.dumps({
                    "model": args.model,
                    "update": update,
                    "percent": percentage,
                    "key": evaluated["key_lookup"],
                    "numeric": evaluated["pattern"]["numeric"],
                    "symbolic": evaluated["pattern"]["symbolic"],
                    "gate": checked["pass"],
                }), flush=True)
                if checked["pass"]:
                    stopped_early = True
        if not args.pilot and (
            update % int(synthetic["resume_checkpoint_interval"]) == 0
            or update in milestones
        ):
            resume_payload = {
                "checkpoint_format": "foundation-v18-synthetic-v4-resume",
                "model_name": args.model,
                "learning_rate": args.learning_rate,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "python_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "training_hashes": sorted(training_hashes),
                "training_templates": dict(training_templates),
                "training_examples": dict(training_examples),
                "curve": curve,
                "losses": losses,
                "processed_tokens": processed_tokens,
                "peak_ram_mb": peak_ram,
                "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
                "update": update,
            }
            temporary_resume = resume_path.with_suffix(".tmp")
            torch.save(resume_payload, temporary_resume)
            os.replace(temporary_resume, resume_path)
        if stopped_early:
            break

    elapsed = elapsed_before_resume + time.perf_counter() - started
    if args.pilot:
        final_evaluation = curve[-1]["evaluation"]
        final_gate = None
        final_attention = None
        final_suite = None
    else:
        final_suite = evaluate_suite(
            model,
            int(synthetic["evaluation_examples"]),
            seed + 400_000,
            training_hashes,
        )
        final_gate = gate(settings, final_suite)
        final_attention = {
            pairs: {
                distance: attention_retrieval_metrics(
                    model,
                    spec["implementation"],
                    examples[:32],
                )
                for distance, examples in distances.items()
            }
            for pairs, distances in final_suite["lookup_examples"].items()
        }
        final_evaluation = strip_examples(final_suite)

    payload = {
        "checkpoint_format": "foundation-v18-synthetic-v4-pilot" if args.pilot else "foundation-v18-synthetic-v4",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config.to_dict(),
        "model_spec": spec,
        "learning_rate": args.learning_rate,
        "update": completed_updates,
        "diagnostic_only": True,
    }
    if not args.pilot:
        torch.save(payload, checkpoint_path)
    if spec["implementation"] == "custom":
        restored = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    else:
        restored = ReferenceTransformerV18(ReferenceConfigV18(**payload["config"]))
    restored.load_state_dict(payload["model_state"], strict=True)
    strict_reload = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), restored.state_dict().values())
    )
    report = {
        "schema_version": "foundation-v18-synthetic-v4-pilot" if args.pilot else "foundation-v18-synthetic-v4",
        "model": spec,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "architecture_manifest": (
            model.architecture_manifest()
            if spec["implementation"] == "reference"
            else {
                "implementation": "UniPilot custom attention",
                "final_norm": "PRESENT",
                "residual_projection_init_scale": spec["residual_projection_init_scale"],
            }
        ),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": synthetic["adam_betas"],
            "eps": synthetic["adam_eps"],
            "weight_decay": synthetic["weight_decay"],
            "gradient_clip": synthetic["gradient_clip"],
        },
        "training": {
            "pilot": args.pilot,
            "updates": completed_updates,
            "maximum_updates": updates,
            "stopped_early_on_full_convergence": stopped_early,
            "batch_size": batch_size,
            "train_examples": completed_updates * batch_size,
            "unique_train_hashes": len(training_hashes),
            "examples_by_difficulty": dict(sorted(training_examples.items())),
            "template_counts": dict(sorted(training_templates.items())),
            "processed_input_tokens": processed_tokens,
            "wall_seconds": elapsed,
            "input_tokens_per_second": processed_tokens / elapsed,
            "peak_ram_mb": peak_ram,
            "curve": curve,
        },
        "dataset_audit": {
            "exact_train_test_overlap": (
                None if args.pilot else final_evaluation["exact_train_test_overlap"]
            ),
            "template_overlap": (
                "intentional shared task templates with disjoint randomized token instances"
            ),
            "answer_leakage_in_query_suffix": (
                None if args.pilot else final_evaluation["answer_leakage_in_query_suffix"]
            ),
            "numeric_tokens_are_atomic_synthetic_ids": True,
            "symbolic_tokens_are_atomic_synthetic_ids": True,
            "eos_used": False,
            "sequence_packing": False,
            "target_construction": "only final queried answer supervised; all other targets -100",
        },
        "final": {
            "evaluation": final_evaluation,
            "gate": final_gate,
            "attention": final_attention,
        },
        "checkpoint": (
            {
                "saved": False,
                "reason": "short common-LR selection pilot",
                "strict_in_memory_reload": strict_reload,
            }
            if args.pilot else {
                "saved": True,
                "path": checkpoint_path.relative_to(ROOT).as_posix(),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "strict_reload": strict_reload,
                "optimizer_state_present": True,
            }
        ),
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if resume_path.exists():
        resume_path.unlink()
    print(json.dumps({
        "model": args.model,
        "pilot": args.pilot,
        "updates": completed_updates,
        "final_gate": final_gate,
        "result": result_path.relative_to(ROOT).as_posix(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
