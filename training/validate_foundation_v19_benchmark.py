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

from evaluation.measure_foundation_v18_attention import attention_retrieval_metrics
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from foundation.reference_transformer_v18 import ReferenceConfigV18, ReferenceTransformerV18
from foundation.synthetic_context_v3 import (
    LEVEL_PAIRS,
    LEVEL_THRESHOLDS,
    counterfactual_relation,
    example_hash,
    key_lookup_example_v3,
    make_batch_v3,
    mapping_hash,
    removed_query,
    removed_relation,
    shuffled_relation,
    weighted_relation_loss,
    wrong_query,
)
from training.optimizer import create_optimizer


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def model_spec(settings: dict, name: str) -> dict:
    return next(row for row in settings["models"] if row["name"] == name)


def build_model(settings: dict, spec: dict, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    architecture = dict(settings["architecture"])
    common = {
        "model_name": f"Foundation v1.9 Benchmark v3 {spec['name']}",
        "vocab_size": 256,
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
    for key in ("weight_tying", "bias", "dropout"):
        reference_common.pop(key)
    return ReferenceTransformerV18(ReferenceConfigV18(
        dropout=architecture["dropout"],
        bias=architecture["bias"],
        weight_tying=architecture["weight_tying"],
        **reference_common,
    ))


def fixed_examples(
    level: int,
    count: int,
    seed: int,
    *,
    markers: bool,
    split: str,
    token_split: str = "shared",
    vocabulary_stage: str = "full",
    allow_duplicates: bool = False,
    excluded_hashes: set[str] | None = None,
) -> list[tuple[list[int], int, dict]]:
    # The deterministic held-out partition contains 102 one-pair combinations.
    # Use a 100-example census-like sample instead of duplicating rows merely to
    # satisfy a larger requested evaluation batch.
    if level == 1 and split == "heldout" and token_split == "shared":
        count = min(count, 100)
    rng = random.Random(seed)
    excluded = excluded_hashes or set()
    examples = []
    hashes = set()
    attempts = 0
    while len(examples) < count:
        candidate = key_lookup_example_v3(
            rng, level, markers=markers, split=split, token_split=token_split,
            vocabulary_stage=vocabulary_stage,
        )
        digest = example_hash(candidate)
        attempts += 1
        # Level 0 intentionally has one fixed example.
        if level == 0:
            examples.append(candidate)
            continue
        if allow_duplicates or (digest not in hashes and digest not in excluded):
            hashes.add(digest)
            examples.append(candidate)
        if attempts > count * 5000:
            raise RuntimeError(f"unable to construct {count} unique level {level} examples")
    return examples


@torch.inference_mode()
def accuracy_loss(
    model,
    examples: list[tuple[list[int], int, dict]],
    *,
    supervision: str = "answer_only",
    answer_weight: float = 1.0,
) -> dict:
    model.eval()
    correct = 0
    loss_sum = 0.0
    answer_loss_sum = 0.0
    non_answer_loss_sum = 0.0
    for start in range(0, len(examples), 32):
        batch = examples[start:start + 32]
        inputs, targets, weights = make_batch_v3(batch, supervision, answer_weight)
        logits, loss = weighted_relation_loss(model, inputs, targets, weights)
        answer_targets = torch.full_like(targets, -100)
        answer_targets[:, -1] = targets[:, -1]
        answer_weights = torch.zeros_like(weights)
        answer_weights[:, -1] = 1
        _, answer_loss = weighted_relation_loss(model, inputs, answer_targets, answer_weights)
        if supervision == "all_token":
            non_targets = targets.clone()
            non_targets[:, -1] = -100
            non_weights = weights.clone()
            non_weights[:, -1] = 0
            _, non_answer_loss = weighted_relation_loss(model, inputs, non_targets, non_weights)
            non_answer_loss_sum += float(non_answer_loss) * len(batch)
        correct += sum(
            prediction == answer
            for prediction, (_, answer, _) in zip(logits[:, -1].argmax(-1).tolist(), batch)
        )
        loss_sum += float(loss) * len(batch)
        answer_loss_sum += float(answer_loss) * len(batch)
    return {
        "accuracy": correct / len(examples),
        "loss": loss_sum / len(examples),
        "answer_token_loss": answer_loss_sum / len(examples),
        "non_answer_loss": (
            non_answer_loss_sum / len(examples) if supervision == "all_token" else None
        ),
        "examples": len(examples),
    }


def evaluate_levels(
    model,
    *,
    count: int,
    seed: int,
    markers: bool,
    training_hashes: set[str],
    supervision: str = "answer_only",
    answer_weight: float = 1.0,
    maximum_level: int = 5,
    test_split: str = "any",
) -> tuple[dict, dict[int, list[tuple]]]:
    levels = {}
    examples_by_level = {}
    for level in range(maximum_level + 1):
        split = "any" if level == 0 else test_split
        examples = fixed_examples(
            level,
            count,
            seed + level * 10_000,
            markers=markers,
            split=split,
            excluded_hashes=training_hashes,
            allow_duplicates=level == 1 and split == "any",
        )
        examples_by_level[level] = examples
        metrics = accuracy_loss(
            model,
            examples,
            supervision=supervision,
            answer_weight=answer_weight,
        )
        metrics.update({
            "pairs": LEVEL_PAIRS[level],
            "candidate_chance": 1 / LEVEL_PAIRS[level],
            "value_vocabulary_chance": 1 / 32,
            "threshold": LEVEL_THRESHOLDS.get(level),
            "pass": (
                metrics["accuracy"] >= LEVEL_THRESHOLDS[level]
                if level in LEVEL_THRESHOLDS else None
            ),
            "mapping_split": split,
        })
        levels[str(level)] = metrics
    return levels, examples_by_level


def aggregate_attention(model, implementation: str, examples: list[tuple]) -> dict:
    raw = attention_retrieval_metrics(model, implementation, examples[:32])
    heads = [head for layer in raw["layers"] for head in layer["heads"]]
    fields = (
        "normalized_entropy", "correct_key_mass", "correct_value_mass",
        "correct_key_value_mass", "correct_position_mean_rank", "attention_margin",
    )
    return {
        "all_layer_head_mean": {
            field: sum(head[field] for head in heads) / len(heads) for field in fields
        },
        "last_layer_mean": raw["layers"][-1]["mean"],
    }


def evaluate_controls(model, level4_examples: list[tuple]) -> dict:
    correct = accuracy_loss(model, level4_examples)
    variants = {
        "counterfactual": [counterfactual_relation(row) for row in level4_examples],
        "shuffled_relation_original_target": [shuffled_relation(row) for row in level4_examples],
        "correct_query": level4_examples,
        "wrong_query_original_target": [wrong_query(row) for row in level4_examples],
        "wrong_query_new_target": [wrong_query(row, target_new=True) for row in level4_examples],
        "removed_query": [removed_query(row) for row in level4_examples],
        "removed_relation": [removed_relation(row) for row in level4_examples],
    }
    measured = {name: accuracy_loss(model, rows) for name, rows in variants.items()}
    measured["drops"] = {
        "shuffled_relation": correct["accuracy"] - measured["shuffled_relation_original_target"]["accuracy"],
        "removed_relation": correct["accuracy"] - measured["removed_relation"]["accuracy"],
        "wrong_query_original_target": correct["accuracy"] - measured["wrong_query_original_target"]["accuracy"],
        "removed_query": correct["accuracy"] - measured["removed_query"]["accuracy"],
    }
    measured["counterfactual_prediction_changes_with_relation"] = measured["counterfactual"]["accuracy"]
    return measured


def optimizer_for(model, settings: dict):
    optimizer = settings["optimizer"]
    return create_optimizer(
        model,
        float(optimizer["learning_rate"]),
        float(optimizer["weight_decay"]),
    )


def train_update(model, optimizer, examples, supervision: str, answer_weight: float, clip: float):
    inputs, targets, weights = make_batch_v3(examples, supervision, answer_weight)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = weighted_relation_loss(model, inputs, targets, weights)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite v1.9 relation loss")
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    optimizer.step()
    return float(loss.detach()), gradient_norm, int(inputs.numel())


def choose_active_level(
    rng: random.Random, level: int, local_update: int, curriculum: dict,
) -> int:
    active_level = level
    replay_probability = float(curriculum["replay_probability"])
    if level > 0 and replay_probability > 0 and rng.random() < replay_probability:
        active_level = rng.randrange(level)
    relation_replay = float(curriculum.get("relation_replay_probability", 0))
    relation_replay_start = int(
        curriculum.get("relation_replay_start_update_by_level", [0] * 6)[level]
    )
    if (
        level > 2
        and local_update > relation_replay_start
        and relation_replay > 0
        and rng.random() < relation_replay
    ):
        active_level = rng.randrange(2, level)
    return active_level


def run_pilot(settings: dict, spec: dict, variant_name: str, output_dir: Path) -> Path:
    variant = next(row for row in settings["loss_pilot"]["variants"] if row["name"] == variant_name)
    result_path = output_dir / f"pilot-reference-{variant_name}.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite pilot: {result_path}")
    seed = int(settings["seed"])
    model = build_model(settings, spec, seed)
    optimizer = optimizer_for(model, settings)
    rng = random.Random(seed + 101)
    updates = int(variant.get("updates", settings["loss_pilot"]["updates"]))
    batch_size = int(settings["loss_pilot"]["batch_size"])
    level = int(settings["loss_pilot"]["level"])
    markers = bool(variant["markers"])
    curve = []
    losses = []
    training_hashes = set()
    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss / 1024**2
    for update in range(1, updates + 1):
        examples = [
            key_lookup_example_v3(
                rng, level, markers=markers,
                split=variant.get("mapping_train_split", "train"),
                vocabulary_stage=variant.get("vocabulary_stage", "full"),
            )
            for _ in range(batch_size)
        ]
        training_hashes.update(example_hash(row) for row in examples)
        loss, gradient_norm, _ = train_update(
            model, optimizer, examples, variant["supervision"],
            float(variant["answer_weight"]), float(settings["optimizer"]["gradient_clip"]),
        )
        losses.append(loss)
        peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
        if update in {10, 25, 50, 100, 200, 400, 800, 1200, 1600, 2400, 3200, updates}:
            evaluation = fixed_examples(
                level, int(settings["loss_pilot"]["evaluation_examples"]),
                seed + 20_000, markers=markers,
                split=variant.get("mapping_eval_split", "heldout"),
                vocabulary_stage=variant.get("vocabulary_stage", "full"),
                allow_duplicates=variant.get("vocabulary_stage") in {"small", "medium"},
                excluded_hashes=training_hashes,
            )
            metrics = accuracy_loss(
                model, evaluation, supervision=variant["supervision"],
                answer_weight=float(variant["answer_weight"]),
            )
            full_vocabulary_metrics = None
            if variant.get("vocabulary_stage") in {"small", "medium"}:
                full_vocabulary_metrics = accuracy_loss(
                    model,
                    fixed_examples(
                        level, int(settings["loss_pilot"]["evaluation_examples"]),
                        seed + 30_000, markers=markers, split="heldout",
                        vocabulary_stage="full", excluded_hashes=training_hashes,
                    ),
                )
            curve.append({
                "update": update,
                "recent_loss": sum(losses[-50:]) / min(len(losses), 50),
                "gradient_norm": gradient_norm,
                "heldout": metrics,
                "full_vocabulary_heldout": full_vocabulary_metrics,
                "attention": aggregate_attention(model, spec["implementation"], evaluation),
            })
            print(json.dumps({
                "pilot": variant_name, "update": update,
                "accuracy": metrics["accuracy"], "answer_loss": metrics["answer_token_loss"],
            }), flush=True)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "synthetic-context-benchmark-v3-loss-pilot",
        "variant": variant,
        "model": spec,
        "parameters": model.parameter_count(),
        "level": level,
        "updates": updates,
        "train_examples": updates * batch_size,
        "unique_train_examples": len(training_hashes),
        "curve": curve,
        "wall_seconds": elapsed,
        "peak_ram_mb": peak_ram,
        "checkpoint_saved": False,
        "final_blind_used": False,
        "production_changed": False,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result_path


def save_resume(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_curriculum(settings: dict, spec: dict, output_dir: Path, resume: bool) -> Path:
    if spec["name"] != "reference_mha":
        reference_result = output_dir / "reference_mha.json"
        if not reference_result.exists():
            raise RuntimeError("Reference-first gate: reference result is missing")
        reference = load_json(reference_result)
        if not reference["validity_gate"]["pass"]:
            raise RuntimeError("Reference-first gate: benchmark v3 is not VALID")
    result_path = output_dir / f"{spec['name']}.json"
    checkpoint_path = output_dir / f"{spec['name']}.pt"
    resume_path = output_dir / f"{spec['name']}.resume.pt"
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite completed v1.9 run: {spec['name']}")
    if resume and not resume_path.exists():
        raise RuntimeError(f"resume state not found: {resume_path}")
    if not resume and resume_path.exists():
        raise RuntimeError(f"resume state exists; use --resume: {resume_path}")

    seed = int(settings["seed"])
    model = build_model(settings, spec, seed)
    optimizer = optimizer_for(model, settings)
    rng = random.Random(seed + 303)
    curriculum = settings["curriculum"]
    batch_size = int(curriculum["batch_size"])
    training_hashes: set[str] = set()
    training_mapping_hashes: set[str] = set()
    examples_by_level = Counter()
    examples_by_vocabulary_stage = Counter()
    curve = []
    losses = []
    processed_tokens = 0
    total_update = 0
    start_level = 0
    update_in_level = 0
    elapsed_before_resume = 0.0
    reset_at_level_2_done = False
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss / 1024**2
    if resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        rng.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        training_hashes = set(state["training_hashes"])
        training_mapping_hashes = set(state["training_mapping_hashes"])
        examples_by_level = Counter(state["examples_by_level"])
        examples_by_vocabulary_stage = Counter(state.get("examples_by_vocabulary_stage", {}))
        curve = state["curve"]
        losses = state["losses"]
        processed_tokens = int(state["processed_tokens"])
        total_update = int(state["total_update"])
        start_level = int(state["level"])
        update_in_level = int(state["update_in_level"])
        elapsed_before_resume = float(state["elapsed_seconds"])
        peak_ram = max(peak_ram, float(state["peak_ram_mb"]))
        reset_at_level_2_done = bool(state.get("reset_at_level_2_done", False))

    started = time.perf_counter()
    level_pass = {}
    for level in range(start_level, 6):
        if (
            level == 2
            and bool(curriculum.get("reset_at_level_2", False))
            and not reset_at_level_2_done
        ):
            # Levels 0/1 establish that the task implementation and random-value
            # copy control work. Level 2 starts the associative-retrieval capacity
            # experiment from a clean, identically seeded model so memorized
            # one-pair shortcuts cannot bias induction-head formation.
            model = build_model(settings, spec, seed)
            optimizer = optimizer_for(model, settings)
            rng = random.Random(seed + 101)
            reset_at_level_2_done = True
        maximum = int(curriculum["maximum_updates_per_level"][level])
        minimum = int(curriculum["minimum_updates_per_level"][level])
        interval = int(curriculum["evaluation_interval"][level])
        level_start = update_in_level + 1 if level == start_level else 1
        passed = False
        for local_update in range(level_start, maximum + 1):
            active_level = choose_active_level(rng, level, local_update, curriculum)
            vocabulary_stage = "full"
            examples = [
                key_lookup_example_v3(
                    rng, active_level, markers=bool(curriculum["markers"]),
                    split=curriculum["mapping_train_split"],
                    vocabulary_stage=vocabulary_stage,
                )
                for _ in range(batch_size)
            ]
            for row in examples:
                training_hashes.add(example_hash(row))
                training_mapping_hashes.add(mapping_hash(row))
                examples_by_level[str(active_level)] += 1
                examples_by_vocabulary_stage[f"level_{active_level}:{vocabulary_stage}"] += 1
                processed_tokens += len(row[0])
            loss, gradient_norm, _ = train_update(
                model, optimizer, examples, curriculum["supervision"],
                float(curriculum["answer_weight"]), float(settings["optimizer"]["gradient_clip"]),
            )
            losses.append(loss)
            total_update += 1
            peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
            should_evaluate = local_update % interval == 0 or local_update == maximum
            if should_evaluate:
                levels, examples_for_attention = evaluate_levels(
                    model,
                    count=128,
                    seed=seed + 100_000,
                    markers=bool(curriculum["markers"]),
                    training_hashes=training_hashes,
                    maximum_level=level,
                    test_split=curriculum["mapping_test_split"],
                )
                required_pass = all(levels[str(index)]["pass"] for index in range(level + 1))
                curve.append({
                    "total_update": total_update,
                    "level": level,
                    "update_in_level": local_update,
                    "recent_loss": sum(losses[-100:]) / min(len(losses), 100),
                    "gradient_norm": gradient_norm,
                    "levels": levels,
                    "current_level_attention": aggregate_attention(
                        model, spec["implementation"], examples_for_attention[level]
                    ),
                    "all_required_levels_pass": required_pass,
                })
                print(json.dumps({
                    "model": spec["name"], "level": level,
                    "update": local_update, "total_update": total_update,
                    "accuracy": levels[str(level)]["accuracy"],
                    "all_pass": required_pass,
                }), flush=True)
                if level < 5 and local_update >= minimum and required_pass:
                    passed = True
                if level == 5 and local_update >= minimum:
                    passed = True
            if total_update % 200 == 0 or passed or local_update == maximum:
                save_resume(resume_path, {
                    "schema_version": "synthetic-context-benchmark-v3-resume",
                    "model_name": spec["name"],
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "python_rng_state": rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "training_hashes": sorted(training_hashes),
                    "training_mapping_hashes": sorted(training_mapping_hashes),
                    "examples_by_level": dict(examples_by_level),
                    "examples_by_vocabulary_stage": dict(examples_by_vocabulary_stage),
                    "curve": curve,
                    "losses": losses,
                    "processed_tokens": processed_tokens,
                    "total_update": total_update,
                    "level": level + 1 if passed else level,
                    "update_in_level": 0 if passed else local_update,
                    "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
                    "peak_ram_mb": peak_ram,
                    "reset_at_level_2_done": reset_at_level_2_done,
                })
            if passed:
                break
        level_pass[str(level)] = passed
        update_in_level = 0
        if not passed:
            break

    final_levels, final_examples = evaluate_levels(
        model,
        count=int(curriculum["evaluation_examples"]),
        seed=seed + 200_000,
        markers=bool(curriculum["markers"]),
        training_hashes=training_hashes,
        maximum_level=5,
        test_split=curriculum["mapping_test_split"],
    )
    required_levels_pass = all(final_levels[str(level)]["pass"] for level in range(5))
    controls = evaluate_controls(model, final_examples[4])
    minimum_drop = float(settings["validity"]["minimum_control_drop"])
    control_pass = all(
        controls["drops"][name] >= minimum_drop
        for name in ("shuffled_relation", "removed_relation")
    )
    validity = {
        "required_levels_pass": required_levels_pass,
        "shuffled_relation_drop_pass": controls["drops"]["shuffled_relation"] >= minimum_drop,
        "removed_relation_drop_pass": controls["drops"]["removed_relation"] >= minimum_drop,
        "minimum_control_drop": minimum_drop,
        "pass": required_levels_pass and control_pass,
    }
    final_attention = {
        str(level): aggregate_attention(model, spec["implementation"], examples)
        for level, examples in final_examples.items()
    }
    elapsed = elapsed_before_resume + time.perf_counter() - started
    checkpoint = {
        "schema_version": "synthetic-context-benchmark-v3-checkpoint",
        "model_name": spec["name"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config.to_dict(),
        "total_update": total_update,
        "diagnostic_only": True,
    }
    torch.save(checkpoint, checkpoint_path)
    if spec["implementation"] == "custom":
        restored = DiagnosticTransformerV17(DiagnosticConfigV17(**checkpoint["config"]))
    else:
        restored = ReferenceTransformerV18(ReferenceConfigV18(**checkpoint["config"]))
    restored.load_state_dict(checkpoint["model_state"], strict=True)
    strict_reload = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), restored.state_dict().values())
    )
    level0_hashes = {example_hash(row) for row in final_examples[0]}
    level1_hashes = {example_hash(row) for row in final_examples[1]}
    heldout_hashes = {
        example_hash(row)
        for level, examples in final_examples.items() if level >= 2
        for row in examples
    }
    heldout_mapping_hashes = {
        mapping_hash(row) for level, examples in final_examples.items() if level > 0 for row in examples
    }
    result = {
        "schema_version": "synthetic-context-benchmark-v3-result",
        "model": spec,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "optimizer": settings["optimizer"],
        "training": {
            "total_updates": total_update,
            "batch_size": batch_size,
            "examples_by_level": dict(sorted(examples_by_level.items())),
            "examples_by_vocabulary_stage": dict(sorted(examples_by_vocabulary_stage.items())),
            "unique_train_examples": len(training_hashes),
            "unique_train_mappings": len(training_mapping_hashes),
            "processed_input_tokens": processed_tokens,
            "wall_seconds": elapsed,
            "input_tokens_per_second": processed_tokens / elapsed,
            "peak_ram_mb": peak_ram,
            "level_pass_during_training": level_pass,
            "reset_at_level_2": bool(curriculum.get("reset_at_level_2", False)),
            "curve": curve,
        },
        "dataset_audit": {
            "exact_train_test_overlap": len(training_hashes & heldout_hashes),
            "fixed_level0_control_overlap": len(training_hashes & level0_hashes),
            "finite_level1_mapping_overlap": len(training_hashes & level1_hashes),
            "mapping_set_overlap": len(training_mapping_hashes & heldout_mapping_hashes),
            "template_overlap": "intentional: same task grammar, disjoint exact sequences and held-out key/value combinations",
            "generalization_target": "within-context novel relations using held-out key/value combinations with shared token vocabulary",
        },
        "final": {
            "levels": final_levels,
            "controls": controls,
            "attention": final_attention,
        },
        "validity_gate": validity,
        "checkpoint": {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "optimizer_state_present": True,
            "strict_reload": strict_reload,
        },
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if resume_path.exists():
        resume_path.unlink()
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v19.json")
    parser.add_argument("--model", default="reference_mha")
    parser.add_argument("--pilot")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default="checkpoints/foundation-v19-benchmark-v3")
    args = parser.parse_args()
    if bool(args.pilot) == bool(args.curriculum):
        raise RuntimeError("select exactly one of --pilot NAME or --curriculum")
    settings = load_json(args.config)
    spec = model_spec(settings, args.model)
    if args.pilot and args.model != "reference_mha":
        raise RuntimeError("benchmark repair pilots are Reference-first")
    torch.set_num_threads(int(settings["cpu_threads"]))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        path = run_pilot(settings, spec, args.pilot, output_dir)
    else:
        path = run_curriculum(settings, spec, output_dir, args.resume)
    print(json.dumps({"result": path.relative_to(ROOT).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
