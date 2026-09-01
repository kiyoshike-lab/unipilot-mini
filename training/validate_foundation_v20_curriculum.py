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
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.measure_foundation_v18_attention import attention_retrieval_metrics
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from foundation.reference_transformer_v18 import ReferenceConfigV18, ReferenceTransformerV18
from foundation.synthetic_context_v3 import (
    LEVEL_PAIRS,
    KEYS,
    VALUES,
    counterfactual_relation,
    example_hash,
    key_lookup_example_v3,
    make_batch_v3,
    mapping_hash,
    mapping_partition,
    removed_query,
    removed_relation,
    shuffled_relation,
    weighted_relation_loss,
    wrong_query,
)


FORMAL_PARAMETERS = 19_514_880


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_spec(settings: dict, name: str) -> dict:
    return next(row for row in settings["models"] if row["name"] == name)


def build_model(settings: dict, spec: dict, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    architecture = dict(settings["architecture"])
    common = {
        "model_name": f"Foundation v2.0 Benchmark v3.1 {spec['name']}",
        "residual_projection_init_scale": spec["residual_projection_init_scale"],
        **architecture,
    }
    if spec["implementation"] == "custom":
        model = DiagnosticTransformerV17(DiagnosticConfigV17(
            token_embedding_scale=1.0,
            position_embedding_scale=1.0,
            norm="layernorm",
            activation="gelu",
            **common,
        ))
    else:
        reference_common = dict(common)
        for key in ("weight_tying", "bias", "dropout"):
            reference_common.pop(key)
        model = ReferenceTransformerV18(ReferenceConfigV18(
            dropout=architecture["dropout"],
            bias=architecture["bias"],
            weight_tying=architecture["weight_tying"],
            **reference_common,
        ))
    if model.parameter_count() != FORMAL_PARAMETERS:
        raise RuntimeError(
            f"formal parameter mismatch: {model.parameter_count():,} != {FORMAL_PARAMETERS:,}"
        )
    return model


def create_synthetic_optimizer(model, settings: dict):
    config = settings["optimizer"]
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
    )


def deterministic_examples(
    level: int,
    count: int,
    seed: int,
    split: str,
    *,
    excluded_hashes: set[str] | None = None,
    excluded_mapping_hashes: set[str] | None = None,
) -> list[tuple[list[int], int, dict]]:
    # Only 102 held-out one-pair relations exist; keep that census duplicate-free.
    if level == 1 and split == "heldout":
        count = min(count, 100)
    excluded = excluded_hashes or set()
    excluded_mappings = excluded_mapping_hashes or set()
    rng = random.Random(seed)
    rows: list[tuple[list[int], int, dict]] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < count:
        row = key_lookup_example_v3(rng, level, markers=True, split=split)
        digest = example_hash(row)
        mapping_digest = mapping_hash(row)
        attempts += 1
        if (
            digest not in seen
            and digest not in excluded
            and mapping_digest not in excluded_mappings
        ):
            seen.add(digest)
            rows.append(row)
        if attempts > count * 10_000:
            raise RuntimeError(f"unable to construct level {level} {split} evaluation set")
    return rows


def answer_only_forward(model, inputs: torch.Tensor, answers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute only the supervised final-position logits for the v3 256-token task.

    The formal model retains its 4,096-token embedding/output matrix and exact
    19,514,880 parameter count. Benchmark v3 defines a 256-token diagnostic
    vocabulary, so inactive output classes and unsupervised positions are not
    part of its answer-only objective.
    """
    hidden = model.embeddings(inputs)
    for block in model.blocks:
        hidden = block(hidden)
    final_hidden = model.final_norm(hidden[:, -1:, :])
    logits = model.output(final_hidden)[:, -1, :256]
    return logits, F.cross_entropy(logits, answers)


@torch.inference_mode()
def accuracy_loss(model, examples: list[tuple]) -> dict:
    model.eval()
    correct = 0
    loss_total = 0.0
    predictions: list[int] = []
    for start in range(0, len(examples), 32):
        batch = examples[start:start + 32]
        inputs, _, _ = make_batch_v3(batch, "answer_only", 1.0)
        answers = torch.tensor([row[1] for row in batch], dtype=torch.long)
        logits, loss = answer_only_forward(model, inputs, answers)
        batch_predictions = logits.argmax(-1).tolist()
        predictions.extend(batch_predictions)
        correct += sum(
            prediction == answer
            for prediction, (_, answer, _) in zip(batch_predictions, batch)
        )
        loss_total += float(loss) * len(batch)
    return {
        "accuracy": correct / len(examples),
        "loss": loss_total / len(examples),
        "examples": len(examples),
        "predictions": predictions,
    }


def compact_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def attention_summary(model, implementation: str, examples: list[tuple]) -> dict:
    raw = attention_retrieval_metrics(model, implementation, examples)
    heads = [head for layer in raw["layers"] for head in layer["heads"]]
    fields = (
        "normalized_entropy",
        "correct_key_mass",
        "correct_value_mass",
        "correct_key_value_mass",
        "correct_position_mean_rank",
        "attention_margin",
        "max_attention_probability",
    )
    return {
        "all_layer_head_mean": {
            field: sum(head[field] for head in heads) / len(heads) for field in fields
        },
        "last_layer_mean": {
            field: raw["layers"][-1]["mean"][field] for field in fields
        },
    }


def train_update(model, optimizer, examples: list[tuple], clip: float) -> tuple[float, float]:
    inputs, _, _ = make_batch_v3(examples, "answer_only", 1.0)
    answers = torch.tensor([row[1] for row in examples], dtype=torch.long)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = answer_only_forward(model, inputs, answers)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite v2.0 relation loss")
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    optimizer.step()
    return float(loss.detach()), gradient_norm


def evaluate_level(
    model,
    settings: dict,
    level: int,
    training_hashes: set[str],
    training_mappings: set[str],
    seed_offset: int,
) -> tuple[dict, list[tuple]]:
    benchmark = settings["benchmark"]
    count = int(benchmark["evaluation_examples"])
    novel = deterministic_examples(
        level,
        count,
        int(settings["seed"]) + seed_offset + level * 10_000,
        benchmark["test_mapping_split"],
        excluded_hashes=training_hashes,
        excluded_mapping_hashes=training_mappings if level >= 2 else None,
    )
    same_partition = deterministic_examples(
        level,
        count,
        int(settings["seed"]) + seed_offset + 100_000 + level * 10_000,
        benchmark["train_mapping_split"],
        excluded_hashes=training_hashes,
    )
    return {
        "pairs": LEVEL_PAIRS[level],
        "candidate_chance": 1 / LEVEL_PAIRS[level],
        "value_vocabulary_chance": 1 / 32,
        "novel_mapping": compact_metrics(accuracy_loss(model, novel)),
        "novel_sequence_same_partition": compact_metrics(accuracy_loss(model, same_partition)),
    }, novel


def evaluate_controls(model, examples: list[tuple], settings: dict) -> dict:
    correct = accuracy_loss(model, examples)
    counterfactual_rows = [counterfactual_relation(row) for row in examples]
    shuffled_rows = [shuffled_relation(row) for row in examples]
    wrong_original_rows = [wrong_query(row) for row in examples]
    wrong_new_rows = [wrong_query(row, target_new=True) for row in examples]
    removed_query_rows = [removed_query(row) for row in examples]
    removed_relation_rows = [removed_relation(row) for row in examples]
    measured = {
        "correct_query": compact_metrics(correct),
        "counterfactual": compact_metrics(accuracy_loss(model, counterfactual_rows)),
        "shuffled_original_target": compact_metrics(accuracy_loss(model, shuffled_rows)),
        "wrong_query_original_target": compact_metrics(accuracy_loss(model, wrong_original_rows)),
        "wrong_query_new_target": compact_metrics(accuracy_loss(model, wrong_new_rows)),
        "removed_query_original_target": compact_metrics(accuracy_loss(model, removed_query_rows)),
        "removed_relation_original_target": compact_metrics(accuracy_loss(model, removed_relation_rows)),
    }
    measured["drops"] = {
        "shuffled": correct["accuracy"] - measured["shuffled_original_target"]["accuracy"],
        "wrong_query_original": correct["accuracy"] - measured["wrong_query_original_target"]["accuracy"],
        "removed_query": correct["accuracy"] - measured["removed_query_original_target"]["accuracy"],
        "removed_relation": correct["accuracy"] - measured["removed_relation_original_target"]["accuracy"],
    }
    thresholds = settings["benchmark"]["controls"]
    checks = {
        "counterfactual": measured["counterfactual"]["accuracy"] >= thresholds["counterfactual_minimum"],
        "shuffled": measured["drops"]["shuffled"] >= thresholds["shuffled_minimum_drop"],
        "removed_relation": measured["removed_relation_original_target"]["accuracy"] <= thresholds["removed_maximum_accuracy"],
        "wrong_query_new_target": measured["wrong_query_new_target"]["accuracy"] >= thresholds["wrong_query_new_target_minimum"],
        "wrong_query_original_target": measured["wrong_query_original_target"]["accuracy"] <= thresholds["wrong_query_original_maximum"],
        "removed_query": measured["removed_query_original_target"]["accuracy"] <= thresholds["removed_query_maximum"],
    }
    measured["gate_checks"] = checks
    measured["pass"] = all(checks.values())
    return measured


def save_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model, optimizer, spec: dict, update: int, run_kind: str) -> dict:
    return {
        "schema_version": "synthetic-context-benchmark-v3.1-checkpoint",
        "run_kind": run_kind,
        "model_name": spec["name"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config.to_dict(),
        "update": update,
        "diagnostic_only": True,
    }


def verify_checkpoint(path: Path, spec: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if spec["implementation"] == "reference":
        restored = ReferenceTransformerV18(ReferenceConfigV18(**payload["config"]))
    else:
        restored = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    incompatible = restored.load_state_dict(payload["model_state"], strict=True)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "strict_reload": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "optimizer_state_present": bool(payload.get("optimizer_state")),
        "parameters": restored.parameter_count(),
    }


def overlap_audit(training_hashes: set[str], training_mappings: set[str], novel: list[tuple]) -> dict:
    novel_hashes = {example_hash(row) for row in novel}
    novel_mappings = {mapping_hash(row) for row in novel}
    train_relations = {
        (key, value)
        for key in KEYS for value in VALUES
        if mapping_partition(key, value) == "train"
    }
    novel_relations = {tuple(pair) for row in novel for pair in row[2]["mapping"]}
    return {
        "exact_sequence_overlap": len(training_hashes & novel_hashes),
        "exact_mapping_combination_overlap": len(training_mappings & novel_mappings),
        "individual_relation_overlap_expected": len(train_relations & novel_relations),
        "template_overlap": "intentional: identical v3 grammar",
        "novel_mapping_definition": "exact unseen multi-relation mapping combination under canonical v3 any/any generation",
    }


def sample_complexity(curve: list[dict]) -> dict:
    result = {}
    for threshold in (0.50, 0.75, 0.90, 0.95, 0.98):
        match = next((row for row in curve if row["accuracy"] >= threshold), None)
        result[f"{threshold:.2f}"] = (
            None if match is None else {
                "updates": match["updates"],
                "examples": match["examples_processed"],
                "tokens": match["tokens_processed"],
            }
        )
    return result


def run_standalone(
    settings: dict,
    spec: dict,
    level: int,
    output_dir: Path,
    resume: bool,
    maximum_updates_override: int | None = None,
) -> Path:
    if spec["name"] != "reference_mha":
        raise RuntimeError("standalone budget determination is Reference-first")
    result_path = output_dir / f"reference_mha-l{level}-standalone.json"
    checkpoint_path = output_dir / f"reference_mha-l{level}-standalone.pt"
    resume_path = output_dir / f"reference_mha-l{level}-standalone.resume.pt"
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite completed standalone run: L{level}")
    if resume != resume_path.exists():
        expectation = "use --resume" if resume_path.exists() else "resume checkpoint is absent"
        raise RuntimeError(expectation)

    seed = int(settings["seed"]) + level * 1_000
    model = build_model(settings, spec, seed)
    optimizer = create_synthetic_optimizer(model, settings)
    rng = random.Random(seed + 1)
    curve: list[dict] = []
    losses: list[float] = []
    training_hashes: set[str] = set()
    training_mappings: set[str] = set()
    start_update = 1
    elapsed_before = 0.0
    peak_ram = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    if resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        rng.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        curve = state["curve"]
        losses = state["losses"]
        training_hashes = set(state["training_hashes"])
        training_mappings = set(state["training_mappings"])
        start_update = int(state["update"]) + 1
        elapsed_before = float(state["elapsed_seconds"])
        peak_ram = max(peak_ram, float(state["peak_ram_mb"]))

    benchmark = settings["benchmark"]
    run_config = benchmark["standalone"][str(level)]
    configured_points = [int(value) for value in run_config["evaluation_points"]]
    maximum = maximum_updates_override or max(configured_points)
    if maximum <= 0:
        raise ValueError("maximum updates must be positive")
    evaluation_points = {value for value in configured_points if value <= maximum}
    evaluation_points.add(maximum)
    batch_size = int(benchmark["batch_size"])
    sequence_tokens = 1 + LEVEL_PAIRS[level] * 5 + 3
    started = time.perf_counter()
    stopped_early = False
    last_update = start_update - 1
    process = psutil.Process(os.getpid())
    for update in range(start_update, maximum + 1):
        rows = [
            key_lookup_example_v3(
                rng, level, markers=True, split=benchmark["train_mapping_split"]
            )
            for _ in range(batch_size)
        ]
        training_hashes.update(example_hash(row) for row in rows)
        training_mappings.update(mapping_hash(row) for row in rows)
        loss, gradient_norm = train_update(
            model, optimizer, rows, float(settings["optimizer"]["gradient_clip"])
        )
        losses.append(loss)
        last_update = update
        peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
        if update in evaluation_points:
            metrics, novel = evaluate_level(
                model, settings, level, training_hashes, training_mappings, 310_000
            )
            novel_metrics = metrics["novel_mapping"]
            attention_rows = novel[:int(benchmark["attention_examples"])]
            row = {
                "updates": update,
                "examples_processed": update * batch_size,
                "unique_examples": len(training_hashes),
                "tokens_processed": update * batch_size * sequence_tokens,
                "accuracy": novel_metrics["accuracy"],
                "loss": novel_metrics["loss"],
                "same_partition_accuracy": metrics["novel_sequence_same_partition"]["accuracy"],
                "recent_training_loss": sum(losses[-100:]) / min(100, len(losses)),
                "gradient_norm": gradient_norm,
                "attention": attention_summary(model, spec["implementation"], attention_rows),
            }
            curve.append(row)
            print(json.dumps({"run": f"L{level}", **{k: row[k] for k in ("updates", "accuracy", "loss")}}), flush=True)
            stopped_early = row["accuracy"] >= float(run_config["early_stop_accuracy"])
            save_atomic(resume_path, {
                "schema_version": "synthetic-context-benchmark-v3.1-resume",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "python_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "curve": curve,
                "losses": losses,
                "training_hashes": sorted(training_hashes),
                "training_mappings": sorted(training_mappings),
                "update": update,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "peak_ram_mb": peak_ram,
            })
            if stopped_early:
                break

    final_metrics, novel = evaluate_level(
        model, settings, level, training_hashes, training_mappings, 410_000
    )
    fixed = deterministic_examples(0, 1, int(settings["seed"]) + 900_000, "any")
    controls = evaluate_controls(model, novel, settings)
    final_checkpoint = checkpoint_payload(model, optimizer, spec, last_update, f"standalone_l{level}")
    torch.save(final_checkpoint, checkpoint_path)
    elapsed = elapsed_before + time.perf_counter() - started
    audit = overlap_audit(training_hashes, training_mappings, novel)
    result = {
        "schema_version": "synthetic-context-benchmark-v3.1-standalone-result",
        "model": spec,
        "parameters": model.parameter_count(),
        "level": level,
        "pairs": LEVEL_PAIRS[level],
        "optimizer": settings["optimizer"],
        "training": {
            "data_generation": "on-the-fly deterministic random stream",
            "updates": last_update,
            "configured_maximum_updates": maximum,
            "examples_processed": last_update * batch_size,
            "unique_examples": len(training_hashes),
            "unique_mapping_combinations": len(training_mappings),
            "tokens_processed": last_update * batch_size * sequence_tokens,
            "batch_size": batch_size,
            "gradient_accumulation": int(benchmark["gradient_accumulation"]),
            "effective_batch": batch_size * int(benchmark["gradient_accumulation"]),
            "wall_seconds": elapsed,
            "peak_ram_mb": peak_ram,
            "stopped_early_at_configured_target": stopped_early,
        },
        "curve": curve,
        "sample_complexity": sample_complexity(curve),
        "final": final_metrics,
        "fixed_mapping": compact_metrics(accuracy_loss(model, fixed)),
        "controls": controls,
        "dataset_audit": audit,
        "pass": final_metrics["novel_mapping"]["accuracy"] >= float(run_config["pass_accuracy"]),
        "checkpoint": verify_checkpoint(checkpoint_path, spec),
        "resume_used": resume,
        "final_blind_used": False,
        "production_changed": False,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    resume_path.unlink(missing_ok=True)
    return result_path


def reference_gate_ready(output_dir: Path) -> bool:
    required = [
        output_dir / "reference_mha-l3-standalone.json",
        output_dir / "reference_mha-l4-standalone.json",
        output_dir / "reference_mha-sequential.json",
    ]
    if not all(path.exists() for path in required):
        return False
    l3, l4, sequential = (load_json(path) for path in required)
    return bool(l3["pass"] and l4["pass"] and sequential["validity_gate"]["pass"])


def isolated_budget(output_dir: Path, level: int, configured: int) -> int:
    path = output_dir / f"reference_mha-l{level}-standalone.json"
    if level not in {3, 4} or not path.exists():
        return configured
    result = load_json(path)
    return min(configured, int(result["training"]["updates"]))


def run_sequential(
    settings: dict,
    spec: dict,
    output_dir: Path,
    resume: bool,
) -> Path:
    if spec["name"] != "reference_mha" and not reference_gate_ready(output_dir):
        raise RuntimeError("Reference-first gate has not passed")
    result_path = output_dir / f"{spec['name']}-sequential.json"
    checkpoint_path = output_dir / f"{spec['name']}-sequential.pt"
    resume_path = output_dir / f"{spec['name']}-sequential.resume.pt"
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite completed sequential run: {spec['name']}")
    if resume != resume_path.exists():
        expectation = "use --resume" if resume_path.exists() else "resume checkpoint is absent"
        raise RuntimeError(expectation)

    seed = int(settings["seed"]) + 20_000
    model = build_model(settings, spec, seed)
    optimizer = create_synthetic_optimizer(model, settings)
    rng = random.Random(seed + 1)
    benchmark = settings["benchmark"]
    sequential = benchmark["sequential"]
    batch_size = int(benchmark["batch_size"])
    training_hashes: set[str] = set()
    training_mappings: set[str] = set()
    examples_by_level: Counter[str] = Counter()
    tokens_by_level: Counter[str] = Counter()
    curve: list[dict] = []
    level_results: dict[str, dict] = {}
    start_level_index = 0
    start_local_update = 1
    total_updates = 0
    elapsed_before = 0.0
    peak_ram = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    if resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        rng.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        training_hashes = set(state["training_hashes"])
        training_mappings = set(state["training_mappings"])
        examples_by_level = Counter(state["examples_by_level"])
        tokens_by_level = Counter(state["tokens_by_level"])
        curve = state["curve"]
        level_results = state["level_results"]
        start_level_index = int(state["level_index"])
        start_local_update = int(state["local_update"]) + 1
        total_updates = int(state["total_updates"])
        elapsed_before = float(state["elapsed_seconds"])
        peak_ram = max(peak_ram, float(state["peak_ram_mb"]))

    levels = [int(value) for value in sequential["levels"]]
    process = psutil.Process(os.getpid())
    started = time.perf_counter()
    for level_index in range(start_level_index, len(levels)):
        level = levels[level_index]
        configured_maximum = int(sequential["maximum_updates"][str(level)])
        maximum = isolated_budget(output_dir, level, configured_maximum)
        minimum = min(maximum, int(sequential["minimum_updates"][str(level)]))
        interval = int(sequential["evaluation_interval"][str(level)])
        local_start = start_local_update if level_index == start_level_index else 1
        passed = False
        recent_losses: list[float] = []
        last_local_update = local_start - 1
        for local_update in range(local_start, maximum + 1):
            rows = [
                key_lookup_example_v3(
                    rng, level, markers=True, split=benchmark["train_mapping_split"]
                )
                for _ in range(batch_size)
            ]
            training_hashes.update(example_hash(row) for row in rows)
            training_mappings.update(mapping_hash(row) for row in rows)
            examples_by_level[str(level)] += batch_size
            tokens_by_level[str(level)] += sum(len(row[0]) for row in rows)
            loss, gradient_norm = train_update(
                model, optimizer, rows, float(settings["optimizer"]["gradient_clip"])
            )
            recent_losses.append(loss)
            total_updates += 1
            last_local_update = local_update
            peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
            if local_update % interval == 0 or local_update == maximum:
                measured_levels = {}
                attention = None
                for prior_level in levels[:level_index + 1]:
                    metrics, novel = evaluate_level(
                        model, settings, prior_level, training_hashes,
                        training_mappings, 510_000
                    )
                    threshold = float(sequential["thresholds"][str(prior_level)])
                    metrics["threshold"] = threshold
                    metrics["pass"] = metrics["novel_mapping"]["accuracy"] >= threshold
                    measured_levels[str(prior_level)] = metrics
                    if prior_level == level:
                        attention = attention_summary(
                            model,
                            spec["implementation"],
                            novel[:int(benchmark["attention_examples"])],
                        )
                current_pass = measured_levels[str(level)]["pass"]
                row = {
                    "level": level,
                    "local_updates": local_update,
                    "total_updates": total_updates,
                    "examples_processed_total": sum(examples_by_level.values()),
                    "tokens_processed_total": sum(tokens_by_level.values()),
                    "recent_training_loss": sum(recent_losses[-100:]) / min(100, len(recent_losses)),
                    "gradient_norm": gradient_norm,
                    "levels": measured_levels,
                    "attention": attention,
                    "past_levels_retained": all(value["pass"] for value in measured_levels.values()),
                }
                curve.append(row)
                print(json.dumps({
                    "run": spec["name"], "level": level, "local_update": local_update,
                    "accuracy": measured_levels[str(level)]["novel_mapping"]["accuracy"],
                    "retained": row["past_levels_retained"],
                }), flush=True)
                passed = local_update >= minimum and current_pass
                next_level_index = level_index + 1 if passed else level_index
                save_atomic(resume_path, {
                    "schema_version": "synthetic-context-benchmark-v3.1-sequential-resume",
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "python_rng_state": rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "training_hashes": sorted(training_hashes),
                    "training_mappings": sorted(training_mappings),
                    "examples_by_level": dict(examples_by_level),
                    "tokens_by_level": dict(tokens_by_level),
                    "curve": curve,
                    "level_results": level_results,
                    "level_index": next_level_index,
                    "local_update": 0 if passed else local_update,
                    "total_updates": total_updates,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                    "peak_ram_mb": peak_ram,
                })
                if passed:
                    level_results[str(level)] = {
                        "passed_during_training": True,
                        "updates": local_update,
                        "examples": local_update * batch_size,
                        "tokens": local_update * batch_size * (1 + LEVEL_PAIRS[level] * 5 + 3),
                    }
                    break
        if not passed:
            level_results[str(level)] = {
                "passed_during_training": False,
                "updates": last_local_update,
                "examples": last_local_update * batch_size,
                "tokens": last_local_update * batch_size * (1 + LEVEL_PAIRS[level] * 5 + 3),
            }
            break
        start_local_update = 1

    final_levels = {}
    final_novel: dict[int, list[tuple]] = {}
    for level in levels:
        metrics, novel = evaluate_level(
            model, settings, level, training_hashes, training_mappings, 610_000
        )
        threshold = float(sequential["thresholds"][str(level)])
        metrics["threshold"] = threshold
        metrics["pass"] = metrics["novel_mapping"]["accuracy"] >= threshold
        final_levels[str(level)] = metrics
        final_novel[level] = novel
    fixed = deterministic_examples(0, 1, int(settings["seed"]) + 900_000, "any")
    controls = evaluate_controls(model, final_novel[4], settings)
    dataset_audit = overlap_audit(training_hashes, training_mappings, final_novel[4])
    required_pass = all(final_levels[str(level)]["pass"] for level in levels)
    novel_pass = all(
        final_levels[str(level)]["novel_mapping"]["accuracy"] >= float(sequential["thresholds"][str(level)])
        for level in levels
    ) and all(
        dataset_audit[key] == 0
        for key in (
            "exact_sequence_overlap",
            "exact_mapping_combination_overlap",
        )
    )
    validity = {
        "required_levels_pass": required_pass,
        "novel_mapping_pass": novel_pass,
        "controls_pass": controls["pass"],
        "pass": required_pass and novel_pass and controls["pass"],
    }
    torch.save(
        checkpoint_payload(model, optimizer, spec, total_updates, "sequential"),
        checkpoint_path,
    )
    elapsed = elapsed_before + time.perf_counter() - started
    result = {
        "schema_version": "synthetic-context-benchmark-v3.1-sequential-result",
        "model": spec,
        "parameters": model.parameter_count(),
        "optimizer": settings["optimizer"],
        "training": {
            "type": "sequential L1 -> L2 -> L3 -> L4",
            "data_generation": "on-the-fly deterministic random stream",
            "batch_size": batch_size,
            "gradient_accumulation": int(benchmark["gradient_accumulation"]),
            "effective_batch": batch_size * int(benchmark["gradient_accumulation"]),
            "total_updates": total_updates,
            "examples_by_level": dict(examples_by_level),
            "tokens_by_level": dict(tokens_by_level),
            "unique_examples": len(training_hashes),
            "unique_mapping_combinations": len(training_mappings),
            "wall_seconds": elapsed,
            "peak_ram_mb": peak_ram,
            "level_results": level_results,
            "curve": curve,
        },
        "final": {
            "levels": final_levels,
            "fixed_mapping": compact_metrics(accuracy_loss(model, fixed)),
            "controls": controls,
        },
        "dataset_audit": dataset_audit,
        "validity_gate": validity,
        "checkpoint": verify_checkpoint(checkpoint_path, spec),
        "resume_used": resume,
        "final_blind_used": False,
        "production_changed": False,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    resume_path.unlink(missing_ok=True)
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v20.json")
    parser.add_argument("--model", default="reference_mha")
    parser.add_argument("--standalone-level", type=int, choices=[3, 4])
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--maximum-updates", type=int)
    parser.add_argument("--output-dir", default="checkpoints/foundation-v20-benchmark-v31")
    args = parser.parse_args()
    if bool(args.standalone_level) == bool(args.sequential):
        raise RuntimeError("select exactly one run mode")
    settings = load_json(args.config)
    if args.learning_rate is not None:
        settings["optimizer"]["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        settings["optimizer"]["weight_decay"] = args.weight_decay
    spec = model_spec(settings, args.model)
    torch.set_num_threads(int(settings["cpu_threads"]))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.standalone_level:
        path = run_standalone(
            settings, spec, args.standalone_level, output_dir, args.resume,
            args.maximum_updates,
        )
    else:
        path = run_sequential(settings, spec, output_dir, args.resume)
    print(json.dumps({"result": path.relative_to(ROOT).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
