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
import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.audit_foundation_v15_architecture import context_ablation
from evaluation.evaluate_foundation_v13 import PRIMARY_MODES
from evaluation.investigate_foundation_v14 import (
    aggregate_generation,
    generate_ids,
)
from evaluation.measure_foundation_v17 import architecture_probe
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.optimizer import create_optimizer
from training.train_foundation_v15_controlled import macro_batch


CHECKPOINT_FORMAT = "foundation-v21-controlled-ab-v1"
EXPECTED_PARAMETERS = 19_514_880
TOKENS_PER_UPDATE = 512
VARIANT_NAMES = ("current", "depth_init")
BOUNDARY_TEXT = ("。", "！", "？", "\n", "<EOS>")
CALIBRATION_TEXT = ("。", "、", "の", "に", "は", "を", "が")


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(values: torch.Tensor) -> str:
    return hashlib.sha256(values.detach().cpu().numpy().tobytes()).hexdigest()


def random_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())


def variant_spec(settings: dict, name: str) -> dict:
    if name not in VARIANT_NAMES:
        raise ValueError(f"unsupported PHASE 32 variant: {name}")
    return next(row for row in settings["variants"] if row["name"] == name)


def build_paired_model(
    settings: dict, tokenizer: FoundationTokenizer, variant: str, seed: int
) -> DiagnosticTransformerV17:
    """Build paired initializations: residual projections share standard-normal draws."""
    spec = variant_spec(settings, variant)
    torch.manual_seed(seed)
    config = DiagnosticConfigV17(
        model_name=f"UniPilot Foundation v2.1 {spec['formal_name']} seed {seed}",
        vocab_size=tokenizer.vocab_size,
        residual_projection_init_scale=1.0,
        **settings["architecture"],
    )
    model = DiagnosticTransformerV17(config)
    scale = float(spec["residual_projection_init_scale"])
    if variant == "depth_init":
        model.config.residual_projection_init_scale = scale
        with torch.no_grad():
            for block in model.blocks:
                block.attention.projection.weight.mul_(scale)
                block.feed_forward.network[2].weight.mul_(scale)
    return model


def paired_initialization_audit(
    settings: dict, tokenizer: FoundationTokenizer, seed: int = 42
) -> dict:
    current = build_paired_model(settings, tokenizer, "current", seed)
    depth = build_paired_model(settings, tokenizer, "depth_init", seed)
    scale = float(variant_spec(settings, "depth_init")["residual_projection_init_scale"])
    changed = []
    standardized_max_error = 0.0
    for (name, left), (right_name, right) in zip(
        current.state_dict().items(), depth.state_dict().items()
    ):
        if name != right_name:
            raise RuntimeError("Current/Depth state layout differs")
        if torch.equal(left, right):
            continue
        changed.append(name)
        standardized_max_error = max(
            standardized_max_error,
            float((left * scale - right).abs().max()),
        )
    expected = [
        name
        for index in range(int(settings["architecture"]["n_layers"]))
        for name in (
            f"blocks.{index}.attention.projection.weight",
            f"blocks.{index}.feed_forward.network.2.weight",
        )
    ]
    result = {
        "seed": seed,
        "current_parameters": current.parameter_count(),
        "depth_parameters": depth.parameter_count(),
        "target_parameters": int(settings["parameter_target"]),
        "parameter_equality": current.parameter_count() == depth.parameter_count(),
        "parameter_target_match": current.parameter_count() == int(settings["parameter_target"]),
        "changed_tensors": changed,
        "expected_changed_tensors": expected,
        "only_residual_output_projections_changed": changed == expected,
        "paired_standardized_max_absolute_error": standardized_max_error,
        "depth_residual_std_formula": "0.02 / sqrt(2 * n_layers)",
        "depth_residual_std": 0.02 * scale,
        "base_std": 0.02,
    }
    if not all((
        result["parameter_equality"],
        result["parameter_target_match"],
        result["only_residual_output_projections_changed"],
        standardized_max_error == 0.0,
    )):
        raise RuntimeError(f"PHASE 32 initialization preflight failed: {result}")
    return result


def _atomic_id(tokenizer: FoundationTokenizer, text: str) -> int:
    values = tokenizer.encode(text)
    if len(values) != 1:
        raise RuntimeError(f"required calibration token is not atomic: {text} -> {values}")
    return values[0]


def frequency_ranks(train: np.memmap, vocab_size: int) -> np.ndarray:
    counts = np.bincount(train, minlength=vocab_size)
    order = np.argsort(counts)[::-1]
    ranks = np.empty(vocab_size, dtype=np.int64)
    ranks[order] = np.arange(vocab_size)
    return ranks


def _pattern_metrics(targets: np.ndarray, predictions: np.ndarray, pattern: list[int]) -> dict:
    width = len(pattern)
    expected = np.asarray(pattern, dtype=np.int64)
    targets_at = [
        index for index in range(len(targets) - width + 1)
        if np.array_equal(targets[index:index + width], expected)
    ]
    predicted_at = [
        index for index in range(len(predictions) - width + 1)
        if np.array_equal(predictions[index:index + width], expected)
    ]
    correct = sum(
        np.array_equal(predictions[index:index + width], expected)
        for index in targets_at
    )
    return {
        "token_ids": [int(value) for value in pattern],
        "targets": len(targets_at),
        "top_1_accuracy": correct / len(targets_at) if targets_at else None,
        "predicted_occurrences": len(predicted_at),
        "predicted_frequency": len(predicted_at) / max(1, len(predictions) - width + 1),
    }


@torch.inference_mode()
def language_metrics(
    model: DiagnosticTransformerV17,
    tokenizer: FoundationTokenizer,
    validation: np.memmap,
    ranks: np.ndarray,
    probe_tokens: int,
) -> dict:
    model.eval()
    vocab = model.config.vocab_size
    probe_tokens = min(probe_tokens, len(validation) - 1)
    named_ids = {text: _atomic_id(tokenizer, text) for text in CALIBRATION_TEXT}
    named_probability = {text: 0.0 for text in named_ids}
    target_rows = []
    top_rows = []
    assigned_rows = []
    total_loss = 0.0
    total = 0
    started = time.perf_counter()
    context = model.config.context_length
    for start in range(0, probe_tokens, context):
        size = min(context, probe_tokens - start)
        values = np.asarray(validation[start:start + size + 1], dtype=np.int64).copy()
        inputs = torch.from_numpy(values[:-1]).unsqueeze(0)
        targets = torch.from_numpy(values[1:])
        logits, loss = model(inputs, targets.unsqueeze(0))
        logits = logits[0]
        probabilities = torch.softmax(logits.float(), dim=-1)
        top = logits.topk(10, dim=-1).indices
        assigned = probabilities.gather(1, targets[:, None]).squeeze(1)
        target_rows.append(targets.cpu())
        top_rows.append(top.cpu())
        assigned_rows.append(assigned.cpu())
        total_loss += float(loss) * size
        total += size
        for text, token_id in named_ids.items():
            named_probability[text] += float(probabilities[:, token_id].sum())
    elapsed = time.perf_counter() - started
    targets = torch.cat(target_rows)
    top = torch.cat(top_rows)
    assigned = torch.cat(assigned_rows)
    target_ranks = ranks[targets.numpy()]
    boundaries = [math.ceil(vocab * value) for value in (.01, .05, .20, .80)]
    definitions = (
        ("top_1_percent", 0, boundaries[0]),
        ("top_5_percent_excluding_top_1", boundaries[0], boundaries[1]),
        ("top_20_percent_excluding_top_5", boundaries[1], boundaries[2]),
        ("middle_20_to_80_percent", boundaries[2], boundaries[3]),
        ("rare_bottom_20_percent", boundaries[3], vocab),
    )
    buckets = {}
    for name, low, high in definitions:
        mask = torch.from_numpy((target_ranks >= low) & (target_ranks < high))
        bucket_targets = targets[mask]
        count = int(mask.sum())
        buckets[name] = {
            "rank_range": [low, high - 1],
            "targets": count,
            "top_1_accuracy": float((top[mask, 0] == bucket_targets).float().mean()) if count else None,
            "top_5_accuracy": float((top[mask, :5] == bucket_targets[:, None]).any(-1).float().mean()) if count else None,
            "top_10_accuracy": float((top[mask] == bucket_targets[:, None]).any(-1).float().mean()) if count else None,
            "mean_correct_token_probability": float(assigned[mask].mean()) if count else None,
            "cross_entropy": float(-assigned[mask].clamp_min(1e-30).log().mean()) if count else None,
        }
    punctuation = {}
    for text, token_id in named_ids.items():
        mask = targets == token_id
        count = int(mask.sum())
        punctuation[text] = {
            "token_id": token_id,
            "actual_frequency": count / total,
            "top_1_predicted_frequency": float((top[:, 0] == token_id).float().mean()),
            "mean_probability": named_probability[text] / total,
            "accuracy": float((top[mask, 0] == token_id).float().mean()) if count else None,
        }
    targets_np = targets.numpy()
    predictions_np = top[:, 0].numpy()
    boundary_patterns = {
        text: ([tokenizer.eos_id] if text == "<EOS>" else tokenizer.encode(text))
        for text in BOUNDARY_TEXT
    }
    sentence_boundaries = {
        text if text != "\n" else "newline": _pattern_metrics(
            targets_np, predictions_np, pattern
        )
        for text, pattern in boundary_patterns.items()
    }
    loss_value = total_loss / total
    outside_mask = torch.from_numpy(target_ranks >= boundaries[0])
    return {
        "tokens": total,
        "loss": loss_value,
        "perplexity": math.exp(min(loss_value, 50)),
        "top_1_accuracy": float((top[:, 0] == targets).float().mean()),
        "top_5_accuracy": float((top[:, :5] == targets[:, None]).any(-1).float().mean()),
        "top_10_accuracy": float((top == targets[:, None]).any(-1).float().mean()),
        "mean_correct_token_probability": float(assigned.mean()),
        "wall_seconds": elapsed,
        "tokens_per_second": total / elapsed,
        "frequency_buckets": buckets,
        "top_1_percent_outside_accuracy": float(
            (top[outside_mask, 0] == targets[outside_mask]).float().mean()
        ),
        "punctuation": punctuation,
        "period_comma_prediction_mass": (
            punctuation["。"] ["top_1_predicted_frequency"]
            + punctuation["、"] ["top_1_predicted_frequency"]
        ),
        "sentence_boundaries": sentence_boundaries,
    }


def activation_summary(probe: dict) -> dict:
    layers = []
    for row in probe["layers"]:
        layers.append({
            "layer": row["layer"],
            "residual_input_rms": row["pre_attention"]["rms"],
            "attention_output_rms": row["attention_output"]["rms"],
            "post_attention_rms": row["post_attention_residual"]["rms"],
            "mlp_output_rms": row["mlp_output"]["rms"],
            "final_residual_rms": row["post_mlp_residual"]["rms"],
        })
    return {
        "embedding_rms": probe["embedding"]["combined"]["rms"],
        "layers": layers,
        "layer_9_rms": layers[-1]["final_residual_rms"],
        "final_residual_rms": layers[-1]["final_residual_rms"],
        "final_norm_rms": probe["final_norm"]["normalized"]["rms"],
        "logit_std": probe["logits"]["std"],
        "all_finite": probe["all_finite"],
        "nan": not probe["all_finite"],
        "inf": not probe["all_finite"],
        "explosion": max(row["final_residual_rms"] for row in layers) >= 20.0,
        "collapse": probe["final_norm"]["normalized"]["std"] <= 0.1,
    }


def fixed_generation_prompts(
    validation: np.memmap, tokenizer: FoundationTokenizer, count: int
) -> list[dict]:
    rows = []
    bos_positions = np.flatnonzero(validation == tokenizer.bos_id)
    for position in bos_positions:
        start = int(position)
        tail = np.flatnonzero(validation[start + 1:] == tokenizer.eos_id)
        if not len(tail):
            continue
        end = start + 1 + int(tail[0])
        document = np.asarray(validation[start:end + 1], dtype=np.int64).tolist()
        if len(document) < 48:
            continue
        prompt_ids = document[:32]
        rows.append({
            "id": f"validation-document-{len(rows) + 1:02d}",
            "prompt_ids": prompt_ids,
            "prompt": tokenizer.decode(prompt_ids, skip_special=True),
            "reference": tokenizer.decode(document[32:96], skip_special=True),
        })
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"could not select {count} fixed generation prompts")
    return rows


@torch.inference_mode()
def generation_evaluation(
    model: DiagnosticTransformerV17,
    tokenizer: FoundationTokenizer,
    prompts: list[dict],
    seed: int,
    max_new_tokens: int,
) -> dict:
    output = {}
    for mode_index, mode in enumerate(PRIMARY_MODES):
        items = []
        for prompt_index, prompt in enumerate(prompts):
            generated = generate_ids(
                model,
                tokenizer,
                prompt["prompt_ids"],
                mode,
                seed + mode_index * 100_000 + prompt_index,
                max_new_tokens=max_new_tokens,
            )
            items.append({
                "id": prompt["id"],
                "prompt": prompt["prompt"],
                "reference": prompt["reference"],
                **generated,
            })
        output[mode["name"]] = {
            "settings": mode,
            "metrics": aggregate_generation(items),
            "items": items,
        }
    return output


def save_checkpoint(
    path: Path,
    *,
    model: DiagnosticTransformerV17,
    optimizer,
    variant: str,
    seed: int,
    update: int,
    permutation: torch.Tensor,
    history: list[dict],
    training_seconds: float,
    settings: dict,
) -> dict:
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config.to_dict(),
        "variant": variant,
        "seed": seed,
        "update": update,
        "tokens_processed": update * TOKENS_PER_UPDATE,
        "permutation": permutation,
        "random_state": random_state(),
        "history": history,
        "training_seconds": training_seconds,
        "production_changed": False,
        "final_blind_used": False,
        "maximum_allowed_tokens_per_run": settings["maximum_allowed_tokens_per_run"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    digest = file_sha256(path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored = DiagnosticTransformerV17(DiagnosticConfigV17(**loaded["config"]))
    restored.load_state_dict(loaded["model_state"], strict=True)
    strict_reload = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), restored.state_dict().values())
    )
    del loaded, restored
    try:
        reported_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        reported_path = path.as_posix()
    return {
        "path": reported_path,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "strict_reload": strict_reload,
        "optimizer_state_present": True,
        "integrity": "PASS" if strict_reload else "FAIL",
    }


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_training(
    *,
    settings: dict,
    variant: str,
    seed: int,
    output_dir: Path,
    token_budget: int | None = None,
    validation_tokens: int | None = None,
    include_generation: bool = True,
    resume: Path | None = None,
) -> dict:
    training = settings["training"]
    token_budget = int(token_budget or training["token_budget"])
    if token_budget > int(settings["maximum_allowed_tokens_per_run"]):
        raise RuntimeError("PHASE 32 forbids training beyond 256k tokens per run")
    if token_budget % TOKENS_PER_UPDATE:
        raise ValueError("token budget must be divisible by 512")
    validation_tokens = int(validation_tokens or settings["evaluation"]["validation_tokens"])
    torch.set_num_threads(int(settings["cpu_threads"]))
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    corpus = load_json(settings["corpus_manifest"])
    train_meta = corpus["splits"]["train"]
    validation_meta = corpus["splits"]["validation"]
    train = np.memmap(ROOT / train_meta["path"], dtype=np.uint16, mode="r")
    validation = np.memmap(ROOT / validation_meta["path"], dtype=np.uint16, mode="r")
    model = build_paired_model(settings, tokenizer, variant, seed)
    if model.parameter_count() != EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameter target mismatch: {model.parameter_count()}")
    optimizer = create_optimizer(
        model,
        float(training["peak_learning_rate"]),
        float(training["weight_decay"]),
    )
    macro_count = (len(train) - 1) // TOKENS_PER_UPDATE
    permutation = torch.randperm(
        macro_count, generator=torch.Generator().manual_seed(seed)
    )
    max_update = token_budget // TOKENS_PER_UPDATE
    if max_update > len(permutation):
        raise RuntimeError("clean corpus has insufficient unique macroblocks")
    update = 0
    history: list[dict] = []
    training_seconds = 0.0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
            raise RuntimeError("unexpected PHASE 32 checkpoint format")
        if payload["variant"] != variant or int(payload["seed"]) != seed:
            raise RuntimeError("PHASE 32 resume run identity mismatch")
        if payload["config"] != model.config.to_dict():
            raise RuntimeError("PHASE 32 resume model config mismatch")
        if not torch.equal(payload["permutation"], permutation):
            raise RuntimeError("PHASE 32 resume data ordering mismatch")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        restore_random_state(payload["random_state"])
        update = int(payload["update"])
        history = payload["history"]
        training_seconds = float(payload["training_seconds"])
    else:
        # Model construction consumes RNG differently by variant; training RNG must not.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    ranks = frequency_ranks(train, tokenizer.vocab_size)
    audit_tokens = torch.from_numpy(
        np.asarray(
            validation[8192:8192 + int(settings["evaluation"]["activation_probe_tokens"])],
            dtype=np.int64,
        ).copy()
    ).unsqueeze(0)
    prompts = fixed_generation_prompts(
        validation, tokenizer, int(settings["evaluation"]["generation_examples"])
    )
    configured_milestones = {
        int(value) for value in training["milestone_tokens"] if int(value) <= token_budget
    }
    configured_milestones.update((0, token_budget))
    milestone_updates = {value // TOKENS_PER_UPDATE for value in configured_milestones}
    peak_ram = _rss_mb()
    recent_losses: list[float] = []

    def evaluate(current_update: int, gradient_norm: float | None, lr: float) -> dict:
        nonlocal peak_ram
        measured = language_metrics(
            model, tokenizer, validation, ranks, validation_tokens
        )
        probe = architecture_probe(model, audit_tokens)
        context = context_ablation(
            model, validation, int(settings["evaluation"]["context_probe_targets"])
        )
        row = {
            "update": current_update,
            "tokens_processed": current_update * TOKENS_PER_UPDATE,
            "corpus_fraction": current_update * TOKENS_PER_UPDATE / int(train_meta["tokens"]),
            "corpus_percentage": (
                100 * current_update * TOKENS_PER_UPDATE / int(train_meta["tokens"])
            ),
            "recent_train_loss": (
                sum(recent_losses) / len(recent_losses) if recent_losses else None
            ),
            "validation": measured,
            "learning_rate": lr,
            "gradient_norm": gradient_norm,
            "training_tokens_per_second": (
                current_update * TOKENS_PER_UPDATE / training_seconds
                if training_seconds > 0 else None
            ),
            "peak_ram_mb": max(peak_ram, _rss_mb()),
            "activation_health": activation_summary(probe),
            "context_utilization": context,
        }
        if (
            include_generation
            and seed == int(settings["representative_generation_seed"])
        ):
            row["generation"] = generation_evaluation(
                model,
                tokenizer,
                prompts,
                seed + current_update * 1000,
                int(settings["evaluation"]["generation_max_new_tokens"]),
            )
        peak_ram = max(peak_ram, _rss_mb())
        row["peak_ram_mb"] = peak_ram
        return row

    if update == 0 and not history:
        row = evaluate(0, None, 0.0)
        checkpoint = output_dir / variant / f"seed-{seed}" / "checkpoint-tokens-0.pt"
        row["checkpoint"] = save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            variant=variant,
            seed=seed,
            update=0,
            permutation=permutation,
            history=[row],
            training_seconds=training_seconds,
            settings=settings,
        )
        history.append(row)
        print(json.dumps({"variant": variant, "seed": seed, "milestone": 0}), flush=True)
    for current_update in range(update + 1, max_update + 1):
        inputs, targets = macro_batch(
            train, int(permutation[current_update - 1]), int(model.config.context_length)
        )
        lr = float(training["peak_learning_rate"]) * min(
            1.0, current_update / int(training["warmup_updates"])
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_started = time.perf_counter()
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite PHASE 32 loss: {variant} seed {seed}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip"])
        ))
        optimizer.step()
        training_seconds += time.perf_counter() - train_started
        recent_losses.append(float(loss.detach()))
        peak_ram = max(peak_ram, _rss_mb())
        if current_update not in milestone_updates:
            continue
        row = evaluate(current_update, gradient_norm, lr)
        recent_losses.clear()
        checkpoint = (
            output_dir / variant / f"seed-{seed}"
            / f"checkpoint-tokens-{current_update * TOKENS_PER_UPDATE}.pt"
        )
        row["checkpoint"] = save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            variant=variant,
            seed=seed,
            update=current_update,
            permutation=permutation,
            history=[*history, row],
            training_seconds=training_seconds,
            settings=settings,
        )
        history.append(row)
        print(json.dumps({
            "variant": variant,
            "seed": seed,
            "tokens": row["tokens_processed"],
            "validation_loss": row["validation"]["loss"],
            "top_1": row["validation"]["top_1_accuracy"],
            "punctuation_mass": row["validation"]["period_comma_prediction_mass"],
            "layer_9_rms": row["activation_health"]["layer_9_rms"],
            "context_advantage": row["context_utilization"]["full_vs_last_1_loss_advantage"],
        }), flush=True)
    final = history[-1]
    result = {
        "schema_version": "foundation-v21-controlled-ab-run-v1",
        "variant": variant_spec(settings, variant),
        "seed": seed,
        "parameters": model.parameter_count(),
        "config": model.config.to_dict(),
        "initialization": model.initialization_manifest(),
        "training": {
            "resumed_from": resume.as_posix() if resume is not None else None,
            "tokens_processed": final["tokens_processed"],
            "updates": final["update"],
            "effective_batch_tokens": TOKENS_PER_UPDATE,
            "data_order_sha256": tensor_sha256(permutation),
            "optimizer": training["optimizer"],
            "betas": training["betas"],
            "epsilon": training["epsilon"],
            "weight_decay": training["weight_decay"],
            "gradient_clip": training["gradient_clip"],
            "peak_learning_rate": training["peak_learning_rate"],
            "warmup_updates": training["warmup_updates"],
            "schedule_after_warmup": training["schedule_after_warmup"],
            "training_seconds": training_seconds,
            "history": history,
        },
        "corpus": {
            "manifest": settings["corpus_manifest"],
            "train_path": train_meta["path"],
            "train_documents": train_meta["documents"],
            "train_tokens": train_meta["tokens"],
            "train_sha256": train_meta["sha256"],
            "validation_path": validation_meta["path"],
            "validation_sha256": validation_meta["sha256"],
            "corpus_added": False,
            "corpus_recleaned": False,
        },
        "tokenizer": {
            "path": settings["tokenizer"],
            "vocab_size": tokenizer.vocab_size,
            "retrained": False,
        },
        "best_validation_loss": min(
            row["validation"]["loss"] for row in history
        ),
        "final": final,
        "representative_generation_run": (
            seed == int(settings["representative_generation_seed"])
        ),
        "final_blind_used": False,
        "foundation_base_complete": False,
        "production_changed": False,
        "campus_changed": False,
        "external_ai_api": "OFF",
    }
    return result


def result_path(output_root: Path, variant: str, seed: int) -> Path:
    return output_root / f"{variant}-seed-{seed}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v21.json")
    parser.add_argument("--variant", choices=VARIANT_NAMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", default="evaluation/foundation-v21-runs")
    parser.add_argument("--checkpoint-root", default="checkpoints/foundation-v21-ab")
    parser.add_argument("--resume")
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--validation-tokens", type=int)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    settings = load_json(args.config)
    if args.seed not in settings["seeds"]:
        raise RuntimeError(f"seed is outside the fixed PHASE 32 set: {args.seed}")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    audit = paired_initialization_audit(settings, tokenizer)
    destination = ROOT / args.output_root
    destination.mkdir(parents=True, exist_ok=True)
    path = result_path(destination, args.variant, args.seed)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite completed PHASE 32 run: {path}")
    result = run_training(
        settings=settings,
        variant=args.variant,
        seed=args.seed,
        output_dir=ROOT / args.checkpoint_root,
        token_budget=args.token_budget,
        validation_tokens=args.validation_tokens,
        include_generation=not args.skip_generation,
        resume=Path(args.resume).resolve() if args.resume else None,
    )
    result["initialization_audit"] = audit
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        reported_result = path.relative_to(ROOT).as_posix()
    except ValueError:
        reported_result = path.as_posix()
    print(json.dumps({
        "result": reported_result,
        "variant": args.variant,
        "seed": args.seed,
        "tokens": result["final"]["tokens_processed"],
        "final_validation_loss": result["final"]["validation"]["loss"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
