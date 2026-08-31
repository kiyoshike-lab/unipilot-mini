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

from evaluation.audit_foundation_v15_architecture import context_sensitivity
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from foundation.reference_transformer_v18 import ReferenceConfigV18, ReferenceTransformerV18
from training.optimizer import create_optimizer
from training.train_foundation_v15_controlled import macro_batch
from training.validate_foundation_v18_synthetic import build_model, load_json, model_spec


def rms(values: torch.Tensor) -> float:
    return float(values.detach().float().square().mean().sqrt())


@torch.inference_mode()
def residual_probe(model, implementation: str, token_ids: torch.Tensor) -> dict:
    model.eval()
    hidden = model.embeddings(token_ids)
    rows = []
    for index, block in enumerate(model.blocks):
        layer_input = hidden
        normalized = block.norm1(layer_input)
        if implementation == "custom":
            attention, _ = block.attention(normalized)
        else:
            length = normalized.size(1)
            attention, _ = block.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=block.causal_mask[:length, :length],
                need_weights=False,
            )
            attention = block.attention_output_dropout(attention)
        post_attention = layer_input + attention
        mlp = block.feed_forward(block.norm2(post_attention))
        hidden = post_attention + mlp
        rows.append({
            "layer": index,
            "input_rms": rms(layer_input),
            "attention_rms": rms(attention),
            "post_attention_rms": rms(post_attention),
            "mlp_rms": rms(mlp),
            "output_rms": rms(hidden),
        })
    return {
        "embedding_rms": rows[0]["input_rms"],
        "layers": rows,
        "layer9_rms": rows[-1]["output_rms"],
        "final_norm_rms": rms(model.final_norm(hidden)),
    }


@torch.inference_mode()
def japanese_metrics(
    model,
    tokenizer: FoundationTokenizer,
    train: np.memmap,
    validation: np.memmap,
    probe_tokens: int,
) -> dict:
    model.eval()
    vocab = model.config.vocab_size
    counts = np.bincount(train, minlength=vocab)
    order = np.argsort(counts)[::-1]
    ranks = np.empty(vocab, dtype=np.int64)
    ranks[order] = np.arange(vocab)
    boundaries = [math.ceil(vocab * value) for value in (.01, .05, .20, .80)]
    definitions = (
        ("top_1_percent", 0, boundaries[0]),
        ("top_5_percent_excluding_top_1", boundaries[0], boundaries[1]),
        ("top_20_percent_excluding_top_5", boundaries[1], boundaries[2]),
        ("middle_20_to_80_percent", boundaries[2], boundaries[3]),
        ("rare_bottom_20_percent", boundaries[3], vocab),
    )
    targets_all = []
    top_all = []
    probability_all = []
    loss_sum = 0.0
    total = 0
    context = model.config.context_length
    for start in range(0, probe_tokens, context):
        size = min(context, probe_tokens - start)
        values = np.asarray(validation[start:start + size + 1], dtype=np.int64).copy()
        inputs = torch.from_numpy(values[:-1]).unsqueeze(0)
        targets = torch.from_numpy(values[1:])
        logits, loss = model(inputs, targets.unsqueeze(0))
        logits = logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        targets_all.append(targets)
        top_all.append(logits.topk(10, dim=-1).indices)
        probability_all.append(probabilities.gather(1, targets[:, None]).squeeze(1))
        loss_sum += float(loss) * size
        total += size
    targets = torch.cat(targets_all)
    top = torch.cat(top_all)
    assigned = torch.cat(probability_all)
    target_ranks = ranks[targets.numpy()]
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
        }
    punctuation_ids = {}
    for text in ("。", "、"):
        token_ids = tokenizer.encode(text)
        if len(token_ids) != 1:
            raise RuntimeError(f"punctuation is not atomic: {text} -> {token_ids}")
        punctuation_ids[text] = token_ids[0]
    punctuation_mass = sum(
        float((top[:, 0] == token_id).float().mean())
        for token_id in punctuation_ids.values()
    )
    return {
        "tokens": total,
        "loss": loss_sum / total,
        "perplexity": math.exp(min(loss_sum / total, 50)),
        "top_1_accuracy": float((top[:, 0] == targets).float().mean()),
        "top_5_accuracy": float((top[:, :5] == targets[:, None]).any(-1).float().mean()),
        "top_10_accuracy": float((top == targets[:, None]).any(-1).float().mean()),
        "mean_correct_token_probability": float(assigned.mean()),
        "frequency_buckets": buckets,
        "punctuation_top1_mass": punctuation_mass,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v18.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="checkpoints/foundation-v18-short-japanese")
    args = parser.parse_args()
    settings = load_json(args.config)
    short = settings["japanese_diagnostic"]
    spec = model_spec(settings, args.model)
    manifest = load_json(settings["diagnostic_corpus_manifest"])
    train_meta = manifest["artifacts"]["packed"]["train"]
    validation_meta = manifest["artifacts"]["packed"]["validation"]
    train = np.memmap(ROOT / train_meta["path"], dtype=np.uint16, mode="r")
    validation = np.memmap(ROOT / validation_meta["path"], dtype=np.uint16, mode="r")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.model}.json"
    checkpoint_path = output_dir / f"{args.model}.pt"
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite v1.8 Japanese run: {args.model}")
    seed = int(short["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(settings["cpu_threads"]))
    model = build_model(
        settings,
        spec,
        tokenizer.vocab_size,
        int(short["context_length"]),
        seed,
    )
    optimizer = create_optimizer(
        model, float(short["learning_rate"]), float(short["weight_decay"])
    )
    updates = int(short["token_budget"]) // int(short["effective_batch_tokens"])
    macro_count = (len(train) - 1) // int(short["effective_batch_tokens"])
    permutation = torch.randperm(macro_count, generator=torch.Generator().manual_seed(seed))
    milestones = {
        int(tokens) // int(short["effective_batch_tokens"])
        for tokens in short["milestone_tokens"]
    }
    audit_tokens = torch.from_numpy(np.asarray(validation[:128], dtype=np.int64).copy()).unsqueeze(0)

    def evaluate(update: int) -> dict:
        return {
            "update": update,
            "tokens_processed": update * int(short["effective_batch_tokens"]),
            "metrics": japanese_metrics(
                model, tokenizer, train, validation, int(short["validation_tokens"])
            ),
            "context_sensitivity": context_sensitivity(model, validation),
            "residual": residual_probe(model, spec["implementation"], audit_tokens),
        }

    history = [evaluate(0)]
    losses = []
    started = time.perf_counter()
    for update in range(1, updates + 1):
        inputs, targets = macro_batch(
            train,
            int(permutation[update - 1]),
            int(short["context_length"]),
        )
        learning_rate = float(short["learning_rate"]) * min(
            1.0, update / int(short["warmup_updates"])
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite Japanese diagnostic loss: {args.model}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(short["gradient_clip"])
        ))
        optimizer.step()
        losses.append(float(loss.detach()))
        if update in milestones:
            row = evaluate(update)
            row["recent_train_loss"] = sum(losses) / len(losses)
            row["gradient_norm"] = gradient_norm
            row["learning_rate"] = learning_rate
            losses.clear()
            history.append(row)
            print(json.dumps({
                "model": args.model,
                "tokens": row["tokens_processed"],
                "metrics": row["metrics"],
                "layer9_rms": row["residual"]["layer9_rms"],
            }), flush=True)
    payload = {
        "checkpoint_format": "foundation-v18-japanese-diagnostic-v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config.to_dict(),
        "model_spec": spec,
        "update": updates,
        "diagnostic_only": True,
    }
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
        "schema_version": "foundation-v18-japanese-diagnostic-v1",
        "model": spec,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "corpus": {
            "manifest": settings["diagnostic_corpus_manifest"],
            "train_sha256": train_meta["sha256"],
            "validation_sha256": validation_meta["sha256"],
            "same_token_permutation_seed": seed,
        },
        "training": {
            "updates": updates,
            "tokens_processed": int(short["token_budget"]),
            "effective_batch_tokens": int(short["effective_batch_tokens"]),
            "wall_seconds": time.perf_counter() - started,
            "history": history,
        },
        "checkpoint": {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
            "strict_reload": strict_reload,
            "optimizer_state_present": True,
        },
        "full_foundation_256k": False,
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "model": args.model,
        "final": history[-1],
        "result": result_path.relative_to(ROOT).as_posix(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
