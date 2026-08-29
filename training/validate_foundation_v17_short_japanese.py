from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.investigate_foundation_v14 import frequency_baselines
from evaluation.measure_foundation_v16 import frequency_metrics
from evaluation.measure_foundation_v17 import architecture_probe, validation_metrics
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.optimizer import create_optimizer
from training.train_foundation_v15_controlled import generation_probe, macro_batch
from training.validate_foundation_v16_short_japanese import sentence_boundary_metrics


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v17.json")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", default="checkpoints/foundation-v17-short-japanese")
    args = parser.parse_args()
    settings = load_json(args.config)
    short = settings["short_japanese"]
    manifest = load_json(settings["diagnostic_corpus_manifest"])
    variant = next(row for row in settings["variants"] if row["name"] == args.variant)
    train_meta = manifest["artifacts"]["packed"]["train"]
    validation_meta = manifest["artifacts"]["packed"]["validation"]
    train = np.memmap(ROOT / train_meta["path"], dtype=np.uint16, mode="r")
    validation = np.memmap(ROOT / validation_meta["path"], dtype=np.uint16, mode="r")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.variant}.json"
    checkpoint_path = output_dir / f"{args.variant}.pt"
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite v1.7 short Japanese: {args.variant}")

    seed = int(short["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(settings["cpu_threads"]))
    architecture = dict(settings["architecture"])
    architecture["context_length"] = int(short["context_length"])
    model = DiagnosticTransformerV17(DiagnosticConfigV17(
        model_name=f"Foundation v1.7 short Japanese {args.variant}",
        vocab_size=tokenizer.vocab_size,
        token_embedding_scale=variant["token_embedding_scale"],
        position_embedding_scale=variant["position_embedding_scale"],
        residual_projection_init_scale=variant["residual_projection_init_scale"],
        **architecture,
    ))
    optimizer = create_optimizer(model, 1e-4, .1)
    updates = int(short["token_budget"]) // int(short["effective_batch_tokens"])
    macro_count = (len(train) - 1) // 512
    permutation = torch.randperm(macro_count, generator=torch.Generator().manual_seed(seed))
    history = [{
        "update": 0,
        "tokens_processed": 0,
        "validation": validation_metrics(model, validation, 8192),
    }]
    recent = []
    started = time.perf_counter()
    for update in range(1, updates + 1):
        x, y = macro_batch(train, int(permutation[update - 1]), int(short["context_length"]))
        lr = 1e-4 * min(1.0, update / 20)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite v1.7 short Japanese loss")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        recent.append(float(loss.item()))
        if update in {10, 64, 100, 128}:
            history.append({
                "update": update,
                "tokens_processed": update * 512,
                "train_loss": sum(recent) / len(recent),
                "gradient_norm": gradient_norm,
                "validation": validation_metrics(model, validation, 8192),
            })
            recent.clear()
    baselines = frequency_baselines(train, validation, tokenizer.vocab_size, 8192, alpha=.1)
    final_frequency = frequency_metrics(model, tokenizer, train, validation, 8192)
    boundaries = sentence_boundary_metrics(model, tokenizer, validation, 8192)
    generation = generation_probe(model, tokenizer, validation, seed)
    audit_tokens = torch.from_numpy(
        np.asarray(validation[8192:8320], dtype=np.int64).copy()
    ).unsqueeze(0)
    probe = architecture_probe(model, audit_tokens)
    payload = {
        "checkpoint_format": "foundation-v17-short-japanese-v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config.to_dict(),
        "variant": variant,
        "tokens_processed": int(short["token_budget"]),
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
        "schema_version": "foundation-v17-short-japanese-diagnostic-v1",
        "variant": variant,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "initialization": model.initialization_manifest(),
        "corpus": {
            "manifest": settings["diagnostic_corpus_manifest"],
            "segments": manifest["selection"]["segments"],
            "added_to_foundation_corpus": False,
            "train_sha256": train_meta["sha256"],
            "validation_sha256": validation_meta["sha256"],
        },
        "training": {
            "seed": seed,
            "tokens_processed": int(short["token_budget"]),
            "history": history,
            "wall_seconds": time.perf_counter() - started,
        },
        "final": history[-1]["validation"],
        "frequency": final_frequency,
        "sentence_boundaries": boundaries,
        "generation": generation,
        "probe": probe,
        "baselines": {
            "unigram": baselines["unigram"],
            "bigram": baselines["bigram"],
            "method": baselines["method"],
        },
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
        "final": report["final"],
        "punctuation_mass": final_frequency["period_comma_top1_mass"],
        "non_top1": final_frequency["non_top_1_percent_any_top_1_accuracy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
