"""PHASE 34 training/inference and KV-cache parity audit for fixed 512k checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import CHECKPOINT_FORMAT, file_sha256, load_json


SEEDS = (42, 123, 2026)
TOLERANCE = 1e-5


def checkpoint_path(seed: int) -> Path:
    return ROOT / f"checkpoints/foundation-v22-current/current/seed-{seed}/checkpoint-tokens-512000.pt"


def load_model(path: Path) -> tuple[DiagnosticTransformerV17, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"unexpected checkpoint format: {path}")
    model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


@torch.inference_mode()
def audit_seed(seed: int, prefix: torch.Tensor, targets: torch.Tensor) -> dict:
    path = checkpoint_path(seed)
    model, payload = load_model(path)
    training_logits, training_loss = model(prefix, targets)
    inference_logits, inference_loss = model(prefix)
    cached_full_logits, _, cache = model(prefix, use_cache=True)
    prefix_without_last = prefix[:, :-1]
    last_token = prefix[:, -1:]
    _, _, partial_cache = model(prefix_without_last, use_cache=True)
    cached_incremental_logits, _, _ = model(
        last_token, past_key_values=partial_cache, use_cache=True
    )
    training_inference_max_abs = float((training_logits - inference_logits).abs().max())
    cache_full_max_abs = float((inference_logits - cached_full_logits).abs().max())
    cache_incremental_max_abs = float(
        (inference_logits[:, -1] - cached_incremental_logits[:, -1]).abs().max()
    )
    dropout_modules = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    tied = model.output.weight.data_ptr() == model.embeddings.token.weight.data_ptr()
    result = {
        "seed": seed,
        "checkpoint": path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": file_sha256(path),
        "strict_reload": True,
        "model_eval": not model.training,
        "all_dropout_disabled": all(not module.training for module in dropout_modules),
        "causal_mask_shape": list(model.blocks[0].attention.causal_mask.shape),
        "position_offset_cached": int(partial_cache[0][0].size(2)),
        "embedding_weight_tied": tied,
        "final_layer_norm": type(model.final_norm).__name__,
        "lm_head": type(model.output).__name__,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "training_forward_loss": float(training_loss),
        "inference_forward_loss": inference_loss,
        "training_vs_inference_max_abs": training_inference_max_abs,
        "full_vs_cached_full_max_abs": cache_full_max_abs,
        "full_vs_cached_incremental_max_abs": cache_incremental_max_abs,
        "training_inference_pass": training_inference_max_abs <= TOLERANCE,
        "kv_cache_pass": max(cache_full_max_abs, cache_incremental_max_abs) <= TOLERANCE,
    }
    result["pass"] = all((
        result["strict_reload"], result["model_eval"], result["all_dropout_disabled"],
        tied, result["training_inference_pass"], result["kv_cache_pass"],
    ))
    del model, payload, cache, partial_cache
    return result


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v22.json")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    corpus = load_json(settings["corpus_manifest"])
    validation_meta = corpus["splits"]["validation"]
    validation = np.memmap(ROOT / validation_meta["path"], dtype=np.uint16, mode="r")
    values = np.asarray(validation[4096:4096 + 65], dtype=np.int64).copy()
    prefix = torch.from_numpy(values[:-1]).unsqueeze(0)
    targets = torch.from_numpy(values[1:]).unsqueeze(0)
    rows = [audit_seed(seed, prefix, targets) for seed in SEEDS]
    final_blind = ROOT / settings["final_blind"]["path"]
    result = {
        "schema": "foundation-v23-inference-parity-v1",
        "phase": 34,
        "tolerance": TOLERANCE,
        "tokenizer": {
            "path": settings["tokenizer"],
            "vocab_size": tokenizer.vocab_size,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
        },
        "rows": rows,
        "inference_parity": "PASS" if all(row["training_inference_pass"] for row in rows) else "FAIL",
        "kv_cache_parity": "PASS" if all(row["kv_cache_pass"] for row in rows) else "FAIL",
        "checkpoint_integrity": "PASS" if all(row["strict_reload"] for row in rows) else "FAIL",
        "final_blind": {
            "sha256": file_sha256(final_blind),
            "content_opened": False,
        },
    }
    result["pass"] = result["inference_parity"] == result["kv_cache_parity"] == "PASS"
    output = ROOT / "evaluation/foundation-v23-inference-parity.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "inference_parity": result["inference_parity"],
        "kv_cache_parity": result["kv_cache_parity"],
        "max_abs": max(row["full_vs_cached_incremental_max_abs"] for row in rows),
    }, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
