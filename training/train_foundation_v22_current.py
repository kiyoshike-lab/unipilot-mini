"""PHASE 33: continue the selected Current architecture from verified 256k state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from training.optimizer import create_optimizer
from training.train_foundation_v21_ab import (
    CHECKPOINT_FORMAT,
    TOKENS_PER_UPDATE,
    build_paired_model,
    file_sha256,
    load_json,
    run_training,
    stateless_scheduler_state,
    tensor_sha256,
)


V21_CHECKPOINT_ROOT = ROOT / "checkpoints/foundation-v21-ab/current"
V22_CHECKPOINT_ROOT = ROOT / "checkpoints/foundation-v22-current"


def default_resume_path(seed: int) -> Path:
    return V21_CHECKPOINT_ROOT / f"seed-{seed}" / "checkpoint-tokens-256000.pt"


def expected_permutation(settings: dict, seed: int) -> torch.Tensor:
    corpus = load_json(settings["corpus_manifest"])
    train_tokens = int(corpus["splits"]["train"]["tokens"])
    macro_count = (train_tokens - 1) // TOKENS_PER_UPDATE
    return torch.randperm(macro_count, generator=torch.Generator().manual_seed(seed))


def _checkpoint_path_label(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def preflight_resume(settings: dict, seed: int, checkpoint_path: Path) -> dict:
    """Verify every state required to continue without data or scheduler discontinuity."""
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"resume checkpoint is missing: {checkpoint_path}")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    model = build_paired_model(settings, tokenizer, "current", seed)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_order = expected_permutation(settings, seed)
    update = int(payload.get("update", -1))
    tokens = int(payload.get("tokens_processed", -1))
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise RuntimeError("resume checkpoint format mismatch")
    if payload.get("variant") != "current" or int(payload.get("seed", -1)) != seed:
        raise RuntimeError("resume checkpoint identity mismatch")
    permitted_updates = {
        500, 625, 750, 875, 1000, 1125, 1250, 1500, 1750, 2000,
        2500, 3000, 3500, 4000,
    }
    if update not in permitted_updates or tokens != update * TOKENS_PER_UPDATE:
        raise RuntimeError("resume checkpoint is not a permitted PHASE 32-35 milestone")
    if payload.get("config") != model.config.to_dict():
        raise RuntimeError("resume model config mismatch")
    if not torch.equal(payload.get("permutation"), expected_order):
        raise RuntimeError("resume sampler/permutation mismatch")
    if not payload.get("optimizer_state", {}).get("state"):
        raise RuntimeError("resume optimizer state is absent")
    required_rng = {"python", "numpy", "torch_cpu"}
    if set(payload.get("random_state", {})) != required_rng:
        raise RuntimeError("resume RNG state is incomplete")
    expected_scheduler = stateless_scheduler_state(settings, update)
    if "scheduler_state" in payload and payload["scheduler_state"] != expected_scheduler:
        raise RuntimeError("resume scheduler state mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer = create_optimizer(
        model,
        float(settings["training"]["peak_learning_rate"]),
        float(settings["training"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state"])
    optimizer_lr = float(optimizer.param_groups[0]["lr"])
    if abs(optimizer_lr - expected_scheduler["learning_rate"]) > 1e-12:
        raise RuntimeError("resume optimizer LR is discontinuous")
    if update >= len(expected_order):
        raise RuntimeError("resume sampler position exceeds the deterministic permutation")
    state_hash = tensor_sha256(expected_order)
    audit = {
        "checkpoint": _checkpoint_path_label(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_format": payload["checkpoint_format"],
        "variant": payload["variant"],
        "seed": seed,
        "update": update,
        "tokens_processed": tokens,
        "model_state_strict_reload": True,
        "optimizer_state_present": True,
        "random_state_present": sorted(required_rng),
        "scheduler": expected_scheduler,
        "scheduler_state_source": "checkpoint" if "scheduler_state" in payload else "derived_legacy_stateless_schedule",
        "optimizer_learning_rate": optimizer_lr,
        "permutation_sha256": state_hash,
        "last_macroblock_index": int(expected_order[update - 1]) if update else None,
        "next_macroblock_index": int(expected_order[update]),
        "next_update": update + 1,
        "duplicate_data_prevented": True,
        "status": "PASS",
    }
    del payload, model, optimizer
    return audit


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v22.json")
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026))
    parser.add_argument("--resume", help="A verified 256k/320k/384k/448k Current checkpoint.")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    settings = load_json(args.config)
    seeds = [args.seed] if args.seed is not None else list(settings["seeds"])
    resume_paths = {
        seed: Path(args.resume).resolve() if args.resume and args.seed == seed else default_resume_path(seed)
        for seed in seeds
    }
    audits = {str(seed): preflight_resume(settings, seed, path) for seed, path in resume_paths.items()}
    if args.preflight_only:
        write_json(ROOT / "evaluation/foundation-v22-resume-preflight.json", {
            "schema": "foundation-v22-resume-preflight-v1",
            "phase": 33,
            "tokenizer_changed": False,
            "corpus_changed": False,
            "architecture_changed": False,
            "audits": audits,
        })
        print(json.dumps({"status": "PASS", "seeds": seeds}, indent=2))
        return 0

    for seed in seeds:
        result_path = ROOT / "evaluation/foundation-v22-runs" / f"current-seed-{seed}.json"
        if result_path.exists():
            raise RuntimeError(f"refusing to overwrite completed PHASE 33 result: {result_path}")
        result = run_training(
            settings=settings,
            variant="current",
            seed=seed,
            output_dir=V22_CHECKPOINT_ROOT,
            include_generation=(seed == int(settings["representative_generation_seed"]) and not args.skip_generation),
            resume=resume_paths[seed],
        )
        result["phase"] = 33
        result["schema"] = "foundation-v22-current-512k-run-v1"
        result["resume_preflight"] = audits[str(seed)]
        result["continuity"] = {
            "resumed_from_tokens": audits[str(seed)]["tokens_processed"],
            "first_new_update": audits[str(seed)]["next_update"],
            "learning_rate_continuous": True,
            "data_order_continuous": True,
            "no_new_warmup": True,
        }
        result["production_changed"] = False
        result["campus_changed"] = False
        result["final_blind_used"] = False
        write_json(result_path, result)
        print(json.dumps({
            "seed": seed,
            "tokens": result["final"]["tokens_processed"],
            "loss": result["final"]["validation"]["loss"],
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
