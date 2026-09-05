"""PHASE 39 safe FP32 continuation runner for CPU or CUDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.device import describe_device, resolve_device
from training.train_foundation_v21_ab import load_json, run_training
from training.train_foundation_v22_current import preflight_resume, write_json


SEEDS = (42, 123, 2026)
START_TOKENS = 10_240_000
APPROVED_TARGET_TOKENS = 15_360_000
CONFIG_PATH = "configs/unipilot-foundation-v28.json"


def default_resume_path(seed: int) -> Path:
    return ROOT / (
        f"checkpoints/foundation-v26-current/current/seed-{seed}/"
        f"checkpoint-tokens-{START_TOKENS}.pt"
    )


def validate_target_tokens(value: int) -> int:
    if value != APPROVED_TARGET_TOKENS:
        raise ValueError(
            "PHASE 38 Gate authorizes exactly 15,360,000 target tokens; "
            f"received {value:,}"
        )
    if value % 512:
        raise ValueError("target tokens must be divisible by 512")
    return value


def verify_phase38_gate() -> dict:
    summary = load_json("evaluation/foundation-v27-summary.json")
    if int(summary["training_curve"][-1]["tokens"]) != START_TOKENS:
        raise RuntimeError("PHASE 38 final token count is not 10,240,000")
    if summary["final_gate"] != "CONTINUE_15M_GENERATION_LAG":
        raise RuntimeError("PHASE 38 Gate does not authorize the 15.360M continuation")
    if not summary["checkpoint_integrity"]["integrity_pass"]:
        raise RuntimeError("PHASE 38 checkpoint integrity did not pass")
    return {
        "final_tokens": START_TOKENS,
        "final_gate": summary["final_gate"],
        "checkpoint_integrity": "PASS",
    }


def require_gpu_migration_gate() -> dict:
    path = ROOT / "evaluation/gpu-migration-report.json"
    if not path.is_file():
        raise RuntimeError("GPU migration report is missing; run the migration validator first")
    report = load_json(path)
    if report.get("gpu_migration_gate") != "GPU_MIGRATION_PASS":
        raise RuntimeError("GPU_MIGRATION_PASS is required before formal PHASE 39 training")
    if report.get("speed_gate") != "PASS":
        raise RuntimeError("GPU Speed Gate PASS is required before formal PHASE 39 training")
    return {
        "report": path.relative_to(ROOT).as_posix(),
        "gate": report["gpu_migration_gate"],
        "speed_gate": report["speed_gate"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--target-tokens", type=int, default=APPROVED_TARGET_TOKENS)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.resume is not None and args.seed is None:
        parser.error("--resume requires --seed so checkpoint identity is unambiguous")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target_tokens = validate_target_tokens(args.target_tokens)
    phase38 = verify_phase38_gate()
    settings = load_json(CONFIG_PATH)
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    actual_device = resolve_device(args.device)
    resume_paths = {
        seed: (
            args.resume.resolve()
            if args.resume is not None and seed == args.seed
            else default_resume_path(seed)
        )
        for seed in seeds
    }
    audits = {
        str(seed): preflight_resume(settings, seed, resume_paths[seed])
        for seed in seeds
    }
    if any(int(row["tokens_processed"]) != START_TOKENS for row in audits.values()):
        raise RuntimeError("PHASE 39 must resume exactly at 10,240,000 tokens")
    preflight = {
        "schema": "foundation-v28-resume-preflight-v1",
        "phase": 39,
        "phase38": phase38,
        "target_tokens": target_tokens,
        "requested_device": args.device,
        "device": describe_device(actual_device),
        "precision_mode": "fp32",
        "audits": audits,
        "architecture_changed": False,
        "corpus_changed": False,
        "tokenizer_changed": False,
        "new_warmup": False,
    }
    preflight_name = (
        f"foundation-v28-resume-preflight-seed-{args.seed}.json"
        if args.seed is not None
        else "foundation-v28-resume-preflight.json"
    )
    write_json(ROOT / "evaluation" / preflight_name, preflight)
    if args.preflight_only:
        print(json.dumps({"status": "PASS", **preflight}, indent=2, default=str))
        return 0

    migration_gate = require_gpu_migration_gate()
    for seed in seeds:
        result_path = ROOT / "evaluation/foundation-v28-runs" / f"current-seed-{seed}.json"
        if result_path.exists():
            raise RuntimeError(f"refusing to overwrite completed PHASE 39 result: {result_path}")
        result = run_training(
            settings=settings,
            variant="current",
            seed=seed,
            output_dir=ROOT / "checkpoints/foundation-v28-current",
            token_budget=target_tokens,
            include_generation=(seed == int(settings["representative_generation_seed"])),
            resume=resume_paths[seed],
            device=actual_device,
            precision_mode="fp32",
        )
        result.update({
            "schema": "foundation-v28-current-run-v1",
            "phase": 39,
            "phase38": phase38,
            "gpu_migration": migration_gate,
            "resume_preflight": audits[str(seed)],
            "continuity": {
                "resumed_from_tokens": START_TOKENS,
                "first_new_update": START_TOKENS // 512 + 1,
                "learning_rate_continuous": True,
                "data_order_continuous": True,
                "no_new_warmup": True,
            },
            "production_changed": False,
            "campus_changed": False,
            "final_blind_used": False,
        })
        write_json(result_path, result)
        print(json.dumps({
            "phase": 39,
            "seed": seed,
            "device": str(actual_device),
            "tokens": result["final"]["tokens_processed"],
            "validation_loss": result["final"]["validation"]["loss"],
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
