"""PHASE 35: continue all three formal Current seeds to exactly 1,024k tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_foundation_v21_ab import load_json, run_training
from training.train_foundation_v22_current import preflight_resume, write_json


RESUME_PATHS = {
    42: ROOT / "checkpoints/foundation-v23-pilot/current/seed-42/checkpoint-tokens-640000.pt",
    123: ROOT / "checkpoints/foundation-v22-current/current/seed-123/checkpoint-tokens-512000.pt",
    2026: ROOT / "checkpoints/foundation-v22-current/current/seed-2026/checkpoint-tokens-512000.pt",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v24.json")
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026))
    parser.add_argument("--resume", help="Explicit verified checkpoint for interruption recovery.")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    settings = load_json(args.config)
    seeds = [args.seed] if args.seed is not None else list(settings["seeds"])
    resume_paths = {
        seed: Path(args.resume).resolve() if args.resume and args.seed == seed else RESUME_PATHS[seed]
        for seed in seeds
    }
    audits = {str(seed): preflight_resume(settings, seed, resume_paths[seed]) for seed in seeds}
    expected_tokens = {42: 640_000, 123: 512_000, 2026: 512_000}
    if args.resume is None:
        for seed in seeds:
            if audits[str(seed)]["tokens_processed"] != expected_tokens[seed]:
                raise RuntimeError(f"seed {seed} did not start at its fixed PHASE 35 resume point")
    if args.preflight_only:
        output = {
            "schema": "foundation-v24-resume-preflight-v1",
            "phase": 35,
            "target_tokens": 1_024_000,
            "architecture_changed": False,
            "corpus_changed": False,
            "tokenizer_changed": False,
            "audits": audits,
        }
        write_json(ROOT / "evaluation/foundation-v24-resume-preflight.json", output)
        print(json.dumps({"status": "PASS", "resume_tokens": {seed: audits[str(seed)]["tokens_processed"] for seed in seeds}}, indent=2))
        return 0
    for seed in seeds:
        result_path = ROOT / f"evaluation/foundation-v24-runs/current-seed-{seed}.json"
        if result_path.exists():
            raise RuntimeError(f"refusing to overwrite completed PHASE 35 run: {result_path}")
        result = run_training(
            settings=settings,
            variant="current",
            seed=seed,
            output_dir=ROOT / "checkpoints/foundation-v24-current",
            token_budget=1_024_000,
            include_generation=(seed == int(settings["representative_generation_seed"])),
            resume=resume_paths[seed],
        )
        result.update({
            "schema": "foundation-v24-current-1024k-run-v1",
            "phase": 35,
            "resume_preflight": audits[str(seed)],
            "continuity": {
                "resumed_from_tokens": audits[str(seed)]["tokens_processed"],
                "first_new_update": audits[str(seed)]["next_update"],
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
            "seed": seed,
            "tokens": result["final"]["tokens_processed"],
            "validation_loss": result["final"]["validation"]["loss"],
            "top_1": result["final"]["validation"]["top_1_accuracy"],
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
