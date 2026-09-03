"""PHASE 36: continue all formal Current seeds from 1.024M to 2.048M tokens."""
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
    seed: ROOT / f"checkpoints/foundation-v24-current/current/seed-{seed}/checkpoint-tokens-1024000.pt"
    for seed in (42, 123, 2026)
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v25.json")
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
    if args.resume is None and any(
        audits[str(seed)]["tokens_processed"] != 1_024_000 for seed in seeds
    ):
        raise RuntimeError("PHASE 36 must resume every seed from exactly 1.024M tokens")
    if args.preflight_only:
        output = {
            "schema": "foundation-v25-resume-preflight-v1",
            "phase": 36,
            "target_tokens": 2_048_000,
            "architecture_changed": False,
            "corpus_changed": False,
            "tokenizer_changed": False,
            "audits": audits,
        }
        write_json(ROOT / "evaluation/foundation-v25-resume-preflight.json", output)
        print(json.dumps({
            "status": "PASS",
            "resume_tokens": {seed: audits[str(seed)]["tokens_processed"] for seed in seeds},
        }, indent=2))
        return 0
    for seed in seeds:
        result_path = ROOT / f"evaluation/foundation-v25-runs/current-seed-{seed}.json"
        if result_path.exists():
            raise RuntimeError(f"refusing to overwrite completed PHASE 36 run: {result_path}")
        result = run_training(
            settings=settings,
            variant="current",
            seed=seed,
            output_dir=ROOT / "checkpoints/foundation-v25-current",
            token_budget=2_048_000,
            include_generation=(seed == int(settings["representative_generation_seed"])),
            resume=resume_paths[seed],
        )
        result.update({
            "schema": "foundation-v25-current-2048k-run-v1",
            "phase": 36,
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
