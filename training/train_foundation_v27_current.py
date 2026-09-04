"""PHASE 38: 5.120M→7.168M gate, then 10.240M only on PASS."""
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


SEEDS = (42, 123, 2026)
TARGETS = {"gate": 7_168_000, "final": 10_240_000}


def resume_paths(stage: str) -> dict[int, Path]:
    tokens = 5_120_000 if stage == "gate" else 7_168_000
    return {seed: ROOT / f"checkpoints/foundation-v26-current/current/seed-{seed}/checkpoint-tokens-{tokens}.pt" for seed in SEEDS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "final"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    settings = load_json("configs/unipilot-foundation-v27.json")
    seeds = [args.seed] if args.seed is not None else list(settings["seeds"])
    paths = resume_paths(args.stage)
    audits = {str(seed): preflight_resume(settings, seed, paths[seed]) for seed in seeds}
    expected_start = 5_120_000 if args.stage == "gate" else 7_168_000
    if any(audits[str(seed)]["tokens_processed"] != expected_start for seed in seeds):
        raise RuntimeError(f"PHASE 38 {args.stage} stage has an incorrect resume point")
    if args.preflight_only:
        write_json(ROOT / f"evaluation/foundation-v27-{args.stage}-resume-preflight.json", {"schema": "foundation-v27-resume-preflight-v1", "phase": 38, "stage": args.stage, "target_tokens": TARGETS[args.stage], "audits": audits, "architecture_changed": False, "corpus_changed": False, "tokenizer_changed": False})
        print(json.dumps({"status": "PASS", "stage": args.stage}, indent=2))
        return 0
    output_name = "foundation-v27-gate-runs" if args.stage == "gate" else "foundation-v27-runs"
    for seed in seeds:
        result_path = ROOT / f"evaluation/{output_name}/current-seed-{seed}.json"
        if result_path.exists():
            raise RuntimeError(f"refusing to overwrite completed PHASE 38 result: {result_path}")
        result = run_training(settings=settings, variant="current", seed=seed, output_dir=ROOT / "checkpoints/foundation-v26-current", token_budget=TARGETS[args.stage], include_generation=(seed == int(settings["representative_generation_seed"])), resume=paths[seed])
        result.update({"schema": "foundation-v27-current-run-v1", "phase": 38, "stage": args.stage, "resume_preflight": audits[str(seed)], "continuity": {"resumed_from_tokens": expected_start, "no_new_warmup": True, "learning_rate_continuous": True, "data_order_continuous": True}, "production_changed": False, "campus_changed": False, "final_blind_used": False})
        write_json(result_path, result)
        print(json.dumps({"stage": args.stage, "seed": seed, "tokens": result["final"]["tokens_processed"], "validation_loss": result["final"]["validation"]["loss"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
