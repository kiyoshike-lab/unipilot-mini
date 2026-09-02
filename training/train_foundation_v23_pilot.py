"""PHASE 34 gated representative-seed continuation from 512k to 640k."""
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_foundation_v21_ab import load_json, run_training
from training.train_foundation_v22_current import preflight_resume, write_json


RESUME = ROOT / "checkpoints/foundation-v22-current/current/seed-42/checkpoint-tokens-512000.pt"


def main() -> int:
    readiness_path = ROOT / "evaluation/foundation-v23-pilot-readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not readiness.get("pilot_allowed"):
        raise RuntimeError("PHASE 34 readiness gate forbids the 640k pilot")
    settings = load_json("configs/unipilot-foundation-v23-pilot.json")
    audit = preflight_resume(settings, 42, RESUME)
    if audit["tokens_processed"] != 512_000 or audit["next_update"] != 1001:
        raise RuntimeError("pilot must resume exactly after the 512k milestone")
    output = ROOT / "evaluation/foundation-v23-pilot-run.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite completed PHASE 34 pilot: {output}")
    result = run_training(
        settings=settings,
        variant="current",
        seed=42,
        output_dir=ROOT / "checkpoints/foundation-v23-pilot",
        token_budget=640_000,
        include_generation=True,
        resume=RESUME,
    )
    result.update({
        "schema": "foundation-v23-640k-pilot-run-v1",
        "phase": 34,
        "pilot": True,
        "resume_preflight": audit,
        "continuity": {
            "resumed_from_tokens": 512_000,
            "first_new_update": 1001,
            "learning_rate_continuous": True,
            "data_order_continuous": True,
            "no_new_warmup": True,
        },
        "one_million_training_started": False,
        "production_changed": False,
        "campus_changed": False,
        "final_blind_used": False,
    })
    write_json(output, result)
    print(json.dumps({
        "tokens": result["final"]["tokens_processed"],
        "validation_loss": result["final"]["validation"]["loss"],
        "top_1": result["final"]["validation"]["top_1_accuracy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
