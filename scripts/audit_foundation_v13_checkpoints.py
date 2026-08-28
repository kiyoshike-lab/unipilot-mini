from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.train_foundation_v13 import CHECKPOINT_FORMAT, sha256


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_checkpoint(path: Path, expected_step: int, settings: dict, corpus: dict) -> dict:
    manifest_path = path.with_suffix(".manifest.json")
    checks = {
        "checkpoint_exists": path.exists(),
        "manifest_exists": manifest_path.exists(),
    }
    if not all(checks.values()):
        return {"step": expected_step, "path": str(path), "checks": checks, "status": "FAIL"}
    manifest = load_json(manifest_path)
    digest = sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "checkpoint_format", "model_state", "optimizer_state", "scheduler_state",
        "global_step", "step", "random_state", "sampler_state", "config",
        "foundation_v13_manifest",
    }
    random_required = {"python", "numpy", "torch_cpu"}
    sampler_required = {
        "size", "seed", "epoch", "position", "permutation", "generator_state",
    }
    scheduler = payload.get("scheduler_state", {})
    sampler = payload.get("sampler_state", {})
    model = UniPilotTransformer(ModelConfig(**payload["config"]))
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    finite = all(
        bool(torch.isfinite(value).all().item())
        for value in payload["model_state"].values()
        if value.dtype.is_floating_point
    )
    optimizer_state = payload.get("optimizer_state", {})
    tokens_processed = expected_step * settings["batch_size"] * settings["model"]["context_length"]
    checks.update({
        "required_payload_keys": required <= payload.keys(),
        "format": payload.get("checkpoint_format") == CHECKPOINT_FORMAT,
        "step": payload.get("global_step") == payload.get("step") == expected_step,
        "manifest_step": manifest.get("global_step") == expected_step,
        "sha256": manifest.get("checkpoint_sha256") == digest,
        "bytes": manifest.get("checkpoint_bytes") == path.stat().st_size,
        "model_state_loads": not missing and not unexpected,
        "model_state_finite": finite,
        "parameters": manifest.get("parameters") == model.parameter_count() == 19_514_880,
        "optimizer_state": bool(optimizer_state.get("state")) and bool(
            optimizer_state.get("param_groups")
        ),
        "scheduler_global_step": scheduler.get("global_step") == expected_step,
        "scheduler_config": (
            scheduler.get("base_learning_rate") == settings["learning_rate"]
            and scheduler.get("warmup_steps") == settings["warmup_steps"]
            and scheduler.get("schedule_steps") == settings["schedule_steps"]
            and scheduler.get("minimum_ratio") == settings["minimum_learning_rate_ratio"]
        ),
        "random_states": random_required <= payload.get("random_state", {}).keys(),
        "sampler_state": sampler_required <= sampler.keys(),
        "sampler_position": sampler.get("position") == expected_step,
        "sampler_permutation_size": int(sampler.get("permutation", torch.empty(0)).numel())
                                    == int(sampler.get("size", -1)),
        "config": payload.get("config") == manifest.get("model_config"),
        "tokenizer": manifest.get("tokenizer") == corpus["tokenizer"],
        "corpus": manifest.get("corpus_manifest") == settings["corpus_manifest"],
        "tokens_processed": manifest.get("tokens_processed") == tokens_processed,
        "final_blind_unused": manifest.get("final_blind_used") is False,
        "external_ai_off": manifest.get("external_ai_api") == "OFF",
        "production_unchanged": manifest.get("production_changed") is False,
    })
    del model, payload
    return {
        "step": expected_step,
        "path": path.relative_to(ROOT).as_posix(),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v13.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/foundation-v13-clean-250")
    parser.add_argument("--output", default="evaluation/foundation-v13-checkpoint-integrity.json")
    args = parser.parse_args()
    settings = load_json(ROOT / args.config)
    corpus = load_json(ROOT / settings["corpus_manifest"])
    steps = list(settings["checkpoint_steps"])
    results = [
        audit_checkpoint(
            ROOT / args.checkpoint_dir / f"checkpoint-step-{step}.pt",
            step, settings, corpus,
        )
        for step in steps
    ]
    report = {
        "schema_version": "foundation-v13-checkpoint-integrity-v1",
        "expected_steps": steps,
        "results": results,
        "all_checkpoints": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_checkpoints"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
