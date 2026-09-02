"""Strict PHASE 35 checkpoint integrity and resume-state verification."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import (
    CHECKPOINT_FORMAT,
    file_sha256,
    load_json,
    stateless_scheduler_state,
    save_checkpoint,
)
from training.train_foundation_v22_current import expected_permutation
from training.optimizer import create_optimizer


EXPECTED = {
    42: (768_000, 896_000, 1_024_000),
    123: (640_000, 768_000, 896_000, 1_024_000),
    2026: (640_000, 768_000, 896_000, 1_024_000),
}


def result_hashes() -> dict[str, str]:
    hashes = {}
    for seed in EXPECTED:
        path = ROOT / f"evaluation/foundation-v24-runs/current-seed-{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["training"]["history"]:
            checkpoint = row.get("checkpoint")
            if checkpoint:
                hashes[checkpoint["path"]] = checkpoint["sha256"]
    return hashes


def resume_reproducibility() -> dict:
    """Compare a continuous next step with the same step after save/reload."""
    torch.manual_seed(35)
    config = DiagnosticConfigV17(
        model_name="phase35 resume verification",
        vocab_size=64,
        context_length=16,
        embedding_dim=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        dropout=0.0,
    )
    continuous = DiagnosticTransformerV17(config)
    continuous_optimizer = create_optimizer(continuous, 1e-4, 0.1)
    x = torch.arange(16).remainder(64)[None]
    y = torch.arange(1, 17).remainder(64)[None]
    continuous_optimizer.zero_grad(set_to_none=True)
    _, loss = continuous(x, y)
    loss.backward()
    continuous_optimizer.step()
    settings = {
        "maximum_allowed_tokens_per_run": 1_024_000,
        "training": {"peak_learning_rate": 1e-4, "warmup_updates": 20},
    }
    with tempfile.TemporaryDirectory(prefix="foundation-v24-resume-") as folder:
        checkpoint = Path(folder) / "checkpoint.pt"
        save_checkpoint(
            checkpoint,
            model=continuous,
            optimizer=continuous_optimizer,
            variant="current",
            seed=35,
            update=1,
            permutation=torch.arange(32),
            history=[],
            training_seconds=0.0,
            settings=settings,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    resumed_optimizer = create_optimizer(resumed, 1e-4, 0.1)
    resumed.load_state_dict(payload["model_state"], strict=True)
    resumed_optimizer.load_state_dict(payload["optimizer_state"])
    for model, optimizer in ((continuous, continuous_optimizer), (resumed, resumed_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        _, next_loss = model(x, y)
        next_loss.backward()
        optimizer.step()
    bitwise_equal = all(
        torch.equal(left, right)
        for left, right in zip(continuous.parameters(), resumed.parameters())
    )
    return {
        "method": "continuous next optimizer step vs save/reload/resume next optimizer step",
        "compared_steps": 1,
        "bitwise_equal_parameters": bitwise_equal,
        "status": "PASS" if bitwise_equal else "FAIL",
    }


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v24.json")
    hashes = result_hashes()
    rows = []
    for seed, milestones in EXPECTED.items():
        permutation = expected_permutation(settings, seed)
        for tokens in milestones:
            path = ROOT / f"checkpoints/foundation-v24-current/current/seed-{seed}/checkpoint-tokens-{tokens}.pt"
            if not path.is_file():
                raise RuntimeError(f"missing PHASE 35 checkpoint: {path}")
            relative = path.relative_to(ROOT).as_posix()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
            model.load_state_dict(payload["model_state"], strict=True)
            strict_reload = all(
                torch.equal(left, right)
                for left, right in zip(payload["model_state"].values(), model.state_dict().values())
            )
            update = tokens // 512
            digest = file_sha256(path)
            checks = {
                "hash_match": hashes.get(relative) == digest,
                "format_match": payload.get("checkpoint_format") == CHECKPOINT_FORMAT,
                "identity_match": payload.get("variant") == "current" and int(payload.get("seed", -1)) == seed,
                "processed_tokens_match": int(payload.get("tokens_processed", -1)) == tokens and int(payload.get("update", -1)) == update,
                "strict_reload": strict_reload,
                "optimizer_state_present": bool(payload.get("optimizer_state", {}).get("state")),
                "scheduler_state_match": payload.get("scheduler_state") == stateless_scheduler_state(settings, update),
                "rng_state_complete": set(payload.get("random_state", {})) == {"python", "numpy", "torch_cpu"},
                "sampler_match": torch.equal(payload.get("permutation"), permutation),
                "next_dataset_position_valid": update < len(permutation),
            }
            row = {
                "path": relative,
                "seed": seed,
                "tokens": tokens,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "next_macroblock_index": int(permutation[update]),
                "checks": checks,
                "pass": all(checks.values()),
            }
            rows.append(row)
            print(json.dumps({"checkpoint": relative, "pass": row["pass"]}), flush=True)
            del payload, model
            gc.collect()
    result = {
        "schema": "foundation-v24-checkpoint-verification-v1",
        "phase": 35,
        "expected_checkpoints": 11,
        "verified_checkpoints": len(rows),
        "integrity_pass": len(rows) == 11 and all(row["pass"] for row in rows),
        "resume_reproducibility": resume_reproducibility(),
        "rows": rows,
    }
    output = ROOT / "evaluation/foundation-v24-checkpoint-verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": len(rows), "integrity_pass": result["integrity_pass"]}, indent=2))
    return 0 if result["integrity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
