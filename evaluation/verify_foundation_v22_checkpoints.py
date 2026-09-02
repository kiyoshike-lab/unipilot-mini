"""Strict integrity and resumability audit for all new PHASE 33 milestones."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

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
)


SEEDS = (42, 123, 2026)
MILESTONES = (320_000, 384_000, 448_000, 512_000)


def known_hashes(run_root: Path) -> dict[str, str]:
    result = {}
    for path in run_root.glob("current-seed-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["training"]["history"]:
            checkpoint = row.get("checkpoint")
            if checkpoint:
                result[checkpoint["path"]] = checkpoint["sha256"]
    return result


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v22.json")
    hashes = known_hashes(ROOT / "evaluation/foundation-v22-runs")
    rows = []
    for seed in SEEDS:
        for tokens in MILESTONES:
            path = ROOT / "checkpoints/foundation-v22-current/current" / f"seed-{seed}" / f"checkpoint-tokens-{tokens}.pt"
            if not path.is_file():
                raise RuntimeError(f"missing PHASE 33 checkpoint: {path}")
            relative = path.relative_to(ROOT).as_posix()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            restored = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
            restored.load_state_dict(payload["model_state"], strict=True)
            digest = file_sha256(path)
            update = tokens // 512
            scheduler = stateless_scheduler_state(settings, update)
            strict_reload = all(
                torch.equal(left, right)
                for left, right in zip(payload["model_state"].values(), restored.state_dict().values())
            )
            row = {
                "path": relative,
                "seed": seed,
                "tokens": tokens,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "expected_sha256": hashes.get(relative),
                "hash_match": hashes.get(relative) == digest,
                "format_match": payload.get("checkpoint_format") == CHECKPOINT_FORMAT,
                "identity_match": payload.get("variant") == "current" and int(payload.get("seed", -1)) == seed and int(payload.get("tokens_processed", -1)) == tokens,
                "strict_reload": strict_reload,
                "optimizer_state_present": bool(payload.get("optimizer_state", {}).get("state")),
                "random_state_complete": set(payload.get("random_state", {})) == {"python", "numpy", "torch_cpu"},
                "scheduler_state_match": payload.get("scheduler_state") == scheduler,
                "resume_state_complete": all(name in payload for name in ("optimizer_state", "permutation", "random_state", "scheduler_state", "update")),
            }
            row["pass"] = all(row[name] for name in (
                "hash_match", "format_match", "identity_match", "strict_reload",
                "optimizer_state_present", "random_state_complete", "scheduler_state_match",
                "resume_state_complete",
            ))
            rows.append(row)
            print(json.dumps({"checkpoint": relative, "pass": row["pass"]}), flush=True)
            del payload, restored
            gc.collect()
    result = {
        "schema": "foundation-v22-checkpoint-verification-v1",
        "expected_checkpoints": len(SEEDS) * len(MILESTONES),
        "verified_checkpoints": len(rows),
        "strict_reload_pass": all(row["strict_reload"] for row in rows),
        "resume_state_complete": all(row["resume_state_complete"] for row in rows),
        "integrity_pass": all(row["pass"] for row in rows),
        "rows": rows,
    }
    output = ROOT / "evaluation/foundation-v22-checkpoint-verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"integrity_pass": result["integrity_pass"], "verified": len(rows)}, indent=2))
    return 0 if result["integrity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
