"""Strict PHASE 37 checkpoint integrity and resume-state verification."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.verify_foundation_v24_checkpoints import resume_reproducibility
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import CHECKPOINT_FORMAT, file_sha256, load_json, stateless_scheduler_state
from training.train_foundation_v22_current import expected_permutation


SEEDS = (42, 123, 2026)
MILESTONES = (2_560_000, 3_072_000, 3_584_000, 4_096_000, 4_608_000, 5_120_000)


def result_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for seed in SEEDS:
        for directory in ("foundation-v26-gate-runs", "foundation-v26-runs"):
            payload = json.loads((ROOT / f"evaluation/{directory}/current-seed-{seed}.json").read_text(encoding="utf-8"))
            for row in payload["training"]["history"]:
                checkpoint = row.get("checkpoint")
                if checkpoint:
                    hashes[checkpoint["path"]] = checkpoint["sha256"]
    return hashes


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v26.json")
    hashes = result_hashes()
    rows = []
    for seed in SEEDS:
        permutation = expected_permutation(settings, seed)
        for tokens in MILESTONES:
            path = ROOT / f"checkpoints/foundation-v26-current/current/seed-{seed}/checkpoint-tokens-{tokens}.pt"
            if not path.is_file():
                raise RuntimeError(f"missing PHASE 37 checkpoint: {path}")
            relative = path.relative_to(ROOT).as_posix()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
            model.load_state_dict(payload["model_state"], strict=True)
            strict_reload = all(torch.equal(left, right) for left, right in zip(payload["model_state"].values(), model.state_dict().values()))
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
            rows.append({"path": relative, "seed": seed, "tokens": tokens, "bytes": path.stat().st_size, "sha256": digest, "next_macroblock_index": int(permutation[update]), "checks": checks, "pass": all(checks.values())})
            del payload, model
            gc.collect()
    reproducibility = resume_reproducibility()
    result = {
        "schema": "foundation-v26-checkpoint-verification-v1", "phase": 37,
        "expected_checkpoints": 18, "verified_checkpoints": len(rows),
        "integrity_pass": len(rows) == 18 and all(row["pass"] for row in rows),
        "resume_reproducibility": reproducibility, "rows": rows,
    }
    (ROOT / "evaluation/foundation-v26-checkpoint-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": len(rows), "integrity_pass": result["integrity_pass"], "resume": reproducibility["status"]}, indent=2))
    return 0 if result["integrity_pass"] and reproducibility["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
