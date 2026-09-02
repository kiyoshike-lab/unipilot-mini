from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import CHECKPOINT_FORMAT, file_sha256


VARIANTS = ("current", "depth_init")
SEEDS = (42, 123, 2026)
MILESTONES = (0, 64_000, 128_000, 192_000, 256_000)


def expected_hashes(run_root: Path) -> dict[str, str]:
    result = {}
    for path in run_root.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["training"]["history"]:
            checkpoint = row.get("checkpoint")
            if checkpoint:
                result[checkpoint["path"]] = checkpoint["sha256"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="checkpoints/foundation-v21-ab")
    parser.add_argument("--run-root", default="evaluation/foundation-v21-runs")
    parser.add_argument("--output", default="evaluation/foundation-v21-checkpoint-verification.json")
    args = parser.parse_args()
    known_hashes = expected_hashes(ROOT / args.run_root)
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for tokens in MILESTONES:
                path = (
                    ROOT / args.checkpoint_root / variant / f"seed-{seed}"
                    / f"checkpoint-tokens-{tokens}.pt"
                )
                if not path.exists():
                    raise RuntimeError(f"missing PHASE 32 checkpoint: {path}")
                relative = path.relative_to(ROOT).as_posix()
                digest = file_sha256(path)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                format_match = payload.get("checkpoint_format") == CHECKPOINT_FORMAT
                identity_match = (
                    payload.get("variant") == variant
                    and int(payload.get("seed")) == seed
                    and int(payload.get("tokens_processed")) == tokens
                )
                restored = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
                restored.load_state_dict(payload["model_state"], strict=True)
                strict_reload = all(
                    torch.equal(expected, actual)
                    for expected, actual in zip(
                        payload["model_state"].values(), restored.state_dict().values()
                    )
                )
                required_resume_state = all(
                    name in payload
                    for name in ("optimizer_state", "permutation", "random_state", "update")
                )
                expected_digest = known_hashes.get(relative)
                hash_match = expected_digest is None or digest == expected_digest
                row = {
                    "path": relative,
                    "variant": variant,
                    "seed": seed,
                    "tokens": tokens,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "expected_sha256": expected_digest,
                    "hash_match": hash_match,
                    "format_match": format_match,
                    "identity_match": identity_match,
                    "strict_reload": strict_reload,
                    "resume_state_complete": required_resume_state,
                    "pass": all((
                        hash_match, format_match, identity_match, strict_reload,
                        required_resume_state,
                    )),
                }
                rows.append(row)
                print(json.dumps({
                    "checkpoint": relative,
                    "pass": row["pass"],
                }), flush=True)
                del payload, restored
                gc.collect()
    result = {
        "schema_version": "foundation-v21-checkpoint-verification-v1",
        "expected_checkpoints": 30,
        "verified_checkpoints": len(rows),
        "known_hashes_verified": sum(row["expected_sha256"] is not None for row in rows),
        "strict_reload_pass": all(row["strict_reload"] for row in rows),
        "integrity_pass": all(row["pass"] for row in rows),
        "resume_state_complete": all(row["resume_state_complete"] for row in rows),
        "rows": rows,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "verified": len(rows),
        "integrity_pass": result["integrity_pass"],
        "output": output.relative_to(ROOT).as_posix(),
    }, indent=2))
    return 0 if result["integrity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
