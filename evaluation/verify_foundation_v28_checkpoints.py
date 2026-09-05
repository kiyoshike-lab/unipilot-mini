"""Verify PHASE 39 CUDA checkpoints remain complete and CPU portable."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import file_sha256, stateless_scheduler_state


SEEDS = (42, 123, 2026)
TOKENS = (12_288_000, 15_360_000)


def main() -> int:
    settings = json.loads(
        (ROOT / "configs/unipilot-foundation-v28.json").read_text(encoding="utf-8")
    )
    rows = []
    for seed in SEEDS:
        for tokens in TOKENS:
            path = ROOT / (
                f"checkpoints/foundation-v28-current/current/seed-{seed}/"
                f"checkpoint-tokens-{tokens}.pt"
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
            model.load_state_dict(payload["model_state"], strict=True)
            update = tokens // 512
            checks = {
                "strict_cpu_reload": True,
                "optimizer": bool(payload.get("optimizer_state", {}).get("state")),
                "rng": {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(
                    payload.get("random_state", {})
                ),
                "sampler": torch.is_tensor(payload.get("permutation")),
                "scheduler": payload.get("scheduler_state") == stateless_scheduler_state(settings, update),
                "tokens": int(payload.get("tokens_processed", 0)) == tokens,
                "device_metadata": payload.get("device_metadata", {}).get("device", "").startswith("cuda"),
                "precision_fp32": payload.get("precision_mode") == "fp32",
                "sha256": bool(file_sha256(path)),
            }
            rows.append({
                "seed": seed,
                "tokens": tokens,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(path),
                "checks": checks,
                "pass": all(checks.values()),
            })
            del payload, model
    expected = len(SEEDS) * len(TOKENS)
    output = {
        "schema": "foundation-v28-checkpoint-verification-v1",
        "phase": 39,
        "expected_checkpoints": expected,
        "verified_checkpoints": len(rows),
        "integrity_pass": len(rows) == expected and all(row["pass"] for row in rows),
        "rows": rows,
    }
    destination = ROOT / "evaluation/foundation-v28-checkpoint-verification.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": len(rows), "integrity_pass": output["integrity_pass"]}))
    return 0 if output["integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
