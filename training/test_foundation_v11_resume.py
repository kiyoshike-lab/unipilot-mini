from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_foundation_v11 import sha256, train_run


def tensor_state_differences(left, right, prefix: str = "") -> tuple[bool, float, list[str]]:
    mismatches: list[str] = []
    maximum = 0.0
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        exact = torch.equal(left, right)
        if left.numel() and left.dtype.is_floating_point:
            maximum = float((left - right).abs().max().item())
        if not exact:
            mismatches.append(prefix)
        return exact, maximum, mismatches
    if isinstance(left, dict) and isinstance(right, dict):
        exact = left.keys() == right.keys()
        if not exact:
            mismatches.append(prefix + ".keys")
        for key in left.keys() & right.keys():
            item_exact, item_max, item_mismatches = tensor_state_differences(
                left[key], right[key], f"{prefix}.{key}"
            )
            exact &= item_exact
            maximum = max(maximum, item_max)
            mismatches.extend(item_mismatches)
        return exact, maximum, mismatches
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        exact = len(left) == len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            item_exact, item_max, item_mismatches = tensor_state_differences(
                left_item, right_item, f"{prefix}[{index}]"
            )
            exact &= item_exact
            maximum = max(maximum, item_max)
            mismatches.extend(item_mismatches)
        return exact, maximum, mismatches
    exact = left == right
    if not exact:
        mismatches.append(prefix)
    return exact, maximum, mismatches


def main() -> int:
    settings = json.loads((ROOT / "configs/unipilot-foundation-v11.json").read_text(
        encoding="utf-8"
    ))
    with tempfile.TemporaryDirectory(prefix="unipilot-v11-resume-") as temp:
        base = Path(temp)
        run_a = train_run(settings=settings, max_steps=40, output_dir=base / "a",
                          checkpoint_steps={0, 20, 40}, cpu_threads=1)
        run_b_first = train_run(settings=settings, max_steps=20, output_dir=base / "b",
                                checkpoint_steps={0, 20}, cpu_threads=1)
        run_b = train_run(settings=settings, max_steps=40, output_dir=base / "b",
                          checkpoint_steps={40}, resume=run_b_first["checkpoint"], cpu_threads=1)
        losses_a = run_a["step_losses"]
        losses_b = run_b_first["step_losses"] + run_b["step_losses"]
        loss_exact = losses_a == losses_b
        max_loss_difference = max(abs(left - right) for left, right in zip(losses_a, losses_b))
        weights_exact, max_weight_difference, weight_mismatches = tensor_state_differences(
            run_a["model"].state_dict(), run_b["model"].state_dict(), "model"
        )
        optimizer_exact, max_optimizer_difference, optimizer_mismatches = tensor_state_differences(
            run_a["optimizer"].state_dict(), run_b["optimizer"].state_dict(), "optimizer"
        )
        payload_a = torch.load(run_a["checkpoint"], map_location="cpu", weights_only=False)
        payload_b = torch.load(run_b["checkpoint"], map_location="cpu", weights_only=False)
        scheduler_exact = payload_a["scheduler_state"] == payload_b["scheduler_state"]
        sampler_exact, _, sampler_mismatches = tensor_state_differences(
            payload_a["sampler_state"], payload_b["sampler_state"], "sampler"
        )
        required_keys = {
            "model_state", "optimizer_state", "scheduler_state", "global_step",
            "random_state", "sampler_state", "config", "foundation_v11_manifest",
        }
        checkpoint_integrity = required_keys <= payload_b.keys() and payload_b["global_step"] == 40
        checks = {
            "loss_sequence_bitwise_equal": loss_exact,
            "weights_bitwise_equal": weights_exact,
            "optimizer_bitwise_equal": optimizer_exact,
            "scheduler_equal": scheduler_exact,
            "sampler_equal": sampler_exact,
            "checkpoint_v2_integrity": checkpoint_integrity,
            "python_rng_saved": "python" in payload_b["random_state"],
            "numpy_rng_saved": "numpy" in payload_b["random_state"],
            "torch_cpu_rng_saved": "torch_cpu" in payload_b["random_state"],
        }
        report = {
            "schema_version": "foundation-v11-resume-reproducibility-v1",
            "experiment": {"A": "scratch -> 40", "B": "scratch -> 20 -> resume -> 40",
                           "seed": settings["seed"], "cpu_threads": 1},
            "loss_steps_compared": len(losses_a),
            "maximum_loss_difference": max_loss_difference,
            "maximum_weight_difference": max_weight_difference,
            "maximum_optimizer_difference": max_optimizer_difference,
            "weight_mismatches": weight_mismatches[:20],
            "optimizer_mismatches": optimizer_mismatches[:20],
            "sampler_mismatches": sampler_mismatches[:20],
            "checkpoint_a_sha256": sha256(run_a["checkpoint"]),
            "checkpoint_b_sha256": sha256(run_b["checkpoint"]),
            "checks": checks, "resume_integrity": "PASS" if all(checks.values()) else "FAIL",
            "external_ai_api": "OFF", "production_changed": False,
        }
    output = ROOT / "evaluation/foundation-v11-resume-reproducibility.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["resume_integrity"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
