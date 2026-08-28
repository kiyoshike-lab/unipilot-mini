from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_foundation_v13 import train_run


def compare(left, right, prefix: str = "") -> tuple[bool, float, list[str]]:
    mismatches = []
    maximum = 0.0
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        exact = torch.equal(left, right)
        if left.numel() and left.dtype.is_floating_point:
            maximum = float((left - right).abs().max().item())
        if not exact:
            mismatches.append(prefix)
        return exact, maximum, mismatches
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        exact = np.array_equal(left, right)
        if not exact:
            mismatches.append(prefix)
        return exact, maximum, mismatches
    if isinstance(left, dict) and isinstance(right, dict):
        exact = left.keys() == right.keys()
        if not exact:
            mismatches.append(prefix + ".keys")
        for key in left.keys() & right.keys():
            item_exact, item_maximum, item_mismatches = compare(
                left[key], right[key], f"{prefix}.{key}"
            )
            exact &= item_exact
            maximum = max(maximum, item_maximum)
            mismatches.extend(item_mismatches)
        return exact, maximum, mismatches
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        exact = len(left) == len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            item_exact, item_maximum, item_mismatches = compare(
                left_item, right_item, f"{prefix}[{index}]"
            )
            exact &= item_exact
            maximum = max(maximum, item_maximum)
            mismatches.extend(item_mismatches)
        return exact, maximum, mismatches
    exact = left == right
    if not exact:
        mismatches.append(prefix)
    return bool(exact), maximum, mismatches


def history_dynamics(history: list[dict], steps: set[int]) -> list[dict]:
    keys = (
        "step", "train_loss", "validation_loss", "perplexity", "learning_rate",
        "gradient_norm", "tokens_processed", "corpus_fraction", "corpus_percentage",
        "epoch_equivalent",
    )
    return [{key: row[key] for key in keys} for row in history if row["step"] in steps]


def main() -> int:
    settings = json.loads((ROOT / "configs/unipilot-foundation-v13.json").read_text(
        encoding="utf-8"
    ))
    checkpoint_dir = ROOT / "checkpoints/foundation-v13-clean-250"
    source_200 = checkpoint_dir / "checkpoint-step-200.pt"
    source_250 = checkpoint_dir / "checkpoint-step-250.pt"
    original = torch.load(source_250, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory(prefix="unipilot-v13-resume-") as temp:
        resumed = train_run(
            settings=settings,
            max_steps=250,
            output_dir=Path(temp),
            metric_steps={225, 250},
            checkpoint_steps={250},
            metrics_output=None,
            resume=source_200,
            device="cpu",
            cpu_threads=int(settings["cpu_threads"]),
        )
        replay = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
        weights_exact, max_weight, weight_mismatches = compare(
            original["model_state"], replay["model_state"], "model"
        )
        optimizer_exact, max_optimizer, optimizer_mismatches = compare(
            original["optimizer_state"], replay["optimizer_state"], "optimizer"
        )
        scheduler_exact, _, scheduler_mismatches = compare(
            original["scheduler_state"], replay["scheduler_state"], "scheduler"
        )
        sampler_exact, _, sampler_mismatches = compare(
            original["sampler_state"], replay["sampler_state"], "sampler"
        )
        random_exact, _, random_mismatches = compare(
            original["random_state"], replay["random_state"], "random"
        )
        original_history = history_dynamics(
            original["foundation_v13_manifest"]["history"], {225, 250}
        )
        replay_history = history_dynamics(
            replay["foundation_v13_manifest"]["history"], {225, 250}
        )
        history_exact, max_history, history_mismatches = compare(
            original_history, replay_history, "history"
        )
        checks = {
            "model_weights_bitwise_equal": weights_exact,
            "optimizer_bitwise_equal": optimizer_exact,
            "scheduler_equal": scheduler_exact,
            "sampler_equal": sampler_exact,
            "random_state_equal": random_exact,
            "step_225_250_dynamics_equal": history_exact,
            "global_step_250": replay["global_step"] == original["global_step"] == 250,
        }
        report = {
            "schema_version": "foundation-v13-resume-reproducibility-v1",
            "experiment": "formal step 200 checkpoint -> replay 250 vs formal scratch step 250",
            "source_checkpoint": source_200.relative_to(ROOT).as_posix(),
            "reference_checkpoint": source_250.relative_to(ROOT).as_posix(),
            "seed": settings["seed"],
            "cpu_threads": settings["cpu_threads"],
            "maximum_weight_difference": max_weight,
            "maximum_optimizer_difference": max_optimizer,
            "maximum_history_difference": max_history,
            "weight_mismatches": weight_mismatches[:20],
            "optimizer_mismatches": optimizer_mismatches[:20],
            "scheduler_mismatches": scheduler_mismatches[:20],
            "sampler_mismatches": sampler_mismatches[:20],
            "random_mismatches": random_mismatches[:20],
            "history_mismatches": history_mismatches[:20],
            "checks": checks,
            "resume_integrity": "PASS" if all(checks.values()) else "FAIL",
            "final_blind_used": False,
            "external_ai_api": "OFF",
            "production_changed": False,
        }
    output = ROOT / "evaluation/foundation-v13-resume-reproducibility.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["resume_integrity"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
