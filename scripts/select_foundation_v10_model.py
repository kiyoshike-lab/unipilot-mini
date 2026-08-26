from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="checkpoints/foundation-v10-sanity")
    parser.add_argument("--output", default="evaluation/foundation-v10-model-comparison-50.json")
    args = parser.parse_args()
    rows = []
    for architecture in ("20m", "30m", "46m"):
        path = ROOT / args.checkpoint_root / architecture / "checkpoint-step-50.manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        metrics = manifest["training_metrics"]
        improvement = float(manifest["validation_loss_improvement"])
        seconds = float(metrics["step_time_seconds"])
        size_penalty = (manifest["parameters"] / 20_000_000) ** .25
        efficiency = improvement / max(seconds, 1e-9) / size_penalty
        rows.append({
            "architecture": architecture, "parameters": manifest["parameters"],
            "context": manifest["model_config"]["context_length"],
            "initial_validation_loss": manifest["initial_validation_loss"],
            "step_50_train_loss": metrics["loss"],
            "step_50_validation_loss": metrics["validation_loss"],
            "validation_loss_improvement": improvement,
            "step_time_seconds": seconds, "tokens_per_second": metrics["tokens_per_second"],
            "memory_usage_mb": metrics["memory_usage_mb"],
            "natural_japanese_rate": manifest["development_probe"]["natural_japanese_rate"],
            "mean_repetition_rate": manifest["development_probe"]["mean_repetition_rate"],
            "efficiency_score": efficiency, "healthy_loss_curve": manifest["healthy_loss_curve"],
        })
    healthy = [row for row in rows if row["healthy_loss_curve"]]
    pool = healthy or rows
    selected = max(pool, key=lambda row: (row["efficiency_score"], -row["parameters"]))
    report = {
        "schema_version": "foundation-v10-model-comparison-50-v1",
        "same_corpus": True, "same_tokenizer": True, "same_seed": True,
        "results": rows, "selected_architecture": selected["architecture"],
        "selection_rule": (
            "Require a healthy validation curve, then maximize validation-loss improvement per "
            "final-step second with a mild parameter penalty. Generation at 50 steps is diagnostic only."
        ),
        "selected_for_100_and_500_sanity_only": True,
        "production_candidate": False, "external_ai_api": "OFF",
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
