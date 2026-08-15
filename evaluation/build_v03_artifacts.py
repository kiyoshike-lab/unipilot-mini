from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"


def load(name: str) -> dict:
    return json.loads((EVAL / name).read_text(encoding="utf-8"))


def metric_row(label: str, result: dict) -> dict:
    metrics = result["metrics"]
    return {
        "model": label,
        "prompts": len(result["generations"]),
        "validation_loss": result["validation_loss"],
        "perplexity": result["perplexity"],
        "japanese_ratio": metrics["japanese_character_ratio"],
        "repetition_rate": metrics["repetition_rate"],
        "keyword_relevance": metrics["keyword_relevance"],
        "category_accuracy": metrics["category_accuracy"],
        "meaningful_response_rate": metrics["meaningful_response_rate"],
        "eos_rate": metrics["eos_reached_rate"],
        "generation_tokens_per_second": metrics["generation_tokens_per_second"],
        "human_score": None,
    }


def comparison() -> None:
    v02 = load("results-v02-1000-v03suite.json")
    v03 = load("results-v03-5000.json")
    v03_by_id = {row["id"]: row for row in v03["generations"]}
    pairs = []
    for old in v02["generations"][:30]:
        new = v03_by_id[old["id"]]
        pairs.append({"id": old["id"], "prompt": old["prompt"], "category": old["category"],
                      "v02": old["answer"], "v03": new["answer"],
                      "evaluation": {"v02_relevance": old["relevance_score"], "v03_relevance": new["relevance_score"],
                                     "v02_meaningful": old["meaningful_response"], "v03_meaningful": new["meaningful_response"],
                                     "v03_eos_reached": new["eos_reached"]}})
    payload = {"note": "The aggregate v0.2 comparison uses 50 prompts; v0.3 final uses all 300. These 30 pairs use identical prompt IDs.",
               "metrics": [metric_row("v0.2-1000", v02), metric_row("v0.3-5000", v03)], "comparisons": pairs}
    (EVAL / "v02-v03-generations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# UniPilot Mini v0.2 / v0.3 生成比較", "", "集計値はv0.2が固定問題の先頭50問、v0.3が固定300問です。以下30件は同一IDの問題を対応させています。", "",
             "| Model | Loss | PPL | JP | Repetition | Keyword | Category | Meaningful | EOS | tok/s | Human |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["metrics"]:
        lines.append(f"| {row['model']} | {row['validation_loss']:.4f} | {row['perplexity']:.2f} | {row['japanese_ratio']:.2%} | {row['repetition_rate']:.2%} | {row['keyword_relevance']:.2f}% | {row['category_accuracy']:.2%} | {row['meaningful_response_rate']:.2%} | {row['eos_rate']:.2%} | {row['generation_tokens_per_second']:.2f} | 未採点 |")
    for index, pair in enumerate(pairs, 1):
        ev = pair["evaluation"]
        lines += ["", f"## {index}. {pair['prompt']}", "", f"Category: `{pair['category']}`", "", "v0.2:", "", pair["v02"], "", "v0.3:", "", pair["v03"], "",
                  f"Evaluation: relevance {ev['v02_relevance']:.1f} → {ev['v03_relevance']:.1f}; meaningful {ev['v02_meaningful']} → {ev['v03_meaningful']}; v0.3 EOS {ev['v03_eos_reached']}."]
    (ROOT / "V02_V03_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plots() -> None:
    with (ROOT / "checkpoints/v03-scratch-001/training_log.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    steps = [int(row["step"]) for row in rows]
    stages = [row["stage"] for row in rows]
    colors = {"A": "#38bdf8", "B": "#a78bfa", "C": "#34d399"}
    for filename, fields, ylabel in [
        ("v03-training-loss.png", ["train_loss"], "Loss"),
        ("v03-validation-loss.png", ["general_validation_loss", "university_validation_loss", "conversation_validation_loss"], "Validation loss"),
    ]:
        plt.figure(figsize=(9, 5))
        for field in fields:
            plt.plot(steps, [float(row[field]) for row in rows], marker="o", label=field.replace("_validation_loss", "").replace("_", " "))
        for step, stage in zip(steps, stages):
            plt.scatter(step, float(rows[steps.index(step)][fields[0]]), color=colors[stage], s=25)
        plt.xlabel("Training step"); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
        plt.savefig(EVAL / filename, dpi=160); plt.close()
    stage_results = [("v0.2", load("results-v02-1000-v03suite.json")), ("Stage A", load("results-v03-stage-a.json")),
                     ("Stage B", load("results-v03-stage-b.json")), ("Stage C", load("results-v03-5000.json"))]
    plt.figure(figsize=(9, 5))
    labels = [item[0] for item in stage_results]
    plt.plot(labels, [item[1]["metrics"]["relevance_score"] for item in stage_results], marker="o", label="relevance score")
    plt.plot(labels, [item[1]["metrics"]["keyword_relevance"] for item in stage_results], marker="o", label="keyword relevance %")
    plt.plot(labels, [100 * item[1]["metrics"]["meaningful_response_rate"] for item in stage_results], marker="o", label="meaningful %")
    plt.ylabel("Score / percent"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(EVAL / "v03-relevance.png", dpi=160); plt.close()


if __name__ == "__main__":
    comparison(); plots()
