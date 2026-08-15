from __future__ import annotations

import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from evaluation.metrics_v03 import semantic_score
from inference.generate import generate_text, load_model
from training.dataset_v03 import SYSTEM_TEXT


ROOT = Path(__file__).resolve().parents[1]; EVAL = ROOT / "evaluation"


def load(name): return json.loads((EVAL / name).read_text(encoding="utf-8"))


def special_comparisons():
    prompts = [
        ("明日試験なんだけど何したらいい？", "exam"), ("課題が3つあるんだけどどれからやればいい？", "assignment"),
        ("教授に欠席メールを送りたい", "email"), ("単位が心配", "credit"), ("レポートが終わらない", "report"),
        ("履修をどう決めればいい？", "registration"), ("空きコマ何したらいい？", "study"), ("明日の試験って何時？", "general"),
        ("今日何したらいい？", "schedule"), ("出席が少なくて心配", "attendance")]
    schemas = {row["category"]: row for row in load("fixed_prompts_v03.json")}
    models = [("v03", *load_model("checkpoints/v03-scratch-001/stage-c/checkpoint-step-5000.pt", "tokenizer/vocab-v02-512.json")),
              ("v04", *load_model("checkpoints/v04-eos15/checkpoint-step-2000.pt", "tokenizer/vocab-v02-512.json"))]
    rows = []
    for index, (prompt, category) in enumerate(prompts):
        item = {**schemas[category], "id": f"special-v04-{index:02d}", "prompt": prompt}
        entry = {"id": item["id"], "prompt": prompt, "category": category, "expected_keywords": item["expected_keywords"]}
        for label, model, tokenizer, _, _ in models:
            random.seed(42 + index); torch.manual_seed(42 + index)
            formatted = f"<BOS><SYSTEM>\n{SYSTEM_TEXT}\n<USER>\n{prompt}\n<ASSISTANT>\n"
            answer, speed = generate_text(model, tokenizer, formatted, 128 if label == "v03" else 96, .7, 40, .9, 1.0 if label == "v03" else 1.1)
            entry[label] = {"answer": answer, "eos_reached": speed["eos_reached"], **semantic_score(answer, item)}
        rows.append(entry)
    return rows


def comparison():
    old = load("results-v03-5000.json"); new = load("results-v04-best-2000.json")
    old_by = {row["id"]: row for row in old["generations"]}; selected = []
    categories = sorted({row["category"] for row in new["generations"]})
    for category in categories:
        selected.extend([row for row in new["generations"] if row["category"] == category][:2])
    rows = special_comparisons()
    for item in selected:
        prior = old_by[item["id"]]
        rows.append({"id": item["id"], "prompt": item["prompt"], "category": item["category"],
                     "expected_keywords": item["expected_keywords"],
                     "v03": {"answer": prior["answer"], "eos_reached": prior["eos_reached"], "relevance_score": prior["relevance_score"],
                             "meaningful_response": prior["meaningful_response"], "category_correct": prior["category_correct"]},
                     "v04": {"answer": item["answer"], "eos_reached": item["eos_reached"], "relevance_score": item["relevance_score"],
                             "meaningful_response": item["meaningful_response"], "category_correct": item["category_correct"]}})
    payload = {"v03_metrics": old["metrics"], "v04_metrics": new["metrics"], "comparisons": rows}
    (EVAL / "v03-v04-generations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# UniPilot Mini v0.3 / v0.4 生成比較", "", "先頭10件は指定された重要質問、残り20件は固定評価の各カテゴリ2件です。", ""]
    for index, row in enumerate(rows, 1):
        lines += [f"## {index}. {row['prompt']}", "", f"Expected category: `{row['category']}`  ", f"Expected keywords: {', '.join(row['expected_keywords'])}", "",
                  "v0.3:", "", row["v03"]["answer"], "", "v0.4:", "", row["v04"]["answer"], "",
                  f"Automatic metrics: relevance {row['v03']['relevance_score']:.1f} → {row['v04']['relevance_score']:.1f}; meaningful {row['v03']['meaningful_response']} → {row['v04']['meaningful_response']}; EOS {row['v03']['eos_reached']} → {row['v04']['eos_reached']}.", ""]
    (ROOT / "V03_V04_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def plots():
    stages = [(500, load("results-v04-eos15-500.json")), (1000, load("results-v04-eos15-1000.json")), (2000, load("results-v04-best-2000.json"))]
    metrics = [("eos_reached_rate", "EOS vs Step", "v04-eos-vs-step.png", 100),
               ("runaway_generation_rate", "Runaway vs Step", "v04-runaway-vs-step.png", 100),
               ("meaningful_response_rate", "Meaningful vs Step", "v04-meaningful-vs-step.png", 100),
               ("keyword_relevance", "Keyword vs Step", "v04-keyword-vs-step.png", 1),
               ("category_accuracy", "Category Accuracy vs Step", "v04-category-vs-step.png", 100),
               ("repetition_rate", "Repetition vs Step", "v04-repetition-vs-step.png", 100)]
    for key, title, filename, scale in metrics:
        plt.figure(figsize=(7, 4)); plt.plot([s for s, _ in stages], [r["metrics"][key] * scale for _, r in stages], marker="o")
        plt.title(title); plt.xlabel("Step"); plt.ylabel("Percent"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(EVAL / filename, dpi=160); plt.close()


if __name__ == "__main__": comparison(); plots()
