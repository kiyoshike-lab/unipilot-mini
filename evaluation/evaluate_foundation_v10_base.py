from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import sys
import time

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.train_foundation_v10 import bigram_overlap, generate, natural_text


def test_documents(limit: int = 60) -> list[dict]:
    selected = []
    with gzip.open(ROOT / "data/foundation_v10/documents/test.jsonl.gz", "rt",
                   encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if len(row["text"]) >= 500:
                selected.append(row)
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        raise RuntimeError(f"Foundation test documents too small: {len(selected)}")
    return selected


def reasoning_rows() -> list[dict]:
    rows = []
    for index in range(10):
        left, right = index + 7, index + 3
        rows.append({"category": "comparison", "prompt":
                     f"青い箱には{left}個、白い箱には{right}個の玉がある。玉が多いのは",
                     "expected_keywords": ["青い箱"], "reference": "青い箱である。"})
    names = (("春", "夏"), ("午前", "午後"), ("月曜日", "火曜日"), ("第一章", "第二章"))
    for index in range(8):
        first, second = names[index % len(names)]
        rows.append({"category": "ordering", "prompt":
                     f"{first}は{second}より先である。先に来るのは",
                     "expected_keywords": [first], "reference": f"{first}である。"})
    conditions = (("雨が降る", "傘を持つ", "雨が降っている", "傘"),
                  ("気温が下がる", "上着を着る", "気温が下がった", "上着"),
                  ("電池が切れる", "充電する", "電池が切れた", "充電"),
                  ("締切になる", "提出する", "締切になった", "提出"))
    for index in range(8):
        condition, action, fact, expected = conditions[index % len(conditions)]
        rows.append({"category": "condition", "prompt":
                     f"{condition}なら{action}。{fact}ので、次にすることは",
                     "expected_keywords": [expected], "reference": action + "。"})
    for index in range(10):
        left, right = index + 2, index + 4
        answer = left + right
        rows.append({"category": "arithmetic", "prompt":
                     f"りんごが{left}個あり、さらに{right}個加えた。合計は",
                     "expected_keywords": [str(answer)], "reference": f"{answer}個である。"})
    classes = (("犬", "動物"), ("りんご", "果物"), ("東京", "都市"),
               ("鉄", "金属"), ("国語", "教科"), ("桜", "植物"), ("バス", "乗り物"))
    for subject, expected in classes:
        rows.append({"category": "classification", "prompt": f"{subject}を分類すると、",
                     "expected_keywords": [expected], "reference": f"{expected}である。"})
    causes = (("水を冷やし続けた", "凍る"), ("日光が当たった", "温まる"),
              ("練習を重ねた", "上達"), ("雨が続いた", "増水"),
              ("睡眠が不足した", "眠気"), ("摩擦が生じた", "熱"),
              ("種に水を与えた", "発芽"))
    for cause, expected in causes:
        rows.append({"category": "causal", "prompt": f"{cause}ため、その結果、",
                     "expected_keywords": [expected], "reference": expected + "。"})
    return rows


def build_evaluation() -> list[dict]:
    docs = test_documents()
    rows = []
    for index, row in enumerate(docs[:25]):
        boundary = min(180, max(80, len(row["text"]) // 4))
        rows.append({"category": "continuation", "prompt": row["text"][:boundary],
                     "reference": row["text"][boundary:boundary + 180],
                     "expected_keywords": [], "source_document": row["id"]})
    for index, row in enumerate(docs[25:35]):
        rows.append({"category": "explanation", "prompt": f"{row['title']}とは、",
                     "reference": row["text"][:200], "expected_keywords": [row["title"]],
                     "source_document": row["id"]})
    for index, row in enumerate(docs[35:40]):
        sentence = re.split(r"(?<=[。！？])", row["text"])[0]
        rows.append({"category": "paraphrase", "prompt": f"別の表現で説明すると、{sentence}\nつまり、",
                     "reference": sentence, "expected_keywords": [row["title"]],
                     "source_document": row["id"]})
    for index, row in enumerate(docs[40:45]):
        rows.append({"category": "summary", "prompt": row["text"][:300] + "\n要点は、",
                     "reference": row["text"][:180], "expected_keywords": [row["title"]],
                     "source_document": row["id"]})
    procedures = (("手を洗う", "水", "石けん"), ("文章を見直す", "読み返す", "修正"),
                  ("部屋を整理する", "分類", "片付け"), ("予定を立てる", "期限", "順序"),
                  ("簡単な計算を確かめる", "計算", "確認"))
    for title, first, second in procedures:
        rows.append({"category": "procedure", "prompt": f"{title}手順は、まず",
                     "reference": f"{first}を確認し、次に{second}を行う。",
                     "expected_keywords": [first, second]})
    rows.extend(reasoning_rows())
    if len(rows) != 100:
        raise RuntimeError(f"Base evaluation must contain 100 rows, got {len(rows)}")
    for index, row in enumerate(rows, 1):
        row["id"] = f"foundation-v10-base-{index:03d}"
        row["campus_question"] = False
        row["used_for_training"] = False
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="evaluation/foundation-v10-base-100-step-500.json")
    parser.add_argument("--dataset", default="data/foundation_v10/evaluation/base-japanese-100.json")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    payload = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["config"])
    model = UniPilotTransformer(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    manifest = payload["foundation_v10_manifest"]
    tokenizer = FoundationTokenizer.load(ROOT / manifest["tokenizer"])
    items = build_evaluation()
    dataset_path = ROOT / args.dataset
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps({
        "schema_version": "foundation-v10-base-japanese-100-v1",
        "created_after_training": True, "used_for_training": False,
        "campus_questions": 0, "items": items,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results = []
    started = time.perf_counter()
    for row in items:
        generated, timing = generate(model, tokenizer, row["prompt"], 64)
        natural, repetition = natural_text(generated)
        keyword_hit = any(keyword in generated for keyword in row["expected_keywords"])
        overlap = bigram_overlap(generated, row["reference"])
        relevant = keyword_hit or (row["category"] == "continuation" and overlap >= .02)
        stripped = generated.rstrip()
        complete = bool(timing["eos_reached"] or re.search(r"[。！？.!?]$", stripped)) and len(stripped) >= 10
        runaway = repetition >= .35 or (
            timing["tokens"] >= 64 and not re.search(r"[。！？.!?]$", stripped)
        )
        results.append({**row, "generated": generated, "natural_japanese": natural,
                        "relevant": relevant, "complete": complete, "runaway": runaway,
                        "repetition_rate": repetition, "reference_bigram_overlap": overlap,
                        "keyword_hit": keyword_hit, **timing})
    elapsed = time.perf_counter() - started
    count = len(results)
    categories = {}
    for category in sorted(set(row["category"] for row in results)):
        selected = [row for row in results if row["category"] == category]
        categories[category] = {
            "questions": len(selected),
            "natural_japanese_rate": sum(row["natural_japanese"] for row in selected) / len(selected),
            "relevance_rate": sum(row["relevant"] for row in selected) / len(selected),
            "completion_rate": sum(row["complete"] for row in selected) / len(selected),
            "runaway_rate": sum(row["runaway"] for row in selected) / len(selected),
        }
    metrics = {
        "natural_japanese_rate": sum(row["natural_japanese"] for row in results) / count,
        "relevance_rate": sum(row["relevant"] for row in results) / count,
        "completion_rate": sum(row["complete"] for row in results) / count,
        "runaway_rate": sum(row["runaway"] for row in results) / count,
        "eos_rate": sum(row["eos_reached"] for row in results) / count,
        "mean_repetition_rate": sum(row["repetition_rate"] for row in results) / count,
        "mean_first_token_seconds": sum(row["first_token_seconds"] for row in results) / count,
        "mean_tokens_per_second": sum(row["tokens_per_second"] for row in results) / count,
        "total_evaluation_seconds": elapsed,
        "peak_process_rss_mb": psutil.Process().memory_info().rss / 1024**2,
    }
    gate = {
        "natural_japanese_gte_95": metrics["natural_japanese_rate"] >= .95,
        "relevance_gte_80": metrics["relevance_rate"] >= .80,
        "completion_gte_90": metrics["completion_rate"] >= .90,
        "runaway_lte_5": metrics["runaway_rate"] <= .05,
    }
    report = {
        "schema_version": "foundation-v10-base-100-result-v1",
        "checkpoint": args.checkpoint, "model": config.model_name,
        "parameters": model.parameter_count(), "step": payload["step"],
        "questions": count, "campus_questions": 0, "metrics": metrics,
        "categories": categories, "base_gate_conditions": gate,
        "base_gate": "PASS" if all(gate.values()) else "FAIL",
        "automated_heuristic_evaluation": True, "human_review_required_before_promotion": True,
        "items": results, "external_ai_api": "OFF", "production_changed": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "items"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
