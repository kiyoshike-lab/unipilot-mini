from __future__ import annotations

import argparse
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
from inference.sampling import apply_repetition_penalty
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.train_foundation_v10 import bigram_overlap, natural_text


MANUAL_PROMPTS = [
    ("大学では、授業だけでなく", ["学", "活動"]),
    ("日本の首都は", ["東京"]),
    ("水は", ["物質", "液体", "水素", "酸素"]),
    ("人工知能とは", ["コンピュータ", "知能", "技術"]),
    ("効率よく勉強するためには", ["計画", "復習", "理解"]),
    ("コンピュータは", ["情報", "計算", "処理"]),
    ("経済とは", ["社会", "生産", "消費"]),
    ("歴史を学ぶ理由は", ["過去", "社会", "理解"]),
    ("科学は、観察と", ["実験", "検証"]),
    ("文章を書くときは", ["構成", "根拠", "読み手"]),
    ("図書館では", ["本", "資料", "利用"]),
    ("地球は太陽の周りを", ["公転"]),
    ("春の次に来る季節は", ["夏"]),
    ("健康を保つためには", ["睡眠", "運動", "食事"]),
    ("情報を確認するときは", ["出典", "資料", "確認"]),
    ("数学では、数や図形の", ["性質", "関係"]),
    ("言語は人と人の間で", ["伝", "情報", "意思"]),
    ("読書によって", ["知識", "理解"]),
    ("計画を立てる際には", ["目標", "時間", "順序"]),
    ("データを分析すると", ["傾向", "特徴", "結果"]),
    ("日本語の文章では", ["文", "意味", "表現"]),
    ("物体を落とすと", ["重力", "地面", "落下"]),
    ("植物は光を使って", ["光合成", "成長"]),
    ("社会にはさまざまな", ["人", "制度", "文化"]),
    ("研究を始める前に", ["目的", "先行研究", "問い"]),
]


def build_prompts() -> list[dict]:
    rows = [{"id": f"manual-{index:02d}", "kind": "manual", "prompt": prompt,
             "expected_keywords": keywords, "reference": ""}
            for index, (prompt, keywords) in enumerate(MANUAL_PROMPTS, 1)]
    with gzip.open(ROOT / "data/foundation_v11/documents/test.jsonl.gz", "rt",
                   encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if len(row["text"]) < 500:
                continue
            boundary = min(180, max(100, len(row["text"]) // 5))
            rows.append({"id": f"heldout-{row['id']}", "kind": "heldout_continuation",
                         "prompt": row["text"][:boundary], "expected_keywords": [],
                         "reference": row["text"][boundary:boundary + 180]})
            if len(rows) == 50:
                break
    if len(rows) != 50:
        raise RuntimeError(f"expected 50 completion prompts, got {len(rows)}")
    return rows


def load_model(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["config"])
    model = UniPilotTransformer(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    manifest = payload.get("foundation_v11_manifest") or payload["foundation_v10_manifest"]
    tokenizer = FoundationTokenizer.load(ROOT / manifest["tokenizer"])
    return model, tokenizer, payload, manifest


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, mode: str, generator: torch.Generator,
             max_new_tokens: int = 64) -> tuple[str, dict]:
    ids = tokenizer.encode(prompt, add_bos=True)
    generated = []
    past = None
    eos = False
    started = time.perf_counter()
    first = None
    forbidden = [index for index in range(len(tokenizer.special_tokens))
                 if index != tokenizer.eos_id]
    for _ in range(max_new_tokens):
        current_ids = ids[-model.config.context_length:] if past is None else [ids[-1]]
        current = torch.tensor([current_ids], dtype=torch.long)
        logits, _, past = model(current, past_key_values=past, use_cache=True)
        scores = apply_repetition_penalty(logits[0, -1], ids[-64:], 1.1).clone()
        scores[forbidden] = -torch.inf
        if mode == "greedy":
            next_id = int(scores.argmax().item())
        else:
            values, indices = torch.topk(scores / .8, k=min(40, scores.numel()))
            choice = int(torch.multinomial(torch.softmax(values, dim=-1), 1,
                                           generator=generator).item())
            next_id = int(indices[choice].item())
        if first is None:
            first = time.perf_counter() - started
        ids.append(next_id)
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            eos = True
            break
    elapsed = time.perf_counter() - started
    return tokenizer.decode(generated, skip_special=True), {
        "tokens": len(generated), "eos_reached": eos,
        "first_token_seconds": first or 0.0, "total_seconds": elapsed,
        "tokens_per_second": len(generated) / max(elapsed, 1e-9),
    }


def evaluate_checkpoint(name: str, checkpoint: Path, prompts: list[dict]) -> dict:
    model, tokenizer, payload, manifest = load_model(checkpoint)
    modes = {}
    for mode in ("greedy", "sampling"):
        generator = torch.Generator().manual_seed(22012026)
        results = []
        for row in prompts:
            text, timing = generate(model, tokenizer, row["prompt"], mode, generator)
            natural, repetition = natural_text(text)
            overlap = bigram_overlap(text, row["reference"]) if row["reference"] else 0.0
            keyword_hit = any(keyword in text for keyword in row["expected_keywords"])
            coherence = keyword_hit or (row["kind"] == "heldout_continuation" and overlap >= .02)
            stripped = text.rstrip()
            character_valid = bool(stripped) and "\ufffd" not in text and not re.search(
                r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text
            )
            complete = len(stripped) >= 10 and bool(
                timing["eos_reached"] or re.search(r"[。！？.!?]$", stripped)
            )
            runaway = timing["tokens"] >= 64 and not timing["eos_reached"]
            results.append({**row, "generated": text, "character_valid": character_valid,
                            "natural_japanese": natural, "semantic_coherence": coherence,
                            "complete": complete, "runaway": runaway,
                            "repetition_rate": repetition, "reference_bigram_overlap": overlap,
                            "keyword_hit": keyword_hit, **timing})
        count = len(results)
        modes[mode] = {
            "metrics": {
                "character_validity_rate": sum(row["character_valid"] for row in results) / count,
                "natural_japanese_rate": sum(row["natural_japanese"] for row in results) / count,
                "semantic_coherence_rate": sum(row["semantic_coherence"] for row in results) / count,
                "completion_rate": sum(row["complete"] for row in results) / count,
                "eos_rate": sum(row["eos_reached"] for row in results) / count,
                "runaway_rate": sum(row["runaway"] for row in results) / count,
                "mean_repetition_rate": sum(row["repetition_rate"] for row in results) / count,
                "mean_tokens_per_second": sum(row["tokens_per_second"] for row in results) / count,
                "mean_first_token_seconds": sum(row["first_token_seconds"] for row in results) / count,
            },
            "items": results,
        }
    return {
        "name": name, "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "step": int(payload.get("global_step", payload["step"])),
        "parameters": model.parameter_count(), "tokenizer": manifest["tokenizer"],
        "modes": modes, "peak_process_rss_mb": psutil.Process().memory_info().rss / 1024**2,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirty", default="checkpoints/foundation-v10-sanity/20m/checkpoint-step-100.pt")
    parser.add_argument("--clean", default="checkpoints/foundation-v11-clean-100/checkpoint-step-100.pt")
    parser.add_argument("--output", default="evaluation/foundation-v11-dirty-clean-generation.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    prompts = build_prompts()
    dataset = ROOT / "data/foundation_v11/evaluation/base-completion-50.json"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(json.dumps({
        "schema_version": "foundation-v11-base-completion-50-v1", "items": prompts,
        "final_blind_used": False, "used_for_training": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "foundation-v11-dirty-clean-generation-v1", "prompts": len(prompts),
        "sampling": {"temperature": .8, "top_k": 40, "seed": 22012026},
        "results": [evaluate_checkpoint("dirty_v1.0", ROOT / args.dirty, prompts),
                    evaluate_checkpoint("clean_v1.1", ROOT / args.clean, prompts)],
        "automated_heuristics": True, "final_blind_used": False,
        "external_ai_api": "OFF", "production_changed": False,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"schema_version": report["schema_version"], "prompts": len(prompts),
                      "results": [{"name": row["name"], "step": row["step"],
                                   "metrics": {mode: data["metrics"]
                                               for mode, data in row["modes"].items()}}
                                  for row in report["results"]]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
