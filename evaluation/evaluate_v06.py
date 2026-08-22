from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader

from evaluation.metrics_v03 import CATEGORY_KEYWORDS as LEGACY_CATEGORY_KEYWORDS
from inference.generate import generate_text, load_model
from training.dataset_v03 import CurriculumDataset, SYSTEM_TEXT, dynamic_collate


RUBRIC = (
    "natural_japanese", "answer_completion", "question_relevance", "unnecessary_information",
    "factual_reliability", "conciseness", "helpfulness", "accuracy", "category_correctness",
    "instruction_following", "readability", "overall_quality",
)
SUBJECT_WORDS = ("法学", "経済学", "心理学", "情報科学")
UNCERTAINTY_WORDS = ("異な", "確認", "不明", "断定", "公式", "案内", "窓口", "シラバス", "規程")
ACTION_WORDS = ("確認", "相談", "整理", "連絡", "進め", "比べ", "記録", "優先", "調べ", "問い合わせ")
POLICY_HALLUCINATION = re.compile(r"全国(?:の)?大学で|どの大学でも|必ず(?:合格|留年|認定|変更|入室)|全員(?:が)?(?:対象|無料|返還不要)")


def japanese_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    japanese = sum(bool(re.match(r"[ぁ-んァ-ヶー一-龥々、。！？「」『』]", char)) for char in visible)
    return japanese / len(visible)


def repetition_rate(text: str) -> float:
    chunks = [text[index:index + 3] for index in range(max(0, len(text) - 2))]
    return 0.0 if not chunks else 1 - len(set(chunks)) / len(chunks)


def score_from_ratio(value: float) -> int:
    if value >= 0.95:
        return 5
    if value >= 0.75:
        return 4
    if value >= 0.5:
        return 3
    if value > 0:
        return 2
    return 0


def category_words(category: str, expected: list[str]) -> list[str]:
    return list(dict.fromkeys([*expected, *LEGACY_CATEGORY_KEYWORDS.get(category, [])]))


def rubric_scores(item: dict, answer: str, generated_tokens: int, eos_reached: bool) -> tuple[dict, dict]:
    expected = item.get("expected_keywords", [])
    forbidden = item.get("forbidden_keywords", [])
    expected_hits = sum(word in answer for word in expected)
    keyword_ratio = expected_hits / max(1, len(expected))
    category_hits = sum(word in answer for word in category_words(item["category"], expected))
    extra_subjects = [word for word in SUBJECT_WORDS if word in answer and word not in item["prompt"]]
    forbidden_hits = [word for word in forbidden if word in answer]
    broken = "�" in answer or bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", answer))
    repeat = repetition_rate(answer)
    complete = bool(answer.strip()) and answer.rstrip().endswith(("。", "！", "？", "ます", "です"))
    uncertain = any(word in answer for word in UNCERTAINTY_WORDS)
    policy_hallucination = bool(POLICY_HALLUCINATION.search(answer))
    hallucination = bool(extra_subjects or forbidden_hits or policy_hallucination)
    unnecessary = bool(extra_subjects or forbidden_hits or repeat > 0.1)
    length_type = item.get("length_type", "normal")
    if length_type == "simple":
        length_score = 5 if 10 <= generated_tokens <= 80 else (3 if generated_tokens <= 100 else 1)
    elif length_type == "detailed":
        # Mini's 256-token context cannot safely promise 200-400 answer tokens;
        # this score rewards detail within the measured context and reports the limitation separately.
        length_score = 5 if 50 <= generated_tokens <= 128 else (3 if generated_tokens >= 30 else 1)
    else:
        length_score = 5 if 20 <= generated_tokens <= 128 else (3 if generated_tokens > 0 else 0)
    natural = score_from_ratio(japanese_ratio(answer))
    if broken or repeat > 0.15:
        natural = min(natural, 2)
    completion = 5 if eos_reached and complete else (4 if complete else (2 if answer else 0))
    relevance = score_from_ratio(keyword_ratio)
    if expected_hits == 0 and answer:
        relevance = 1
    no_extra = 5 if not unnecessary else max(0, 5 - 2 * (len(extra_subjects) + len(forbidden_hits)) - int(repeat > 0.1))
    reliability = 5 if not hallucination else max(0, 5 - 2 * (len(extra_subjects) + len(forbidden_hits)) - 3 * int(policy_hallucination))
    if item.get("requires_uncertainty") and not uncertain:
        reliability = min(reliability, 2)
    helpfulness = min(5, 1 + sum(word in answer for word in ACTION_WORDS)) if answer else 0
    accuracy = min(relevance, reliability)
    category = 5 if category_hits >= 2 else (4 if category_hits == 1 else (1 if answer else 0))
    instruction = min(length_score, reliability if item.get("requires_uncertainty") else 5)
    readability = 5 if complete and repeat <= 0.03 and not broken else (4 if complete and repeat <= 0.08 else (2 if answer else 0))
    first_eleven = {
        "natural_japanese": natural, "answer_completion": completion, "question_relevance": relevance,
        "unnecessary_information": no_extra, "factual_reliability": reliability, "conciseness": length_score,
        "helpfulness": helpfulness, "accuracy": accuracy, "category_correctness": category,
        "instruction_following": instruction, "readability": readability,
    }
    scores = {**first_eleven, "overall_quality": round(sum(first_eleven.values()) / len(first_eleven), 2)}
    signals = {
        "expected_keyword_rate": keyword_ratio, "keyword_hit": expected_hits > 0, "category_keyword_hits": category_hits,
        "forbidden_hits": forbidden_hits, "extra_subjects": extra_subjects, "policy_hallucination": policy_hallucination,
        "hallucination": hallucination, "unnecessary": unnecessary, "uncertainty_present": uncertain,
        "complete": complete, "broken": broken, "japanese_ratio": japanese_ratio(answer), "repetition_rate": repeat,
    }
    return scores, signals


@torch.inference_mode()
def validation_loss(model, tokenizer, device) -> float:
    dataset = CurriculumDataset("data/v06/instruction/validation.jsonl", tokenizer, model.config.context_length, True)
    loader = DataLoader(dataset, batch_size=1, collate_fn=dynamic_collate)
    values = []
    for inputs, targets, _, _ in loader:
        _, loss = model(inputs.to(device), targets.to(device))
        values.append(loss.item())
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/vocab-v02-512.json")
    parser.add_argument("--prompts", default="evaluation/fixed_prompts_v06.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(6062026)
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer, "cpu")
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if args.limit:
        prompts = prompts[:args.limit]
    rows = []
    for item in prompts:
        formatted = f"<BOS><SYSTEM>\n{SYSTEM_TEXT}\n<USER>\n{item['prompt']}\n<ASSISTANT>\n"
        answer, timing = generate_text(model, tokenizer, formatted, args.max_new_tokens, 0.0, 40, 0.9, 1.1)
        scores, signals = rubric_scores(item, answer, timing["tokens"], timing["eos_reached"])
        rows.append({**item, "answer": answer, **timing, "rubric": scores, "signals": signals})
    count = len(rows)
    dimensions = {name: sum(row["rubric"][name] for row in rows) / count for name in RUBRIC}
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    category_metrics = {
        category: {"questions": len(items), "relevance_rate": sum(row["rubric"]["question_relevance"] >= 4 for row in items) / len(items),
                   "accuracy_rate": sum(row["rubric"]["accuracy"] >= 4 for row in items) / len(items),
                   "hallucination_rate": sum(row["signals"]["hallucination"] for row in items) / len(items)}
        for category, items in sorted(by_category.items())
    }
    metrics = {
        "questions": count,
        "natural_rate": sum(row["rubric"]["natural_japanese"] >= 4 for row in rows) / count,
        "completion_rate": sum(row["signals"]["complete"] for row in rows) / count,
        "relevance_rate": sum(row["rubric"]["question_relevance"] >= 4 for row in rows) / count,
        "keyword_rate": sum(row["signals"]["keyword_hit"] for row in rows) / count,
        "category_accuracy": sum(row["rubric"]["category_correctness"] >= 4 for row in rows) / count,
        "accuracy_rate": sum(row["rubric"]["accuracy"] >= 4 for row in rows) / count,
        "unnecessary_information_rate": sum(row["signals"]["unnecessary"] for row in rows) / count,
        "hallucination_rate": sum(row["signals"]["hallucination"] for row in rows) / count,
        "runaway_rate": sum(not row["eos_reached"] and row["tokens"] >= args.max_new_tokens for row in rows) / count,
        "repetition_rate": sum(row["signals"]["repetition_rate"] for row in rows) / count,
        "eos_rate": sum(row["eos_reached"] for row in rows) / count,
        "broken_rate": sum(row["signals"]["broken"] for row in rows) / count,
        "mean_tokens_per_second": sum(row["tokens_per_sec"] for row in rows) / count,
        "rubric_mean_0_to_5": dimensions,
    }
    val = validation_loss(model, tokenizer, device)
    result = {
        "schema_version": 1, "evaluation": "unipilot-eval-v06-300-rubric12", "rubric": list(RUBRIC),
        "checkpoint": args.checkpoint, "model": model.config.model_name, "parameters": model.parameter_count(),
        "vocab": model.config.vocab_size, "context": model.config.context_length, "step": payload.get("step"),
        "validation_loss_v06": val, "perplexity_v06": math.exp(min(val, 20)), "metrics": metrics,
        "category_metrics": category_metrics, "context_limit_note": "Detailed 200-400-token answers are not a safe promise at context 256.",
        "automated_rubric_limit": "Transparent keyword/rule proxy; not a substitute for blinded human scoring.",
        "generations": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    human = {
        "checkpoint": args.checkpoint, "instructions": "Blind-review each answer on the same 12 dimensions, 0-5. Do not infer scores from automated fields.",
        "target": "mean overall_quality >= 4.0/5", "completed": False,
        "items": [{"id": row["id"], "category": row["category"], "prompt": row["prompt"], "answer": row["answer"],
                   "human_scores": {name: None for name in RUBRIC}, "notes": ""} for row in rows],
    }
    output.with_name(output.stem + "-human.json").write_text(json.dumps(human, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("generations", "category_metrics")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
