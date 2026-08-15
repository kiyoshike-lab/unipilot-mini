from __future__ import annotations

import re
import unicodedata

from evaluation.metrics_v02 import japanese_character_ratio, repetition_rate


CATEGORY_KEYWORDS = {
    "assignment": ["課題", "締切", "提出"], "exam": ["試験", "範囲", "復習", "勉強"],
    "credit": ["単位", "シラバス", "教務", "成績"], "email": ["件名", "先生", "メール", "連絡", "よろしく"],
    "attendance": ["出席", "欠席", "シラバス"], "registration": ["履修", "必修", "選択", "シラバス"],
    "report": ["レポート", "構成", "引用", "資料"], "study": ["勉強", "復習", "計画", "確認"],
    "schedule": ["予定", "締切", "優先", "時間"], "general": ["情報", "分かりません", "確認", "案内"],
}


def broken_text_metrics(text: str) -> dict:
    visible = [char for char in text if not char.isspace()]
    controls = sum(unicodedata.category(char).startswith("C") for char in visible)
    replacements = text.count("�")
    symbols = sum(unicodedata.category(char).startswith("S") for char in visible)
    allowed_ascii = sum(char.isascii() and (char.isalnum() or char in ".,!?%:-_<>/()") for char in visible)
    japanese = sum(bool(re.match(r"[\u3040-\u30ff\u3400-\u9fff、。！？「」『』〈〉]", char)) for char in visible)
    artifact = max(0, len(visible) - japanese - allowed_ascii - symbols)
    return {"invalid_utf8": False, "replacement_characters": replacements, "control_characters": controls,
            "excessive_symbol_ratio": symbols / max(1, len(visible)), "isolated_artifact_ratio": artifact / max(1, len(visible)),
            "broken": bool(replacements or controls or symbols / max(1, len(visible)) > 0.2 or artifact / max(1, len(visible)) > 0.15)}


def infer_category(text: str) -> tuple[str, int]:
    scores = {category: sum(keyword in text for keyword in keywords) for category, keywords in CATEGORY_KEYWORDS.items()}
    category = max(scores, key=scores.get)
    return (category, scores[category]) if scores[category] else ("unknown", 0)


def semantic_score(answer: str, item: dict) -> dict:
    if not answer.strip():
        return {"expected_keyword_rate": 0.0, "category_keyword_rate": 0.0, "forbidden_hits": 0,
                "relevance_score": 0.0, "predicted_category": "unknown", "category_correct": False,
                "meaningful_response": False, "email_structure_rate": None}
    expected = item["expected_keywords"]; forbidden = item["forbidden_keywords"]
    expected_hits = sum(word in answer for word in expected); forbidden_hits = sum(word in answer for word in forbidden)
    category_words = CATEGORY_KEYWORDS[item["category"]]; category_hits = sum(word in answer for word in category_words)
    predicted, _ = infer_category(answer); category_correct = predicted == item["category"]
    jp_ratio = japanese_character_ratio(answer); broken = broken_text_metrics(answer)["broken"]
    score = 45 * expected_hits / max(1, len(expected)) + (10 if expected_hits >= 2 else 0) + (20 if category_correct else 0)
    score += 15 * min(1.0, jp_ratio / 0.95) + (10 if not broken else -15) - 20 * forbidden_hits
    score = max(0.0, min(100.0, score))
    email_rate = None
    if item["category"] == "email":
        fields = ["件名" in answer, "先生" in answer, any(word in answer for word in ["欠席", "遅刻", "課題", "相談", "質問", "連絡"]),
                  any(word in answer for word in ["よろしく", "お願いいたします"])]
        email_rate = sum(fields) / 4
    # Fluency alone is not evidence that the answer addresses the prompt.  Require
    # at least one prompt-specific keyword or a correctly inferred category.
    relevance_evidence = expected_hits > 0 or category_correct
    meaningful = score >= 25 and relevance_evidence and jp_ratio >= 0.9 and len(answer) >= 15 and forbidden_hits == 0 and not broken
    return {"expected_keyword_rate": expected_hits / max(1, len(expected)), "category_keyword_rate": category_hits / max(1, len(category_words)),
            "forbidden_hits": forbidden_hits, "relevance_score": score, "predicted_category": predicted,
            "category_correct": category_correct, "meaningful_response": meaningful, "email_structure_rate": email_rate}


def aggregate(rows: list[dict], max_new_tokens: int) -> dict:
    email = [row["email_structure_rate"] for row in rows if row["email_structure_rate"] is not None]
    return {
        "response_not_empty": sum(bool(row["answer"].strip()) for row in rows) / len(rows),
        "eos_reached_rate": sum(row["eos_reached"] for row in rows) / len(rows),
        "average_generated_tokens": sum(row["tokens"] for row in rows) / len(rows),
        "runaway_generation_rate": sum(not row["eos_reached"] and row["tokens"] >= max_new_tokens for row in rows) / len(rows),
        "repetition_rate": sum(row["repetition_rate"] for row in rows) / len(rows),
        "japanese_character_ratio": sum(row["japanese_character_ratio"] for row in rows) / len(rows),
        "keyword_relevance": 100 * sum(row["expected_keyword_rate"] for row in rows) / len(rows),
        "category_keyword_hit": 100 * sum(row["category_keyword_rate"] for row in rows) / len(rows),
        "forbidden_keyword_hits": sum(row["forbidden_hits"] for row in rows),
        "relevance_score": sum(row["relevance_score"] for row in rows) / len(rows),
        "category_accuracy": sum(row["category_correct"] for row in rows) / len(rows),
        "meaningful_response_rate": sum(row["meaningful_response"] for row in rows) / len(rows),
        "broken_response_rate": sum(row["broken"]["broken"] for row in rows) / len(rows),
        "email_structure_rate": sum(email) / len(email) if email else None,
        "generation_tokens_per_second": sum(row["tokens_per_sec"] for row in rows) / len(rows),
    }
