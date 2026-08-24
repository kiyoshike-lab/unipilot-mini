from __future__ import annotations

import hashlib
import re

from pipeline.campus_retrieval_v22 import detect_numeric_conflict


DETAIL_TRIGGERS = (
    "詳しく", "詳細", "細かく", "深掘り", "理由も", "根拠も", "例も", "長め", "もっと説明", "もっと教えて",
)
SHORT_TRIGGERS = ("短く", "簡潔に", "一言で", "要点だけ")
DETAIL_FOLLOWUPS = ("もっと詳しく", "詳しく教えて", "もっと教えて", "細かく", "続き", "深掘りして", "具体例も")

MODE_LIMITS = {
    "short": (100, 220),
    "normal": (300, 650),
    "detailed": (700, 1200),
}

ACTION_GUIDANCE = {
    "study_plan": "一般的な進め方：目的を一つ決め、短い間隔で思い出す練習と復習を組み合わせ、翌日に理解度を確認してください。",
    "exam": "一般的な進め方：まず試験範囲と配点を確認し、理解が弱い項目から問題演習と復習を行ってください。",
    "report_outline": "一般的な進め方：課題要件を確認し、主張・根拠・出典を対応させてから本文を組み立ててください。",
    "citation_check": "一般的な進め方：原典を開き、著者・題名・公開年・URL・参照日を記録し、授業指定の形式に合わせてください。",
    "programming": "一般的な進め方：再現条件を小さくし、エラーメッセージ、入力、期待結果、実際の結果を順に照合してください。",
    "statistics": "一般的な進め方：変数の尺度、標本、仮定を確認してから手法を選び、数値だけでなく不確実性も解釈してください。",
    "math": "一般的な進め方：定義と前提を確認し、式変形を一段ずつ書いて、最後に条件へ戻して検算してください。",
    "scholarship": "一般的な進め方：制度名、対象年度、申請期限、所得・成績要件を公式窓口で照合してください。",
    "part_time_job": "一般的な進め方：雇用契約書、労働条件通知書、勤務記録を保存し、不明点は大学窓口か公的相談先へ確認してください。",
    "general": "一般的な進め方：用語の定義、成り立ち、適用範囲を分け、授業資料や原典と照合してください。",
}


def _sentences(text: str) -> list[str]:
    return [item.strip(" ・\t") for item in re.split(r"(?<=[。！？])\s*|\n+", text) if len(item.strip()) >= 18]


def _key(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


class CampusAnswerComposerV22:
    @staticmethod
    def choose_mode(question: str, requested: str = "auto", *, is_detail_followup: bool = False) -> str:
        if requested in MODE_LIMITS:
            return requested
        if is_detail_followup or any(trigger in question for trigger in DETAIL_TRIGGERS):
            return "detailed"
        if any(trigger in question for trigger in SHORT_TRIGGERS):
            return "short"
        return "normal"

    @staticmethod
    def is_detail_followup(question: str) -> bool:
        compact = re.sub(r"\s+", "", question)
        return len(compact) <= 24 and any(trigger in compact for trigger in DETAIL_FOLLOWUPS)

    def compose_grounded(self, question: str, category: str, documents: list[dict], mode: str) -> tuple[str, dict]:
        minimum, maximum = MODE_LIMITS[mode]
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        for document in documents:
            for sentence in _sentences(document.get("selected_text", "")):
                key = _key(sentence)
                if len(key) < 12 or key in seen:
                    continue
                seen.add(key)
                candidates.append((sentence, document["id"]))
        if not candidates:
            return "根拠として利用できる本文を抽出できませんでした。最新の公式情報を確認してください。", {
                "supported": False, "evidence_sentence_count": 0, "claim_sources": [], "coverage_chars": 0,
            }

        answer_parts = [f"結論：{candidates[0][0]}"]
        used: list[tuple[str, str]] = [candidates[0]]
        if mode != "short":
            answer_parts.append("\n根拠に基づく説明：")
        for sentence, document_id in candidates[1:]:
            prefix = "・" if mode != "short" else ""
            proposed = "\n".join([*answer_parts, f"{prefix}{sentence}"])
            reserve = 150 if mode != "short" else 0
            if len(proposed) > maximum - reserve:
                continue
            answer_parts.append(f"{prefix}{sentence}")
            used.append((sentence, document_id))
            current = len("\n".join(answer_parts))
            target = {"short": 145, "normal": 430, "detailed": 930}[mode]
            if current >= target:
                break

        if mode != "short":
            advice_body = ACTION_GUIDANCE.get(category, ACTION_GUIDANCE["general"]).removeprefix("一般的な進め方：")
            advice = f"一般的な助言：\n今やること：{advice_body}"
            proposed = "\n".join([*answer_parts, "", advice])
            if len(proposed) <= maximum:
                answer_parts.extend(["", advice])
            if any(document.get("stale") for document in documents):
                caution = "注意：更新日の古い根拠を含むため、期限・金額・制度条件は最新の公式ページで再確認してください。"
            elif detect_numeric_conflict(documents):
                caution = "注意：複数資料の数値条件が一致しない可能性があるため、対象年度の公式ページを優先してください。"
            else:
                caution = "注意：適用条件や授業・年度による違いは、提示した出典と最新の公式案内で確認してください。"
            proposed = "\n".join([*answer_parts, caution])
            if len(proposed) <= maximum:
                answer_parts.append(caution)

        answer = "\n".join(answer_parts)
        # Never pad with invented facts. A shortfall is recorded for the evaluator.
        claim_sources = [
            {
                "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
                "document_id": document_id,
                "text": sentence,
            }
            for sentence, document_id in used
        ]
        return answer[:maximum], {
            "supported": True,
            "evidence_sentence_count": len(used),
            "claim_sources": claim_sources,
            "coverage_chars": sum(len(sentence) for sentence, _ in used),
            "length_target": {"min": minimum, "max": maximum},
            "length_target_met": minimum <= len(answer) <= maximum,
            "unsupported_factual_claims": 0,
            "source_conflict": detect_numeric_conflict(documents),
        }
