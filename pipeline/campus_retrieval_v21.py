from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from pipeline.campus_categories_v2 import CATEGORY_TO_LEVEL1, LEVEL1_GROUPS
from pipeline.campus_retrieval import CampusFAQRetriever, load_jsonl


QUERY_EXPANSION = {
    "exam": "試験 テスト 受験 範囲 準備",
    "assignment": "課題 提出 要件 期限",
    "credit": "単位 取得 不足 落とす 卒業 条件",
    "gpa": "GPA 計算 成績 GP 単位",
    "grade_simulator": "必要点 合格点 配点 残り評価 計算",
    "attendance": "欠席 出席 公欠 記録 確認",
    "lateness": "遅刻 遅延 到着 対応",
    "professor_email": "教授 先生 メール 連絡 文面 件名",
    "absence_email": "欠席 メール 連絡 文面",
    "lateness_email": "遅刻 メール 到着 連絡 文面",
    "late_submission_email": "課題 提出遅延 メール お詫び 延長",
    "registration": "履修 登録 必修 時間割 シラバス",
    "schedule": "予定 スケジュール 時間 配分",
    "study_plan": "試験 勉強 学習 計画 復習",
    "assignment_priority": "課題 優先順位 締切 所要時間",
    "deadline_organizer": "締切 期限 一覧 管理",
    "report_outline": "レポート 構成 章立て 序論 本論 結論",
    "citation_check": "引用 出典 参考文献 書誌情報",
    "presentation_outline": "プレゼン 発表 スライド 構成 時間",
    "career_schedule": "就活 選考 面接 締切 日程",
    "es_outline": "ES エントリーシート 自己PR 志望動機 構成",
    "toeic_plan": "TOEIC 英語 学習 計画 模試",
    "internship": "インターン 応募 実習 選考",
    "scholarship": "奨学金 給付 貸与 申請 JASSO",
    "tuition": "学費 授業料 納付 分納 延納 減免",
    "part_time_job": "アルバイト バイト シフト 学業",
    "campus_life": "大学生活 学内 サークル 研究室 ゼミ",
    "relationship": "人間関係 友達 共同作業 相談",
    "programming": "プログラミング コード エラー デバッグ",
    "ai_usage": "生成AI AI利用 授業 課題 ルール",
    "math": "数学 数式 計算 証明",
    "statistics": "統計 確率 分散 標準偏差 回帰 検定",
    "university_policy": "大学 学則 規程 公式 条件",
    "faq_search": "FAQ よくある質問 確認先 窓口",
    "general": "大学生活 相談 次の行動",
}

PHRASE_EXPANSION = {
    "単位やば": "単位 取得 不足 落とす 条件",
    "教授なんて送": "教授 メール 連絡 文面",
    "先生なんて送": "教授 メール 連絡 文面",
    "gpa出して": "GPA 計算 成績 単位",
    "間に合わない": "期限 遅延 次の行動",
    "落としそう": "不足 条件 確認",
    "れぽ": "レポート 構成 提出",
}


def expand_query(question: str, category: str | None) -> str:
    compact = re.sub(r"\s+", "", question.lower())
    additions = [value for key, value in PHRASE_EXPANSION.items() if key in compact]
    if category in QUERY_EXPANSION:
        additions.append(QUERY_EXPANSION[category])
    return question + " " + " ".join(additions)


class CampusFAQRetrieverV21(CampusFAQRetriever):
    methods = ("bm25", "word_tfidf", "character_tfidf", "word_character_hybrid", "bm25_character",
               "category_filtered", "query_expansion", "router_aware")

    def __init__(self, rows: list[dict], config_path: str | Path = "data/campus_v21/retrieval/retrieval-config.json"):
        super().__init__(rows)
        self.ids = [row["id"] for row in rows]
        self.positions = {identifier: index for index, identifier in enumerate(self.ids)}
        documents = [row["question"] + " " + " ".join(row.get("keywords", [])) for row in rows]
        self.char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
        self.char_matrix = self.char_vectorizer.fit_transform(documents)
        self.word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=1,
                                                token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
        self.word_matrix = self.word_vectorizer.fit_transform(documents)
        path = Path(config_path)
        if path.exists():
            selected = json.loads(path.read_text(encoding="utf-8"))
            self.selected_method = selected["selected_method"]
            self.no_match_threshold = float(selected["selected_threshold"])
        else:
            self.selected_method, self.no_match_threshold = "router_aware", .24

    @classmethod
    def from_jsonl(cls, path: str | Path = "data/campus_v2/faq/reviewed.jsonl",
                   config_path: str | Path = "data/campus_v21/retrieval/retrieval-config.json") -> "CampusFAQRetrieverV21":
        return cls(load_jsonl(path), config_path)

    def _bm25_scores(self, question: str) -> np.ndarray:
        scores = np.zeros(len(self.rows), dtype=np.float64)
        results = self.index.search(question, top_k=min(120, len(self.rows)))
        maximum = max((float(item["score"]) for item in results), default=0.0)
        if maximum > 0:
            for item in results:
                scores[self.positions[item["id"]]] = max(0.0, float(item["score"]) / maximum)
        return scores

    def _vector_scores(self, question: str) -> tuple[np.ndarray, np.ndarray]:
        char = linear_kernel(self.char_vectorizer.transform([question]), self.char_matrix).ravel()
        word = linear_kernel(self.word_vectorizer.transform([question]), self.word_matrix).ravel()
        return char, word

    def _allowed(self, category: str | None, confidence_band: str) -> np.ndarray:
        allowed = np.ones(len(self.rows), dtype=bool)
        if not category or confidence_band != "high":
            return allowed
        level1 = CATEGORY_TO_LEVEL1.get(category)
        neighbors = set(LEVEL1_GROUPS.get(level1, ()))
        neighbors.add(category)
        return np.array([row["category"] in neighbors for row in self.rows], dtype=bool)

    def search_method(self, question: str, category: str | None, method: str, top_k: int = 3,
                      confidence_band: str = "high", threshold: float = 0.0) -> list[dict]:
        expanded = expand_query(question, category)
        char_query = expanded if method in ("query_expansion", "router_aware") else question
        char, word = self._vector_scores(char_query) if method != "bm25" else (
            np.zeros(len(self.rows)), np.zeros(len(self.rows)))
        bm25 = (self._bm25_scores(expanded if method in ("query_expansion", "router_aware") else question)
                if method in ("bm25", "bm25_character", "query_expansion", "router_aware")
                else np.zeros(len(self.rows)))
        if method == "bm25": scores = bm25
        elif method == "word_tfidf": scores = word
        elif method == "character_tfidf": scores = char
        elif method == "word_character_hybrid": scores = .82 * char + .18 * word
        elif method == "bm25_character": scores = .78 * char + .22 * bm25
        elif method == "category_filtered": scores = char.copy()
        elif method == "query_expansion": scores = .82 * char + .18 * bm25
        elif method == "router_aware": scores = .84 * char + .16 * bm25
        else: raise ValueError(method)

        if method in ("category_filtered", "router_aware"):
            allowed = self._allowed(category, confidence_band)
            if confidence_band == "high":
                scores = np.where(allowed, scores, -1.0)
            elif category:
                scores = scores + np.array([.06 if row["category"] == category else 0.0 for row in self.rows])
        ordered = np.argsort(-scores)[:top_k]
        if not len(ordered) or float(scores[ordered[0]]) < threshold:
            return []
        result = []
        for position in ordered:
            if scores[position] < 0:
                continue
            item = self.rows[int(position)]
            result.append({**item, "retrieval_score": float(scores[position]),
                           "confidence": float(scores[position]), "category_match": item["category"] == category,
                           "retrieval_method": method, "query_expanded": expanded != question})
        return result

    def search(self, question: str, category: str, top_k: int = 3,
               confidence_band: str = "high") -> list[dict]:
        return self.search_method(question, category, self.selected_method, top_k, confidence_band,
                                  self.no_match_threshold)
