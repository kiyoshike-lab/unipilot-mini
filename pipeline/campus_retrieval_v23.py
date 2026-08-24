from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from pipeline.campus_categories_v2 import CATEGORY_TO_LEVEL1
from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_retrieval_v21 import expand_query
from pipeline.campus_retrieval_v22 import (
    KNOWLEDGE_FILES,
    CampusKnowledgeRetrieverV22,
    query_variants,
)
from retrieval.bm25 import tokens


V23_RETRIEVAL_STRATEGIES = (
    "bm25",
    "tfidf",
    "current_reranked",
    "hybrid",
    "category_aware_hybrid",
    "multi_query_hybrid",
)

CONFIDENCE_ORDER = {"REJECT": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

TYPO_REWRITES = {
    "レポ卜": "レポート", "メ一ル": "メール", "りしゅ": "履修", "ていしゅつ": "提出",
    "しけん": "試験", "かくにん": "確認", "gpaやば": "GPA 成績 単位",
}

CONVERSATIONAL_REWRITES = {
    "単位やば": "単位取得条件 成績 欠席 必要点数",
    "落単": "単位取得条件 成績評価 欠席",
    "テストやば": "試験日 試験範囲 配点 勉強計画",
    "課題やば": "課題 締切 提出条件 優先順位",
    "なんて送": "教授 メール 件名 本文 用件",
    "履修むり": "履修登録 必修 前提科目 登録期限",
    "金ない": "学費 奨学金 学生支援 相談窓口",
}

AUTHORITY_BY_TYPE = {
    "official_university": 1.00,
    "official_government": .98,
    "official_public": .95,
    "public_institution": .92,
    "wikipedia": .58,
}

POLICY_CUES = (
    "制度", "規程", "学則", "申請", "締切", "奨学金", "学費", "公欠", "履修", "単位",
    "上限", "対象", "条件", "公式",
)


def normalise_query(text: str) -> str:
    value = text.strip()
    for before, after in TYPO_REWRITES.items():
        value = value.replace(before, after)
    return value


def rewrite_queries(question: str, primary_category: str, secondary_category: str | None = None) -> list[str]:
    """Generate search candidates without collapsing ambiguity into a single asserted intent."""
    normalised = normalise_query(question)
    variants = [question, normalised, expand_query(normalised, primary_category)]
    compact = re.sub(r"\s+", "", normalised.lower())
    variants.extend(rewrite for cue, rewrite in CONVERSATIONAL_REWRITES.items() if cue.lower() in compact)
    variants.extend(query_variants(normalised, primary_category))
    if secondary_category and secondary_category != primary_category:
        variants.append(expand_query(normalised, secondary_category))
    return list(dict.fromkeys(value.strip() for value in variants if value and value.strip()))[:8]


def knowledge_quality(row: dict, *, stale: bool) -> dict[str, Any]:
    source_type = str(row.get("source_type") or "")
    publisher = str(row.get("publisher") or row.get("source") or "")
    authority = AUTHORITY_BY_TYPE.get(source_type, .72)
    if "JASSO" in publisher or "日本学生支援機構" in publisher:
        authority = max(authority, .97)
    elif any(name in publisher for name in ("文部科学省", "厚生労働省", "デジタル庁")):
        authority = max(authority, .98)
    freshness = 0.25 if stale else 1.0
    title = str(row.get("title") or "")
    text = str(row.get("text") or "")
    specificity = min(1.0, .35 + min(len(title), 40) / 100 + min(len(text), 300) / 600)
    license_status = 1.0 if row.get("license") and row.get("source_url") else 0.0
    official = source_type.startswith("official_") or authority >= .9
    score = .42 * authority + .20 * freshness + .20 * specificity + .18 * license_status
    return {
        "source_authority": round(authority, 3),
        "freshness": round(freshness, 3),
        "specificity": round(specificity, 3),
        "university_specific": bool(row.get("university_specific")),
        "official_or_public": official,
        "license_status": "verified" if license_status else "missing",
        "overall": round(score, 3),
    }


class CampusKnowledgeRetrieverV23(CampusKnowledgeRetrieverV22):
    """Precision-first two-stage local retrieval with an explicit confidence gate."""

    def __init__(self, rows: list[dict], hard_negative_path: str | Path | None = None):
        super().__init__(rows)
        for row in self.rows:
            row["knowledge_quality"] = knowledge_quality(row, stale=self._stale(row))
        self._row_tokens = [set(tokens(f"{row['title']} {row['text']}")) for row in self.rows]
        self._quality = np.array([row["knowledge_quality"]["overall"] for row in self.rows])
        self._authority = np.array([row["knowledge_quality"]["source_authority"] for row in self.rows])
        self._negative_pair_rate: dict[tuple[str, str], float] = {}
        if hard_negative_path and Path(hard_negative_path).exists():
            payload = json.loads(Path(hard_negative_path).read_text(encoding="utf-8"))
            pair_counts = Counter(
                (row.get("expected_category", ""), row.get("retrieved_category", ""))
                for row in payload.get("items", [])
            )
            expected_counts = Counter(expected for expected, _ in pair_counts.elements())
            for pair, count in pair_counts.items():
                self._negative_pair_rate[pair] = count / max(1, expected_counts[pair[0]])

    @classmethod
    def from_files(
        cls,
        paths: tuple[str, ...] = KNOWLEDGE_FILES,
        hard_negative_path: str | Path = "data/campus_v23/retrieval/hard-negatives.json",
    ) -> "CampusKnowledgeRetrieverV23":
        rows: list[dict] = []
        for path in paths:
            rows.extend(load_jsonl(Path(path)))
        return cls(rows, hard_negative_path)

    def _category_compatibility(self, primary: str, secondary: str | None) -> np.ndarray:
        primary_level = CATEGORY_TO_LEVEL1.get(primary)
        values = []
        for row in self.rows:
            category = row.get("category")
            if category == primary:
                values.append(1.0)
            elif secondary and category == secondary:
                values.append(.72)
            elif primary_level and CATEGORY_TO_LEVEL1.get(category) == primary_level:
                values.append(.38)
            else:
                values.append(0.0)
        return np.array(values)

    def _hard_negative_penalty(self, primary: str) -> np.ndarray:
        return np.array([
            min(.08, self._negative_pair_rate.get((primary, str(row.get("category") or "")), 0.0) * .08)
            if row.get("category") != primary else 0.0
            for row in self.rows
        ])

    @staticmethod
    def _confidence(score: float, margin: float, semantic: float, category_compatibility: float,
                    query_length: int, policy: str) -> str:
        thresholds = {
            "precision": {"high": .72, "medium": .62, "semantic": .20, "margin": .060, "medium_margin": .030},
            "balanced": {"high": .65, "medium": .54, "semantic": .15, "margin": .030, "medium_margin": .014},
            "recall": {"high": .60, "medium": .46, "semantic": .11, "margin": .015, "medium_margin": .005},
        }.get(policy, {"high": .65, "medium": .54, "semantic": .15, "margin": .030, "medium_margin": .014})
        short_penalty = .05 if query_length <= 8 else 0.0
        if (score >= thresholds["high"] + short_penalty and semantic >= thresholds["semantic"]
                and margin >= thresholds["margin"]):
            return "HIGH"
        if (score >= thresholds["medium"] + short_penalty and semantic >= thresholds["semantic"] * .75
                and category_compatibility >= .35 and margin >= thresholds["medium_margin"]):
            return "MEDIUM"
        if score >= thresholds["medium"] - .08 and semantic >= thresholds["semantic"] * .55:
            return "LOW"
        return "REJECT"

    def search(
        self,
        question: str,
        category: str,
        *,
        university: str | None = None,
        top_k: int = 5,
        response_mode: str = "normal",
        threshold: float = 0.0,
        strategy: str = "category_aware_hybrid",
        secondary_category: str | None = None,
        intent: str | None = None,
        confidence_policy: str = "precision",
        stage1_k: int = 20,
    ) -> tuple[list[dict], dict]:
        started = time.perf_counter()
        if strategy not in V23_RETRIEVAL_STRATEGIES:
            raise ValueError(f"unknown retrieval strategy: {strategy}")
        variants = rewrite_queries(question, category, secondary_category)
        active = variants if strategy == "multi_query_hybrid" else variants[:2]
        if strategy in ("bm25", "tfidf", "current_reranked"):
            active = [normalise_query(question)]
        char_parts = [linear_kernel(self.char_vectorizer.transform([value]), self.char_matrix).ravel()
                      for value in active]
        word_parts = [linear_kernel(self.word_vectorizer.transform([value]), self.word_matrix).ravel()
                      for value in active]
        bm25_parts = [self._bm25(value) for value in active]
        char_original = char_parts[0]
        char_multi = np.maximum.reduce(char_parts)
        word_multi = np.maximum.reduce(word_parts)
        bm25_multi = np.maximum.reduce(bm25_parts)
        category_compatibility = self._category_compatibility(category, secondary_category)
        query_tokens = set(tokens(" ".join(active)))
        original_query_tokens = set(tokens(question))
        lexical = np.array([
            min(1.0, len(query_tokens & row_tokens) / max(1, min(len(query_tokens), 8)))
            for row_tokens in self._row_tokens
        ])
        original_lexical = np.array([
            min(1.0, len(original_query_tokens & row_tokens) / max(1, min(len(original_query_tokens), 8)))
            for row_tokens in self._row_tokens
        ])
        compact_question = re.sub(r"[\s\W_]+", "", question.lower())
        title_bonus = np.array([
            1.0 if len(re.sub(r"[\s\W_]+", "", row["title"])) >= 2
            and re.sub(r"[\s\W_]+", "", row["title"].lower()) in compact_question else 0.0
            for row in self.rows
        ])
        quoted_raw = re.findall(r"『([^』]{2,160})』", question)
        quoted_fragments = [
            re.sub(r"[\s\W_]+", "", fragment.lower())
            for fragment in quoted_raw
        ]
        quoted_cores = [
            re.sub(r"[\s\W_]+", "", re.split(r"[|—]", fragment, maxsplit=1)[0].lower())
            for fragment in quoted_raw
        ]
        quoted_title_bonus = np.array([
            1.0 if any(
                fragment in re.sub(r"[\s\W_]+", "", row["title"].lower())
                or re.sub(r"[\s\W_]+", "", row["title"].lower()) in fragment
                or core in re.sub(r"[\s\W_]+", "", row["title"].lower())
                for fragment, core in zip(quoted_fragments, quoted_cores)
            ) else 0.0
            for row in self.rows
        ])
        policy_query = any(cue in question for cue in POLICY_CUES)

        if strategy == "bm25":
            stage1 = bm25_multi
        elif strategy == "tfidf":
            stage1 = .58 * char_multi + .42 * word_multi
        elif strategy == "current_reranked":
            stage1 = (.38 * char_original + .15 * word_multi + .20 * bm25_multi + .10 * lexical
                      + .35 * title_bonus + .45 * quoted_title_bonus)
        else:
            stage1 = (.27 * char_multi + .21 * word_multi + .27 * bm25_multi + .09 * lexical
                      + .06 * self._quality + .28 * title_bonus + .42 * quoted_title_bonus)
        if strategy in ("category_aware_hybrid", "multi_query_hybrid"):
            stage1 = stage1 + .05 * category_compatibility
        allowed = self._allowed(university)
        if not university:
            # University-specific evidence stays hidden unless the query itself names that source/title.
            allowed = np.logical_or(allowed, np.array([
                bool(row.get("university_specific"))
                and len(re.sub(r"[\s\W_]+", "", row.get("title", ""))) >= 4
                and re.sub(r"[\s\W_]+", "", row.get("title", "").lower()) in compact_question
                for row in self.rows
            ]))
        stage1 = np.where(allowed, stage1, -1.0)
        stage1_positions = np.argsort(-stage1)[: min(max(10, stage1_k), len(stage1))]

        semantic = .42 * char_multi + .30 * word_multi + .28 * bm25_multi
        original_semantic = .55 * char_original + .45 * word_parts[0]
        authority_weight = .12 if policy_query else .06
        hard_negative_penalty = self._hard_negative_penalty(category) * (1.0 - np.maximum(title_bonus, quoted_title_bonus))
        stage2 = (
            .50 * stage1 + .15 * semantic + .08 * original_lexical + .04 * lexical
            + .08 * category_compatibility + .24 * title_bonus + .38 * quoted_title_bonus
            + authority_weight * self._authority + .04 * self._quality - hard_negative_penalty
        )
        # Stage 2 only reranks the broad Stage 1 shortlist.
        ordered = sorted(stage1_positions, key=lambda position: float(stage2[position]), reverse=True)
        selected: list[dict] = []
        for position in ordered:
            if len(selected) >= top_k:
                break
            score = float(stage2[position])
            if score < threshold:
                continue
            row = self.rows[int(position)]
            if sum(item["source_url"] == row["source_url"] for item in selected) >= 2:
                continue
            selected_text = self.select_context(question, row["text"], response_mode)
            if not selected_text:
                continue
            selected.append({
                **row,
                "retrieval_score": score,
                "rerank_score": score,
                "stage1_score": float(stage1[position]),
                "selected_text": selected_text,
                "stale": self._stale(row),
                "score_components": {
                    "character": float(char_multi[position]),
                    "word": float(word_multi[position]),
                    "bm25": float(bm25_multi[position]),
                    "lexical": float(lexical[position]),
                    "original_lexical": float(original_lexical[position]),
                    "original_semantic": float(original_semantic[position]),
                    "title_bonus": float(title_bonus[position]),
                    "quoted_title_bonus": float(quoted_title_bonus[position]),
                    "category_compatibility": float(category_compatibility[position]),
                    "authority": float(self._authority[position]),
                    "knowledge_quality": float(self._quality[position]),
                    "hard_negative_penalty": float(hard_negative_penalty[position]),
                },
            })
        top_score = float(selected[0]["rerank_score"]) if selected else 0.0
        runner_up = next((
            float(row["rerank_score"]) for row in selected[1:]
            if row.get("source_url") != selected[0].get("source_url")
        ), 0.0) if selected else 0.0
        top_semantic = 0.0
        top_category = 0.0
        if selected:
            position = self.positions[selected[0]["id"]]
            top_semantic = float(max(original_semantic[position], title_bonus[position], quoted_title_bonus[position]))
            top_category = float(category_compatibility[position])
        confidence = self._confidence(
            top_score, top_score - runner_up, top_semantic, top_category,
            len(re.sub(r"[\s\W_]+", "", question)), confidence_policy,
        )
        return selected, {
            "latency_ms": (time.perf_counter() - started) * 1000,
            "method": strategy,
            "two_stage": True,
            "stage1_candidates": len(stage1_positions),
            "query_count": len(active),
            "queries": active,
            "primary_category": category,
            "secondary_category": secondary_category,
            "intent": intent or category,
            "confidence": confidence,
            "confidence_policy": confidence_policy,
            "accepted": CONFIDENCE_ORDER[confidence] >= CONFIDENCE_ORDER["MEDIUM"],
            "top_score": round(top_score, 6),
            "score_margin": round(top_score - runner_up, 6),
            "top_semantic": round(top_semantic, 6),
            "top_category_compatibility": round(top_category, 3),
            "knowledge_sources": len({row["source_url"] for row in self.source_rows}),
            "knowledge_chunks": len(self.rows),
            "raw_top": [
                {"id": row["id"], "parent_id": row.get("parent_id"), "title": row["title"],
                 "category": row.get("category"), "source_url": row.get("source_url"),
                 "rerank_score": row["rerank_score"], "stage1_score": row["stage1_score"]}
                for row in selected[:5]
            ],
        }
