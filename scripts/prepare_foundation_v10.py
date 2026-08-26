"""Build contamination-safe Foundation v1.0 document splits from licensed sources."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zlib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    "data/foundation_v10/raw/wikipedia-dump-ja.jsonl.gz",
    "data/foundation_v10/raw/wikibooks-dump-ja.jsonl.gz",
    "data/foundation_v10/raw/wikipedia-ja.jsonl.gz",
)
HOLDOUTS = (
    "data/foundation_v09/evaluation/final-blind-1000.json",
    "data/campus_v23/holdouts/blind-500.json",
    "data/campus_v23/holdouts/stress-200.json",
    "data/campus_v22/generalization/blind-300.json",
    "data/campus_v22/generalization/stress-100.json",
    "data/standard_50m_short/blind-200.json",
    "data/v08/blind/evaluation.json",
    "evaluation/human-comparison-campus-v21.json",
    "evaluation/campus-v21-human-results.json",
    "evaluation/campus-v21-quick-human-results.json",
)
CATEGORY_KEYWORDS = {
    "mathematics": ("数学", "算数", "代数", "幾何", "解析", "確率", "統計", "数論"),
    "science": ("科学", "物理", "化学", "生物", "地学", "医学", "天文", "工学"),
    "computing": ("情報", "計算機", "コンピュータ", "プログラム", "アルゴリズム", "通信"),
    "history": ("歴史", "史学", "時代", "世紀", "古代", "中世", "近代"),
    "society_economics": ("社会", "経済", "政治", "法律", "法学", "金融", "産業", "行政"),
    "language_writing": ("日本語", "言語", "文学", "文章", "作文", "文法", "語学", "英語"),
    "education": ("教育", "学習", "教材", "教科", "学校", "大学", "講義", "試験"),
    "procedure": ("方法", "手順", "入門", "使い方", "実験", "演習", "作り方"),
}


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^0-9a-zぁ-んァ-ヶ一-龥々]", "", value)


def split_for(key: str) -> str:
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    if value < 9700:
        return "train"
    if value < 9850:
        return "validation"
    return "test"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def question_values(payload) -> list[str]:
    values: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            values.extend(question_values(item))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"question", "prompt", "user"} and isinstance(value, str):
                values.append(value)
            elif isinstance(value, (list, dict)):
                values.extend(question_values(value))
    return values


def holdout_questions() -> tuple[list[str], dict]:
    questions: list[str] = []
    counts = {}
    for relative in HOLDOUTS:
        path = ROOT / relative
        if not path.exists():
            counts[relative] = {"questions": 0, "missing": True}
            continue
        selected = [value.strip() for value in question_values(read_json(path)) if value.strip()]
        counts[relative] = {"questions": len(selected), "missing": False}
        questions.extend(selected)
    unique = list(dict.fromkeys(questions))
    return unique, {"files": counts, "questions": len(questions), "unique_questions": len(unique)}


def ngrams(value: str, size: int = 3) -> set[str]:
    normalized = normalize(value)
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[index:index + size] for index in range(len(normalized) - size + 1)}


def holdout_index(questions: list[str]) -> tuple[list[set[str]], dict[str, set[int]]]:
    grams = [ngrams(question) for question in questions]
    inverted: dict[str, set[int]] = {}
    for index, value in enumerate(grams):
        if not value:
            continue
        ordered = sorted(value, key=lambda item: hashlib.sha1(item.encode("utf-8")).digest())
        for gram in ordered[:min(8, len(ordered))]:
            inverted.setdefault(gram, set()).add(index)
    return grams, inverted


def contamination(text: str, question_grams: list[set[str]], inverted: dict[str, set[int]]) -> float:
    maximum = 0.0
    segments = [part for part in re.split(r"\n+|(?<=[。！？])", text) if len(part.strip()) >= 12]
    for segment in segments:
        grams = ngrams(segment)
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(inverted.get(gram, ()))
        for index in candidates:
            reference = question_grams[index]
            similarity = len(grams & reference) / max(1, len(grams | reference))
            maximum = max(maximum, similarity)
            if similarity >= 0.78:
                return similarity
    return maximum


def simhash(text: str) -> int:
    value = normalize(text[:8000])
    vector = [0] * 32
    for index in range(0, max(0, len(value) - 4), 3):
        hashed = zlib.crc32(value[index:index + 5].encode("utf-8"))
        for bit in range(32):
            vector[bit] += 1 if hashed & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def semantic_duplicate(value: int, buckets: list[dict[int, list[int]]], hashes: list[int]) -> bool:
    candidates: set[int] = set()
    for band in range(4):
        key = (value >> (band * 8)) & 0xFF
        candidates.update(buckets[band].get(key, ()))
    return any((value ^ hashes[index]).bit_count() <= 3 for index in candidates)


def add_hash(value: int, index: int, buckets: list[dict[int, list[int]]]) -> None:
    for band in range(4):
        key = (value >> (band * 8)) & 0xFF
        buckets[band].setdefault(key, []).append(index)


def category_for(row: dict) -> str:
    haystack = row.get("title", "") + " " + " ".join(row.get("categories") or [])
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return category
    return "general_encyclopedia"


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*", default=list(DEFAULT_SOURCES))
    parser.add_argument("--output-dir", default="data/foundation_v10/documents")
    parser.add_argument("--report", default="evaluation/foundation-v10-data-audit.json")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    questions, holdout_report = holdout_questions()
    question_grams, question_index = holdout_index(questions)
    handles = {split: gzip.open(output / f"{split}.jsonl.gz", "wt", encoding="utf-8",
                                newline="\n") for split in ("train", "validation", "test")}
    exact: set[str] = set()
    page_keys: set[tuple[str, int]] = set()
    semantic_hashes: list[int] = []
    buckets: list[dict[int, list[int]]] = [{}, {}, {}, {}]
    accepted = Counter()
    characters = Counter()
    source_characters = Counter()
    category_characters = Counter()
    excluded = Counter()
    maximum_holdout_similarity = 0.0
    source_inputs = []
    try:
        for relative in args.sources:
            path = ROOT / relative
            if not path.exists():
                source_inputs.append({"path": relative, "missing": True})
                continue
            input_rows = 0
            for row in rows(path):
                input_rows += 1
                page_key = (str(row.get("source_type", "")).split("_official_dump")[0],
                            int(row.get("page_id", 0)))
                if page_key in page_keys:
                    excluded["duplicate_page"] += 1
                    continue
                fingerprint = row.get("content_sha256") or hashlib.sha256(
                    row["text"].encode("utf-8")).hexdigest()
                if fingerprint in exact:
                    excluded["exact_duplicate"] += 1
                    continue
                overlap = contamination(row["text"], question_grams, question_index)
                maximum_holdout_similarity = max(maximum_holdout_similarity, overlap)
                if overlap >= 0.78:
                    excluded["semantic_holdout_overlap"] += 1
                    continue
                value = simhash(row["title"] + "\n" + row["text"])
                if semantic_duplicate(value, buckets, semantic_hashes):
                    excluded["semantic_duplicate"] += 1
                    continue
                exact.add(fingerprint)
                page_keys.add(page_key)
                add_hash(value, len(semantic_hashes), buckets)
                semantic_hashes.append(value)
                split = split_for(f"{row['source_type']}:{row['page_id']}")
                category = category_for(row)
                selected = {**row, "split": split, "category": category,
                            "contamination_checked": True,
                            "max_holdout_similarity": round(overlap, 6),
                            "semantic_dedup_checked": True}
                handles[split].write(json.dumps(selected, ensure_ascii=False) + "\n")
                accepted[split] += 1
                characters[split] += len(row["text"])
                source_characters[row["source_type"]] += len(row["text"])
                category_characters[category] += len(row["text"])
            source_inputs.append({"path": relative, "missing": False, "input_rows": input_rows})
    finally:
        for handle in handles.values():
            handle.close()
    report = {
        "schema_version": "foundation-v10-data-audit-v1",
        "sources": source_inputs,
        "documents": dict(accepted), "unique_documents": len(semantic_hashes),
        "characters": dict(characters), "total_characters": sum(characters.values()),
        "characters_by_source": dict(source_characters),
        "characters_by_category": dict(category_characters), "excluded": dict(excluded),
        "holdout_audit": {**holdout_report,
                          "maximum_segment_question_similarity": maximum_holdout_similarity,
                          "threshold": 0.78, "questions_used_only_as_fingerprints": True,
                          "answers_opened_or_used": False},
        "split_policy": "SHA256 document key: train 97%, validation 1.5%, test 1.5%",
        "pretraining_format": "plain Japanese next-token text; no user/assistant wrapper",
        "external_ai_api": "OFF", "production_changed": False,
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
