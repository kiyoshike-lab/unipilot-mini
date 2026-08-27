"""Build strict, contamination-safe Foundation v1.1 document splits."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.mediawiki_cleaner import strict_quality_reason
from scripts.prepare_foundation_v10 import (
    add_hash,
    category_for,
    contamination,
    holdout_index,
    question_values,
    rows,
    semantic_duplicate,
    simhash,
    split_for,
)

DEFAULT_SOURCES = (
    "data/foundation_v11/raw/wikipedia-dump-ja.jsonl.gz",
    "data/foundation_v11/raw/wikibooks-dump-ja.jsonl.gz",
    "data/foundation_v11/raw/wikipedia-ja.jsonl.gz",
)
SAFE_HOLDOUTS = (
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


def safe_holdout_questions() -> tuple[list[str], dict]:
    questions: list[str] = []
    counts = {}
    for relative in SAFE_HOLDOUTS:
        path = ROOT / relative
        if not path.exists():
            counts[relative] = {"questions": 0, "missing": True}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = [value.strip() for value in question_values(payload) if value.strip()]
        counts[relative] = {"questions": len(selected), "missing": False}
        questions.extend(selected)
    unique = list(dict.fromkeys(questions))
    return unique, {"files": counts, "questions": len(questions),
                    "unique_questions": len(unique)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*", default=list(DEFAULT_SOURCES))
    parser.add_argument("--output-dir", default="data/foundation_v11/documents")
    parser.add_argument("--report", default="evaluation/foundation-v11-data-audit.json")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    questions, holdout_report = safe_holdout_questions()
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
    maximum_similarity = 0.0
    source_inputs = []
    try:
        for relative in args.sources:
            path = ROOT / relative
            input_rows = 0
            for row in rows(path):
                input_rows += 1
                reason = strict_quality_reason(row.get("title", ""), row["text"])
                if reason:
                    excluded[f"quality_{reason}"] += 1
                    continue
                page_key = (str(row.get("source_type", "")).split("_official_dump")[0],
                            int(row.get("page_id", 0)))
                if page_key in page_keys:
                    excluded["duplicate_page"] += 1
                    continue
                fingerprint = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                if fingerprint in exact:
                    excluded["exact_duplicate"] += 1
                    continue
                overlap = contamination(row["text"], question_grams, question_index)
                maximum_similarity = max(maximum_similarity, overlap)
                if overlap >= .78:
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
                selected = {**row, "content_sha256": fingerprint, "split": split,
                            "category": category, "contamination_checked": True,
                            "max_holdout_similarity": round(overlap, 6),
                            "semantic_dedup_checked": True,
                            "foundation_version": "v1.1-clean"}
                handles[split].write(json.dumps(selected, ensure_ascii=False) + "\n")
                accepted[split] += 1
                characters[split] += len(row["text"])
                source_characters[row["source_type"]] += len(row["text"])
                category_characters[category] += len(row["text"])
            source_inputs.append({"path": relative, "input_rows": input_rows})
    finally:
        for handle in handles.values():
            handle.close()
    report = {
        "schema_version": "foundation-v11-data-audit-v1",
        "sources": source_inputs, "documents": dict(accepted),
        "unique_documents": len(semantic_hashes), "characters": dict(characters),
        "total_characters": sum(characters.values()),
        "characters_by_source": dict(source_characters),
        "characters_by_category": dict(category_characters), "excluded": dict(excluded),
        "holdout_audit": {**holdout_report,
                          "maximum_segment_question_similarity": maximum_similarity,
                          "threshold": .78, "questions_used_only_as_fingerprints": True,
                          "answers_opened_or_used": False,
                          "final_blind_content_opened": False},
        "split_policy": "SHA256 document key: train 97%, validation 1.5%, test 1.5%",
        "document_level_and_semantic_dedup_before_split": True,
        "strict_zero_markup_residue": True,
        "pretraining_format": "plain Japanese next-token text; no instruction wrapper",
        "external_ai_api": "OFF", "production_changed": False,
    }
    report_path = ROOT / args.report
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
