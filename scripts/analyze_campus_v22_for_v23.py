#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_retrieval_v22 import (
    KNOWLEDGE_FILES,
    CampusKnowledgeRetrieverV22,
    detect_numeric_conflict,
)
from pipeline.campus_retrieval_v23 import knowledge_quality
from retrieval.bm25 import tokens


OUT = ROOT / "data/campus_v23/retrieval"
EVALUATION_OUT = ROOT / "evaluation"
BENCHMARK = ROOT / "data/campus_v22/benchmarks/knowledge-1000.jsonl"
BLIND_RESULTS = ROOT / "evaluation/campus-v22-generalization-blind-300.json"

VAGUE = ("あれ", "やば", "どうしよう", "わから", "ちょっと", "そのうち")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalise(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def failure_reason(question: str, expected_category: str, results: list[dict], expected_url: str) -> str:
    if not results:
        return "INSUFFICIENT_KNOWLEDGE"
    first = results[0]
    if first.get("stale"):
        return "STALE_KNOWLEDGE"
    if len(normalise(question)) <= 16:
        return "QUERY_TOO_SHORT"
    if any(cue in question for cue in VAGUE):
        return "AMBIGUOUS_QUERY"
    if first.get("category") != expected_category:
        return "CATEGORY_MISMATCH"
    title_tokens = set(tokens(first.get("title", "")))
    if title_tokens and len(title_tokens & set(tokens(question))) >= max(1, min(2, len(title_tokens))):
        return "LEXICAL_COLLISION"
    urls = [row.get("source_url") for row in results]
    if expected_url in urls[1:]:
        return "WRONG_RERANK"
    if any(row.get("parent_id") == first.get("parent_id") and row["id"] != first["id"] for row in results[1:]):
        return "OVERLAPPING_CHUNK"
    if expected_url not in urls:
        return "INSUFFICIENT_KNOWLEDGE"
    return "OTHER"


def analyse_false_matches() -> tuple[list[dict], dict]:
    retriever = CampusKnowledgeRetrieverV22.from_files()
    benchmark = load_jsonl(BENCHMARK)
    sample = benchmark[::max(1, len(benchmark) // 300)][:300]
    failures = []
    for row in sample:
        results, meta = retriever.search(row["question"], row["category"], top_k=5,
                                         response_mode="normal", strategy="reranked")
        expected_url = row["expected_source_url"]
        if results and results[0].get("source_url") == expected_url:
            continue
        top = results[0] if results else {}
        reason = failure_reason(row["question"], row["category"], results, expected_url)
        failures.append({
            "id": row["id"],
            "query": row["question"],
            "expected_topic": row.get("group") or row["category"],
            "expected_category": row["category"],
            "expected_source": expected_url,
            "retrieved_chunk_id": top.get("id"),
            "retrieved_parent_id": top.get("parent_id"),
            "retrieved_chunk": top.get("selected_text"),
            "retrieved_title": top.get("title"),
            "retrieved_source": top.get("source_url"),
            "retrieved_category": top.get("category"),
            "retrieval_score": top.get("retrieval_score", 0.0),
            "rerank_score": top.get("retrieval_score", 0.0),
            "category": row["category"],
            "failure_reason": reason,
            "top5": [
                {"id": item["id"], "parent_id": item.get("parent_id"), "title": item["title"],
                 "category": item.get("category"), "source_url": item.get("source_url"),
                 "score": item.get("retrieval_score", 0.0)}
                for item in results
            ],
            "latency_ms": meta.get("latency_ms"),
        })
    counts = Counter(row["failure_reason"] for row in failures)
    summary = {
        "questions": len(sample),
        "false_matches": len(failures),
        "false_match_rate": round(len(failures) / len(sample), 4),
        "cause_counts": dict(counts.most_common()),
    }
    return failures, summary


def analyse_conflicts() -> dict:
    rows = [row for path in KNOWLEDGE_FILES for row in load_jsonl(ROOT / path)]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row.get("category", ""), row.get("sub_category", ""))].append(
            {**row, "selected_text": row.get("text", "")}
        )
    candidates = []
    for (category, sub_category), group in groups.items():
        if len({row.get("source_url") for row in group}) < 2 or not detect_numeric_conflict(group):
            continue
        sources = []
        universities = set()
        revisions = set()
        for row in group:
            numbers = sorted(set(re.findall(r"\d+(?:\.\d+)?(?:%|％|円|日|回|単位)", row.get("text", ""))))
            if not numbers:
                continue
            university = row.get("university_name")
            revision = row.get("revision_timestamp") or row.get("last_verified_at")
            if university:
                universities.add(university)
            if revision:
                revisions.add(str(revision)[:10])
            sources.append({
                "id": row["id"], "title": row["title"], "url": row["source_url"],
                "publisher": row.get("publisher"), "university": university,
                "revision_or_date": revision, "numbers": numbers,
                "quality": knowledge_quality(row, stale=False),
            })
        scope_separable = len(universities) >= 2
        date_separable = len(revisions) >= 2
        candidates.append({
            "id": f"campus-v23-conflict-{len(candidates) + 1:02d}",
            "category": category,
            "sub_category": sub_category,
            "source_count": len(sources),
            "scope_analysis": "different_universities" if scope_separable else "same_or_unspecified_scope",
            "date_analysis": "different_revision_dates" if date_separable else "same_or_unknown_date",
            "resolution": "KEEP_SEPARATE_BY_SCOPE" if scope_separable else (
                "KEEP_SEPARATE_BY_DATE" if date_separable else "UNRESOLVED_DO_NOT_ASSERT"),
            "unresolved": not scope_separable and not date_separable,
            "automatic_truth_selection": False,
            "sources": sources,
        })
    return {
        "candidate_groups": len(candidates),
        "unresolved_groups": sum(row["unresolved"] for row in candidates),
        "automatic_truth_selection": False,
        "items": candidates,
    }


def analyse_critical_blind() -> dict:
    payload = json.loads(BLIND_RESULTS.read_text(encoding="utf-8"))
    items = []
    for row in payload["items"]:
        judge = row["judge"]
        if not (judge["quality_label"] == "bad" or judge["hallucination_suspected"]
                or judge.get("university_policy_assertion")):
            continue
        components = []
        if row.get("predicted_category") != row.get("category"):
            components.append("ROUTER")
        if "PARTIAL_ANSWER" in judge.get("issues", []):
            components.append("PLANNER")
        if row.get("route") == "tool" and judge.get("unsupported_claims"):
            components.extend(("TOOL", "VALIDATOR"))
        if row.get("route") == "rag" and judge.get("unsupported_claims"):
            components.extend(("RETRIEVAL", "KNOWLEDGE"))
        items.append({
            "id": row["id"], "question": row["question"], "expected_category": row["category"],
            "predicted_category": row.get("predicted_category"), "route": row["route"],
            "score": row["score"], "issues": judge.get("issues", []),
            "unsupported_claims": judge.get("unsupported_claims", []),
            "root_components": list(dict.fromkeys(components or ["OTHER"])),
            "general_fix": (
                "planner category candidates and tool applicability gate" if "ROUTER" in components else
                "atomic requirement coverage" if "PLANNER" in components else
                "tool output grounding and final validator"
            ),
            "question_specific_rule_added": False,
        })
    return {"critical_errors": len(items), "items": items}


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    failures, summary = analyse_false_matches()
    false_payload = {
        "schema_version": "campus-v23-v22-false-match-analysis-v1",
        "generated_at": generated_at,
        "development_data": True,
        "baseline_method": "v2.2 reranked",
        "summary": summary,
        "items": failures,
    }
    write_json(OUT / "v22-false-matches.json", false_payload)
    write_json(OUT / "hard-negatives.json", {
        "schema_version": "campus-v23-hard-negatives-v1",
        "generated_at": generated_at,
        "source": "all v2.2 fixed-300 false matches",
        "automatic_training": False,
        "items": [{
            "id": row["id"], "query": row["query"],
            "expected_category": row["expected_category"], "expected_source": row["expected_source"],
            "retrieved_category": row["retrieved_category"], "negative_chunk_id": row["retrieved_chunk_id"],
            "negative_source": row["retrieved_source"], "negative_text": row["retrieved_chunk"],
            "failure_reason": row["failure_reason"],
        } for row in failures],
    })
    conflicts = analyse_conflicts()
    write_json(OUT / "numeric-conflicts.json", {
        "schema_version": "campus-v23-numeric-conflicts-v1", "generated_at": generated_at, **conflicts,
    })
    critical = analyse_critical_blind()
    write_json(EVALUATION_OUT / "campus-v23-v22-critical-root-causes.json", {
        "schema_version": "campus-v23-v22-critical-root-causes-v1", "generated_at": generated_at, **critical,
    })
    print(json.dumps({"false_match": summary, "conflicts": {
        "groups": conflicts["candidate_groups"], "unresolved": conflicts["unresolved_groups"],
    }, "critical": critical["critical_errors"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
