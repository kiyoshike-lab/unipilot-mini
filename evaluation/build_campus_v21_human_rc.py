from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.campus_v21 import UniPilotCampusV21


RC_COMMIT = "0dc18789be28613a8c651cfefde63fb659ee2019"
LOGIC_PATHS = (
    "pipeline/campus_v21.py", "pipeline/campus_router_v21.py", "pipeline/campus_retrieval_v21.py",
    "pipeline/campus_composer_v21.py", "pipeline/campus_v2.py",
    "data/campus_v21/router/adversarial-train-1500.jsonl",
    "data/campus_v21/router/clarification-config.json",
    "data/campus_v21/retrieval/retrieval-config.json", "data/campus_v2/faq/reviewed.jsonl",
)
CHECKPOINT = "checkpoints/v04-eos15/unipilot-mini-v04-inference.pt"
TOKENIZER = "tokenizer/vocab-v02-512.json"
SPECIALIST = {
    "gpa": "GPA計算", "grade_simulator": "必要点数計算", "professor_email": "教授メール",
    "absence_email": "欠席メール", "lateness_email": "遅刻メール", "study_plan": "試験計画",
    "assignment_priority": "課題優先順位", "deadline_organizer": "締切整理",
    "university_policy": "大学制度の安全な案内",
}

EASY_CATEGORIES = (
    "gpa", "grade_simulator", "professor_email", "absence_email", "lateness_email", "study_plan",
    "assignment_priority", "deadline_organizer", "university_policy", "exam", "assignment", "credit",
    "attendance", "lateness", "registration", "report_outline", "citation_check", "presentation_outline",
    "toeic_plan", "career_schedule", "internship", "scholarship", "part_time_job", "ai_usage", "campus_life",
)
MEDIUM_CATEGORIES = (
    "gpa", "grade_simulator", "professor_email", "absence_email", "lateness_email", "study_plan",
    "assignment_priority", "deadline_organizer", "university_policy", "exam", "assignment", "credit",
    "attendance", "registration", "report_outline", "citation_check", "presentation_outline", "toeic_plan",
    "career_schedule", "internship", "scholarship", "part_time_job", "ai_usage", "relationship", "general",
)
HARD_CATEGORIES = (
    "gpa", "grade_simulator", "professor_email", "absence_email", "lateness_email", "study_plan",
    "assignment_priority", "deadline_organizer", "university_policy", "exam", "assignment", "credit",
    "attendance", "lateness", "registration", "report_outline", "citation_check", "presentation_outline",
    "toeic_plan", "career_schedule", "internship", "scholarship", "ai_usage", "programming", "statistics",
)
COMPOUND_CATEGORIES = (
    "gpa", "grade_simulator", "absence_email", "lateness_email", "study_plan", "assignment_priority",
    "deadline_organizer", "university_policy", "toeic_plan", "career_schedule", "report_outline",
    "citation_check", "presentation_outline",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def freeze_manifest() -> dict:
    hashes = {}
    for name in LOGIC_PATHS:
        committed_blob = git_output("rev-parse", f"{RC_COMMIT}:{name}")
        current_blob = git_output("hash-object", name)
        if committed_blob != current_blob:
            raise RuntimeError(f"RC answer logic changed after freeze: {name}")
        hashes[name] = sha256(ROOT / name)
    checkpoint = ROOT / CHECKPOINT
    tokenizer = ROOT / TOKENIZER
    manifest = {
        "release_candidate": "UniPilot Campus v2.1 RC1", "rc_source_commit": RC_COMMIT,
        "frozen_at": "2026-08-24", "answer_logic_mutation_policy": "prohibited during human evaluation",
        "logic_sha256": hashes,
        "checkpoint": {"path": CHECKPOINT, "size_bytes": checkpoint.stat().st_size,
                       "sha256": sha256(checkpoint), "model": "v0.4", "step": 2000},
        "tokenizer": {"path": TOKENIZER, "sha256": sha256(tokenizer), "vocab": 512},
        "external_ai_api": "OFF", "production_changed": False,
    }
    manifest["combined_logic_sha256"] = hashlib.sha256(
        "".join(f"{key}:{value}\n" for key, value in sorted(hashes.items())).encode()).hexdigest()
    return manifest


def semantic_key(question: str) -> str:
    return re.sub(r"\s+\d+$", "", re.sub(r"[\s　]+", "", question.lower()))


def pick(rows: list[dict], used_ids: set[str], used_questions: set[str], *, category: str | None = None,
         difficulty: str | None = None, surface: str | None = None) -> dict:
    for row in rows:
        if row["id"] in used_ids or semantic_key(row["prompt"]) in used_questions:
            continue
        if category is not None and row["category"] != category:
            continue
        if difficulty is not None and row.get("difficulty") != difficulty:
            continue
        if surface is not None and row.get("surface_type") != surface:
            continue
        used_ids.add(row["id"]); used_questions.add(semantic_key(row["prompt"]))
        return row
    raise RuntimeError(f"no unique RC question for category={category}, difficulty={difficulty}, surface={surface}")


def challenge_tags(row: dict) -> list[str]:
    tags = []
    surface = row.get("surface_type")
    category = row["category"]
    if surface == "colloquial": tags.append("colloquial")
    if surface == "ambiguous": tags.append("ambiguous")
    if surface == "compound": tags.extend(("multi_condition", "compound"))
    if category in ("general", "relationship", "campus_life"): tags.append("free_campus_consultation")
    if category in ("math", "statistics", "programming", "ai_usage"): tags.append("general_knowledge_or_reasoning")
    if category in ("report_outline", "presentation_outline", "es_outline", "professor_email"):
        tags.append("long_form_writing")
    if len(row["prompt"]) >= 55: tags.append("long_explanation")
    return list(dict.fromkeys(tags))


def empty_pairwise() -> dict:
    axes = ("correctness", "specificity", "actionability", "readability", "would_use")
    return {competitor: {axis: "unscored" for axis in axes} for competitor in ("chatgpt", "gemini")}


def empty_issues() -> dict:
    return {key: False for key in (
        "critical_error", "factual_error", "unanswered", "university_policy_assertion",
        "unnecessary_information", "unusable_answer", "router_error", "tool_error", "faq_error",
        "retrieval_error", "model_error",
    )}


def empty_ux() -> dict:
    return {key: "not_evaluated" for key in (
        "tool_card", "copy_action", "input_flow", "clarification", "streaming", "latency",
    )}


def automatic_evaluation(source: dict, result: dict) -> dict:
    action = result.get("route_action", result.get("executed_action"))
    ambiguous = bool(source.get("ambiguous") or source.get("surface_type") == "ambiguous")
    routed = result.get("router", {}).get("primary", result["category"])
    routing_success = action == "CLARIFY" if ambiguous else routed == source["category"]
    forbidden = source.get("forbidden_claims", []) + [
        "必ず追試", "必ず単位", "全国の大学で", "どの大学でも同じ", "欠席3回で単位を落とす",
    ]
    invented = any(claim in result["text"] for claim in forbidden)
    relevant = set(source.get("relevant_faq_ids", []))
    returned = {row["id"] for row in result.get("retrieval", [])}
    retrieval_status = ("not_applicable" if not relevant else "correct" if relevant & returned else
                        "wrong" if returned else "no_retrieval")
    return {"routing_success": routing_success,
            "answer_correct": bool(routing_success and result.get("validator", {}).get("valid", True) and not invented),
            "hallucination": invented, "retrieval_status": retrieval_status}


def main() -> None:
    manifest = freeze_manifest()
    blind = json.loads((ROOT / "data/campus_v2/blind/evaluation-2000.json").read_text(encoding="utf-8"))
    used_ids, used_questions = set(), set()
    selected: list[tuple[str, dict]] = []
    for category in EASY_CATEGORIES:
        selected.append(("easy", pick(blind, used_ids, used_questions, category=category, difficulty="easy")))
    for category in MEDIUM_CATEGORIES:
        selected.append(("medium", pick(blind, used_ids, used_questions, category=category, difficulty="medium")))
    for category in HARD_CATEGORIES:
        selected.append(("hard", pick(blind, used_ids, used_questions, category=category,
                                      difficulty="hard", surface="hard_negation_typo")))
    for category in COMPOUND_CATEGORIES:
        selected.append(("compound_ambiguous", pick(blind, used_ids, used_questions, category=category,
                                                    surface="compound")))
    for _ in range(12):
        selected.append(("compound_ambiguous", pick(blind, used_ids, used_questions, surface="ambiguous")))
    assert Counter(bucket for bucket, _ in selected) == {
        "easy": 25, "medium": 25, "hard": 25, "compound_ambiguous": 25}

    old_path = ROOT / "evaluation/human-comparison-campus-v21.json"
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        if any(row.get("evaluation_status") == "SCORED_MANUALLY" or
               any(value is not None for value in row.get("scores", {}).values()) for row in old):
            raise RuntimeError("human evaluation already started; refusing to replace RC questions")

    pipeline = UniPilotCampusV21()
    human_rows = []
    for index, (bucket, source) in enumerate(selected):
        result = pipeline.answer(source["prompt"], max_new_tokens=32)
        automatic = automatic_evaluation(source, result)
        human_rows.append({
            "id": f"campus-v21-rc-human-{index:03d}", "source_id": source["id"],
            "question": source["prompt"], "category": source["category"],
            "intent_labels": source.get("intent_labels", [source["category"]]),
            "evaluation_bucket": bucket, "difficulty": source.get("difficulty"),
            "surface_type": source.get("surface_type"), "challenge_tags": challenge_tags(source),
            "specialist_domain": SPECIALIST.get(source["category"]),
            "campus_answer": result["text"], "campus_answer_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
            "campus_metadata": {"category": result["category"], "action": result.get("route_action"),
                                "route": result["route"], "tool": result.get("tool"),
                                "cards": result.get("cards", []), "validator": result.get("validator", {}),
                                "latency_ms": result["timing"]["total_seconds"] * 1000,
                                "external_ai_api": result.get("external_ai_api")},
            "automatic_evaluation": automatic,
            "scores": {"correctness": None, "relevance": None, "actionable": None,
                       "naturalness": None, "would_use_again": None},
            "issue_flags": empty_issues(), "issues_reviewed": False, "pairwise": empty_pairwise(),
            "competitor_scores": {"chatgpt": None, "gemini": None},
            "chatgpt_answer": "", "gemini_answer": "", "ux": empty_ux(), "notes": "",
            "evaluation_status": "PENDING_MANUAL_REVIEW", "rc_source_commit": RC_COMMIT,
        })

    distribution = Counter(row["evaluation_bucket"] for row in human_rows)
    categories = Counter(row["category"] for row in human_rows)
    audit = {
        "questions": len(human_rows), "bucket_distribution": dict(distribution),
        "surface_distribution": dict(Counter(row["surface_type"] for row in human_rows)),
        "category_distribution": dict(categories),
        "specialist_questions": sum(row["specialist_domain"] is not None for row in human_rows),
        "challenge_tag_distribution": dict(Counter(tag for row in human_rows for tag in row["challenge_tags"])),
        "unique_semantic_questions": len({semantic_key(row["question"]) for row in human_rows}),
        "required_topics_present": {category: categories[category] > 0 for category in (
            "exam", "assignment", "credit", "gpa", "attendance", "lateness", "registration",
            "professor_email", "report_outline", "citation_check", "presentation_outline", "study_plan",
            "toeic_plan", "career_schedule", "internship", "scholarship", "part_time_job", "ai_usage",
            "campus_life")},
        "compound_present": any(row["surface_type"] == "compound" for row in human_rows),
        "ambiguous_present": any(row["surface_type"] == "ambiguous" for row in human_rows),
        "colloquial_present": any(row["surface_type"] == "colloquial" for row in human_rows),
        "easy_only": len(distribution) == 1 and "easy" in distribution,
        "rc_source_commit": RC_COMMIT,
    }
    assert len(human_rows) == audit["unique_semantic_questions"] == 100
    assert all(audit["required_topics_present"].values())
    assert audit["compound_present"] and audit["ambiguous_present"] and audit["colloquial_present"]
    old_path.write_text(json.dumps(human_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "evaluation/campus-v21-human-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "evaluation/campus-v21-rc-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest["combined_logic_sha256"], "audit": audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
