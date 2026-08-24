"""Build the frozen Campus v2.1 RC known-issue queue without changing answer logic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.campus_v21 import UniPilotCampusV21


MANIFEST = ROOT / "evaluation/campus-v21-rc-manifest.json"
FAILURES = ROOT / "evaluation/campus-v21-full-failure-analysis.json"
RETRIEVAL = ROOT / "evaluation/campus-v21-retrieval.json"
BLIND = ROOT / "data/campus_v2/blind/evaluation-2000.json"
OUTPUT = ROOT / "evaluation/campus-v21-rc-known-issues.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def review(previous: dict[tuple[str, str], dict], group: str, item_id: str) -> dict:
    return previous.get((group, item_id), {
        "status": "pending", "severity": "unreviewed", "blocks_production": False, "notes": "",
    })


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in manifest["logic_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"RC answer logic changed after freeze: {relative}")

    prior: dict[tuple[str, str], dict] = {}
    if OUTPUT.exists():
        old = json.loads(OUTPUT.read_text(encoding="utf-8"))
        for group, items in old.get("groups", {}).items():
            for item in items:
                prior[(group, item["id"])] = item.get("human_review", {})

    analysis = json.loads(FAILURES.read_text(encoding="utf-8"))
    v21 = next(row for row in analysis["blind_2000"] if row["variant"] == "Campus v2.1")
    failures = v21["all_failures"]
    source = {row["id"]: row for row in json.loads(BLIND.read_text(encoding="utf-8"))}
    pipeline = UniPilotCampusV21(model=None, tokenizer=None)

    hallucination = []
    for failure in (row for row in failures if row["gold"] == "tuition"):
        answer = pipeline.answer(failure["question"])
        hallucination.append({
            "id": failure["id"], "question": failure["question"], "answer": answer["text"],
            "gold_category": failure["gold"], "predicted_category": failure["predicted"],
            "action": failure["action"], "automatic_flag": "HALLUCINATION",
            "expected_guardrail": "大学固有の期限・金額・手続を断定せず、公式窓口での確認を案内する",
            "source_forbidden_claims": source[failure["id"]].get("forbidden_claims", []),
            "human_review": review(prior, "hallucination", failure["id"]),
        })

    router = []
    for failure in (row for row in failures if not row["routing_success"]):
        answer = pipeline.answer(failure["question"])
        router.append({
            "id": failure["id"], "question": failure["question"], "answer": answer["text"],
            "gold_category": failure["gold"], "predicted_category": failure["predicted"],
            "top1": failure["top1"], "top2": failure["top2"], "margin": failure["margin"],
            "action": failure["action"], "automatic_flag": "ROUTER_ERROR",
            "known_pattern": "toeic_plan -> study_plan (compound English-study + lateness email)",
            "human_review": review(prior, "router", failure["id"]),
        })

    retrieval_payload = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    retrieval = []
    for failure in retrieval_payload["test"]["failures"]:
        retrieval.append({
            "id": failure["id"], "question": failure["query"], "automatic_flag": failure["reason"],
            "returned": failure.get("returned", []), "relevant": failure.get("relevant", []),
            "score": failure.get("score"),
            "human_review": review(prior, "retrieval", failure["id"]),
        })

    if (len(hallucination), len(router), len(retrieval)) != (13, 3, 7):
        raise RuntimeError(f"unexpected issue counts: {(len(hallucination), len(router), len(retrieval))}")
    payload = {
        "release_candidate": manifest["release_candidate"], "rc_source_commit": manifest["rc_source_commit"],
        "answer_logic_mutation": "prohibited", "review_status": "PENDING",
        "counts": {"hallucination": 13, "router": 3, "retrieval": 7, "total": 23},
        "groups": {"hallucination": hallucination, "router": router, "retrieval": retrieval},
        "external_ai_api": "OFF", "production_changed": False,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
