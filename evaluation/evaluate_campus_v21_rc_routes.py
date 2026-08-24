"""Measure frozen RC route mix and local latency on the balanced Human100 set."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.campus_v21 import UniPilotCampusV21


MANIFEST = ROOT / "evaluation/campus-v21-rc-manifest.json"
HUMAN = ROOT / "evaluation/human-comparison-campus-v21.json"
OUTPUT = ROOT / "evaluation/campus-v21-rc-route-speed.json"


def percentile(values: list[float], proportion: float) -> float:
    return sorted(values)[min(len(values) - 1, round((len(values) - 1) * proportion))]


def canonical(action: str) -> str:
    if action.startswith("TOOL"):
        return "TOOL"
    if action.startswith("RAG"):
        return "RAG"
    if action == "FAQ":
        return "FAQ"
    if action == "MODEL":
        return "MODEL"
    if action == "CLARIFY":
        return "CLARIFY"
    return "OTHER"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in manifest["logic_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"RC answer logic changed: {relative}")
    rows = json.loads(HUMAN.read_text(encoding="utf-8"))
    pipeline = UniPilotCampusV21(model=None, tokenizer=None)
    raw = Counter(); routes = Counter(); categories = Counter(); latencies = []
    records = []
    for row in rows:
        started = time.perf_counter()
        result = pipeline.answer(row["question"], session_id=f"route-{row['id']}")
        latency_ms = (time.perf_counter() - started) * 1000
        action = result.get("route_action", "OTHER")
        raw[action] += 1; routes[canonical(action)] += 1; categories[result["category"]] += 1
        latencies.append(latency_ms)
        records.append({"id": row["id"], "question": row["question"], "category": result["category"],
                        "route_action": action, "canonical_route": canonical(action),
                        "executed_action": result.get("executed_action"), "latency_ms": latency_ms,
                        "under_one_second": latency_ms < 1000})
    total = len(rows)
    payload = {
        "release_candidate": manifest["release_candidate"], "rc_source_commit": manifest["rc_source_commit"],
        "evaluation_set": "balanced Campus v2.1 Human100 (25 easy / 25 medium / 25 hard / 25 compound-ambiguous)",
        "questions": total,
        "canonical_route_mix": {name: {"count": routes.get(name, 0), "share": routes.get(name, 0) / total}
                                for name in ("TOOL", "FAQ", "RAG", "MODEL", "CLARIFY")},
        "raw_route_action_counts": dict(sorted(raw.items())),
        "planned_model_assisted_actions": {"count": sum("MODEL" in record["route_action"] for record in records),
                                           "share": sum("MODEL" in record["route_action"] for record in records) / total},
        "actual_model_generation": {"count": sum(record["executed_action"] == "MODEL" for record in records),
                                    "share": sum(record["executed_action"] == "MODEL" for record in records) / total,
                                    "non_model_share": sum(record["executed_action"] != "MODEL" for record in records) / total},
        "local_latency": {"mean_ms": statistics.mean(latencies), "p50_ms": percentile(latencies, .50),
                          "p95_ms": percentile(latencies, .95), "max_ms": max(latencies),
                          "under_one_second_count": sum(value < 1000 for value in latencies),
                          "under_one_second_share": sum(value < 1000 for value in latencies) / total},
        "category_counts": dict(sorted(categories.items())), "records": records,
        "measurement_scope": "local deterministic Campus routing/FAQ/tool path; not Render network latency",
        "external_ai_api": "OFF", "production_changed": False,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"route_mix": payload["canonical_route_mix"], "latency": payload["local_latency"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
