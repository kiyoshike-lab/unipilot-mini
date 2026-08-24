from __future__ import annotations

from itertools import product
import json
from pathlib import Path

from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_router_v21 import CampusRouterV21, should_clarify


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    examples = load_jsonl(ROOT / "data" / "campus_v2" / "router" / "train.jsonl")
    examples += load_jsonl(ROOT / "data" / "campus_v21" / "router" / "adversarial-train-1500.jsonl")
    validation = json.loads((ROOT / "data" / "campus_v21" / "router" / "clarification-validation-1200.json").read_text(encoding="utf-8"))
    router = CampusRouterV21(examples, config_path=ROOT / "data" / "campus_v21" / "router" / "missing.json")
    features = []
    for index, item in enumerate(validation):
        _, values = router.analyze(item["question"])
        features.append((item, values))
        if (index + 1) % 300 == 0:
            print("features", index + 1)

    candidates = []
    for vague_length, vague_confidence, vague_margin, uncertain_length in product(
            (16, 24, 36, 60), (.65, .75, .85, .95), (.5, 1.0, 1.5, 2.0), (0, 8, 12)):
        config = {
            "minimum_signals_to_route": 1, "vague_max_length": vague_length,
            "vague_confidence_floor": vague_confidence, "vague_margin_floor": vague_margin,
            "uncertain_max_length": uncertain_length, "uncertain_confidence_floor": .58,
            "uncertain_margin_floor": .45, "selection_source": "clarification-validation-1200-grid-search",
        }
        ambiguous = [should_clarify(values, config) for item, values in features if item["ambiguous"]]
        determinate = [should_clarify(values, config) for item, values in features if not item["ambiguous"]]
        ambiguous_accuracy = sum(ambiguous) / len(ambiguous)
        unnecessary = sum(determinate) / len(determinate)
        overall = (sum(ambiguous) + len(determinate) - sum(determinate)) / len(features)
        candidates.append({"config": config, "ambiguous_handling": ambiguous_accuracy,
                           "unnecessary_clarify": unnecessary, "overall": overall,
                           "constraints_passed": ambiguous_accuracy >= .97 and unnecessary <= .02})
    eligible = [row for row in candidates if row["constraints_passed"]]
    selected = max(eligible or candidates, key=lambda row: (row["overall"], row["ambiguous_handling"],
                                                             -row["unnecessary_clarify"], -row["config"]["vague_max_length"]))
    output = {"validation": len(validation), "search_candidates": len(candidates),
              "objective": "maximize overall after ambiguous>=97% and unnecessary clarify<=2%",
              "selected_config": selected["config"], "selected_metrics": {key: selected[key] for key in (
                  "ambiguous_handling", "unnecessary_clarify", "overall", "constraints_passed")},
              "top_candidates": sorted(candidates, key=lambda row: row["overall"], reverse=True)[:20],
              "external_ai_api": "OFF"}
    path = ROOT / "data" / "campus_v21" / "router" / "clarification-config.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["selected_metrics"], ensure_ascii=False), json.dumps(output["selected_config"], ensure_ascii=False))


if __name__ == "__main__":
    main()
