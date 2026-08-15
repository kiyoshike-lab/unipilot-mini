from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics_v03 import aggregate, broken_text_metrics, semantic_score
from evaluation.metrics_v02 import japanese_character_ratio, repetition_rate


def recompute(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["generations"]
    for row in rows:
        row["repetition_rate"] = repetition_rate(row["answer"])
        row["japanese_character_ratio"] = japanese_character_ratio(row["answer"])
        row["broken"] = broken_text_metrics(row["answer"])
        row.update(semantic_score(row["answer"], row))
    payload["metrics"] = aggregate(rows, payload["generation_settings"]["max_new_tokens"])
    payload["metric_definition_version"] = "v03.1-relevance-evidence-required"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "metrics": payload["metrics"]}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute v0.3 saved generations without regenerating text")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        recompute(path)


if __name__ == "__main__":
    main()
