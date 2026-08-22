from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from retrieval.bm25 import LocalBM25


CASES = (
    ("GPAとは何ですか", ("GPA",)),
    ("卒業論文の意味", ("卒業論文",)),
    ("著作物を引用する方法", ("引用", "著作権")),
    ("人工知能について知りたい", ("人工知能",)),
    ("図書館とは", ("図書館",)),
    ("就職活動の概要", ("就職活動",)),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/v06/knowledge/wikipedia.jsonl")
    parser.add_argument("--output", default="evaluation/retrieval-benchmark-v06.json")
    args = parser.parse_args()
    started = time.perf_counter()
    index = LocalBM25.from_jsonl(args.corpus)
    build_seconds = time.perf_counter() - started
    rows = []
    for query, expected in CASES:
        query_started = time.perf_counter()
        results = index.search(query, top_k=3)
        rows.append({"query": query, "expected_titles": expected, "results": results,
                     "top3_hit": any(result["title"] in expected for result in results),
                     "latency_ms": 1000 * (time.perf_counter() - query_started)})
    report = {"corpus": args.corpus, "documents": len(index.rows), "build_seconds": build_seconds,
              "top3_accuracy": sum(row["top3_hit"] for row in rows) / len(rows), "cases": rows,
              "production_enabled": False,
              "reason": "Offline candidate only; current-rule answers still require current official sources."}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
