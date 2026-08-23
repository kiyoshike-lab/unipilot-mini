from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics

from tokenizer.tokenizer import BPETokenizer


TARGETS = {"short": (40, 80), "normal": (100, 200), "detailed": (200, 400)}


def main() -> None:
    tokenizer = BPETokenizer.load("tokenizer/vocab-standard-v08-1024.json")
    samples = defaultdict(list)
    for line in Path("data/v08/curriculum/C/train.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        mode = row["length_mode"]
        if len(samples[mode]) < 100:
            samples[mode].append(len(tokenizer.encode(row["assistant"])))
        if all(len(samples[name]) >= 100 for name in TARGETS):
            break
    results = []
    for mode, (minimum, maximum) in TARGETS.items():
        values = samples[mode]
        results.append({
            "mode": mode, "target_tokens": [minimum, maximum], "samples": len(values),
            "mean_tokens": statistics.fmean(values), "min_tokens": min(values), "max_tokens": max(values),
            "p05_tokens": sorted(values)[4], "p95_tokens": sorted(values)[94],
            "target_hit_rate": sum(minimum <= value <= maximum for value in values) / len(values),
        })
    report = {
        "results": results, "all_targets_passed": all(row["target_hit_rate"] >= 0.8 for row in results),
        "decision": "API modes and hard caps are implemented, but training answer lengths must be rewritten before long training if targets are not met.",
    }
    Path("evaluation/length-control-v08.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
