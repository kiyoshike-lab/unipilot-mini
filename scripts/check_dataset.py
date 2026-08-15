from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


JAPANESE_OR_ALLOWED = re.compile(r"^[\s\u0000-\u007f\u3000-\u30ff\u3400-\u9fff\uff00-\uffef〈〉％]+$")


def content(row: dict) -> tuple[str, str]:
    return (str(row.get("user", row.get("text", ""))).strip(), str(row.get("assistant", row.get("text", ""))).strip())


def canonical(text: str) -> str:
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"[\s、。！？,.!?]", "", text)
    return text


def inspect(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    exact = Counter(hashlib.sha256("\n".join(content(row)).encode()).hexdigest() for row in rows)
    near = Counter(canonical("\n".join(content(row))) for row in rows)
    invalid = []
    for row in rows:
        user, assistant = content(row)
        reasons = []
        if not user or not assistant: reasons.append("empty")
        if row.get("kind") == "dialogue" and len(assistant) < 20: reasons.append("short_answer")
        if not JAPANESE_OR_ALLOWED.match(user + assistant): reasons.append("abnormal_character")
        if re.search(r"\b\d{7,}@|\b0\d{9,10}\b|@(?:gmail|yahoo)", user + assistant, re.I): reasons.append("possible_personal_data")
        if reasons: invalid.append({"id": row.get("id"), "reasons": reasons})
    families_by_split: dict[str, set] = {}
    for row in rows: families_by_split.setdefault(row.get("template_family", ""), set()).add(row.get("split", ""))
    leaked = [family for family, splits in families_by_split.items() if len(splits) > 1]
    input_lengths = [len(content(row)[0]) for row in rows]
    output_lengths = [len(content(row)[1]) for row in rows]
    result = {
        "total_samples": len(rows), "unique_samples": len(exact), "duplicates": sum(value - 1 for value in exact.values()),
        "duplicate_rate": sum(value - 1 for value in exact.values()) / max(1, len(rows)),
        "near_duplicate_pairs_by_canonical_form": sum(value - 1 for value in near.values()),
        "average_input_length": statistics.mean(input_lengths), "average_output_length": statistics.mean(output_lengths),
        "category_distribution": dict(Counter(row.get("category", "unknown") for row in rows)),
        "kind_distribution": dict(Counter(row.get("kind", "unknown") for row in rows)),
        "split_distribution": dict(Counter(row.get("split", "unknown") for row in rows)),
        "invalid_samples": len(invalid), "template_family_leaks": len(leaked), "leaked_families": leaked[:20],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", default=["data/train/v02.jsonl", "data/validation/v02.jsonl", "data/test/v02.jsonl"])
    parser.add_argument("--output", default="evaluation/dataset-quality-v02.json")
    args = parser.parse_args()
    result = inspect([Path(path) for path in args.files])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["invalid_samples"] or result["duplicates"] or result["template_family_leaks"]:
        raise SystemExit(1)


if __name__ == "__main__": main()
