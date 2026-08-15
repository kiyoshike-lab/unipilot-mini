from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics


ROOT = Path("data/v03")


def check() -> dict:
    rows = []
    files = sorted(ROOT.glob("stage_*/*.jsonl"))
    for path in files:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    content_hashes = Counter()
    invalid = []; families = defaultdict(set); answers = []
    quality_values = defaultdict(list)
    for row in rows:
        content = row.get("text") or row.get("user", "") + "\n" + row.get("assistant", "")
        content_hashes[hashlib.sha256(content.encode()).hexdigest()] += 1
        families[row["template_family"]].add(row["split"])
        reasons = []
        if not content.strip(): reasons.append("empty")
        if "�" in content or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", content): reasons.append("broken_character")
        if any(marker in content for marker in ["例番号", "予定番号"]): reasons.append("synthetic_index_leak")
        quality = row.get("quality", {})
        for key in ["length_score", "diversity_score", "keyword_alignment"]:
            if key in quality: quality_values[key].append(float(quality[key]))
        if not quality.get("format_valid"): reasons.append("format")
        if row["kind"] == "conversation":
            answers.append(row["assistant"])
            if not row.get("eos_required"): reasons.append("eos_missing")
            if not row.get("intent"): reasons.append("intent_missing")
            if not row.get("expected_keywords"): reasons.append("keywords_missing")
            if quality.get("keyword_alignment", 0) <= 0: reasons.append("question_answer_mismatch")
            if any(bad in row["assistant"] for bad in ["出席率70%なら", "必ず単位が取れ"]): reasons.append("unsafe_rule_claim")
        if reasons: invalid.append({"id": row.get("id"), "reasons": reasons})
    fixed = json.loads(Path("evaluation/fixed_prompts_v03.json").read_text(encoding="utf-8"))
    fixed_invalid = [item["id"] for item in fixed if not all(key in item for key in ["expected_keywords", "forbidden_keywords", "category", "intent"])]
    human = json.loads(Path("evaluation/human-eval-v03.json").read_text(encoding="utf-8"))
    result = {
        "dataset_version": "unipilot-dataset-v03", "total_samples": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)), "split_counts": dict(Counter(row["split"] for row in rows)),
        "kind_counts": dict(Counter(row["kind"] for row in rows)), "category_counts": dict(Counter(row.get("category", "general_japanese") for row in rows)),
        "intents": dict(Counter(row.get("intent", "NONE") for row in rows)),
        "exact_duplicates": sum(value - 1 for value in content_hashes.values()),
        "template_family_leaks": sum(len(splits) > 1 for splits in families.values()), "invalid_samples": len(invalid),
        "conversation_eos_valid_rate": sum(row.get("eos_required") is True for row in rows if row["kind"] == "conversation") / max(1, len(answers)),
        "unique_answer_rate": len(set(answers)) / max(1, len(answers)),
        "average_quality": {key: statistics.mean(values) for key, values in quality_values.items()},
        "fixed_prompts": len(fixed), "fixed_prompt_schema_errors": len(fixed_invalid), "human_eval_items": len(human),
        "invalid_examples": invalid[:20],
    }
    return result


def main():
    result = check(); output = Path("evaluation/dataset-quality-v03.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if any(result[key] for key in ["exact_duplicates", "template_family_leaks", "invalid_samples", "fixed_prompt_schema_errors"]): raise SystemExit(1)


if __name__ == "__main__": main()
