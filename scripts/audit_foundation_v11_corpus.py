from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.mediawiki_cleaner import residue_signals, strict_quality_reason


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_rows() -> list[dict]:
    output = []
    for split in ("train", "validation", "test"):
        with gzip.open(ROOT / f"data/foundation_v11/documents/{split}.jsonl.gz", "rt",
                       encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                row["audited_split"] = split
                output.append(row)
    return output


def signals(row: dict) -> dict:
    text = row["text"]
    japanese = len(re.findall(r"[ぁ-んァ-ヶ一-龥々]", text)) / max(1, len(text))
    markup = residue_signals(text)
    reference_residue = bool(re.search(
        r"(?im)^\s*(?:脚注|出典|参考文献|外部リンク)\s*[。:]?\s*$|<references?\b", text
    ))
    navigation_residue = bool(re.search(
        r"(?im)^\s*(?:次へ|前へ|目次|カテゴリ|ナビゲーション)\s*$", text
    ))
    broken = "\ufffd" in text or bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text))
    readable = japanese >= .35 and len(text) >= 500 and len(re.findall(r"[。！？]", text)) >= 2
    return {
        "readable_japanese": readable,
        "markup_free": not any(markup.values()),
        "reference_free": not reference_residue,
        "navigation_free": not navigation_residue,
        "broken_text_free": not broken,
        "strict_quality_reason": strict_quality_reason(row.get("title", ""), text),
        "residue_signals": markup,
    }


def main() -> int:
    rows = all_rows()
    all_signal_rows = [signals(row) for row in rows]
    residue = Counter()
    strict_failures = Counter()
    for item in all_signal_rows:
        residue.update(item["residue_signals"])
        if item["strict_quality_reason"]:
            strict_failures[item["strict_quality_reason"]] += 1
    rng = random.Random(11012026)
    sample_indices = sorted(rng.sample(range(len(rows)), 100))
    sample = []
    for index in sample_indices:
        row = rows[index]
        item = signals(row)
        sample.append({
            "id": row["id"], "split": row["audited_split"],
            "source_type": row["source_type"], "source": row["source"],
            "source_url": row["source_url"], "license": row["license"],
            "characters": len(row["text"]), "opening_text": row["text"][:300],
            "quality_signals": item,
        })
    sample_rates = {
        key: sum(row["quality_signals"][key] for row in sample) / len(sample)
        for key in ("readable_japanese", "markup_free", "reference_free",
                    "navigation_free", "broken_text_free")
    }
    final_blind = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    final_hash = sha256(final_blind)
    checks = {
        "all_documents_strict_quality_pass": not strict_failures,
        "all_residue_counts_zero": not any(residue.values()),
        "sample_readable_japanese_100_percent": sample_rates["readable_japanese"] == 1,
        "sample_markup_free_100_percent": sample_rates["markup_free"] == 1,
        "sample_reference_free_100_percent": sample_rates["reference_free"] == 1,
        "sample_navigation_free_100_percent": sample_rates["navigation_free"] == 1,
        "sample_broken_text_free_100_percent": sample_rates["broken_text_free"] == 1,
        "final_blind_hash_matches_without_parsing": final_hash == (
            "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"
        ),
    }
    report = {
        "schema_version": "foundation-v11-corpus-quality-audit-v1",
        "documents": len(rows), "full_corpus_residue_counts": dict(residue),
        "full_corpus_strict_quality_failures": dict(strict_failures),
        "random_sample_seed": 11012026, "random_sample_documents": len(sample),
        "random_sample_rates": sample_rates, "random_sample": sample,
        "final_blind": {"sha256": final_hash, "content_parsed": False,
                        "used_for_training_or_evaluation": False},
        "checks": checks, "corpus_quality": "PASS" if all(checks.values()) else "FAIL",
        "external_ai_api": "OFF", "production_changed": False,
    }
    json_path = ROOT / "evaluation/foundation-v11-corpus-quality-audit.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    lines = [
        "# Foundation v1.1 Random 100 Document Audit", "",
        f"- Corpus Quality: **{report['corpus_quality']}**",
        f"- Documents: {len(rows):,}",
        f"- Residue counts: `{json.dumps(dict(residue), ensure_ascii=False)}`",
        "- Sample rates: " + ", ".join(f"{key}={value:.0%}"
                                          for key, value in sample_rates.items()), "",
    ]
    for number, row in enumerate(sample, 1):
        excerpt = " ".join(row["opening_text"].split())
        lines.extend([
            f"## {number}. {row['id']}", "",
            f"- Source: {row['source_type']}",
            f"- Split: {row['split']}",
            f"- Signals: `{json.dumps(row['quality_signals'], ensure_ascii=False)}`", "",
            excerpt, "",
        ])
    md_path = ROOT / "evaluation/foundation-v11-random-100-audit.md"
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "random_sample"}, ensure_ascii=False, indent=2))
    return 0 if report["corpus_quality"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
