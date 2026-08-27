from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.mediawiki_cleaner import clean_mediawiki, strict_quality_reason
from scripts.collect_foundation_v10_wikimedia import clean_extract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/foundation_v10/raw/wikipedia-ja.jsonl.gz")
    parser.add_argument("--output", default="data/foundation_v11/raw/wikipedia-ja.jsonl.gz")
    parser.add_argument("--report", default="evaluation/foundation-v11-wikipedia-api-reclean.json")
    args = parser.parse_args()
    input_path, output_path = ROOT / args.input, ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = modified = characters = 0
    excluded = Counter()
    with gzip.open(input_path, "rt", encoding="utf-8") as source, gzip.open(
        output_path, "wt", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            row = json.loads(line)
            cleaned, metrics = clean_mediawiki(row["text"])
            cleaned, line_metrics = clean_extract(cleaned)
            reason = strict_quality_reason(row.get("title", ""), cleaned)
            if reason:
                excluded[reason] += 1
                continue
            fingerprint = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            selected = {**row, "text": cleaned, "content_sha256": fingerprint,
                        "cleaning": {**metrics, "line_cleanup": line_metrics,
                                     "cleaner": "foundation-v11-stack-v1"}}
            target.write(json.dumps(selected, ensure_ascii=False) + "\n")
            accepted += 1
            modified += int(cleaned != row["text"])
            characters += len(cleaned)
    report = {
        "schema_version": "foundation-v11-api-reclean-v1", "input": args.input,
        "output": args.output, "accepted_documents": accepted,
        "accepted_characters": characters, "modified_documents": modified,
        "excluded": dict(excluded), "metadata_preserved": True,
        "strict_zero_residue_gate": True, "external_ai_api": "OFF",
    }
    path = ROOT / args.report
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
