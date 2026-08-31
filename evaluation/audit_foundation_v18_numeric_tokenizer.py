from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer


DEFAULT_SAMPLES = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "12", "32", "64", "100", "123", "2026", "1 2 3 4",
    "12,345", "3.14", "-7", "GPA 3.5",
)


def audit(tokenizer: FoundationTokenizer, samples=DEFAULT_SAMPLES) -> dict:
    rows = []
    for text in samples:
        token_ids = tokenizer.encode(text)
        rows.append({
            "text": text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "decoded": tokenizer.decode(token_ids),
            "round_trip_exact": tokenizer.decode(token_ids) == text,
            "atomic": len(token_ids) == 1,
        })
    single_digits = [
        row for row in rows if len(row["text"]) == 1 and row["text"].isdigit()
    ]
    return {
        "schema_version": "foundation-v18-numeric-tokenizer-audit-v1",
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "samples": rows,
        "single_digit_atomic_rate": (
            sum(row["atomic"] for row in single_digits) / len(single_digits)
            if single_digits else None
        ),
        "round_trip_exact_rate": sum(row["round_trip_exact"] for row in rows) / len(rows),
        "synthetic_v4_numeric_representation": {
            "uses_foundation_tokenizer": False,
            "representation": "atomic synthetic token IDs 32-63",
            "conclusion": (
                "Foundation tokenizer fragmentation cannot cause Synthetic v4 numeric "
                "continuation failures; the synthetic control uses atomic IDs."
            ),
        },
        "production_tokenizer_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", default="tokenizer/foundation-v11-base-4096.json"
    )
    parser.add_argument(
        "--output", default="evaluation/foundation-v18-numeric-tokenizer-audit.json"
    )
    args = parser.parse_args()
    tokenizer = FoundationTokenizer.load(ROOT / args.tokenizer)
    result = audit(tokenizer)
    output = ROOT / args.output
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "single_digit_atomic_rate": result["single_digit_atomic_rate"],
        "round_trip_exact_rate": result["round_trip_exact_rate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
