from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer, SPECIAL_TOKENS


def main() -> int:
    benchmark = json.loads((ROOT / "evaluation/foundation-v11-tokenizer-benchmark.json").read_text(
        encoding="utf-8"
    ))
    selected_vocab = int(benchmark["selected_vocab"])
    selected = FoundationTokenizer.load(
        ROOT / f"tokenizer/foundation-v11-base-{selected_vocab}.json"
    )
    normal_texts = [
        "自然な日本語を正確に符号化する。",
        "BOSやEOSは通常本文から自動挿入されない。",
        "大学では資料を読み、根拠を確認する。",
        "記号「」【】とcode[index]も往復する。",
    ]
    manifest = json.loads((
        ROOT / f"data/foundation_v11/packed/vocab-{selected_vocab}/manifest.json"
    ).read_text(encoding="utf-8"))
    train = np.memmap(ROOT / manifest["splits"]["train"]["path"], dtype=np.uint16, mode="r")
    packed_special_counts = {
        token: int(np.count_nonzero(train == selected.special_to_id[token]))
        for token in SPECIAL_TOKENS
    }
    per_vocab = []
    for vocab in (1024, 2048, 4096):
        tokenizer = FoundationTokenizer.load(ROOT / f"tokenizer/foundation-v11-base-{vocab}.json")
        per_vocab.append({
            "vocab": vocab, "special_token_ids": tokenizer.special_to_id,
            "roundtrip_rate": sum(
                tokenizer.decode(tokenizer.encode(text)) == text for text in normal_texts
            ) / len(normal_texts),
            "normal_text_contains_eos": any(
                tokenizer.eos_id in tokenizer.encode(text) for text in normal_texts
            ),
        })
    checks = {
        "special_token_ids_are_expected": selected.special_to_id == {
            "<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3,
            "<USER>": 4, "<ASSISTANT>": 5, "<SYSTEM>": 6,
        },
        "special_token_ids_unique": len(set(selected.special_to_id.values())) == len(SPECIAL_TOKENS),
        "all_vocab_roundtrip_100_percent": all(row["roundtrip_rate"] == 1 for row in per_vocab),
        "normal_text_never_encodes_eos": not any(row["normal_text_contains_eos"] for row in per_vocab),
        "packed_bos_matches_documents": packed_special_counts["<BOS>"] ==
                                         manifest["splits"]["train"]["documents"],
        "packed_eos_matches_documents": packed_special_counts["<EOS>"] ==
                                         manifest["splits"]["train"]["documents"],
        "packed_has_no_pad_unk_or_dialogue_specials": all(
            packed_special_counts[token] == 0
            for token in ("<PAD>", "<UNK>", "<USER>", "<ASSISTANT>", "<SYSTEM>")
        ),
    }
    report = {
        "schema_version": "foundation-v11-special-token-audit-v1",
        "selected_vocab": selected_vocab, "per_vocab": per_vocab,
        "packed_train_special_counts": packed_special_counts,
        "checks": checks, "tokenizer_gate": "PASS" if all(checks.values()) else "FAIL",
        "external_ai_api": "OFF", "production_changed": False,
    }
    output = ROOT / "evaluation/foundation-v11-special-token-audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["tokenizer_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
