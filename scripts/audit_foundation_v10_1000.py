from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_documents(path: Path, limit: int) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def scan_corpus_debris(patterns: dict[str, re.Pattern]) -> dict:
    split_results = {}
    total_counts = {name: 0 for name in patterns}
    total_documents = 0
    for split in ("train", "validation", "test"):
        path = ROOT / f"data/foundation_v10/documents/{split}.jsonl.gz"
        counts = {name: 0 for name in patterns}
        documents = 0
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                documents += 1
                for name, pattern in patterns.items():
                    if pattern.search(row["text"]):
                        counts[name] += 1
                        total_counts[name] += 1
        total_documents += documents
        split_results[split] = {"documents": documents, "documents_with_signal": counts}
    return {
        "documents": total_documents,
        "documents_with_signal": total_counts,
        "splits": split_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/foundation-v10-sanity/20m/checkpoint-step-500.pt",
    )
    parser.add_argument(
        "--final-blind",
        default="data/foundation_v09/evaluation/final-blind-1000.json",
    )
    parser.add_argument(
        "--output", default="evaluation/foundation-v10-1000-preflight-audit.json"
    )
    args = parser.parse_args()

    manifest_path = ROOT / "data/foundation_v10/packed/vocab-4096/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = FoundationTokenizer.load(ROOT / manifest["tokenizer"])
    train_path = ROOT / manifest["splits"]["train"]["path"]
    tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    documents = load_documents(ROOT / "data/foundation_v10/documents/train.jsonl.gz", 20)

    expected_prefix: list[int] = []
    document_samples = []
    debris_patterns = {
        "replacement_character": re.compile("\ufffd"),
        "html_tag": re.compile(r"</?[A-Za-z][^>]*>"),
        "wiki_template": re.compile(r"\{\{|\}\}"),
        "wiki_link": re.compile(r"\[\[|\]\]"),
        "wiki_table": re.compile(r"\{\||\|\}"),
        "reference_tag": re.compile(r"<ref\b", re.IGNORECASE),
        "heading_markup": re.compile(r"(?m)^={2,}.*={2,}\s*$"),
    }
    anomaly_counts = {name: 0 for name in debris_patterns}
    for index, row in enumerate(documents):
        ids = tokenizer.encode(row["text"], add_bos=True, add_eos=True)
        expected_prefix.extend(ids)
        anomalies = [name for name, pattern in debris_patterns.items() if pattern.search(row["text"])]
        for name in anomalies:
            anomaly_counts[name] += 1
        document_samples.append({
            "index": index + 1,
            "id": row["id"],
            "source_type": row["source_type"],
            "characters": len(row["text"]),
            "tokens": len(ids),
            "begins_with_bos": ids[0] == tokenizer.bos_id,
            "ends_with_eos": ids[-1] == tokenizer.eos_id,
            "roundtrip_exact": tokenizer.decode(ids[1:-1]) == row["text"],
            "debris_signals": anomalies,
            "start_excerpt": row["text"][:120],
            "end_excerpt": row["text"][-120:],
        })

    expected_array = np.asarray(expected_prefix, dtype=np.uint16)
    corpus_debris = scan_corpus_debris(debris_patterns)
    prefix_matches = bool(np.array_equal(tokens[:len(expected_array)], expected_array))
    bos_count = int(np.count_nonzero(tokens == tokenizer.bos_id))
    eos_count = int(np.count_nonzero(tokens == tokenizer.eos_id))
    eos_to_bos = int(np.count_nonzero(
        (tokens[:-1] == tokenizer.eos_id) & (tokens[1:] == tokenizer.bos_id)
    ))
    blocks = (len(tokens) - 1) // 512
    covered_targets = tokens[1:blocks * 512 + 1]
    eos_targets = int(np.count_nonzero(covered_targets == tokenizer.eos_id))

    checkpoint_path = ROOT / args.checkpoint
    checkpoint_manifest_path = checkpoint_path.with_suffix(".manifest.json")
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_lr = 3e-4 * 0.1
    optimizer_groups = payload["optimizer_state"]["param_groups"]
    optimizer_lrs = [float(group["lr"]) for group in optimizer_groups]
    final_blind_path = ROOT / args.final_blind
    final_blind_hash = sha256(final_blind_path)

    special_ids = tokenizer.special_to_id
    checks = {
        "special_token_ids_unique": len(set(special_ids.values())) == len(special_ids),
        "bos_id_is_1": tokenizer.bos_id == 1,
        "eos_id_is_2": tokenizer.eos_id == 2,
        "first_token_is_bos": int(tokens[0]) == tokenizer.bos_id,
        "last_token_is_eos": int(tokens[-1]) == tokenizer.eos_id,
        "bos_count_matches_documents": bos_count == manifest["splits"]["train"]["documents"],
        "eos_count_matches_documents": eos_count == manifest["splits"]["train"]["documents"],
        "all_interdocument_boundaries_are_eos_bos": eos_to_bos == eos_count - 1,
        "first_20_packed_sequences_match_sources": prefix_matches,
        "first_20_roundtrip_exact": all(row["roundtrip_exact"] for row in document_samples),
        "first_20_have_no_markup_or_mojibake": not any(anomaly_counts.values()),
        "eos_is_a_loss_target": eos_targets > 0,
        "eos_target_coverage_above_99_9_percent": eos_targets / eos_count >= .999,
        "checkpoint_size_matches_manifest": (
            checkpoint_path.stat().st_size == checkpoint_manifest["training_checkpoint_bytes"]
        ),
        "checkpoint_config_matches_manifest": payload["config"] == checkpoint_manifest["model_config"],
        "checkpoint_model_step_is_500": payload["step"] == checkpoint_manifest["step"] == 500,
        "optimizer_state_present": bool(payload["optimizer_state"]["state"]),
        "legacy_checkpoint_rng_absent_and_recovery_required": "random_state" not in payload,
        "scheduler_lr_matches_step_500_floor": all(
            abs(value - expected_lr) <= 5e-9 for value in optimizer_lrs
        ),
        "final_blind_hash_matches_without_parsing": (
            final_blind_hash == "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"
        ),
    }
    audit = {
        "schema_version": "foundation-v10-1000-preflight-audit-v1",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_rng_status": "legacy_missing_recover_by_replaying_100_to_500_dropout_draws",
        "optimizer_state_entries": len(payload["optimizer_state"]["state"]),
        "optimizer_group_learning_rates": optimizer_lrs,
        "scheduler": {
            "kind": "stateless_warmup_cosine",
            "restored_by": "global_step",
            "global_step": payload["step"],
            "floor_learning_rate": expected_lr,
        },
        "tokenizer": {
            "path": manifest["tokenizer"],
            "vocab": tokenizer.vocab_size,
            "special_token_ids": special_ids,
            "roundtrip_samples": len(documents),
        },
        "packing": {
            "train_tokens": len(tokens),
            "documents": manifest["splits"]["train"]["documents"],
            "bos_count": bos_count,
            "eos_count": eos_count,
            "eos_to_bos_boundaries": eos_to_bos,
            "eos_loss_targets": eos_targets,
            "eos_loss_target_rate": eos_targets / eos_count,
            "tail_tokens_outside_full_training_blocks": len(tokens) - (blocks * 512 + 1),
            "document_boundary": "BOS document EOS BOS next-document EOS",
            "first_20_packed_prefix_matches": prefix_matches,
            "first_20_debris_counts": anomaly_counts,
            "full_corpus_debris_audit": corpus_debris,
            "samples": document_samples,
        },
        "loss": {
            "cross_entropy_ignore_index": -100,
            "eos_id": tokenizer.eos_id,
            "eos_is_not_ignored": tokenizer.eos_id != -100,
            "eos_targets_observed": eos_targets,
        },
        "final_blind": {
            "path": args.final_blind,
            "sha256": final_blind_hash,
            "content_parsed": False,
            "used_for_training_or_evaluation": False,
        },
        "checks": checks,
        "preflight": "PASS" if all(checks.values()) else "FAIL",
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**{key: value for key, value in audit.items() if key != "packing"},
                      "packing": {key: value for key, value in audit["packing"].items()
                                  if key != "samples"}}, ensure_ascii=False, indent=2))
    return 0 if audit["preflight"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
