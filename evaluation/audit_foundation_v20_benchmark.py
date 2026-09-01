from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.synthetic_context_oracle_v31 import oracle_answer_v31
from foundation.synthetic_context_v3 import (
    ANSWER,
    KEY,
    KEYS,
    LEVEL_PAIRS,
    LEVEL_THRESHOLDS,
    PAIR,
    QUERY,
    REMOVED,
    VALUE,
    VALUES,
    ambiguity_audit,
    causal_learnability,
    example_hash,
    key_lookup_example_v3,
    mapping_hash,
    token_label,
)


FINAL_BLIND_EXPECTED = "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_example_hash(examples: list[tuple]) -> str:
    payload = "\n".join(example_hash(row) for row in examples).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def oracle_audit(count_per_level: int, seed: int) -> dict:
    levels = {}
    for level in range(6):
        rng = random.Random(seed + level * 100_000)
        correct = 0
        causal_failures = 0
        ambiguity_count = 0
        hashes: set[str] = set()
        mappings: set[str] = set()
        for _ in range(count_per_level):
            row = key_lookup_example_v3(rng, level, markers=True, split="any")
            correct += oracle_answer_v31(row[0]) == row[1]
            causal_failures += not causal_learnability(row)["pass"]
            ambiguity_count += ambiguity_audit(row)["ambiguous"]
            hashes.add(example_hash(row))
            mappings.add(mapping_hash(row))
        levels[str(level)] = {
            "pairs": LEVEL_PAIRS[level],
            "examples": count_per_level,
            "unique_examples": len(hashes),
            "unique_mapping_combinations": len(mappings),
            "accuracy": correct / count_per_level,
            "causal_failures": causal_failures,
            "ambiguity_count": ambiguity_count,
            "pass": correct == count_per_level and causal_failures == 0 and ambiguity_count == 0,
        }
    return {
        "implementation": "deterministic marker parser; no model and no generator metadata",
        "levels": levels,
        "pass": all(row["pass"] for row in levels.values()),
    }


def test_generator_manifest(seed: int) -> dict:
    sets = {}
    for level in range(6):
        rows_by_split = {}
        train_mapping_hashes: set[str] = set()
        for split_index, split in enumerate(("any", "any")):
            rng = random.Random(seed + level * 10_000 + split_index * 1_000)
            count = 100 if level == 1 and split == "heldout" else 256
            rows = []
            hashes = set()
            while len(rows) < count:
                row = key_lookup_example_v3(rng, level, markers=True, split=split)
                digest = example_hash(row)
                mapping_digest = mapping_hash(row)
                is_test = split_index == 1
                mapping_is_novel = level == 0 or mapping_digest not in train_mapping_hashes
                if (level == 0 or digest not in hashes) and (not is_test or mapping_is_novel):
                    rows.append(row)
                    hashes.add(digest)
                    if not is_test:
                        train_mapping_hashes.add(mapping_digest)
            rows_by_split["train" if split_index == 0 else "test"] = rows
        train, heldout = rows_by_split["train"], rows_by_split["test"]
        train_relations = {tuple(pair) for row in train for pair in row[2]["mapping"]}
        heldout_relations = {tuple(pair) for row in heldout for pair in row[2]["mapping"]}
        sets[str(level)] = {
            "train_examples": len(train),
            "heldout_examples": len(heldout),
            "train_ordered_hash_sha256": ordered_example_hash(train),
            "heldout_ordered_hash_sha256": ordered_example_hash(heldout),
            "exact_sequence_overlap": len(
                {example_hash(row) for row in train} & {example_hash(row) for row in heldout}
            ),
            "exact_mapping_combination_overlap": len(
                {mapping_hash(row) for row in train} & {mapping_hash(row) for row in heldout}
            ),
            "relation_overlap": len(train_relations & heldout_relations),
        }
    return sets


def main() -> int:
    config_path = ROOT / "configs/unipilot-foundation-v20.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = {
        "schema_version": "synthetic-context-benchmark-v3.1-oracle-audit",
        "oracle": oracle_audit(10_000, int(config["seed"]) + 31_000),
        "generator_sets": test_generator_manifest(int(config["seed"]) + 131_000),
        "task_semantics_changed_from_v3": False,
        "final_blind_content_opened": False,
    }
    if not audit["oracle"]["pass"]:
        raise RuntimeError("BENCHMARK INVALID: oracle did not achieve 100%")

    audit_path = ROOT / "evaluation/foundation-v20-oracle-audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_blind_path = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    final_blind_sha = sha256_file(final_blind_path)
    source_paths = [
        "foundation/synthetic_context_v3.py",
        "foundation/synthetic_context_oracle_v31.py",
        "training/validate_foundation_v20_curriculum.py",
        "configs/unipilot-foundation-v20.json",
        "tests/test_foundation_v20_oracle.py",
    ]
    sequence_format = (
        "<KEY_LOOKUP> (<PAIR> <KEY> key <VALUE> value){N} "
        "<QUERY> query-key <ANSWER>; predict value at <ANSWER> position"
    )
    manifest = {
        "schema_version": "synthetic-context-benchmark-v3.1-manifest",
        "name": "Synthetic Context Benchmark v3.1",
        "inherits": "Synthetic Context Benchmark v3",
        "task_semantics_changed": False,
        "benchmark_manifest_path": "evaluation/foundation-v20-synthetic-context-benchmark-v31-manifest.json",
        "vocabulary": {
            "diagnostic_size": 256,
            "formal_model_size": int(config["architecture"]["vocab_size"]),
            "keys": [min(KEYS), max(KEYS)],
            "values": [min(VALUES), max(VALUES)],
            "markers": {
                "PAIR": PAIR, "KEY": KEY, "VALUE": VALUE,
                "QUERY": QUERY, "ANSWER": ANSWER, "REMOVED": REMOVED,
            },
            "foundation_tokenizer_changed": False,
        },
        "sequence_format": sequence_format,
        "answer_mask": {
            "type": "answer_only",
            "supervised_position": "final <ANSWER> input position",
            "all_other_targets": -100,
            "answer_weight": 1.0,
        },
        "levels": {
            str(level): {
                "pairs": pairs,
                "chance_candidate_accuracy": 1 / pairs,
                "chance_value_vocabulary_accuracy": 1 / len(VALUES),
                "threshold": LEVEL_THRESHOLDS.get(level),
                "role": "diagnostic only" if level == 5 else "validity gate",
            }
            for level, pairs in LEVEL_PAIRS.items()
        },
        "train_generator": {
            "implementation": "foundation.synthetic_context_v3.key_lookup_example_v3",
            "mode": "on-the-fly deterministic random generation",
            "mapping_split": "any (canonical v3)",
            "mapping_memorization_prevention": "per-example random mapping",
        },
        "test_generator": {
            "implementation": "foundation.synthetic_context_v3.key_lookup_example_v3",
            "mode": "fixed deterministic evaluation sets",
            "mapping_split": "any (canonical v3)",
            "novel_mapping": "exact multi-relation mapping combination excluded from training history",
            "sets": audit["generator_sets"],
        },
        "seeds": {
            "base_model": int(config["seed"]),
            "oracle_base": int(config["seed"]) + 31_000,
            "generator_manifest_base": int(config["seed"]) + 131_000,
            "standalone_l3_model": int(config["seed"]) + 3_000,
            "standalone_l4_model": int(config["seed"]) + 4_000,
            "sequential_model": int(config["seed"]) + 20_000,
        },
        "oracle": {
            "path": "foundation/synthetic_context_oracle_v31.py",
            "audit_path": audit_path.relative_to(ROOT).as_posix(),
            "result": audit["oracle"],
        },
        "optimizer": config["optimizer"],
        "source_sha256": {
            path: sha256_file(ROOT / path) for path in source_paths
        },
        "final_blind": {
            "path": final_blind_path.relative_to(ROOT).as_posix(),
            "sha256": final_blind_sha,
            "expected_sha256": FINAL_BLIND_EXPECTED,
            "match": final_blind_sha == FINAL_BLIND_EXPECTED,
            "content_opened": False,
        },
    }
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest["manifest_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path = ROOT / "evaluation/foundation-v20-synthetic-context-benchmark-v31-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "oracle_pass": True,
        "audit": audit_path.relative_to(ROOT).as_posix(),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_file_sha256": sha256_file(manifest_path),
        "final_blind_match": final_blind_sha == FINAL_BLIND_EXPECTED,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
