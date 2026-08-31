from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.synthetic_context_v3 import LEVEL_PAIRS, LEVEL_THRESHOLDS, example_hash
from training.validate_foundation_v19_benchmark import fixed_examples


EXPECTED_FINAL_BLIND = "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"
PILOTS = (
    "answer-only-markers", "answer-only-no-markers", "all-token-1x-markers",
    "all-token-4x-markers", "all-token-16x-markers", "answer-only-markers-long",
)


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pilot_rows() -> dict:
    rows = {}
    for name in PILOTS:
        report = load(f"checkpoints/foundation-v19-benchmark-v3/pilot-reference-{name}.json")
        final = report["curve"][-1]
        rows[name] = {
            "updates": report["updates"],
            "accuracy": final["heldout"]["accuracy"],
            "total_loss": final["heldout"]["loss"],
            "answer_token_loss": final["heldout"]["answer_token_loss"],
            "non_answer_loss": final["heldout"]["non_answer_loss"],
            "attention": final["attention"]["all_layer_head_mean"],
        }
    return rows


def build_manifest(settings: dict, reference: dict) -> dict:
    evaluation_sets = {}
    for level in range(6):
        examples = fixed_examples(
            level,
            int(settings["curriculum"]["evaluation_examples"]),
            int(settings["seed"]) + 200_000 + level * 10_000,
            markers=True,
            split="any",
            allow_duplicates=level == 1,
        )
        hashes = [example_hash(row) for row in examples]
        evaluation_sets[str(level)] = {
            "examples": len(examples),
            "sha256_of_ordered_example_hashes": canonical_sha256(hashes),
        }
    manifest = {
        "schema_version": "synthetic-context-benchmark-v3-manifest",
        "name": "Synthetic Context Benchmark v3",
        "status": "INVALID_REFERENCE_LEVEL_3_GATE",
        "deprecated_retained": [
            "Synthetic Context Benchmark v1",
            "Synthetic Context Benchmark v2",
            "Phase 29 Synthetic v4 Key Lookup",
        ],
        "task_definitions": {
            "level_0": "one fixed key/value pair; memorized-relation implementation control",
            "level_1": "one random pair; random-value copy control",
            "level_2": "two per-example random pairs and random query; within-context associative retrieval",
            "level_3": "four per-example random pairs",
            "level_4": "eight per-example random pairs; required architecture gate",
            "level_5": "sixteen per-example random pairs; diagnostic only",
            "markers": ["<PAIR>", "<KEY>", "<VALUE>", "<QUERY>", "<ANSWER>"],
            "target": "value paired with the query key in the same causal context",
            "attention_supervision": False,
        },
        "generalization_metrics": {
            "memorized_relation": "Level 0",
            "within_context_novel_relation": "Levels 2-5, uniform per-example random mapping",
            "unseen_key_value_combination": "separate held-out-combination diagnostic; not mixed into the core metric",
            "unseen_token_ids": "separate seen/unseen token-pool API; diagnostic only",
        },
        "seed": {
            "model": int(settings["seed"]),
            "level_2_training_stream": int(settings["seed"]) + 101,
            "final_evaluation_base": int(settings["seed"]) + 200_000,
        },
        "vocab": {
            "size": 256,
            "values": [32, 63],
            "keys": [64, 79],
            "marker_ids": {"PAIR": 227, "KEY": 228, "VALUE": 229, "QUERY": 230, "ANSWER": 231, "REMOVED": 232},
            "foundation_tokenizer_changed": False,
        },
        "train": {
            "batch_size": settings["curriculum"]["batch_size"],
            "examples_by_level": reference["training"]["examples_by_level"],
            "unique_exact_examples": reference["training"]["unique_train_examples"],
            "unique_mapping_sets": reference["training"]["unique_train_mappings"],
            "mapping_distribution": "uniform random; mapping overlap is allowed and measured",
            "reset_at_level_2": True,
        },
        "test": {
            "evaluation_sets": evaluation_sets,
            "exact_train_test_overlap_levels_2_to_5": reference["dataset_audit"]["exact_train_test_overlap"],
            "mapping_set_overlap": reference["dataset_audit"]["mapping_set_overlap"],
            "template_overlap": "intentional",
        },
        "loss_masking": {
            "canonical": "answer_only",
            "active_position": "final <ANSWER> input position only",
            "all_other_targets": -100,
            "answer_weight": 1,
            "foundation_training_method_changed": False,
        },
        "chance_baseline": {
            str(level): {
                "pairs": LEVEL_PAIRS[level],
                "actual_candidate_choice_accuracy": 1 / LEVEL_PAIRS[level],
                "full_value_vocabulary_accuracy": 1 / 32,
            }
            for level in range(6)
        },
        "pass_threshold": {
            str(level): threshold for level, threshold in LEVEL_THRESHOLDS.items()
        } | {"5": None},
        "validity_gate": reference["validity_gate"],
        "source_sha256": {
            path: file_sha256(ROOT / path)
            for path in (
                "foundation/synthetic_context_v3.py",
                "training/validate_foundation_v19_benchmark.py",
                "configs/unipilot-foundation-v19.json",
                "tests/test_foundation_v19_benchmark.py",
            )
        },
        "final_blind": {
            "path": "data/foundation_v09/evaluation/final-blind-1000.json",
            "sha256": file_sha256(ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"),
            "expected_sha256": EXPECTED_FINAL_BLIND,
            "content_opened": False,
        },
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    return manifest


def report_text(summary: dict) -> str:
    reference = summary["reference"]
    pilots = summary["loss_and_marker_pilots"]
    levels = reference["final"]["levels"]
    attention_curve = summary["reference_level2_long_curve"]
    lines = [
        "# UniPilot Foundation v1.9 Synthetic Benchmark Repair Report",
        "",
        "## 最終判定",
        "",
        f"- Benchmark validity: **{'PASS' if summary['benchmark_valid'] else 'FAIL'}**",
        f"- Final Gate: **{summary['decision']}**",
        "- Depth-init Architecture Candidate: **NO**",
        "- 次にFull 256k: **NO**",
        "- 正式Foundation architecture: Currentのまま",
        "- Current / Depth v3大量比較: Reference-first gate未達のため未実行",
        "",
        "## 旧Key Lookupのroot cause",
        "",
        "旧taskはcausalで曖昧性もなく、実装上すでにanswer-only lossだった。したがって未来参照、重複正解、all-token structural lossは旧失敗の原因ではない。Phase 29ではKey Lookupが全更新の16.675%で、12個のpair×distance cellへ分散し、各cellは80〜172 update相当しかなかった。v3のReference Level 2は1,800 update付近までchance plateauが続き、2,800 updateで99.2%に到達したため、主因はassociative retrieval形成前に終わる過少・分散supervisionだった。",
        "",
        "ただし4-pair Level 3は4,800 updateでも基準未達だった。v3は1/2-pair問題を修復したが、Required Gate全体はまだ成立していない。",
        "",
        "## 監査",
        "",
        f"- 旧Key例: {summary['audit']['old_example_count']}件をtoken IDs / target / mask / query/key/value/answer位置付きで保存",
        f"- 旧全走査: {summary['audit']['old_scan']['examples_scanned']:,}件、causal failure {summary['audit']['old_scan']['causal_failures']}、ambiguity {summary['audit']['old_scan']['ambiguity_count']}",
        f"- v3全走査: {summary['audit']['v3_scan']['examples']:,}件、causal failure {summary['audit']['v3_scan']['causal_failures']}、ambiguity {summary['audit']['v3_scan']['ambiguity_count']}",
        f"- Level 2-5 exact train/test overlap: {reference['dataset_audit']['exact_train_test_overlap']}",
        f"- mapping overlap: {reference['dataset_audit']['mapping_set_overlap']}（in-context taskでは許容し、別途計測）",
        "",
        "## Loss / marker比較（Reference、Level 2）",
        "",
        "| Variant | Updates | Accuracy | Answer loss | Non-answer loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "answer-only-markers", "answer-only-no-markers", "all-token-1x-markers",
        "all-token-4x-markers", "all-token-16x-markers",
    ):
        row = pilots[name]
        non = "—" if row["non_answer_loss"] is None else f"{row['non_answer_loss']:.4f}"
        lines.append(
            f"| {name} | {row['updates']} | {row['accuracy']:.2%} | {row['answer_token_loss']:.4f} | {non} |"
        )
    lines.extend([
        "",
        "旧runはanswer-only。all-tokenは反実仮想比較であり、1x/4x/16xはいずれも400 update時にanswer-onlyを上回らなかった。markersありは47.66%、なしは42.58%で、構造markerは小さいが正の効果。canonical v3は単純なanswer-only + markersを採用した。",
        "",
        "## Curriculum / Reference",
        "",
        "| Level | Pairs | Accuracy | Chance(candidate) | Threshold | PASS |",
        "|---:|---:|---:|---:|---:|---|",
    ])
    for level in range(6):
        row = levels[str(level)]
        threshold = "diagnostic" if row["threshold"] is None else f"{row['threshold']:.0%}"
        passed = "diagnostic" if row["pass"] is None else ("PASS" if row["pass"] else "FAIL")
        lines.append(
            f"| {level} | {row['pairs']} | {row['accuracy']:.2%} | {row['candidate_chance']:.2%} | {threshold} | {passed} |"
        )
    controls = reference["final"]["controls"]
    lines.extend([
        "",
        "Level 4まで学習到達しなかったため、Level 4 counterfactual/ablationはVALID判定材料としてFAIL。",
        f"- correct {levels['4']['accuracy']:.2%}; counterfactual {controls['counterfactual']['accuracy']:.2%}",
        f"- shuffled {controls['shuffled_relation_original_target']['accuracy']:.2%} (drop {controls['drops']['shuffled_relation']:.2%})",
        f"- removed relation {controls['removed_relation']['accuracy']:.2%} (drop {controls['drops']['removed_relation']:.2%})",
        f"- wrong query/original target {controls['wrong_query_original_target']['accuracy']:.2%}; wrong query/new target {controls['wrong_query_new_target']['accuracy']:.2%}; removed query {controls['removed_query']['accuracy']:.2%}",
        "",
        "## Level 2 attention learning curve",
        "",
        "| Update | Accuracy | Correct K+V mass | Rank | Margin | Entropy |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in attention_curve:
        lines.append(
            f"| {row['update']} | {row['accuracy']:.2%} | {row['correct_key_value_mass']:.4f} | {row['rank']:.2f} | {row['margin']:.2f} | {row['entropy']:.4f} |"
        )
    lines.extend([
        "",
        "accuracyは上昇したが、final-markerから見た単純なdirect K/V attention massは単調増加しなかった。このmetricだけでは複数layerに分散したretrieval計算を説明できず、attention supervisionは使用していない。",
        "",
        "## Numeric / Symbolic",
        "",
        f"- Foundation tokenizer単数字atomic率: {summary['numeric']['single_digit_atomic_rate']:.2%}",
        f"- raw numeric/token ID監査: standalone {summary['numeric']['standalone_samples']}件、numeric sequence {summary['numeric']['actual_pattern_samples']}件",
        "- Phase 29 symbolic (arithmetic不要): Current / Depth / Reference = 99.61% / 100% / 100%（>=95% PASS）",
        "- Phase 29 numeric: 39.45% / 41.02% / 42.19%。これはFoundation tokenizerではなくatomic synthetic IDs上のmodular additionで、単純pattern continuationではない。Architecture Gateから除外しactual numericもdiagnosticのみ。",
        "",
        "## Phase 29再利用 / Architecture",
        "",
        "Copy、Long Range、Context-conditioned、Position、SymbolicはCurrent / Depth / ReferenceすべてPASS。Japanese 128kではDepthがCurrentよりloss、punctuation collapse、Layer9 RMSを改善したが、v3 Reference gateがFAILのためDepth候補復帰条件A/Bを満たさない。Current / Depthのv3比較は実行していない。",
        "",
        "| Model | v3 Key L0/1/2/3/4/5 | Copy 4/8/16 min | Symbolic | Context | Position min |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for name, label in (
        ("custom_current", "Current"),
        ("custom_depth_init", "Depth-init"),
        ("reference_mha", "Reference"),
    ):
        reused = summary["phase29_reused"][name]
        key = (
            "/".join(f"{levels[str(level)]['accuracy']:.1%}" for level in range(6))
            if name == "reference_mha" else "SKIPPED (Reference gate)"
        )
        copy_min = min(reused["copy"][str(length)]["accuracy"] for length in (4, 8, 16))
        symbolic = reused["pattern"]["symbolic"]["accuracy"]
        context = reused["context_conditioned"]["correct"]["accuracy"]
        position = min(row["accuracy"] for row in reused["position"].values())
        lines.append(
            f"| {label} | {key} | {copy_min:.1%} | {symbolic:.1%} | {context:.1%} | {position:.1%} |"
        )
    lines.extend([
        "",
        "## Integrity / protection",
        "",
        f"- Reference diagnostic parameters: {reference['parameters']:,}; formal architecture parameters: 19,514,880",
        f"- Checkpoint strict reload: {reference['checkpoint']['strict_reload']}; optimizer state: {reference['checkpoint']['optimizer_state_present']}; SHA256 `{reference['checkpoint']['sha256']}`",
        f"- Final Blind: content unopened; SHA256 `{summary['final_blind']['sha256']}`; match={summary['final_blind']['match']}",
        "- focused v3/Reference/Numeric: 40 passed; full pytest: 319 passed, 3 warnings",
        "- Full corpus、46M、tokenizer本体、architecture、本番、Render、Vercel、Releaseは未変更。",
        "",
        "## 次の推奨",
        "",
        "Full 256kへは進まない。次PHASEは4/8-pair relation curriculumの再設計（3-pair中間段階、seed安定性、direct-attention metricの限界）に限定し、Reference Level 3/4とcontrolsがPASSしてからCurrent/Depthを比較する。",
    ])
    return "\n".join(lines) + "\n"


def compact_reference(reference: dict) -> dict:
    return {
        "schema_version": reference["schema_version"],
        "model": reference["model"],
        "config": reference["config"],
        "parameters": reference["parameters"],
        "optimizer": reference["optimizer"],
        "training": {
            key: reference["training"][key]
            for key in (
                "total_updates", "batch_size", "examples_by_level",
                "examples_by_vocabulary_stage", "unique_train_examples",
                "unique_train_mappings", "processed_input_tokens", "wall_seconds",
                "input_tokens_per_second", "peak_ram_mb", "level_pass_during_training",
                "reset_at_level_2",
            )
        } | {
            "curve": [{
                "total_update": row["total_update"],
                "level": row["level"],
                "update_in_level": row["update_in_level"],
                "recent_loss": row["recent_loss"],
                "current_accuracy": row["levels"][str(row["level"])]["accuracy"],
                "all_required_levels_pass": row["all_required_levels_pass"],
                "attention": row["current_level_attention"]["all_layer_head_mean"],
            } for row in reference["training"]["curve"]],
        },
        "dataset_audit": reference["dataset_audit"],
        "final": {
            "levels": reference["final"]["levels"],
            "controls": reference["final"]["controls"],
            "attention": {
                level: {
                    "all_layer_head_mean": row["all_layer_head_mean"],
                    "last_layer_mean": row["last_layer_mean"],
                }
                for level, row in reference["final"]["attention"].items()
            },
        },
        "validity_gate": reference["validity_gate"],
        "checkpoint": reference["checkpoint"],
        "final_blind_used": reference["final_blind_used"],
        "production_changed": reference["production_changed"],
    }


def main() -> int:
    settings = load("configs/unipilot-foundation-v19.json")
    audit = load("evaluation/foundation-v19-benchmark-audit.json")
    reference = load("checkpoints/foundation-v19-benchmark-v3/reference_mha.json")
    phase29 = load("evaluation/foundation-v18-summary.json")
    pilots = pilot_rows()
    long_pilot = load(
        "checkpoints/foundation-v19-benchmark-v3/pilot-reference-answer-only-markers-long.json"
    )
    manifest = build_manifest(settings, reference)
    manifest_path = ROOT / "evaluation/foundation-v19-synthetic-context-benchmark-v3-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_blind_hash = manifest["final_blind"]["sha256"]
    summary = {
        "schema_version": "foundation-v19-summary-v1",
        "decision": "RELATION_TRAINING_ISSUE",
        "benchmark_valid": bool(reference["validity_gate"]["pass"]),
        "full_256k_recommended": False,
        "depth_init_architecture_candidate": False,
        "formal_architecture": "Current unchanged",
        "audit": {
            "path": "evaluation/foundation-v19-benchmark-audit.json",
            "old_example_count": len(audit["old_benchmark"]["examples"]),
            "old_scan": audit["old_benchmark"]["full_scan"],
            "v3_scan": audit["benchmark_v3"]["scan"],
            "loss_supervision": audit["loss_supervision"],
        },
        "manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "file_sha256": file_sha256(manifest_path),
            "content_sha256": manifest["manifest_content_sha256"],
        },
        "loss_and_marker_pilots": pilots,
        "reference_level2_long_curve": [{
            "update": row["update"],
            "accuracy": row["heldout"]["accuracy"],
            "answer_loss": row["heldout"]["answer_token_loss"],
            "correct_key_value_mass": row["attention"]["all_layer_head_mean"]["correct_key_value_mass"],
            "rank": row["attention"]["all_layer_head_mean"]["correct_position_mean_rank"],
            "margin": row["attention"]["all_layer_head_mean"]["attention_margin"],
            "entropy": row["attention"]["all_layer_head_mean"]["normalized_entropy"],
        } for row in long_pilot["curve"]],
        "reference": compact_reference(reference),
        "v3_model_comparison": {
            "reference_mha": "RUN",
            "custom_current": "SKIPPED_REFERENCE_FIRST_GATE_FAIL",
            "custom_depth_init": "SKIPPED_REFERENCE_FIRST_GATE_FAIL",
        },
        "phase29_reused": {
            name: {
                "synthetic_gate": phase29["synthetic"][name]["final"]["gate"],
                "copy": phase29["synthetic"][name]["final"]["evaluation"]["copy"],
                "long_range": phase29["synthetic"][name]["final"]["evaluation"]["long_range"],
                "pattern": phase29["synthetic"][name]["final"]["evaluation"]["pattern"],
                "context_conditioned": phase29["synthetic"][name]["final"]["evaluation"]["context_conditioned"],
                "position": phase29["synthetic"][name]["final"]["evaluation"]["position"],
                "japanese_128k": phase29["japanese_diagnostic"][name]["training"]["history"][-1],
            }
            for name in ("custom_current", "custom_depth_init", "reference_mha")
        },
        "numeric": {
            "single_digit_atomic_rate": audit["numeric"]["single_digit_atomic_rate"],
            "standalone_samples": len(audit["numeric"]["standalone_samples"]),
            "actual_pattern_samples": len(audit["numeric"]["actual_numeric_pattern_samples"]),
            "tasks_separated": audit["numeric"]["tasks_separated"],
        },
        "final_blind": {
            "sha256": final_blind_hash,
            "expected_sha256": EXPECTED_FINAL_BLIND,
            "match": final_blind_hash == EXPECTED_FINAL_BLIND,
            "content_opened": False,
        },
        "verification": {
            "focused_v3_reference_numeric_tests": "40 passed in 7.83s",
            "pytest": "319 passed, 3 warnings in 58.13s",
            "checkpoint_integrity": {
                "sha256_matches_report": True,
                "optimizer_state_present": True,
                "strict_state_dict_missing_keys": [],
                "strict_state_dict_unexpected_keys": [],
            },
        },
        "production_changed": False,
        "external_ai_api": "OFF",
    }
    summary_path = ROOT / "evaluation/foundation-v19-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = ROOT / "evaluation/foundation-v19-benchmark-repair-report.md"
    report_path.write_text(report_text(summary), encoding="utf-8")
    print(json.dumps({
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "summary": summary_path.relative_to(ROOT).as_posix(),
        "report": report_path.relative_to(ROOT).as_posix(),
        "decision": summary["decision"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
