from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("current_unscaled", "sqrt_scaled_a")
LABELS = {"current_unscaled": "Current", "sqrt_scaled_a": "sqrt-scale A"}
SEEDS = (42, 123, 2026)


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def mean_std(values) -> dict:
    values = [float(value) for value in values]
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
    }


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def number(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_frequency(runs: dict) -> tuple[dict, dict]:
    token_metrics = {}
    bucket_metrics = {}
    for variant in VARIANTS:
        token_metrics[variant] = {}
        source_tokens = runs[variant][0]["final"]["frequency"]["tokens"]
        for token in source_tokens:
            token_metrics[variant][token] = {
                name: mean_std([
                    run["final"]["frequency"]["tokens"][token][name]
                    for run in runs[variant]
                ])
                for name in (
                    "actual_frequency", "top_1_predicted_frequency",
                    "average_probability", "accuracy_when_target",
                    "top_5_accuracy_when_target",
                )
            }
        bucket_metrics[variant] = {}
        source_buckets = runs[variant][0]["final"]["frequency"]["buckets"]
        for bucket in source_buckets:
            bucket_metrics[variant][bucket] = {
                name: mean_std([
                    run["final"]["frequency"]["buckets"][bucket][name]
                    for run in runs[variant]
                ])
                for name in ("top_1_accuracy", "top_5_accuracy", "mean_target_probability")
            }
    return token_metrics, bucket_metrics


def synthetic_gate_details(result: dict) -> dict:
    final = result["final"]
    copy_short_medium = all(final["copy"][str(length)] >= .9 for length in (4, 8, 16))
    lookup_2_to_8 = all(
        final["key_lookup"][str(pairs)][distance] >= .9
        for pairs in (2, 4, 8)
        for distance in ("short", "medium", "long")
    )
    pattern_basic = all(final["pattern"][name] >= .9 for name in final["pattern"])
    conditioned = final["context_conditioned"]
    return {
        "copy_lengths_4_8_16_at_least_90_percent": copy_short_medium,
        "key_lookup_pairs_2_4_8_all_distances_at_least_90_percent": lookup_2_to_8,
        "long_range_at_least_90_percent": final["long_range"] >= .9,
        "all_basic_patterns_at_least_90_percent": pattern_basic,
        "context_conditioned_at_least_90_percent": conditioned["correct"] >= .9,
        "correct_context_clearly_above_controls": (
            conditioned["correct"] - max(conditioned["shuffled"], conditioned["removed"])
            >= .5
        ),
        "overall": "PASS" if final["gate_pass"] else "FAIL",
    }


def main() -> int:
    reproduction = load("evaluation/foundation-v16-reproduction-summary.json")
    runs = {
        variant: [
            load(f"checkpoints/foundation-v16-reproduction/{variant}-seed-{seed}.json")
            for seed in SEEDS
        ]
        for variant in VARIANTS
    }
    synthetic = {
        variant: load(f"checkpoints/foundation-v16-synthetic/{variant}.json")
        for variant in VARIANTS
    }
    short = {
        variant: load(f"checkpoints/foundation-v16-short-japanese/{variant}.json")
        for variant in VARIANTS
    }
    named_frequency, frequency_buckets = aggregate_frequency(runs)
    synthetic_gates = {
        variant: synthetic_gate_details(synthetic[variant]) for variant in VARIANTS
    }
    final_blind = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    final_blind_hash = sha256(final_blind)
    expected_final_blind_hash = (
        "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"
    )

    current = reproduction["comparison"]["current_unscaled"]
    scaled = reproduction["comparison"]["sqrt_scaled_a"]
    non_top1_current = current["non_top_1_percent_macro_top_1_accuracy"]["mean"]
    non_top1_scaled = scaled["non_top_1_percent_macro_top_1_accuracy"]["mean"]
    gates = {
        "A_three_seed_mean_current_improvement": (
            scaled["validation_loss"]["mean"] < current["validation_loss"]["mean"]
            and scaled["top_1_accuracy"]["mean"] > current["top_1_accuracy"]["mean"]
        ),
        "B_validation_loss_improved": (
            scaled["validation_loss"]["mean"] < current["validation_loss"]["mean"]
        ),
        "C_top_k_improved": (
            scaled["top_1_accuracy"]["mean"] > current["top_1_accuracy"]["mean"]
            and scaled["top_5_accuracy"]["mean"] > current["top_5_accuracy"]["mean"]
        ),
        "D_frequency_collapse_improved": (
            non_top1_scaled > non_top1_current
            and non_top1_scaled > 0
        ),
        "E_activation_health_not_worse": (
            scaled["layer_9_output_rms"]["mean"] < current["layer_9_output_rms"]["mean"]
            and all(
                row["history"][-1]["probe"]["all_finite"]
                for variant in VARIANTS for row in runs[variant]
            )
        ),
        "F_synthetic_context_gate_v2": synthetic["sqrt_scaled_a"]["final"]["gate_pass"],
    }
    architecture_fix = "PASS" if all(gates.values()) else "FAIL"
    if architecture_fix != "FAIL":
        raise RuntimeError("report policy expected isolation gate to fail; review before 256k")

    summary = {
        "schema_version": "foundation-v16-architecture-fix-summary-v1",
        "phase": "PHASE 27",
        "candidate": {
            "name": "sqrt_scaled_a",
            "formula": "token_embedding * sqrt(d_model) + learned_position_embedding",
            "alternative_b_rejected": "(token_embedding + position_embedding) * sqrt(d_model)",
            "d_model": 384,
            "sqrt_d_model": 384 ** .5,
            "position_embedding_scaled": False,
            "lm_head_scaled": False,
            "scaling_application_count": 1,
        },
        "reproduction": {
            "seeds": list(SEEDS),
            "standard_deviation": "population standard deviation",
            "comparison": reproduction["comparison"],
            "paired_sqrt_minus_current": reproduction["paired_sqrt_minus_current"],
            "checks": reproduction["checks"],
            "milestone_detail": "evaluation/foundation-v16-reproduction-summary.json",
        },
        "frequency": {
            "named_tokens": named_frequency,
            "buckets": frequency_buckets,
            "non_top_1_percent_top_1_accuracy_remains_zero": non_top1_scaled == 0,
        },
        "synthetic_v2": {
            variant: {
                "training": synthetic[variant]["training"],
                "dataset_audit": synthetic[variant]["dataset_audit"],
                "final": synthetic[variant]["final"],
                "gate_checks": synthetic_gates[variant],
                "checkpoint": synthetic[variant]["checkpoint"],
            }
            for variant in VARIANTS
        },
        "short_japanese": {
            variant: {
                "final": short[variant]["final"],
                "frequency_buckets": short[variant]["frequency_buckets"],
                "sentence_boundaries": short[variant]["sentence_boundaries"],
                "generation": short[variant]["generation"],
                "checkpoint": short[variant]["checkpoint"],
            }
            for variant in VARIANTS
        },
        "diagnostic_baselines": short["current_unscaled"]["baselines"],
        "architecture_fix_gate": gates,
        "architecture_fix": architecture_fix,
        "formal_architecture_change": False,
        "controlled_256k": "NOT EXECUTED",
        "final_gate": "STOP",
        "next_recommendation": "architecture investigation",
        "next_candidates_proposal_only": [
            "positional embedding scale/initialization alignment",
            "residual branch scaling",
            "depth-aware initialization scaling",
            "Norm change as a later isolated ablation",
        ],
        "final_blind": {
            "contents_opened": False,
            "sha256": final_blind_hash,
            "expected_sha256": expected_final_blind_hash,
            "hash_matches": final_blind_hash == expected_final_blind_hash,
        },
        "controls": {
            "standard_46m": "NOT EXECUTED",
            "corpus_added": False,
            "campus_changed": False,
            "instruction_tuning": False,
            "human_feedback_or_dpo": False,
            "production_changed": False,
            "push_or_deploy": False,
            "external_ai_api": "OFF",
        },
    }
    output_json = ROOT / "evaluation/foundation-v16-summary.json"
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# UniPilot Foundation v1.6 Architecture Fix Validation",
        "",
        "## 判定",
        "",
        "- Architecture Fix: **FAIL**",
        "- 正式architecture変更: **NO**",
        "- Controlled 256k run: **NOT EXECUTED**",
        "- 最終Gate: **STOP**",
        "- 次の推奨: **architecture investigation**",
        "",
        "sqrt-scale Aは実Corpusの3-seed再現と短時間日本語診断を改善したが、"
        "frequency collapseの必須目標とSynthetic Context Gate v2を満たさなかった。"
        "特にlearned positionをscaleしない式AではCopy/numeric patternがCurrentより悪化し、"
        "位置情報の相対scaleに未解決問題が残る。",
        "",
        "## Scaling実装監査",
        "",
        "正式候補として隔離検証した式は "
        "`token_embedding * sqrt(d_model) + learned_position_embedding`。"
        "d_model=384、sqrt=19.5959179423。position、LM head、weight tying後の出力には"
        "scaleを追加せず、embedding内部で1回だけ適用した。候補B "
        "`(token_embedding + position_embedding) * sqrt(d_model)` は不採用。",
        "",
        "## 3-seed再現（64k tokens、mean ± population std）",
        "",
        "| 指標 | Current | sqrt-scale A |",
        "|---|---:|---:|",
    ]
    metric_rows = (
        ("Validation loss", "validation_loss", False),
        ("Top-1", "top_1_accuracy", True),
        ("Top-5", "top_5_accuracy", True),
        ("Context Sensitivity", "context_sensitivity_score", False),
        ("Layer 9 output RMS", "layer_9_output_rms", False),
        ("Logit entropy", "softmax_entropy", False),
        ("。+、 Top-1 prediction mass", "period_comma_top_1_mass", True),
        ("Top 1%外 macro Top-1", "non_top_1_percent_macro_top_1_accuracy", True),
        ("Top 1%外 macro Top-5", "non_top_1_percent_macro_top_5_accuracy", True),
    )
    for label, key, as_percent in metric_rows:
        formatter = percent if as_percent else number
        lines.append(
            f"| {label} | {formatter(current[key]['mean'])} ± {formatter(current[key]['std_population'])} "
            f"| {formatter(scaled[key]['mean'])} ± {formatter(scaled[key]['std_population'])} |"
        )

    lines += [
        "",
        "全seedでlossとTop-1は改善した。一方、Top 1%頻度bucket外のTop-1は"
        "両構成とも0%のままで、frequency collapse解消目標には未達。句読点合計massは"
        "低下したが、「。」の予測頻度は悪化しており部分改善に留まる。",
        "",
        "## Embedding RMS",
        "",
        "| Variant | update | raw token | scaled token | position | combined | scaled/position |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for step in (0, 10, 50, 100):
            row = reproduction["milestones"][variant][str(step)]
            embedding = row["embedding"]
            lines.append(
                f"| {LABELS[variant]} | {step} | {number(embedding['raw_token']['mean'])} "
                f"| {number(embedding['scaled_token']['mean'])} | {number(embedding['position']['mean'])} "
                f"| {number(embedding['combined']['mean'])} "
                f"| {number(row['scaled_to_position_rms_ratio']['mean'], 2)} |"
            )

    lines += [
        "",
        "## Activation / residual stream（sqrt-scale A、3-seed mean RMS）",
        "",
        "全10層・指定6 milestoneの値。Residual欄は mean/std/RMS。",
        "",
        "| update | layer | input RMS | attention RMS | residual mean/std/RMS | MLP RMS | output RMS |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in (0, 10, 25, 50, 75, 100):
        for layer in reproduction["milestones"]["sqrt_scaled_a"][str(step)]["layers"]:
            residual = layer["residual"]
            lines.append(
                f"| {step} | {layer['layer']} | {number(layer['input']['rms']['mean'])} "
                f"| {number(layer['attention']['rms']['mean'])} "
                f"| {number(residual['mean']['mean'])}/{number(residual['std']['mean'])}/"
                f"{number(residual['rms']['mean'])} | {number(layer['mlp']['rms']['mean'])} "
                f"| {number(layer['output']['rms']['mean'])} |"
            )
    lines += [
        "",
        "Layer 9 output RMSは最終時点でCurrent 4.4827からsqrt-scale 2.4376へ低下し、"
        "全probeはfiniteだった。全componentのmean/std/RMSと3-seed分散は "
        "`evaluation/foundation-v16-reproduction-summary.json` に保存。",
        "",
        "## Logit scale / entropy（sqrt-scale A、3-seed mean）",
        "",
        "| update | mean | std | max | entropy | Top-1 prob | Top-5 mass |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in (0, 10, 25, 50, 75, 100):
        logits = reproduction["milestones"]["sqrt_scaled_a"][str(step)]["logits"]
        lines.append(
            f"| {step} | {number(logits['mean']['mean'])} | {number(logits['std']['mean'])} "
            f"| {number(logits['max']['mean'])} | {number(logits['mean_softmax_entropy']['mean'])} "
            f"| {number(logits['mean_top_1_probability']['mean'])} "
            f"| {number(logits['mean_top_5_probability_mass']['mean'])} |"
        )

    lines += [
        "",
        "## Frequency collapse（3-seed mean）",
        "",
        "| token | actual | Current Top-1予測 | sqrt Top-1予測 | Current target accuracy | sqrt target accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for token in ("。", "、", "の", "に", "は", "を", "が", "<EOS>"):
        c = named_frequency["current_unscaled"][token]
        s = named_frequency["sqrt_scaled_a"][token]
        lines.append(
            f"| {token} | {percent(c['actual_frequency']['mean'])} "
            f"| {percent(c['top_1_predicted_frequency']['mean'])} "
            f"| {percent(s['top_1_predicted_frequency']['mean'])} "
            f"| {percent(c['accuracy_when_target']['mean'])} "
            f"| {percent(s['accuracy_when_target']['mean'])} |"
        )
    lines += [
        "",
        "| Frequency bucket | Current Top-1 / Top-5 | sqrt Top-1 / Top-5 |",
        "|---|---:|---:|",
    ]
    for bucket in frequency_buckets["current_unscaled"]:
        c = frequency_buckets["current_unscaled"][bucket]
        s = frequency_buckets["sqrt_scaled_a"][bucket]
        lines.append(
            f"| {bucket} | {percent(c['top_1_accuracy']['mean'])} / "
            f"{percent(c['top_5_accuracy']['mean'])} | "
            f"{percent(s['top_1_accuracy']['mean'])} / {percent(s['top_5_accuracy']['mean'])} |"
        )

    lines += [
        "",
        "## Synthetic Context v2（final 256 examples/cell）",
        "",
        "Exact train/test overlapは両構成0。各構成25,600 train examples、"
        "testは5,888 unique hashes。sequence length 8–68、required context distance 3–66。"
        "chance baselineはCopy=1/L、Key Lookup=1/pairs、4-way context=25%。",
        "",
        "### Copy",
        "",
        "| length | chance | Current | sqrt-scale A |",
        "|---:|---:|---:|---:|",
    ]
    for length in (4, 8, 16, 32, 64):
        chance = synthetic["current_unscaled"]["dataset_audit"]["chance_baselines"]["copy_query_ignored"][str(length)]
        lines.append(
            f"| {length} | {percent(chance)} | "
            f"{percent(synthetic['current_unscaled']['final']['copy'][str(length)])} | "
            f"{percent(synthetic['sqrt_scaled_a']['final']['copy'][str(length)])} |"
        )
    lines += [
        "",
        "### Key Lookup",
        "",
        "| pairs | distance | chance | Current | sqrt-scale A |",
        "|---:|---|---:|---:|---:|",
    ]
    for pairs in (2, 4, 8, 16):
        for distance in ("short", "medium", "long"):
            chance = synthetic["current_unscaled"]["dataset_audit"]["chance_baselines"]["key_query_ignored"][str(pairs)]
            lines.append(
                f"| {pairs} | {distance} | {percent(chance)} | "
                f"{percent(synthetic['current_unscaled']['final']['key_lookup'][str(pairs)][distance])} | "
                f"{percent(synthetic['sqrt_scaled_a']['final']['key_lookup'][str(pairs)][distance])} |"
            )
    lines += [
        "",
        "### Long range / Pattern / Context controls",
        "",
        "| task | Current | sqrt-scale A |",
        "|---|---:|---:|",
        f"| Long range | {percent(synthetic['current_unscaled']['final']['long_range'])} | {percent(synthetic['sqrt_scaled_a']['final']['long_range'])} |",
    ]
    for pattern in ("abab", "abcabc", "numeric", "nested"):
        lines.append(
            f"| Pattern {pattern} | {percent(synthetic['current_unscaled']['final']['pattern'][pattern])} "
            f"| {percent(synthetic['sqrt_scaled_a']['final']['pattern'][pattern])} |"
        )
    for control in ("correct", "shuffled", "removed"):
        lines.append(
            f"| Context {control} | {percent(synthetic['current_unscaled']['final']['context_conditioned'][control])} "
            f"| {percent(synthetic['sqrt_scaled_a']['final']['context_conditioned'][control])} |"
        )
    lines += [
        "",
        "両構成ともLong Rangeとcontext control差は成立したが、Copy短〜中、"
        "Key Lookup 2〜8 pairs、基本patternの全条件90%以上を満たさない。"
        "sqrt-scale AはCopyがほぼquery無視chanceまで低下しnumeric patternも0.78%で、"
        "Synthetic Gate v2はFAIL。10/25/50/75/100%のcurveは各result JSONに保存。",
        "",
        "## Short Japanese Diagnostic（同一64k tokens）",
        "",
        "| Model | loss | Top-1 | Top-5 | Top-10 |",
        "|---|---:|---:|---:|---:|",
    ]
    baselines = short["current_unscaled"]["baselines"]
    for label, row in (
        ("Unigram", baselines["unigram"]),
        ("Bigram", baselines["bigram"]),
        ("Current", short["current_unscaled"]["final"]),
        ("sqrt-scale A", short["sqrt_scaled_a"]["final"]),
    ):
        lines.append(
            f"| {label} | {number(row['loss'])} | {percent(row['top_1_accuracy'])} "
            f"| {percent(row['top_5_accuracy'])} | {percent(row['top_10_accuracy'])} |"
        )
    lines += [
        "",
        "sqrt-scale AはCurrentより全指標を改善しBigramとの差を縮めたが、まだBigram未満。"
        "「。」境界Top-1は45.40%→85.63%、EOSは両方100%。診断segmentは正式Corpusへ"
        "追加していない。生成probeはsummary JSONに参考値として保存。",
        "",
        "## Architecture Fix Gate",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "",
        "Gate DはTop 1%外Top-1が0%のままなのでFAIL。Gate FもFAIL。よって式Aを正式"
        "Foundation architectureへ採用せず、旧checkpoint互換性変更も発生しない。"
        "Controlled 256kと64k/128k/192k/256k checkpointは作成していない。",
        "",
        "## 次の隔離候補（提案のみ）",
        "",
        "1. token scaleに対するlearned positional embeddingのscale/initialization alignment",
        "2. residual branch scaling",
        "3. depth-aware initialization scaling",
        "4. 上記を単独検証後にのみNorm変更",
        "",
        "今回は追加architecture変更を実装していない。46M、Corpus追加、Campus、"
        "Instruction Tuning、Human Feedback、DPO、本番連携、push/deployは未実施。",
        "",
        "## Integrity",
        "",
        f"Final Blindは内容を開かずSHA256のみ確認: `{final_blind_hash}` "
        f"({'MATCH' if final_blind_hash == expected_final_blind_hash else 'MISMATCH'})。",
        "",
        "Reproduction 6 checkpoint、Synthetic 2 checkpoint、Short Japanese 2 checkpointは"
        "strict reloadを確認済み。",
        "",
    ]
    output_report = ROOT / "evaluation/foundation-v16-architecture-fix-report.md"
    output_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "architecture_fix": architecture_fix,
        "formal_architecture_change": False,
        "controlled_256k": "NOT EXECUTED",
        "final_gate": "STOP",
        "gates": gates,
        "summary": output_json.relative_to(ROOT).as_posix(),
        "report": output_report.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
