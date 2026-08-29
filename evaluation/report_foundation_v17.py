from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
THREE_SEED = ("current_unscaled", "sqrt_scaled_a", "depth_scaled_residual_init")
MATRIX = (
    "current_unscaled",
    "sqrt_scaled_a",
    "balanced_position_sqrt",
    "depth_scaled_residual_init",
)
LABEL = {
    "current_unscaled": "Current",
    "sqrt_scaled_a": "sqrt A",
    "balanced_position_sqrt": "sqrt token+position",
    "depth_scaled_residual_init": "depth-scaled residual init",
}


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def aggregate(values) -> dict:
    values = [float(value) for value in values]
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    settings = load("configs/unipilot-foundation-v17.json")
    variants = {row["name"]: row for row in settings["variants"]}
    phase27_runs = {
        variant: [
            load(f"checkpoints/foundation-v16-reproduction/{variant}-seed-{seed}.json")
            for seed in settings["seeds"]
        ]
        for variant in ("current_unscaled", "sqrt_scaled_a")
    }
    phase28_runs = {
        "depth_scaled_residual_init": [
            load(
                "checkpoints/foundation-v17-reproduction/"
                f"depth_scaled_residual_init-seed-{seed}.json"
            )
            for seed in settings["seeds"]
        ],
        "balanced_position_sqrt": [load(
            "checkpoints/foundation-v17-reproduction/"
            "balanced_position_sqrt-seed-42.json"
        )],
    }
    baseline_diagnostics = load("evaluation/foundation-v17-baseline-diagnostics.json")
    synthetic = {
        variant: load(f"checkpoints/foundation-v17-synthetic/{variant}-seed-42.json")
        for variant in MATRIX
    }
    short = {
        "current_unscaled": load(
            "checkpoints/foundation-v16-short-japanese/current_unscaled.json"
        ),
        "sqrt_scaled_a": load(
            "checkpoints/foundation-v16-short-japanese/sqrt_scaled_a.json"
        ),
        "depth_scaled_residual_init": load(
            "checkpoints/foundation-v17-short-japanese/depth_scaled_residual_init.json"
        ),
    }
    real_parameters = {
        "current_unscaled": phase27_runs["current_unscaled"][0]["parameters"],
        "sqrt_scaled_a": phase27_runs["sqrt_scaled_a"][0]["parameters"],
        "balanced_position_sqrt": phase28_runs["balanced_position_sqrt"][0]["parameters"],
        "depth_scaled_residual_init": phase28_runs["depth_scaled_residual_init"][0]["parameters"],
    }

    def real_rows(variant: str) -> list[dict]:
        if variant in phase27_runs:
            diagnostics = baseline_diagnostics["results"][variant]
            rows = []
            for run, diagnostic in zip(phase27_runs[variant], diagnostics):
                final = run["final"]
                non_top1 = [
                    bucket["top_1_accuracy"]
                    for name, bucket in final["frequency"]["buckets"].items()
                    if name != "top_1_percent"
                ]
                rows.append({
                    "seed": run["seed"],
                    "validation": final["validation"],
                    "context": final["context_sensitivity"]["context_sensitivity_score"],
                    "layer9": final["probe"]["layers"][9]["output"]["rms"],
                    "punctuation_mass": final["frequency"]["period_comma_top1_mass"],
                    "non_top1_macro_top1": statistics.fmean(non_top1),
                    "tokens_per_second": diagnostic["validation"]["tokens_per_second"],
                    "peak_ram_mb": run["peak_ram_mb"],
                    "position_gradient_rms": diagnostic["position_gradient"]["rms"],
                    "position_delta_rms": diagnostic["position_parameter_delta"]["rms"],
                    "probe": diagnostic["probe"],
                    "hidden_similarity": diagnostic["hidden_token_similarity"],
                })
            return rows
        rows = []
        for run in phase28_runs[variant]:
            final = run["final"]
            non_top1 = [
                bucket["top_1_accuracy"]
                for name, bucket in final["frequency"]["buckets"].items()
                if name != "top_1_percent"
            ]
            rows.append({
                "seed": run["seed"],
                "validation": final["validation"],
                "context": final["context_sensitivity"]["context_sensitivity_score"],
                "layer9": final["probe"]["layers"][9]["post_mlp_residual"]["rms"],
                "punctuation_mass": final["frequency"]["period_comma_top1_mass"],
                "non_top1_macro_top1": statistics.fmean(non_top1),
                "tokens_per_second": final["validation"]["tokens_per_second"],
                "peak_ram_mb": run["peak_ram_mb"],
                "position_gradient_rms": final["position_learning"]["gradient"]["rms"],
                "position_delta_rms": final["position_learning"]["parameter_delta"]["rms"],
                "probe": final["probe"],
                "hidden_similarity": final["hidden_token_similarity"],
            })
        return rows

    normalized_real = {variant: real_rows(variant) for variant in MATRIX}
    real_summary = {}
    for variant, rows in normalized_real.items():
        paths = {
            "validation_loss": lambda row: row["validation"]["loss"],
            "top_1_accuracy": lambda row: row["validation"]["top_1_accuracy"],
            "top_5_accuracy": lambda row: row["validation"]["top_5_accuracy"],
            "top_10_accuracy": lambda row: row["validation"]["top_10_accuracy"],
            "context_sensitivity": lambda row: row["context"],
            "layer_9_rms": lambda row: row["layer9"],
            "punctuation_mass": lambda row: row["punctuation_mass"],
            "non_top_1_percent_macro_top_1": lambda row: row["non_top1_macro_top1"],
            "tokens_per_second": lambda row: row["tokens_per_second"],
            "peak_ram_mb": lambda row: row["peak_ram_mb"],
            "position_gradient_rms": lambda row: row["position_gradient_rms"],
            "position_delta_rms": lambda row: row["position_delta_rms"],
        }
        real_summary[variant] = {
            name: aggregate([getter(row) for row in rows])
            for name, getter in paths.items()
        }

    position_milestones = {}
    residual_milestones = {}
    for variant in MATRIX:
        position_milestones[variant] = {}
        residual_milestones[variant] = {}
        source_runs = phase27_runs.get(variant, phase28_runs.get(variant))
        for update in (0, 10, 25, 50, 100):
            history_rows = [
                next(row for row in run["history"] if row["update"] == update)
                for run in source_runs
            ]
            if variant in phase27_runs:
                token_key = "scaled_token"
                position_key = "position"
                combined_key = "combined"
                tensor_elements = 128 * settings["architecture"]["embedding_dim"]
                def phase27_component(name: str) -> dict:
                    return {
                        statistic: aggregate([
                            (
                                row["probe"]["embedding"][name]["rms"]
                                * math.sqrt(tensor_elements)
                                if statistic == "norm"
                                else row["probe"]["embedding"][name][statistic]
                            )
                            for row in history_rows
                        ])
                        for statistic in ("mean", "std", "rms", "norm")
                    }
                position_milestones[variant][str(update)] = {
                    "effective_token": phase27_component(token_key),
                    "effective_position": phase27_component(position_key),
                    "combined": phase27_component(combined_key),
                    "ratio": aggregate([
                        row["probe"]["embedding"]["scaled_to_position_rms_ratio"]
                        for row in history_rows
                    ]),
                }
            else:
                def phase28_component(name: str) -> dict:
                    return {
                        statistic: aggregate([
                            row["probe"]["embedding"][name][statistic]
                            for row in history_rows
                        ])
                        for statistic in ("mean", "std", "rms", "norm")
                    }
                position_milestones[variant][str(update)] = {
                    "effective_token": phase28_component("effective_token"),
                    "effective_position": phase28_component("effective_position"),
                    "combined": phase28_component("combined"),
                    "ratio": aggregate([
                        row["probe"]["embedding"][
                            "effective_token_to_position_rms_ratio"
                        ] for row in history_rows
                    ]),
                }
            layers = []
            for layer in range(10):
                if variant in phase27_runs:
                    inputs = [row["probe"]["layers"][layer]["input"]["rms"] for row in history_rows]
                    attention = [row["probe"]["layers"][layer]["attention"]["rms"] for row in history_rows]
                    residual = [row["probe"]["layers"][layer]["residual"]["rms"] for row in history_rows]
                    mlp = [row["probe"]["layers"][layer]["mlp"]["rms"] for row in history_rows]
                    output = [row["probe"]["layers"][layer]["output"]["rms"] for row in history_rows]
                else:
                    inputs = [row["probe"]["layers"][layer]["pre_attention"]["rms"] for row in history_rows]
                    attention = [row["probe"]["layers"][layer]["attention_output"]["rms"] for row in history_rows]
                    residual = [row["probe"]["layers"][layer]["post_attention_residual"]["rms"] for row in history_rows]
                    mlp = [row["probe"]["layers"][layer]["mlp_output"]["rms"] for row in history_rows]
                    output = [row["probe"]["layers"][layer]["post_mlp_residual"]["rms"] for row in history_rows]
                layers.append({
                    "layer": layer,
                    "pre_attention_rms": aggregate(inputs),
                    "attention_output_rms": aggregate(attention),
                    "post_attention_residual_rms": aggregate(residual),
                    "mlp_output_rms": aggregate(mlp),
                    "post_mlp_residual_rms": aggregate(output),
                    "attention_to_residual_ratio": aggregate([
                        branch / max(stream, 1e-12)
                        for branch, stream in zip(attention, inputs)
                    ]),
                    "mlp_to_residual_ratio": aggregate([
                        branch / max(stream, 1e-12)
                        for branch, stream in zip(mlp, residual)
                    ]),
                })
            residual_milestones[variant][str(update)] = layers

    norm_audit = {
        variant: {
            "final_norm": [row["probe"]["final_norm"] for row in rows],
            "layers": [row["probe"]["norms"] for row in rows],
        }
        for variant, rows in normalized_real.items()
    }
    similarity = {}
    for variant, rows in normalized_real.items():
        similarity[variant] = {
            "hidden_norm": aggregate([
                row["hidden_similarity"]["hidden_norm"]["mean"] for row in rows
            ]),
            "correct_token_cosine": aggregate([
                row["hidden_similarity"]["correct_token_cosine"]["mean"] for row in rows
            ]),
            "named_token_cosine": {
                token: aggregate([
                    row["hidden_similarity"]["named_token_cosine"][token]["mean"]
                    for row in rows
                ])
                for token in ("。", "、", "の", "に", "は", "を", "が")
            },
            "named_token_logit": {
                token: aggregate([
                    row["hidden_similarity"]["named_token_logit"][token]["mean"]
                    for row in rows
                ])
                for token in ("。", "、", "の", "に", "は", "を", "が")
            },
        }

    synthetic_summary = {}
    for variant, result in synthetic.items():
        base = result["final"]["base"]
        position = result["final"]["position"]
        synthetic_summary[variant] = {
            "copy": base["copy"],
            "key_lookup": base["key_lookup"],
            "long_range": base["long_range"],
            "pattern": base["pattern"],
            "context_conditioned": base["context_conditioned"],
            "position": position,
            "gate": result["final"]["phase28_gate"],
            "copy_failure_analysis": result["copy_failure_analysis"],
            "numeric_failure_analysis": result["numeric_failure_analysis"],
            "input_tokens_per_second": result["training"]["input_tokens_per_second"],
            "peak_ram_mb": result["training"]["peak_ram_mb"],
            "checkpoint": result["checkpoint"],
            "dataset_audit": result["dataset_audit"],
        }

    short_summary = {
        variant: {
            "validation": result["final"],
            "punctuation_mass": (
                result["frequency"]["period_comma_top1_mass"]
                if "frequency" in result
                else sum(
                    result["selected_token_frequency"][token]["top_1_predicted_frequency"]
                    for token in ("。", "、")
                )
            ),
            "non_top1_any_top1": (
                result["frequency"]["non_top_1_percent_any_top_1_accuracy"]
                if "frequency" in result
                else any(
                    row["top_1_accuracy"] > 0
                    for name, row in result["frequency_buckets"].items()
                    if name != "top_1_percent"
                )
            ),
        }
        for variant, result in short.items()
    }

    depth = real_summary["depth_scaled_residual_init"]
    current = real_summary["current_unscaled"]
    architecture_checks = {
        "three_seed_validation_loss_better_than_current": (
            depth["validation_loss"]["mean"] < current["validation_loss"]["mean"]
        ),
        "three_seed_top_1_better_than_current": (
            depth["top_1_accuracy"]["mean"] > current["top_1_accuracy"]["mean"]
        ),
        "three_seed_top_5_better_than_current": (
            depth["top_5_accuracy"]["mean"] > current["top_5_accuracy"]["mean"]
        ),
        "context_better_than_current": (
            depth["context_sensitivity"]["mean"] > current["context_sensitivity"]["mean"]
        ),
        "layer_9_rms_better_than_current": (
            depth["layer_9_rms"]["mean"] < current["layer_9_rms"]["mean"]
        ),
        "full_corpus_non_top1_accuracy_above_zero": (
            depth["non_top_1_percent_macro_top_1"]["mean"] > 0
        ),
        "full_corpus_punctuation_mass_below_current": (
            depth["punctuation_mass"]["mean"] < current["punctuation_mass"]["mean"]
        ),
        "clean_japanese_loss_better_than_current": (
            short_summary["depth_scaled_residual_init"]["validation"]["loss"]
            < short_summary["current_unscaled"]["validation"]["loss"]
        ),
        "clean_japanese_frequency_gate": (
            short_summary["depth_scaled_residual_init"]["non_top1_any_top1"]
            and short_summary["depth_scaled_residual_init"]["punctuation_mass"]
            < short_summary["current_unscaled"]["punctuation_mass"]
        ),
        "synthetic_gate": synthetic_summary["depth_scaled_residual_init"]["gate"]["pass"],
    }
    architecture_gate = "PASS" if all(architecture_checks.values()) else "FAIL"
    decision = "MULTI_COMPONENT_FIX_REQUIRED"
    final_blind = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    final_blind_sha = digest(final_blind)
    expected_sha = "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"

    matrix = {}
    for variant in MATRIX:
        real = real_summary[variant]
        synth = synthetic_summary[variant]
        matrix[variant] = {
            "configuration": variants[variant],
            "parameters": real_parameters[variant],
            "real_corpus": real,
            "synthetic": {
                "copy": synth["copy"],
                "key_lookup": synth["key_lookup"],
                "pattern": synth["pattern"],
                "position": synth["position"],
                "context_conditioned": synth["context_conditioned"],
                "long_range": synth["long_range"],
                "gate": synth["gate"],
            },
            "synthetic_input_tokens_per_second": synth["input_tokens_per_second"],
            "synthetic_peak_ram_mb": synth["peak_ram_mb"],
        }

    summary = {
        "schema_version": "foundation-v17-residual-position-initialization-summary-v1",
        "phase": "PHASE 28",
        "architecture_matrix": matrix,
        "three_seed_real_corpus": {
            variant: real_summary[variant] for variant in THREE_SEED
        },
        "position_milestones": position_milestones,
        "position_gradient": {
            variant: {
                "gradient_rms": real_summary[variant]["position_gradient_rms"],
                "parameter_delta_rms": real_summary[variant]["position_delta_rms"],
            }
            for variant in MATRIX
        },
        "residual_milestones": residual_milestones,
        "norm_audit": norm_audit,
        "hidden_to_token_similarity": similarity,
        "synthetic": synthetic_summary,
        "short_japanese": short_summary,
        "final_norm": {
            "status": "PRESENT",
            "order": "Embedding -> Blocks -> Final LayerNorm -> LM Head",
            "ablation_executed": False,
        },
        "initialization": {
            "current": "all Linear/Embedding weights std 0.02",
            "depth_scaled_candidate": (
                "attention output and MLP output projection only: "
                "0.02 / sqrt(2 * 10) = 0.004472135955; QKV and MLP input remain 0.02"
            ),
            "runtime_residual_scaling_used": False,
        },
        "architecture_checks": architecture_checks,
        "architecture_decision": decision,
        "architecture_gate": architecture_gate,
        "formal_architecture_change": False,
        "next_architecture_to_adopt": (
            "NONE. Keep current formal Foundation architecture; retain depth-scaled "
            "residual projection initialization as an experimental partial fix only."
        ),
        "combined_ablation_executed": False,
        "combined_ablation_reason": (
            "Neither isolated position fix nor residual-init candidate passed all single-component gates."
        ),
        "full_256k": "NOT EXECUTED",
        "proceed_to_full_256k": "NO",
        "final_blind": {
            "contents_opened": False,
            "sha256": final_blind_sha,
            "hash_matches": final_blind_sha == expected_sha,
        },
        "controls": {
            "production_changed": False,
            "campus_changed": False,
            "tokenizer_changed": False,
            "corpus_added": False,
            "standard_46m": False,
            "push_or_deploy": False,
            "external_ai_api": "OFF",
        },
    }
    summary_path = ROOT / "evaluation/foundation-v17-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# UniPilot Foundation v1.7 Residual / Position / Initialization Isolation",
        "",
        "## 最終判定",
        "",
        f"- Architecture decision: **{decision}**",
        f"- Architecture Gate: **{architecture_gate}**",
        "- 正式architecture変更: **NO**",
        "- Full 256kへ進めるか: **NO**",
        "- Final Norm: **PRESENT**",
        "- Combined ablation: **NOT EXECUTED**（単独候補が総合PASSしなかったため）",
        "",
        "depth-scaled residual projection initは実Corpus、activation、Copy、Positionを改善したが、"
        "Key Lookup、numeric pattern、Full Corpus frequency gateを解決しない。position両scaleは"
        "token/position ratioを戻したもののresidual全体を過大化しSyntheticを悪化させた。",
        "",
        "## 3-seed real corpus（64k、mean ± population std）",
        "",
        "| Config | loss | Top-1 | Top-5 | Top-10 | Context | Layer9 RMS | punctuation mass | non-top1 Top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in THREE_SEED:
        row = real_summary[variant]
        lines.append(
            f"| {LABEL[variant]} | {fmt(row['validation_loss']['mean'])} ± {fmt(row['validation_loss']['std_population'])} "
            f"| {pct(row['top_1_accuracy']['mean'])} ± {pct(row['top_1_accuracy']['std_population'])} "
            f"| {pct(row['top_5_accuracy']['mean'])} ± {pct(row['top_5_accuracy']['std_population'])} "
            f"| {pct(row['top_10_accuracy']['mean'])} ± {pct(row['top_10_accuracy']['std_population'])} "
            f"| {fmt(row['context_sensitivity']['mean'])} ± {fmt(row['context_sensitivity']['std_population'])} "
            f"| {fmt(row['layer_9_rms']['mean'])} ± {fmt(row['layer_9_rms']['std_population'])} "
            f"| {pct(row['punctuation_mass']['mean'])} "
            f"| {pct(row['non_top_1_percent_macro_top_1']['mean'])} |"
        )

    lines += [
        "",
        "## Position scale / gradient",
        "",
        "| Config | update | token mean/std/RMS/norm | position mean/std/RMS/norm | combined RMS | ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in MATRIX:
        for update in (0, 10, 50, 100):
            row = position_milestones[variant][str(update)]
            token = row["effective_token"]
            position = row["effective_position"]
            lines.append(
                f"| {LABEL[variant]} | {update} "
                f"| {fmt(token['mean']['mean'])}/{fmt(token['std']['mean'])}/"
                f"{fmt(token['rms']['mean'])}/{fmt(token['norm']['mean'], 2)} "
                f"| {fmt(position['mean']['mean'])}/{fmt(position['std']['mean'])}/"
                f"{fmt(position['rms']['mean'])}/{fmt(position['norm']['mean'], 2)} "
                f"| {fmt(row['combined']['rms']['mean'])} | {fmt(row['ratio']['mean'], 2)} |"
            )
    lines += [
        "",
        "| Config | position grad RMS | position delta RMS |",
        "|---|---:|---:|",
    ]
    for variant in MATRIX:
        lines.append(
            f"| {LABEL[variant]} | {fmt(real_summary[variant]['position_gradient_rms']['mean'], 8)} "
            f"| {fmt(real_summary[variant]['position_delta_rms']['mean'], 8)} |"
        )
    lines += [
        "",
        "sqrt Aはposition gradient RMSもCurrentより小さく、effective ratioは約20。"
        "両scale候補はratioを約1へ戻すが、representation全体を約19.6倍にするため"
        "residualに対するbranch contributionが低下した。",
        "",
        "## Architecture matrix（Syntheticはseed 42）",
        "",
        "| Config | Params | Validation | Top-1/5 | Copy 4/8/16 | Key min 2/4/8 | Pattern basic/numeric | Position min | Context | Layer9 | punct. | tok/s | RAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in MATRIX:
        real = real_summary[variant]
        synth = synthetic_summary[variant]
        base = synth
        key_min = [
            min(base["key_lookup"][str(pairs)].values()) for pairs in (2, 4, 8)
        ]
        pattern_basic = min(base["pattern"][name] for name in ("abab", "abcabc", "nested"))
        lines.append(
            f"| {LABEL[variant]} | {real_parameters[variant]:,} "
            f"| {fmt(real['validation_loss']['mean'])} "
            f"| {pct(real['top_1_accuracy']['mean'])}/{pct(real['top_5_accuracy']['mean'])} "
            f"| {'/'.join(pct(base['copy'][str(length)]) for length in (4,8,16))} "
            f"| {'/'.join(pct(value) for value in key_min)} "
            f"| {pct(pattern_basic)}/{pct(base['pattern']['numeric'])} "
            f"| {pct(base['position']['minimum_accuracy'])} "
            f"| {pct(base['context_conditioned']['correct'])} "
            f"| {fmt(real['layer_9_rms']['mean'])} | {pct(real['punctuation_mass']['mean'])} "
            f"| {fmt(real['tokens_per_second']['mean'], 0)} "
            f"| {fmt(real['peak_ram_mb']['mean'], 1)} |"
        )

    lines += [
        "",
        "## Residual stream",
        "",
        "Final updateの全layer。attn ratio=attention output/pre-attention residual、"
        "MLP ratio=MLP output/post-attention residual。",
        "",
        "| Config | layer | residual RMS | output RMS | attn ratio | MLP ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in MATRIX:
        for row in residual_milestones[variant]["100"]:
            lines.append(
                f"| {LABEL[variant]} | {row['layer']} "
                f"| {fmt(row['post_attention_residual_rms']['mean'])} "
                f"| {fmt(row['post_mlp_residual_rms']['mean'])} "
                f"| {fmt(row['attention_to_residual_ratio']['mean'])} "
                f"| {fmt(row['mlp_to_residual_ratio']['mean'])} |"
            )
    lines += [
        "",
        "step 0/10/25/50/100の全layer値はsummary JSONに保存。depth-initはattention/MLP"
        " output projectionだけを0.02/sqrt(20)=0.004472へ変更し、QKV/MLP inputは0.02を維持。"
        "runtime residual scalingは使用していない。",
        "",
        "## Copy / Numeric failure",
        "",
        "| Config | Copy classification (96 probes) | Numeric classification (256 probes) |",
        "|---|---|---|",
    ]
    for variant in MATRIX:
        copy_counts = synthetic_summary[variant]["copy_failure_analysis"]["classification_counts"]
        numeric_counts = synthetic_summary[variant]["numeric_failure_analysis"]["classification_counts"]
        lines.append(
            f"| {LABEL[variant]} | `{json.dumps(copy_counts, ensure_ascii=False)}` "
            f"| `{json.dumps(numeric_counts, ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "sqrt AのCopy誤り82/96は別position tokenへの置換で、固定offsetはなく広く分散。"
        "balancedも58/96がposition shift。Current/depth-initは94/96、95/96正解。"
        "Numeric誤りは全構成で主に既出pattern tokenのwrong phaseで、非value/frequency tokenへの"
        "collapseではない。従ってnumeric failureは位置比だけでなくsequence phase推論の問題。",
        "",
        "Synthetic audit: final answer位置だけteacher forcing、他target=-100、EOS/packingなし、"
        "入力末尾は固定ANSWER sentinelで実answerは含まない。train/test overlapは全構成0。",
        "",
        "## Frequency / hidden similarity",
        "",
        "tied LM headでは `logit = ||hidden|| × ||token embedding|| × cosine`。"
        "全構成でcorrect-token cosineより句読点・助詞方向の平均cosineが高く、hiddenが頻出token"
        "方向へ寄る現象を確認した。詳細なtoken別cosine/logitはsummary JSONに保存。",
        "",
        "Clean Japaneseではdepth-initがloss 6.9165→6.8260、Top-1 8.73→9.31%、"
        "punctuation mass 57.86%、Top 1%外Top-1>0を達成。一方Full Corpus 3-seedでは"
        "punctuation massは90.98%へわずかに低下したが、Top 1%外Top-1は0%。",
        "",
        "## Norm audit",
        "",
        "Final LayerNormは全構成で存在し、`Embedding -> Blocks -> Final LayerNorm -> LM Head`。"
        "各layerのinput/normalized mean/std、gamma、betaをsummary JSONに保存し、非finite値や"
        "Final Norm欠落はなかった。PHASE 26で単独改善しなかったためRMSNorm再実験は未実施。",
        "",
        "## Gate",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in architecture_checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "",
        "結論はMULTI_COMPONENT_FIX_REQUIRED。ただし現時点で採用するarchitectureはない。"
        "formal FoundationはCurrent構成を維持し、depth-scaled residual initは次フェーズの"
        "実験用partial fixとしてのみ保持する。単独総合PASSがないため組合せは今回実行していない。",
        "",
        "Full 256k、512k、1M、46M、Corpus/Tokenizer変更、Campus、Instruction/DPO、"
        "Production、push/deployは未実施。",
        "",
        f"Final Blindは内容を開かずSHA256のみ確認: `{final_blind_sha}` "
        f"({'MATCH' if final_blind_sha == expected_sha else 'MISMATCH'})。",
        "",
    ]
    report_path = ROOT / "evaluation/foundation-v17-architecture-isolation-report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "architecture_decision": decision,
        "architecture_gate": architecture_gate,
        "proceed_to_full_256k": "NO",
        "checks": architecture_checks,
        "summary": summary_path.relative_to(ROOT).as_posix(),
        "report": report_path.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
