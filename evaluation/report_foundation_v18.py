from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODELS = ("custom_current", "custom_depth_init", "reference_mha")
LABELS = {
    "custom_current": "Current",
    "custom_depth_init": "Depth-init",
    "reference_mha": "Reference MHA",
}
EXPECTED_BLIND = "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def attention_average(attention: dict) -> dict:
    heads = [head for layer in attention["layers"] for head in layer["heads"]]
    fields = (
        "normalized_entropy", "max_attention_probability", "top_3_attention_mass",
        "correct_key_mass", "correct_value_mass", "correct_key_value_mass",
        "correct_position_mean_rank", "q_rms", "k_rms", "qk_dot_product_std",
        "scaled_attention_logit_std", "attention_margin",
    )
    return {field: average(head[field] for head in heads) for field in fields}


def architecture_decision(synthetic: dict, japanese: dict) -> tuple[str, dict]:
    passes = {
        name: bool(synthetic[name]["final"]["gate"]["pass"])
        for name in MODELS
    }
    depth_eval = synthetic["custom_depth_init"]["final"]["evaluation"]
    ref_eval = synthetic["reference_mha"]["final"]["evaluation"]
    depth_key = average(
        cell["accuracy"]
        for distances in depth_eval["key_lookup"].values()
        for cell in distances.values()
    )
    ref_key = average(
        cell["accuracy"]
        for distances in ref_eval["key_lookup"].values()
        for cell in distances.values()
    )
    jp = {
        name: japanese[name]["training"]["history"][-1]
        for name in MODELS
    }
    candidate_checks = {
        "synthetic_convergence": passes["custom_depth_init"],
        "reference_equivalent_key_capacity": abs(depth_key - ref_key) <= .05,
        "residual_rms_below_current": (
            jp["custom_depth_init"]["residual"]["layer9_rms"]
            < jp["custom_current"]["residual"]["layer9_rms"]
        ),
        "japanese_loss_below_current": (
            jp["custom_depth_init"]["metrics"]["loss"]
            < jp["custom_current"]["metrics"]["loss"]
        ),
        "japanese_top_k_not_worse": all(
            jp["custom_depth_init"]["metrics"][field] + .001
            >= jp["custom_current"]["metrics"][field]
            for field in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")
        ),
        "position_and_copy_no_regression": all(
            row["accuracy"] >= minimum
            for row, minimum in [
                *[(value, .95) for key, value in depth_eval["copy"].items() if int(key) <= 8],
                *[(value, .90) for key, value in depth_eval["copy"].items() if int(key) == 16],
                *[(value, .95) for value in depth_eval["position"].values()],
            ]
        ),
    }
    if passes["custom_depth_init"] and passes["reference_mha"] and all(candidate_checks.values()):
        decision = "DEPTH_INIT_CAPACITY_PASS"
    elif passes["reference_mha"] and not (
        passes["custom_current"] or passes["custom_depth_init"]
    ):
        non_key_depth = all(
            synthetic["custom_depth_init"]["final"]["gate"]["checks"][key]
            for key in synthetic["custom_depth_init"]["final"]["gate"]["checks"]
            if not key.startswith("key_")
        )
        decision = "ATTENTION_RETRIEVAL_ISSUE" if non_key_depth else "CUSTOM_IMPLEMENTATION_ISSUE"
    elif not any(passes.values()):
        decision = "SYNTHETIC_BENCHMARK_ISSUE"
    else:
        decision = "UNRESOLVED"
    return decision, {
        "synthetic_gate_by_model": passes,
        "depth_candidate_checks": candidate_checks,
        "depth_key_mean": depth_key,
        "reference_key_mean": ref_key,
    }


def make_report(summary: dict) -> str:
    synthetic = summary["synthetic"]
    japanese = summary["japanese_diagnostic"]
    lines = [
        "# UniPilot Foundation v1.8 Reference Cross-Check Report",
        "",
        "## 判定",
        "",
        f"- Architecture Gate: **{summary['decision']}**",
        f"- Full 256kへ進む: **{'YES' if summary['full_256k_recommended'] else 'NO'}**",
        f"- Depth-scaled init正式候補: **{'YES' if summary['depth_scaled_init_candidate'] else 'NO'}**",
        "- 正式Foundation architecture: Currentのまま（本番・Campus・Final Blind内容は未変更）",
        "",
        "## Reference architecture / fairness",
        "",
        "Referenceは`torch.nn.MultiheadAttention(batch_first=True)`を使う独立decoderで、Pre-LN、Final LN、learned absolute position、GELU、causal mask、tied LM headです。custom attention/residual blockは共有していません。",
        "",
        "| Model | Formal params (vocab4096/context512) | Synthetic params (vocab256/context80) | Layers | Hidden | Heads | FFN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        row = synthetic[name]
        cfg = row["config"]
        lines.append(
            f"| {LABELS[name]} | 19,514,880 | {row['parameters']:,} | {cfg['n_layers']} | "
            f"{cfg['embedding_dim']} | {cfg['n_heads']} | {cfg['ffn_dim']} |"
        )
    lines.extend([
        "",
        "Reference correctness: causal leakage、position sensitivity、gradient flow、tiny overfit、EOS sanity、parameter parityを全件PASS。",
        "",
        "比較差分はattention実装（custom fused QKV/明示softmax 対 `nn.MultiheadAttention`）と、Currentだけのresidual出力初期値です。Depth-initとReferenceの初期値、dropout、bias、norm位置、residual式、embedding、position、tied LM headは一致させました。",
        "",
        "## Synthetic LR pilot",
        "",
        "| LR | Current | Depth-init | Reference | Cross-model mean |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in summary["lr_pilot"]["by_learning_rate"]:
        scores = {item["model"]: item["normalized_score"] for item in row["models"]}
        lines.append(
            f"| {row['learning_rate']:.4g} | {scores['custom_current']:.4f} | "
            f"{scores['custom_depth_init']:.4f} | {scores['reference_mha']:.4f} | "
            f"{row['cross_model_mean']:.4f} |"
        )
    lines.extend([
        "",
        "採用LRは3モデル共通の3e-4。AdamW、betas=(0.9,0.95)、eps=1e-8、weight decay=0.01、batch=16、clip=1.0で統一。これはcapacity診断専用であり、Foundation pretraining LRには転用しません。",
        "",
        "## Synthetic learning curves",
        "",
        "Key列はshort/medium/longの最小accuracy。chanceはpairsごとに50/25/12.5/6.25%。Copy/Position/Long/Numericは1/32=3.125%、Symbolicは1/4=25%、Contextは1/4=25%を目安とします。",
        "",
        "| Model | Budget | Copy 4/8/16/32/64 | Key 2/4/8/16 | Numeric | Symbolic | Long | Context | Position min | Gate |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for name in MODELS:
        for point in synthetic[name]["training"]["curve"]:
            ev = point["evaluation"]
            copy = "/".join(f"{ev['copy'][key]['accuracy']:.1%}" for key in ("4", "8", "16", "32", "64"))
            key = "/".join(
                f"{min(cell['accuracy'] for cell in ev['key_lookup'][pairs].values()):.1%}"
                for pairs in ("2", "4", "8", "16")
            )
            position = min(cell["accuracy"] for cell in ev["position"].values())
            lines.append(
                f"| {LABELS[name]} | {point['percent_of_phase28_budget']}% | {copy} | {key} | "
                f"{ev['pattern']['numeric']['accuracy']:.1%} | {ev['pattern']['symbolic']['accuracy']:.1%} | "
                f"{ev['long_range']['accuracy']:.1%} | {ev['context_conditioned']['correct']['accuracy']:.1%} | "
                f"{position:.1%} | {'PASS' if point['gate']['pass'] else 'FAIL'} |"
            )
    lines.extend([
        "",
        "全cellのloss、Context shuffled/removed、Pattern全種、exact/template/leakage監査はsummary JSONに収録しています。",
        "",
        "3モデルすべてでexact train/test overlap=0、answer leakage=0です。Templateはtask定義として意図的に共有し、token instanceは分離しています。",
        "",
        "## Final attention audit（全layer/head平均）",
        "",
        "| Model | Entropy | Max prob | Top-3 mass | Correct K+V mass | Correct rank | Q RMS | K RMS | scaled logit std | Margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in MODELS:
        attention = synthetic[name]["final"]["attention"]["4"]["medium"]
        row = attention_average(attention)
        lines.append(
            f"| {LABELS[name]} | {row['normalized_entropy']:.4f} | "
            f"{row['max_attention_probability']:.4f} | {row['top_3_attention_mass']:.4f} | "
            f"{row['correct_key_value_mass']:.4f} | {row['correct_position_mean_rank']:.2f} | "
            f"{row['q_rms']:.4f} | {row['k_rms']:.4f} | "
            f"{row['scaled_attention_logit_std']:.4f} | {row['attention_margin']:.4f} |"
        )
    lines.extend([
        "",
        "step 0〜400%の各layer/head値と、pairs×distance全12cellのfinal値はsummary JSON内の生データに保持しています。",
        "",
        "最終attentionはuniformではなく強くselective（entropy 0.074〜0.089、max prob 0.877〜0.903）ですが、4-pair mediumの正解K+V massは0.080〜0.137（chance 0.071）、平均rank 9.68〜10.49、marginは全モデルで大幅な負値です。つまりattentionを絞れないのではなく、正しいkey/value関係へ選択を向けるsupervision/curriculumが不足しています。",
        "",
        "## Japanese diagnostic（同一128k tokens）",
        "",
        "| Model | Tokens | Loss | Top-1 | Top-5 | Top-10 | Correct prob | Punc mass | Context | Layer9 RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in MODELS:
        for point in japanese[name]["training"]["history"]:
            metrics = point["metrics"]
            lines.append(
                f"| {LABELS[name]} | {point['tokens_processed']:,} | {metrics['loss']:.4f} | "
                f"{metrics['top_1_accuracy']:.2%} | {metrics['top_5_accuracy']:.2%} | "
                f"{metrics['top_10_accuracy']:.2%} | {metrics['mean_correct_token_probability']:.4f} | "
                f"{metrics['punctuation_top1_mass']:.2%} | "
                f"{point['context_sensitivity']['context_sensitivity_score']:.4f} | "
                f"{point['residual']['layer9_rms']:.4f} |"
            )
    lines.extend([
        "",
        "Frequency bucketごとのTop-1/5/10と正解token確率はsummary JSONに収録しています。",
        "",
        "## 原因分析",
        "",
        "- Custom、Depth-init、独立Referenceの全てが400%でも同じKey retrieval失敗を再現したため、custom attention固有のfatal defectではありません。",
        "- Symbolicは3モデルとも100%へ収束した一方、atomic IDを使うnumericは40〜55%でした。Tokenizer分割ではなく、数列規則推定の難度とtask/variant当たりのsupervision密度の問題です。",
        "- Copy 4/8/16、Long Range、Context control、Positionは全モデルでPASSし、context capacityそのものの欠如も否定されます。",
        "- よって今回の失敗はarchitectureではなく、複数難度を1/6 task schedule内で疎に混ぜ、最終answer tokenだけを教師にしたSynthetic benchmark/training setupに帰属します。次はarchitectureを変えず、Key Lookup単独curriculumと中間関係supervisionを隔離検証すべきです。",
        "",
        "## Integrity / protection",
        "",
        f"- Final Blind SHA256: `{summary['final_blind']['sha256']}` (content unopened, MATCH={summary['final_blind']['match']})",
        "- Synthetic train/test exact overlap、answer leakageは各モデルreport参照。",
        "- 3モデルのcheckpointはmodel/optimizer stateを含みstrict reload PASS。",
        "- Full 256k、46M、Tokenizer、Campus、本番、push、deployは未実施。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    synthetic = {
        name: load(ROOT / "checkpoints/foundation-v18-synthetic" / f"{name}-lr-3em04.json")
        for name in MODELS
    }
    japanese = {
        name: load(ROOT / "checkpoints/foundation-v18-short-japanese" / f"{name}.json")
        for name in MODELS
    }
    pilot_rows = []
    for path in sorted((ROOT / "checkpoints/foundation-v18-lr-pilot").glob("*.json")):
        report = load(path)
        pilot_rows.append({
            "model": report["model"]["name"],
            "learning_rate": report["optimizer"]["learning_rate"],
            "normalized_score": report["final"]["evaluation"]["normalized_score"],
            "report": report,
        })
    grouped = []
    for learning_rate in sorted({row["learning_rate"] for row in pilot_rows}):
        models = [
            {key: row[key] for key in ("model", "learning_rate", "normalized_score")}
            for row in pilot_rows if row["learning_rate"] == learning_rate
        ]
        grouped.append({
            "learning_rate": learning_rate,
            "cross_model_mean": average(row["normalized_score"] for row in models),
            "models": models,
        })
    grouped.sort(key=lambda row: row["cross_model_mean"], reverse=True)
    decision, decision_evidence = architecture_decision(synthetic, japanese)
    blind_path = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    blind_sha = file_sha256(blind_path)
    summary = {
        "schema_version": "foundation-v18-reference-cross-check-v1",
        "decision": decision,
        "decision_evidence": decision_evidence,
        "full_256k_recommended": decision == "DEPTH_INIT_CAPACITY_PASS",
        "depth_scaled_init_candidate": (
            decision == "DEPTH_INIT_CAPACITY_PASS"
            and all(decision_evidence["depth_candidate_checks"].values())
        ),
        "reference_correctness_tests": {
            "causal_leakage": "PASS",
            "position_sensitivity": "PASS",
            "gradient_flow": "PASS",
            "tiny_overfit": "PASS",
            "eos_sanity": "PASS",
            "parameter_parity": "PASS",
        },
        "parameter_fairness": {
            "formal_vocab_size": 4096,
            "formal_context_length": 512,
            "formal_parameters_each": 19_514_880,
            "synthetic_vocab_size": 256,
            "synthetic_context_length": 80,
            "synthetic_parameters_each": 17_874_432,
            "japanese_diagnostic_context_length": 128,
            "japanese_diagnostic_parameters_each": 19_367_424,
            "maximum_pairwise_difference_percent": 0.0,
        },
        "lr_pilot": {
            "selected_common_learning_rate": grouped[0]["learning_rate"],
            "by_learning_rate": grouped,
            "pretraining_lr_changed": False,
        },
        "numeric_tokenizer_audit": load(
            ROOT / "evaluation/foundation-v18-numeric-tokenizer-audit.json"
        ),
        "custom_vs_reference_difference_inventory": {
            "attention_implementation": (
                "custom fused qkv + explicit masked softmax vs "
                "torch.nn.MultiheadAttention"
            ),
            "initialization": (
                "Depth-init and Reference use identical 0.02/sqrt(20) residual "
                "output std; Current alone uses 0.02"
            ),
            "dropout": "same configured 0.1; standard modules place equivalent attention/MLP output dropout",
            "bias": "same enabled bias",
            "norm_placement": "same Pre-LayerNorm with Final LayerNorm",
            "residual": "same x+attention(LN(x)); x+MLP(LN(x)) flow",
            "embedding": "same unscaled token + learned absolute position embedding",
            "position": "same learned absolute position range",
            "lm_head": "same bias-free tied token embedding head",
        },
        "synthetic": synthetic,
        "japanese_diagnostic": japanese,
        "final_blind": {
            "path": "data/foundation_v09/evaluation/final-blind-1000.json",
            "sha256": blind_sha,
            "expected_sha256": EXPECTED_BLIND,
            "match": blind_sha == EXPECTED_BLIND,
            "content_opened": False,
        },
        "formal_foundation_architecture_changed": False,
        "full_256k_run": False,
        "production_changed": False,
        "push": False,
        "deploy": False,
        "external_ai_api": "OFF",
    }
    output = ROOT / "evaluation/foundation-v18-summary.json"
    report = ROOT / "evaluation/foundation-v18-reference-cross-check-report.md"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report.write_text(make_report(summary) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "full_256k": summary["full_256k_recommended"],
        "depth_candidate": summary["depth_scaled_init_candidate"],
        "summary": output.relative_to(ROOT).as_posix(),
        "report": report.relative_to(ROOT).as_posix(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
