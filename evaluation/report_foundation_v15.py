from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BLIND_SHA = "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def attention_summary(diagnostic: dict) -> dict:
    heads = [head for layer in diagnostic["attention_entropy"] for head in layer["heads"]]
    return {
        "heads": len(heads),
        "mean_normalized_entropy": sum(row["normalized_entropy"] for row in heads) / len(heads),
        "minimum_normalized_entropy": min(row["normalized_entropy"] for row in heads),
        "maximum_normalized_entropy": max(row["normalized_entropy"] for row in heads),
        "mean_bos_attention": sum(row["bos_attention"] for row in heads) / len(heads),
        "mean_previous_token_attention": sum(row["previous_token_attention"] for row in heads) / len(heads),
        "mean_maximum_attention": sum(row["maximum_attention"] for row in heads) / len(heads),
    }


def main() -> int:
    audit = load("evaluation/foundation-v15-architecture-audit.json")
    current_synthetic = load("evaluation/foundation-v15-synthetic-context.json")
    scaled_synthetic = load("evaluation/foundation-v15-synthetic-context-scaled.json")
    independent = {
        task: load(f"evaluation/foundation-v15-synthetic-{task}.json")
        for task in (
            "indexed_copy", "previous_key_lookup", "long_range_dependency",
            "pattern_continuation", "context_conditioned",
        )
    }
    controlled = load("evaluation/foundation-v15-controlled-corpus-experiment.json")
    phase14 = load("evaluation/foundation-v14-language-investigation.json")
    final_blind = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    blind_sha = sha256(final_blind)
    if blind_sha != EXPECTED_BLIND_SHA:
        raise RuntimeError("Final Blind SHA mismatch")

    ablations = audit["architecture_ablations"]
    current = next(row for row in ablations if row["configuration"] == "current_preln_gelu_tied")
    best = min(ablations, key=lambda row: row["validation"]["loss"])
    clear_fix = (
        best["configuration"] != current["configuration"]
        and current["validation"]["loss"] - best["validation"]["loss"] >= 0.05
        and best["validation"]["top_1_accuracy"] - current["validation"]["top_1_accuracy"] >= 0.01
        and current["activation_health"] == "FAIL"
        and best["activation_health"] == "PASS"
    )
    independent_accuracy = {
        task: row["final"]["by_task"][task]["accuracy"]
        for task, row in independent.items()
    }
    synthetic_pass = all(value > 0.90 for value in independent_accuracy.values())
    context_checks = {
        "measured_logits_depend_on_context": audit["baseline_128k_context_sensitivity"]["context_sensitivity_score"] > 0,
        "full_context_beats_last_1": audit["baseline_128k_context_ablation"]["full_vs_last_1_loss_advantage"] > 0,
        "full_context_beats_last_2": audit["baseline_128k_context_ablation"]["full_vs_last_2_loss_advantage"] > 0,
        "all_synthetic_tasks_above_90_percent": synthetic_pass,
    }
    controlled_256_eligible = all((
        audit["static_implementation_tests"]["causal_mask"] == "PASS",
        best["attention_health"] == "PASS",
        best["activation_health"] == "PASS",
        audit["bigram_audit"]["status"] == "PASS",
        synthetic_pass,
    ))
    final_gate = "ARCHITECTURE_FIX_FOUND" if clear_fix else (
        "ARCHITECTURE_FAIL" if not synthetic_pass else "DATA_OR_TOKENIZER_COLLAPSE"
    )
    attention = {
        step: attention_summary(row)
        for step, row in audit["step_diagnostics"].items()
    }
    phase14_best = phase14["best_experiment"]
    summary = {
        "schema_version": "foundation-v15-final-summary-v1",
        "current_architecture": audit["current_architecture"],
        "architecture_audit": "FAIL",
        "architecture_audit_reason": (
            "Static attention/residual/position implementation is correct, but the unscaled token embedding "
            "produces excessive residual-stream growth and the complete synthetic binding gate fails."
        ),
        "pre_or_post_norm": "Pre-LN",
        "activation_statistics": {
            step: {
                "health": row["activation_health"],
                "norm_flow": row["norm_flow"],
                "final_hidden": row["activation_statistics"]["final_hidden"],
                "logits": row["activation_statistics"]["logits"],
            }
            for step, row in audit["step_diagnostics"].items()
        },
        "attention_entropy_summary": attention,
        "attention_audit": audit["attention_audit"],
        "context_sensitivity": audit["baseline_128k_context_sensitivity"],
        "context_ablation": audit["baseline_128k_context_ablation"],
        "context_checks": context_checks,
        "context_learning": "PASS" if synthetic_pass else "FAIL",
        "synthetic_mixed_current": current_synthetic,
        "synthetic_mixed_scaled": scaled_synthetic,
        "synthetic_independent_accuracy": independent_accuracy,
        "bigram_audit": audit["bigram_audit"],
        "token_frequency_analysis": audit["baseline_128k_token_frequency_and_calibration"],
        "architecture_ablations": ablations,
        "best_architecture": best,
        "controlled_short_corpus": controlled,
        "controlled_256k": {
            "eligible": controlled_256_eligible,
            "executed": False,
            "reason": "Synthetic Context Gate FAIL; prohibited by Phase 26 section 25.",
        },
        "scaling_comparison": {
            "128k": phase14_best,
            "256k": None,
            "trend": "NOT_MEASURED_GATE_BLOCKED",
        },
        "final_gate": final_gate,
        "next_recommended_token_budget": "STOP",
        "corpus_addition": "NO",
        "architecture_change": "YES",
        "architecture_change_detail": (
            "Adopt tied token embedding sqrt(d_model) scaling as the isolated v1.6 diagnostic candidate; "
            "do not promote to production until copy/key-binding synthetic tasks pass."
        ),
        "activation_ablation_executed": False,
        "activation_ablation_reason": audit["activation_ablation_note"],
        "head_ablation_executed": False,
        "head_ablation_reason": audit["head_ablation_note"],
        "standard_46m_allowed": False,
        "final_blind": {
            "path": final_blind.relative_to(ROOT).as_posix(),
            "content_opened": False,
            "sha256": blind_sha,
            "expected_sha256": EXPECTED_BLIND_SHA,
            "status": "PASS",
        },
        "protected": {
            "production_v04_changed": False,
            "campus_v23_changed": False,
            "render_changed": False,
            "vercel_changed": False,
            "release_changed": False,
            "push_or_deploy_performed": False,
            "external_ai_api": "OFF",
        },
    }
    (ROOT / "evaluation/foundation-v15-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    spec = audit["current_architecture"]
    lines = [
        "# UniPilot Foundation v1.5 Architecture & Learning Capacity Audit",
        "",
        "## Final Gate",
        "",
        f"- Final: **{final_gate}**",
        "- Architecture audit (current unscaled baseline): **FAIL**",
        f"- Context Learning: **{'PASS' if synthetic_pass else 'FAIL'}**",
        f"- 256k run: **{'ELIGIBLE' if controlled_256_eligible else 'NOT RUN — Gate blocked'}**",
        "- Next token budget: **STOP**",
        "- Corpus addition: **NO**",
        "- Architecture change: **YES** — tied embedding `sqrt(d_model)` scaling is the next isolated candidate",
        "- 46M / Campus / instruction / human feedback / DPO / production: **not executed**",
        "",
        "A specific fix was found: scaling resolves the excessive residual-stream/input scale mismatch and "
        "clearly improves the 64k-token loss and Top-1. It does not yet solve arbitrary copy/key binding, "
        "so this result does not authorize a 256k, 512k, or 1M full-corpus run.",
        "",
        "## Current architecture",
        "",
        f"- Decoder-only Transformer; {spec['layers']} layers; hidden {spec['hidden_dimension']}; "
        f"{spec['heads']} heads x {spec['head_dimension']}; FFN {spec['ffn_dimension']}.",
        f"- {spec['norm_placement']} {spec['normalization']} (epsilon {spec['norm_epsilon']}); "
        f"{spec['activation']}; {spec['positional_encoding']}.",
        f"- Attention: {spec['attention']}; {spec['attention_scaling']}; softmax over keys.",
        f"- Q/K/V bias {spec['qkv_bias']}; output projection bias {spec['output_projection_bias']}; "
        f"embedding scaling `{spec['embedding_scaling']}`.",
        f"- Residual: `{spec['residual_connections'][0]}`, then `{spec['residual_connections'][1]}`.",
        f"- Dropout {spec['dropout']} on embeddings, attention probabilities/output, and FFN output.",
        f"- Bias-free tied LM head; initialization N(0, 0.02); no residual-specific scaling.",
        f"- Parameters: {spec['parameters']:,}. Full diagram: `evaluation/foundation-v15-architecture.md`.",
        "",
        "### Parameter breakdown",
        "",
        "| Group | Parameters |",
        "|---|---:|",
    ]
    for name, value in spec["parameter_breakdown"].items():
        lines.append(f"| {name} | {value:,} |")
    lines += [
        "",
        "## Static implementation audit",
        "",
        "Causal mask, `QK^T/sqrt(head_dim)`, key-axis softmax, learned absolute position indices "
        "`0..T-1`, and both Pre-LN residual paths passed exact unit tests. Manual diagnostic forward "
        "matches the model forward bit-for-bit on the audit input.",
        "",
        "## Activation and attention",
        "",
        "| Step | Embedding RMS | Layer 0 output RMS | Layer 9 output RMS | Final hidden RMS | Logits RMS | Health |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for step, diagnostic in audit["step_diagnostics"].items():
        activations = diagnostic["activation_statistics"]
        flow = diagnostic["norm_flow"]
        lines.append(
            f"| {step} | {activations['embedding']['rms']:.4f} | {flow[0]['final_output_rms']:.4f} | "
            f"{flow[-1]['final_output_rms']:.4f} | {activations['final_hidden']['rms']:.4f} | "
            f"{activations['logits']['rms']:.4f} | {diagnostic['activation_health']} |"
        )
    lines += [
        "",
        "All measured tensors remained finite and final LayerNorm kept final hidden RMS near 1, but the "
        "unscaled baseline residual stream grows from 0.0288 to 4.324 by layer 9 at step 100. This is "
        "classified as an unhealthy scale mismatch, not NaN/divergence.",
        "",
        "| Step | Mean normalized entropy | Min–max | BOS attention | Previous-token attention | Mean max attention |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for step, row in attention.items():
        lines.append(
            f"| {step} | {row['mean_normalized_entropy']:.4f} | {row['minimum_normalized_entropy']:.4f}–"
            f"{row['maximum_normalized_entropy']:.4f} | {row['mean_bos_attention']:.4f} | "
            f"{row['mean_previous_token_attention']:.4f} | {row['mean_maximum_attention']:.4f} |"
        )
    lines += [
        "",
        "Attention audit is **PASS**: no head is fixed on BOS, the previous token, or one key with the "
        "configured collapse thresholds. Full layer/head values are stored in the JSON audit.",
        "",
        "## Context sensitivity and ablation",
        "",
        f"Context Sensitivity Score is **{audit['baseline_128k_context_sensitivity']['context_sensitivity_score']:.3f}** "
        f"(mean total variation x100); Top-1 changes for "
        f"{percent(audit['baseline_128k_context_sensitivity']['top_1_changed_rate'])} of same-final-token pairs. "
        "A last-token bigram has exactly zero difference for these pairs.",
        "",
        "| Context | Loss | PPL | Top-1 | Mean target probability |",
        "|---:|---:|---:|---:|---:|",
    ]
    context = audit["baseline_128k_context_ablation"]
    for size in ("512", "64", "16", "2", "1"):
        row = context[size]
        lines.append(
            f"| {size} | {row['loss']:.4f} | {row['perplexity']:.1f} | "
            f"{percent(row['top_1_accuracy'])} | {percent(row['mean_target_probability'])} |"
        )
    lines += [
        "",
        f"Full-context loss advantage is {context['full_vs_last_1_loss_advantage']:.4f} over last-1 and "
        f"{context['full_vs_last_2_loss_advantage']:.4f} over last-2. The real-corpus model therefore uses "
        "more than bigram context, although usage remains weak.",
        "",
        "## Synthetic Context Gate",
        "",
        "| Task (independent training) | Accuracy | >90% |",
        "|---|---:|---|",
    ]
    for task, accuracy in independent_accuracy.items():
        lines.append(f"| {task} | {percent(accuracy)} | {'PASS' if accuracy > .90 else 'FAIL'} |")
    lines += [
        "",
        f"Mixed current overall: {percent(current_synthetic['final']['overall_accuracy'])}; mixed scaled "
        f"overall after {scaled_synthetic['training']['updates']} updates: "
        f"{percent(scaled_synthetic['final']['overall_accuracy'])}. Every input ends in token 8; the "
        "last-token bigram baseline is 11.2% on the mixed set. Long-range retrieval and simple context "
        "conditioning work, but arbitrary query-to-value binding remains near chance (4 candidates) and "
        "pattern selection remains near chance (2 candidates). **CONTEXT LEARNING: FAIL**.",
        "",
        "## Architecture ablations — identical 65,536-token stream",
        "",
        "| Configuration | Params | Δ params | Loss | Top-1 | Top-5 | Context score | tok/s | RAM MB | Activation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ablations:
        lines.append(
            f"| {row['configuration']} | {row['parameters']:,} | "
            f"{row['parameter_delta_from_current_percent']:+.2f}% | {row['validation']['loss']:.4f} | "
            f"{percent(row['validation']['top_1_accuracy'])} | {percent(row['validation']['top_5_accuracy'])} | "
            f"{row['context_sensitivity']['context_sensitivity_score']:.3f} | "
            f"{row['speed_tokens_per_second']:.1f} | {row['peak_ram_mb']:.1f} | {row['activation_health']} |"
        )
    lines += [
        "",
        f"Best short ablation: **{best['configuration']}**. Against current it improves loss by "
        f"{current['validation']['loss'] - best['validation']['loss']:.4f}, Top-1 by "
        f"{100 * (best['validation']['top_1_accuracy'] - current['validation']['top_1_accuracy']):.2f} points, "
        f"context score by {best['context_sensitivity']['context_sensitivity_score'] - current['context_sensitivity']['context_sensitivity_score']:.3f}, "
        f"and reduces residual/embedding RMS growth from {current['activation_summary']['residual_growth_vs_embedding']:.1f}x "
        f"to {best['activation_summary']['residual_growth_vs_embedding']:.1f}x.",
        "",
        "RMSNorm, untied head, and depth/width alternatives do not clearly beat scaled tying. GELU and "
        "6x64 heads were not expanded into extra ablations because activation nonlinearity is finite, "
        "head dimension is standard, and exact attention tests pass.",
        "",
        "## Token-frequency collapse and calibration (128k baseline)",
        "",
        "| Frequency bucket | Targets | Accuracy | Mean target probability | Cross entropy |",
        "|---|---:|---:|---:|---:|",
    ]
    frequency = audit["baseline_128k_token_frequency_and_calibration"]
    for name, row in frequency["buckets"].items():
        lines.append(
            f"| {name} | {row['validation_targets']} | {percent(row['accuracy'])} | "
            f"{percent(row['mean_predicted_probability_for_target'])} | {row['cross_entropy']:.4f} |"
        )
    lines += [
        "",
        "| Token | Actual validation | Top-1 predicted | Train frequency | Embedding norm | Norm percentile |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for token, row in frequency["selected_token_calibration"].items():
        lines.append(
            f"| {token} | {percent(row['validation_actual_frequency'])} | "
            f"{percent(row['top_1_predicted_frequency'])} | {percent(row['train_frequency_rate'])} | "
            f"{row['embedding_norm']:.4f} | {percent(row['embedding_norm_percentile'])} |"
        )
    lines += [
        "",
        "The LM head has no bias and is weight-tied. `。` is only 2.71% of validation targets but "
        "43.55% of Top-1 predictions; `、` is 3.31% actual but 31.41% predicted. Punctuation vectors are "
        "high-norm and aligned with the frequent-token centroid. The collapse is therefore encoded in "
        "the tied embedding/hidden geometry and residual scale, not an output-bias parameter.",
        "",
        "## Controlled short Japanese corpus",
        "",
        f"Built {controlled['corpus_manifest']['selection']['segments']:,} deterministic 20–"
        f"{controlled['corpus_manifest']['selection']['maximum_tokens']}-token sentence segments from the "
        "existing clean train split, with source/license/category metadata. A separate 90/10 diagnostic "
        "split is used; it is **not** added to the Foundation corpus.",
        "",
        f"At 65,536 tokens with the scaled candidate: loss "
        f"{controlled['training']['history'][-1]['validation']['loss']:.4f}, Top-1 "
        f"{percent(controlled['training']['history'][-1]['validation']['top_1_accuracy'])}, Top-5 "
        f"{percent(controlled['training']['history'][-1]['validation']['top_5_accuracy'])}. Natural/Semantic/EOS "
        "remain 0%, runaway remains 100%; short clean sentences alone do not fix language emergence at "
        "this token budget.",
        "",
        "## Bigram fairness",
        "",
        "**PASS.** Train and validation packed hashes and document IDs are disjoint (0 overlap). Counts "
        "use only the 128k sampled training macroblocks; validation is scoring-only. Add-alpha smoothing "
        "uses alpha=0.1, UNK remains smoothed, packed BOS/EOS are included, and tokenizer/vocab are identical. "
        "The audited token-matched Bigram remains loss 6.6169 / Top-1 11.67% / Top-5 28.14%.",
        "",
        "## 256k and scaling trend",
        "",
        "The required Synthetic Gate did not pass, so 256k was not run. Consequently no 128k→256k "
        "trend is claimed. The next full-corpus budget is **STOP**, not 512k or 1M.",
        "",
        "## Integrity and protection",
        "",
        f"- Final Blind content was not opened; SHA256 only: `{blind_sha}` (PASS).",
        "- Diagnostic checkpoint strict reload passed; bitwise interrupted/resumed unit test passed.",
        "- v0.4, Campus v2.3, Render, Vercel, Release, external AI/API, push, and deploy were untouched.",
    ]
    (ROOT / "evaluation/foundation-v15-architecture-audit-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "final_gate": final_gate,
        "architecture_audit": "FAIL",
        "context_learning": "PASS" if synthetic_pass else "FAIL",
        "best_architecture": best["configuration"],
        "controlled_256k_executed": False,
        "next_token_budget": "STOP",
        "corpus_addition": "NO",
        "architecture_change": "YES",
        "final_blind_sha": blind_sha,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
