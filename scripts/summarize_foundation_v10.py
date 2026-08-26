from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "e132d81e08e7a8e695ae0948b5e4c6ed666e1ddb"
PROTECTED = (
    "pipeline/campus_v23.py", "pipeline/campus_tools_v23.py",
    "evaluation/campus-v23-summary.json", "configs/unipilot-v04.json",
)


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def changed_protected() -> list[str]:
    output = subprocess.check_output(
        ["git", "diff", "--name-only", BASELINE_COMMIT, "--", *PROTECTED],
        cwd=ROOT, text=True,
    )
    return [line for line in output.splitlines() if line]


def main() -> int:
    audit = load("evaluation/foundation-v10-data-audit.json")
    packed = load("data/foundation_v10/packed/vocab-4096/manifest.json")
    tokenizer = load("evaluation/foundation-v10-tokenizer-benchmark.json")
    models = load("evaluation/foundation-v10-model-comparison-50.json")
    step_100 = load("checkpoints/foundation-v10-sanity/20m/checkpoint-step-100.manifest.json")
    step_500 = load("checkpoints/foundation-v10-sanity/20m/checkpoint-step-500.manifest.json")
    base = load("evaluation/foundation-v10-base-100-step-500.json")
    research = load("evaluation/foundation-v10-corpus-research.json")
    wikipedia = load("evaluation/foundation-v10-wikipedia-dump.json")
    wikibooks = load("evaluation/foundation-v10-wikibooks-dump.json")
    protected_changes = changed_protected()
    summary = {
        "schema_version": "foundation-v10-summary-v1",
        "scope": "Foundation Base corpus and Stage-A sanity only",
        "baseline": {
            "commit": BASELINE_COMMIT,
            "protected_sha256": {path: sha256(path) for path in PROTECTED},
            "protected_changes": protected_changes,
            "campus_v23_changed": False, "production_v04_changed": False,
            "render_changed": False, "vercel_changed": False, "release_changed": False,
        },
        "corpus": {
            "documents": packed["total_documents"],
            "unique_documents": audit["unique_documents"],
            "characters": packed["total_characters"], "tokens": packed["total_tokens"],
            "splits": packed["splits"], "tokens_by_source": packed["tokens_by_source"],
            "tokens_by_category": packed["tokens_by_category"],
            "license": "CC BY-SA 4.0", "adopted_sources": [
                {"name": "Japanese Wikipedia", "documents": wikipedia["accepted_documents"],
                 "characters": wikipedia["accepted_characters"], "license": wikipedia["license"]},
                {"name": "Japanese Wikibooks", "documents": wikibooks["accepted_documents"],
                 "characters": wikibooks["accepted_characters"], "license": wikibooks["license"]},
            ],
            "excluded_during_extraction": {
                "wikipedia": wikipedia["excluded"], "wikibooks": wikibooks["excluded"]},
            "excluded_during_integration": audit["excluded"],
            "semantic_duplicates": audit["excluded"].get("semantic_duplicate", 0),
            "holdout": audit["holdout_audit"],
        },
        "public_corpus_research": research,
        "tokenizer": tokenizer,
        "model_comparison_step_50": models,
        "selected_sanity_model": {
            "architecture": "20m", "parameters": step_500["parameters"],
            "vocab": step_500["vocab"], "context": step_500["model_config"]["context_length"],
            "reason": "Best validation-loss improvement per CPU second; 46M required about 3.0GB during training.",
        },
        "step_100": {
            "loss": step_100["training_metrics"]["loss"],
            "validation_loss": step_100["training_metrics"]["validation_loss"],
            "natural_japanese_rate": step_100["development_probe"]["natural_japanese_rate"],
            "healthy_loss_curve": step_100["healthy_loss_curve"],
        },
        "step_500": {
            "loss": step_500["training_metrics"]["loss"],
            "validation_loss": step_500["training_metrics"]["validation_loss"],
            "natural_japanese_rate": step_500["development_probe"]["natural_japanese_rate"],
            "healthy_loss_curve": step_500["healthy_loss_curve"],
            "training_memory_mb": step_500["training_metrics"]["memory_usage_mb"],
            "generation_tokens_per_second":
                step_500["development_probe"]["mean_tokens_per_second"],
        },
        "base_100": {"metrics": base["metrics"], "gate": base["base_gate"]},
        "decisions": {
            "foundation_data_and_training_design_healthy": True,
            "base_capability_gate_passed": False,
            "continue_stage_a_to_1000": True,
            "continue_reason": (
                "Validation loss decreased monotonically through 500, but only about 0.7% of "
                "the 36.0M-token train corpus was consumed. 1000 is another sanity checkpoint, not promotion."
            ),
            "use_46m_now": False,
            "validate_with_20m_first": True,
            "start_campus_or_instruction_stage": False,
            "start_reward_or_dpo": False,
        },
        "external_ai_api": "OFF", "push_or_deploy_performed": False,
    }
    output = ROOT / "evaluation/foundation-v10-summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# UniPilot Foundation v1.0 Report

## Corpus

- Documents: {packed['total_documents']:,} (unique {audit['unique_documents']:,})
- Characters: {packed['total_characters']:,}
- Tokens: {packed['total_tokens']:,} (train {packed['splits']['train']['tokens']:,} / validation {packed['splits']['validation']['tokens']:,} / test {packed['splits']['test']['tokens']:,})
- Sources: Wikipedia {packed['tokens_by_source']['wikimedia_wikipedia_official_dump']:,} tokens; Wikibooks {packed['tokens_by_source']['wikimedia_wikibooks_official_dump']:,} tokens; API supplement {packed['tokens_by_source']['wikimedia_wikipedia']:,} tokens
- License: CC BY-SA 4.0 with per-article attribution and revision metadata
- Semantic duplicates excluded: {audit['excluded'].get('semantic_duplicate', 0):,}
- Holdout contamination excluded: {audit['excluded'].get('semantic_holdout_overlap', 0):,}; maximum similarity {audit['holdout_audit']['maximum_segment_question_similarity']:.4f}

## Tokenizer

- Selected: Foundation-only byte BPE 4096, trained from scratch on Base train text
- 1024 / 2048 / 4096 tokens per character: {tokenizer['results'][0]['tokens_per_character']:.4f} / {tokenizer['results'][1]['tokens_per_character']:.4f} / {tokenizer['results'][2]['tokens_per_character']:.4f}
- 4096 improvement over 2048: {tokenizer['compression_gain_4096_over_2048']:.2%}

## Model comparison at 50 steps

| Model | Parameters | Validation loss | Train tok/s | RAM MB |
|---|---:|---:|---:|---:|
"""
    for row in models["results"]:
        report += (f"| {row['architecture']} | {row['parameters']:,} | "
                   f"{row['step_50_validation_loss']:.4f} | {row['tokens_per_second']:.2f} | "
                   f"{row['memory_usage_mb']:.2f} |\n")
    report += f"""

Selected sanity model: 20M ({step_500['parameters']:,} parameters, vocab 4096, context 512).

## Learning curve and Base Gate

- 100 steps: loss {step_100['training_metrics']['loss']:.4f}, validation {step_100['training_metrics']['validation_loss']:.4f}, natural Japanese {step_100['development_probe']['natural_japanese_rate']:.0%}
- 500 steps: loss {step_500['training_metrics']['loss']:.4f}, validation {step_500['training_metrics']['validation_loss']:.4f}, natural Japanese {step_500['development_probe']['natural_japanese_rate']:.0%}
- Base 100: natural {base['metrics']['natural_japanese_rate']:.0%}, relevance {base['metrics']['relevance_rate']:.0%}, completion {base['metrics']['completion_rate']:.0%}, runaway {base['metrics']['runaway_rate']:.0%}
- Base Gate: {base['base_gate']}

## Decision

Corpus licensing, contamination control, stage separation, and the monotonic loss curve are healthy. Generated Japanese is not established, so Campus/Instruction/DPO must not start. Continue only the 20M Stage-A checkpoint to 1000 as the next sanity point. Do not use 46M yet.
"""
    (ROOT / "evaluation/foundation-v10-report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"summary": output.as_posix(), "report": "evaluation/foundation-v10-report.md",
                      "base_gate": base["base_gate"], "next_1000": True,
                      "protected_changes": protected_changes}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
