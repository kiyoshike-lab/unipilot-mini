from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/foundation-v09-sanity"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mib(value: int) -> float:
    return round(value / 1024**2, 3)


def main() -> int:
    inventory = read(ROOT / "evaluation/foundation-v09-data-inventory.json")
    tokenizer = read(ROOT / "evaluation/foundation-v09-tokenizer-benchmark.json")
    contexts = {str(context): read(ROOT / f"evaluation/foundation-v09-context/context-{context}.json")
                for context in (256, 512, 1024, 2048)}
    training = read(ROOT / "checkpoints/foundation-v09-sanity/checkpoint-step-100.manifest.json")
    comparison = read(OUT / "validation-200-comparison.json")
    current_rows = sum(row["rows"] for row in inventory["source_files"])
    selected_training = (inventory["selected"]["base_language"]["rows"]
                         + inventory["selected"]["campus_stable_pretraining"]["rows"]
                         + inventory["selected"]["instruction"]["rows"])
    mini = {
        "name": "UniPilot Mini v0.4 / Campus v2.3 system baseline",
        "parameters": 19_814_784, "vocab": 512, "context": 256,
        "embedding_dim": 384, "layers": 11, "heads": 6, "ffn_dim": 1536,
        "production_changed": False,
    }
    standard = {
        "name": "UniPilot Standard Foundation v0.9 sanity candidate",
        "parameters": 46_755_840, "vocab": 4096, "context": 1024,
        "embedding_dim": 512, "layers": 14, "heads": 8, "ffn_dim": 2048,
        "role": "separate research candidate; not production",
    }
    payload = {
        "schema_version": "foundation-v09-phase-summary-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "PHASE 21 items 1-8 only",
        "baseline": inventory["baseline"],
        "data": {
            "current_inventory_rows": current_rows,
            "new_structured_training_rows": selected_training,
            "new_content_claim": "Rows are licensed-source chunks/restructured existing authored data; only 4 answers are explicitly human-approved.",
            "base_language_rows": inventory["selected"]["base_language"]["rows"],
            "campus_stable_rows": inventory["selected"]["campus_stable_pretraining"]["rows"],
            "instruction_rows": inventory["selected"]["instruction"]["rows"],
            "human_approved_rows": inventory["selected"]["human_approved"]["rows"],
            "rag_only_documents": inventory["selected"]["rag_only"]["rag_only_documents"],
            "source_and_license": {
                "base": {"source": "Japanese Wikipedia contributors", "license": "CC BY-SA 4.0"},
                "campus_and_instruction": {"source": "UniPilot project-authored", "license": "CC0-1.0"},
                "rag_only": inventory["selected"]["rag_only"]["rag_only_by_license"],
            },
            "exact_duplicate_excluded": inventory["selected"]["instruction"]["exact_duplicate_excluded"],
            "semantic_holdout_overlap_excluded": inventory["selected"]["instruction"]["semantic_holdout_overlap_excluded"],
            "legacy_quality_excluded": inventory["excluded_legacy"]["rows"],
            "remaining_gaps": inventory["gaps"],
        },
        "evaluation_separation": {
            "validation_questions": 200, "final_blind_questions": 1000,
            "final_blind_sha256": comparison["final_blind_sha256"],
            "final_blind_opened": False,
            "stress": "data/campus_v23/holdouts/stress-200.json",
            "real_student": "data/campus_v21/real-student/evaluation-500.json",
        },
        "architectures": {"mini": mini, "standard_candidate": standard},
        "tokenizer": tokenizer,
        "context_benchmark": contexts,
        "selected_context": {
            "sanity_and_long_term_candidate": 1024,
            "reason": "Supports detailed answers/history while remaining below 512 MiB in the isolated random-weight probe; 2048 reached about 771 MiB."
        },
        "step_100": {
            "executed": True, "stage": training["training_metrics"]["stage"],
            "training_loss": training["training_metrics"]["loss"],
            "validation_loss": training["training_metrics"]["validation_loss"],
            "training_rss_mb": training["training_metrics"]["memory_usage_mb"],
            "training_tokens_per_second": training["training_metrics"]["tokens_per_second"],
            "probe": {key: training["development_probe"][key] for key in (
                "questions", "natural_rate", "mean_characters", "mean_first_token_seconds",
                "mean_tokens_per_second", "text_generation_established")},
            "training_checkpoint_mb": mib(training["training_checkpoint_bytes"]),
            "inference_checkpoint_mb": mib(training["inference_checkpoint_bytes"]),
            "gate": "FAIL",
        },
        "step_500": {
            "executed": False,
            "reason": "Step-100 Base language generation gate failed (0/8 natural continuations).",
        },
        "mini_vs_standard_validation_200": comparison,
        "standard_continue": False,
        "next_training_step": None,
        "decision": "Do not continue this checkpoint to 500/1000/2000/5000. Improve licensed Base corpus and training design first.",
        "external_ai_api": "OFF", "push_or_deploy_performed": False,
        "production_changed": False,
    }
    write(OUT / "summary.json", payload)
    campus = comparison["campus_v23"]
    candidate = comparison["standard"]
    axes = "\n".join(
        f"| {axis} | {campus['axis_percent'][axis]:.2f}% | {candidate['axis_percent'][axis]:.2f}% | "
        f"{comparison['axis_delta_standard_minus_campus'][axis]:+.2f}pt |"
        for axis in ("correctness", "relevance", "completeness", "specificity", "naturalness", "actionable")
    )
    (OUT / "report.md").write_text(
        "# UniPilot Foundation v0.9 — PHASE 21 (1–8)\n\n"
        "Campus v2.3 / production v0.4を固定し、新モデル別系列だけを検証した。外部LLM/APIは未使用。\n\n"
        "## Data\n\n"
        f"- Current inventory: {current_rows:,} rows\n"
        f"- v0.9 structured training: {selected_training:,} rows (Base 1,040 / Campus 330 / Instruction 11,844)\n"
        "- Human approved: 4; RAG-only current/institutional sources: 114\n"
        f"- Exact duplicate / holdout semantic overlap excluded: "
        f"{inventory['selected']['instruction']['exact_duplicate_excluded']} / "
        f"{inventory['selected']['instruction']['semantic_holdout_overlap_excluded']}\n"
        "- Legacy low-quality template rows excluded: 50,000\n"
        "- Final Blind 1,000: sealed and unopened\n\n"
        "## Model and sanity training\n\n"
        "- Mini: 19,814,784 params / vocab 512 / context 256\n"
        "- Standard candidate: 46,755,840 params / vocab 4096 / context 1024\n"
        f"- 100step: train loss {training['training_metrics']['loss']:.4f}, validation loss "
        f"{training['training_metrics']['validation_loss']:.4f}, natural continuation 0/8, Gate FAIL\n"
        "- 500step: not executed because the 100step gate failed\n\n"
        "## Validation 200\n\n"
        "| Axis | Campus v2.3 | Standard step100 | Delta |\n|---|---:|---:|---:|\n"
        f"{axes}\n\n"
        f"- Campus: critical {campus['critical_errors']}, first response "
        f"{campus['average_first_token_seconds']:.3f}s, peak RSS {campus['peak_rss_mb']:.1f}MB\n"
        f"- Standard: critical generation failures {candidate['critical_errors']}, first token "
        f"{candidate['average_first_token_seconds']:.3f}s, {candidate['average_tokens_per_second']:.2f} tok/s, "
        f"peak RSS {candidate['peak_rss_mb']:.1f}MB\n\n"
        "## Decision\n\n"
        "Standard継続: **NO**。このcheckpointの次stepはなし。まず高品質なライセンス済みBase本文を増やし、"
        "短いsanityで日本語成立を再確認する。\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(OUT / "summary.json"), "report": str(OUT / "report.md"),
                      "standard_continue": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
