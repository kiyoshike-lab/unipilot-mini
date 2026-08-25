from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/standard-50m-short"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mib(value: int) -> float:
    return round(value / 1024**2, 3)


def main() -> int:
    architectures = {
        name: read(OUT / f"architecture-standard-v08-{name}.json")
        for name in ("a-45m", "b-51m", "c-58m")
    }
    matrix = {
        f"vocab-{vocab}-context-{context}": read(OUT / f"vocab-{vocab}-context-{context}.json")
        for vocab in (1024, 2048) for context in (512, 1024)
    }
    training = read(ROOT / "checkpoints/standard-50m-short/checkpoint-step-100.manifest.json")
    comparison = read(OUT / "blind-200-comparison.json")
    toeic = read(ROOT / "evaluation/campus-v23-toeic-tool-fix.json")
    selected = {
        "name": "a-45m with vocab 2048",
        "role": "short-validation candidate only; not production",
        "parameters": 45_445_120,
        "vocab": 2048,
        "context": 512,
        "layers": 14,
        "hidden": 512,
        "heads": 8,
        "ffn": 2048,
        "selection_reason": (
            "The 45M family was the lightest/fastest architecture; vocab 2048 reduced Japanese token use "
            "with about 2 MiB checkpoint cost; context 1024 added RAM without benefit on measured prompts."
        ),
    }
    payload = {
        "schema_version": "unipilot-standard-50m-short-phase-summary-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campus_v23_preserved": True,
        "production_v04_changed": False,
        "render_changed": False,
        "vercel_changed": False,
        "release_changed": False,
        "push_or_deploy_performed": False,
        "external_ai_api": "OFF",
        "toeic_tool_fix": {
            "known_critical_before": toeic["critical_errors_before"],
            "known_critical_after": toeic["critical_errors_after"],
            "generalization": "Missing values are requested; only explicit user values are echoed; no fixed study ratio is invented.",
        },
        "architecture_comparison": {name: {
            "parameters": row["parameters"],
            "generation_tokens_per_second": row["inference"]["generation_tokens_per_second"],
            "first_token_seconds": row["inference"]["first_token_seconds"],
            "training_peak_rss_mb": row["rss_after_optimizer_step_mb"],
            "checkpoint_mb": row["checkpoint_mb"],
        } for name, row in architectures.items()},
        "tokenizer_context_comparison": {name: {
            "parameters": row["parameters"],
            "tokens_per_japanese_character": row["tokens_per_japanese_character"],
            "prompt_p95_tokens": row["p95_prompt_plus_keypoint_tokens"],
            "generation_tokens_per_second": row["generation"]["tokens_per_second"],
            "peak_observed_rss_mb": row["peak_observed_rss_mb"],
            "checkpoint_mb": row["checkpoint_mb_fp32"],
        } for name, row in matrix.items()},
        "selected_candidate": selected,
        "step_100": {
            "executed": True,
            "training_loss": training["training_metrics"]["loss"],
            "validation_loss": training["training_metrics"]["validation_loss"],
            "training_rss_mb": training["training_metrics"]["memory_usage_mb"],
            "development_generation": {
                key: training["development_generation_probe"][key]
                for key in ("questions", "natural_rate", "mean_characters", "eos_rate",
                            "mean_tokens_per_second", "text_generation_established")
            },
            "training_checkpoint_mb": mib(training["training_checkpoint_bytes"]),
            "inference_checkpoint_mb": mib(training["inference_checkpoint_bytes"]),
            "gate": "FAIL",
        },
        "step_500": {
            "executed": False,
            "reason": "Step-100 text-generation gate failed (0/8 visible outputs); mandatory early stop applied.",
        },
        "blind_200": {
            "sealed_sha256": comparison["blind_sha256"],
            "evaluated_standard_step": 100,
            "campus_v23": comparison["campus_v23"],
            "standard": comparison["standard"],
            "axis_delta_standard_minus_campus": comparison["axis_delta_standard_minus_campus"],
            "standard_clearly_improved": comparison["standard_clearly_improved"],
        },
        "standard_50m_continuation_value": False,
        "next_training_step": None,
        "decision": "STOP at step 100; do not run 500/1000/2000/5000 in this phase.",
    }
    write(OUT / "summary.json", payload)
    campus = comparison["campus_v23"]
    standard = comparison["standard"]
    axes = "\n".join(
        f"| {axis} | {campus['axis_percent'][axis]:.2f}% | {standard['axis_percent'][axis]:.2f}% | "
        f"{comparison['axis_delta_standard_minus_campus'][axis]:+.2f}pt |"
        for axis in ("correctness", "relevance", "completeness", "specificity", "naturalness", "actionable")
    )
    (OUT / "report.md").write_text(
        "# UniPilot Standard 50M Short Validation\n\n"
        "Campus v2.3と本番v0.4を保持し、外部AI/APIなしで短時間検証だけを実施した。\n\n"
        "## 結果\n\n"
        f"- TOEIC既知重大誤回答: {toeic['critical_errors_before']} -> "
        f"{toeic['critical_errors_after']}\n"
        f"- 採用候補: 45,445,120 parameters / vocab 2048 / context 512\n"
        f"- 100step: train loss {training['training_metrics']['loss']:.4f}, validation loss "
        f"{training['training_metrics']['validation_loss']:.4f}, 可視文章 0/8、Gate FAIL\n"
        "- 500step: 未実行（100step文章成立条件を満たさないため）\n\n"
        "## Independent Blind 200\n\n"
        "| Axis | Campus v2.3 | Standard step100 | Delta |\n"
        "|---|---:|---:|---:|\n"
        f"{axes}\n\n"
        f"- Campus: critical {campus['critical_errors']}, hallucination "
        f"{campus['hallucination_rate'] * 100:.2f}%, unsupported {campus['unsupported_claim_rate'] * 100:.2f}%, "
        f"first response {campus['average_first_token_seconds']:.3f}s, peak RSS {campus['peak_rss_mb']:.1f}MB\n"
        f"- Standard: raw generation success {standard['raw_generation_success_rate'] * 100:.2f}%, "
        f"validator fallback {standard['fallback_rate'] * 100:.2f}%, first token "
        f"{standard['average_first_token_seconds']:.3f}s, {standard['average_tokens_per_second']:.2f} tok/s, "
        f"peak RSS {standard['peak_rss_mb']:.1f}MB\n\n"
        "## 判定\n\n"
        "Standard 50M継続価値: **NO**。次の学習step: **なし（step100で停止）**。"
        "500/1000/2000/5000 step計画は作成しない。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "summary": str(OUT / "summary.json"), "report": str(OUT / "report.md"),
        "decision": payload["decision"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
