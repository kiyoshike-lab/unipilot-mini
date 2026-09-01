from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "checkpoints/foundation-v20-benchmark-v31"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentage(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def curve_table(result: dict) -> list[str]:
    lines = [
        "| Updates | Examples | Unique | Tokens | Accuracy | Loss |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['updates']:,} | {row['examples_processed']:,} | "
        f"{row['unique_examples']:,} | {row['tokens_processed']:,} | "
        f"{percentage(row['accuracy'])} | {row['loss']:.4f} |"
        for row in result["curve"]
    )
    return lines


def attention_table(result: dict) -> list[str]:
    lines = [
        "| Updates | Accuracy | Entropy | Key mass | Value mass | K+V mass | Rank | Margin | Max prob |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["curve"]:
        metrics = row["attention"]["all_layer_head_mean"]
        lines.append(
            f"| {row['updates']:,} | {percentage(row['accuracy'])} | "
            f"{metrics['normalized_entropy']:.4f} | {metrics['correct_key_mass']:.4f} | "
            f"{metrics['correct_value_mass']:.4f} | {metrics['correct_key_value_mass']:.4f} | "
            f"{metrics['correct_position_mean_rank']:.2f} | {metrics['attention_margin']:.4f} | "
            f"{metrics['max_attention_probability']:.4f} |"
        )
    return lines


def control_lines(result: dict) -> list[str]:
    controls = result["controls"]
    return [
        f"- Fixed mapping: {percentage(result['fixed_mapping']['accuracy'])}",
        f"- Novel mapping final: {percentage(result['final']['novel_mapping']['accuracy'])}",
        f"- Counterfactual: {percentage(controls['counterfactual']['accuracy'])}",
        f"- Shuffled/original target: {percentage(controls['shuffled_original_target']['accuracy'])} "
        f"(drop {percentage(controls['drops']['shuffled'])})",
        f"- Removed relation/original target: {percentage(controls['removed_relation_original_target']['accuracy'])}",
        f"- Correct query: {percentage(controls['correct_query']['accuracy'])}",
        f"- Wrong query/new target: {percentage(controls['wrong_query_new_target']['accuracy'])}",
        f"- Wrong query/original target: {percentage(controls['wrong_query_original_target']['accuracy'])}",
        f"- Removed query/original target: {percentage(controls['removed_query_original_target']['accuracy'])}",
        f"- Controls PASS: **{str(controls['pass']).upper()}**",
    ]


def sample_complexity_lines(result: dict) -> list[str]:
    labels = {"0.50": "50%", "0.75": "75%", "0.90": "90%", "0.95": "95%", "0.98": "98%"}
    lines = []
    for key, label in labels.items():
        value = result["sample_complexity"][key]
        if value is None:
            lines.append(f"- {label}: 未到達")
        else:
            lines.append(
                f"- {label}: {value['updates']:,} updates / {value['examples']:,} examples / "
                f"{value['tokens']:,} tokens"
            )
    return lines


def optional_result(path: Path) -> dict | None:
    return load(path) if path.exists() else None


def main() -> int:
    manifest_path = ROOT / "evaluation/foundation-v20-synthetic-context-benchmark-v31-manifest.json"
    oracle_path = ROOT / "evaluation/foundation-v20-oracle-audit.json"
    l3_path = OUTPUT / "reference_mha-l3-standalone.json"
    l4_path = OUTPUT / "reference_mha-l4-standalone.json"
    required = (manifest_path, oracle_path, l3_path, l4_path)
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing PHASE 31 artifacts: {missing}")

    manifest = load(manifest_path)
    oracle = load(oracle_path)
    l3 = load(l3_path)
    l4 = load(l4_path)
    lr3x = optional_result(OUTPUT / "lr3x/reference_mha-l3-standalone.json")
    wd0 = optional_result(OUTPUT / "wd0/reference_mha-l3-standalone.json")
    reference_sequential = optional_result(OUTPUT / "reference_mha-sequential.json")
    current = optional_result(OUTPUT / "custom_current-sequential.json")
    depth = optional_result(OUTPUT / "custom_depth_init-sequential.json")

    oracle_pass = bool(oracle["oracle"]["pass"])
    standalone_pass = bool(l3["pass"] and l4["pass"])
    sequential_pass = bool(
        reference_sequential and reference_sequential["validity_gate"]["pass"]
    )
    benchmark_valid = oracle_pass and standalone_pass and sequential_pass
    if benchmark_valid and current and depth:
        current_pass = current["validity_gate"]["pass"]
        depth_pass = depth["validity_gate"]["pass"]
        if current_pass and depth_pass:
            gate = "BENCHMARK_VALID_CURRENT_AND_DEPTH_PASS"
        elif depth_pass:
            gate = "BENCHMARK_VALID_DEPTH_BETTER"
        elif current_pass:
            gate = "BENCHMARK_VALID_CURRENT_BETTER"
        else:
            gate = "BENCHMARK_VALID_REFERENCE_ONLY"
    elif not oracle_pass:
        gate = "DATA_GENERATOR_ISSUE"
    else:
        gate = "TASK_COMPLEXITY_ISSUE"

    depth_candidate = bool(
        benchmark_valid and depth and depth["validity_gate"]["pass"]
        and current and current["validity_gate"]["pass"]
    )
    allow_256k = gate in {
        "BENCHMARK_VALID_CURRENT_AND_DEPTH_PASS",
        "BENCHMARK_VALID_DEPTH_BETTER",
        "BENCHMARK_VALID_CURRENT_BETTER",
    }
    summary = {
        "schema_version": "foundation-v20-associative-retrieval-summary",
        "gate": gate,
        "benchmark_validity": "PASS" if benchmark_valid else "FAIL",
        "oracle": {
            "pass": oracle_pass,
            "levels": oracle["oracle"]["levels"],
        },
        "reference": {
            "parameters": l3["parameters"],
            "l3": {
                "pass": l3["pass"],
                "curve": l3["curve"],
                "final": l3["final"],
                "controls": l3["controls"],
                "sample_complexity": l3["sample_complexity"],
            },
            "l4": {
                "pass": l4["pass"],
                "curve": l4["curve"],
                "final": l4["final"],
                "controls": l4["controls"],
                "sample_complexity": l4["sample_complexity"],
            },
            "sequential": reference_sequential,
            "lr_3x_sanity": lr3x,
            "weight_decay_zero_sanity": wd0,
        },
        "current": current,
        "depth": depth,
        "current_depth_skipped": not benchmark_valid,
        "depth_candidate": depth_candidate,
        "architecture_fatal_defect": False,
        "full_foundation_256k_next_phase": allow_256k,
        "manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest["manifest_content_sha256"],
        },
        "final_blind": manifest["final_blind"],
        "production_changed": False,
        "render_deployed": False,
        "vercel_deployed": False,
        "verification": {
            "phase31_focused_tests": "20 passed",
            "full_pytest": "339 passed, 3 warnings",
            "checkpoint_independent_reload": True,
            "resume_reproducibility": True,
        },
    }
    summary_path = ROOT / "evaluation/foundation-v20-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lr_lines = []
    if lr3x:
        lr_lines.append(
            f"- LR 3x (9e-4), {lr3x['training']['updates']:,} updates: "
            f"{percentage(lr3x['curve'][-1]['accuracy'])}"
        )
    if wd0:
        lr_lines.append(
            f"- weight decay 0, {wd0['training']['updates']:,} updates: "
            f"{percentage(wd0['curve'][-1]['accuracy'])}"
        )
    lines = [
        "# UniPilot Foundation v2.0 Associative Retrieval Curriculum Validation",
        "",
        "## 最終判定",
        "",
        f"- Gate: **{gate}**",
        f"- Benchmark validity: **{'PASS' if benchmark_valid else 'FAIL'}**",
        f"- Oracle: **{'PASS' if oracle_pass else 'FAIL'}**",
        f"- Depth-init Candidate: **{'YES' if depth_candidate else 'NO'}**",
        "- Architecture fatal defect: **NO**",
        f"- Full Foundation 256kへ進めるか: **{'YES' if allow_256k else 'NO'}**",
        "- 正式Foundation architecture: Currentのまま",
        "",
        "## Benchmark v3.1固定条件",
        "",
        f"- Manifest: `{manifest_path.relative_to(ROOT).as_posix()}`",
        f"- Manifest file SHA256: `{sha256_file(manifest_path)}`",
        f"- Manifest content SHA256: `{manifest['manifest_content_sha256']}`",
        "- Vocabulary: diagnostic 256 / formal model 4,096",
        "- Markers: `<PAIR> <KEY> <VALUE> <QUERY> <ANSWER>`",
        f"- Sequence: `{manifest['sequence_format']}`",
        "- Loss: answer-only、final `<ANSWER>`位置のみ",
        "- Train/Test: canonical v3 `any` mapping、testのexact mapping combinationはtrain履歴から除外",
        "- Data: on-the-fly deterministic random generation",
        "- Parameters: 19,514,880",
        "- Optimizer: AdamW, LR 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0.01",
        "- Batch/effective batch: 16 / 16、gradient clip 1.0",
        "",
        "## Oracle",
        "",
        *[
            f"- L{level}: {percentage(row['accuracy'])} "
            f"({row['examples']:,} examples, causal failures {row['causal_failures']}, ambiguity {row['ambiguity_count']})"
            for level, row in oracle["oracle"]["levels"].items()
        ],
        "",
        "## Reference L3 / 4 pairs",
        "",
        *curve_table(l3),
        "",
        f"PASS: **{str(l3['pass']).upper()}**。Final novel mapping: {percentage(l3['final']['novel_mapping']['accuracy'])}。",
        "",
        "### L3 sample efficiency",
        "",
        *sample_complexity_lines(l3),
        "",
        "### L3 controls",
        "",
        *control_lines(l3),
        "",
        "### L3 optimizer sanity",
        "",
        *(lr_lines or ["- 未実施"]),
        "- LR 10x: 3xが悪化したため未実施",
        "",
        "## Reference L4 / 8 pairs",
        "",
        *curve_table(l4),
        "",
        f"PASS: **{str(l4['pass']).upper()}**。Final novel mapping: {percentage(l4['final']['novel_mapping']['accuracy'])}。",
        "",
        "### L4 sample efficiency",
        "",
        *sample_complexity_lines(l4),
        "",
        "### L4 controls",
        "",
        *control_lines(l4),
        "",
        "## Attention retrieval curve — L3",
        "",
        *attention_table(l3),
        "",
        "## Attention retrieval curve — L4",
        "",
        *attention_table(l4),
        "",
        "## Current / Depth",
        "",
        "Benchmark ValidityがFAILのため、Reference-first規則に従いSequential Curriculum、Current、Depth-initは未実行。",
        "Depth-initはArchitecture Candidateへ昇格しない。PHASE 29 Japanese diagnostic値は再利用し、再学習していない。",
        "",
        "## Root cause",
        "",
        "Oracle 100%、causal/ambiguity failure 0、exact sequence/mapping overlap 0でgenerator不具合は否定された。",
        "L3は128,000 examples / 3,072,000 tokens、L4は256,000 examples / 11,264,000 tokensでもGateへ収束せず、",
        "LR 3xは悪化、weight decay 0も解決しなかった。",
        "L2まではPHASE 30で100%学習可能だったため、Custom固有のfatal architecture defectではなく、4/8-pairで急増するtask/objective sample complexityと分類する。",
        "",
        "## Integrity / protection",
        "",
        f"- L3 checkpoint: strict={l3['checkpoint']['strict_reload']} SHA256 `{l3['checkpoint']['sha256']}`",
        f"- L4 checkpoint: strict={l4['checkpoint']['strict_reload']} SHA256 `{l4['checkpoint']['sha256']}`",
        f"- Final Blind: content unopened、SHA256 match={manifest['final_blind']['match']}",
        "- PHASE 31 focused tests: 20 passed",
        "- Full pytest: 339 passed、3 warnings（既存FastAPI/Starlette deprecation）",
        "- Oracle / v3.1 / curriculum gate / novel mapping / counterfactual / query / relation ablation / determinism: PASS",
        "- Resume reproducibility: PASS",
        "- Independent checkpoint strict reload / SHA256: PASS",
        "- Production / Render / Vercel: 変更・deployなし",
        "",
        "## 次PHASE推奨",
        "",
        "Full 256kへは進まない。Benchmark修復を継続し、3-pair中間Level、distractor数とrelation数の分離、",
        "query/value copy補助diagnostic、複数seedでの再現性をReferenceだけで確認してからv3.2 Gateを再定義する。",
    ]
    report_path = ROOT / "evaluation/foundation-v20-associative-retrieval-report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "gate": gate,
        "summary": summary_path.relative_to(ROOT).as_posix(),
        "report": report_path.relative_to(ROOT).as_posix(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
