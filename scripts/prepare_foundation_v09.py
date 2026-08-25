from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Iterable

from scripts.build_campus_v22_generalization import PROFILES, ngrams


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/foundation_v09"
EVAL_OUT = ROOT / "evaluation"
SEED = 5092026
SYSTEM = (
    "あなたは大学生を支援する完全ローカルのUniPilotです。最初に質問へ直接答え、理由、具体策、"
    "次の行動を示します。不足情報や大学固有制度は推測せず、確認方法を案内します。"
)
EVAL_FORMS = (
    "{time}までに{case}状態です。最初の行動と、その判断理由を教えてください。",
    "{case}のですが、自己判断で決めつけずに進める確認順を示してください。",
    "{case}。今日・今週・返答待ちの三段階に分けるとどうなりますか？",
    "{time}の相談です。{case}とき、避ける判断と安全な手順を分けてください。",
    "{case}ので、今すぐできる一歩と、解決しない場合の次の一歩を知りたいです。",
    "{case}。判断条件と実行順を、理由付きのチェックリストにしてください。",
    "{time}です。{case}ため、事実・未確認事項・次の行動に整理してください。",
    "{case}。公式情報が見つからない場合も含め、断定せずに答えてください。",
    "{case}とき、手元の情報だけで準備できることと、追加で聞く質問は何ですか？",
    "{time}に{case}状況です。優先順位と見直すタイミングを教えてください。",
    "{case}。結論を先に、理由と具体策を続けて説明してください。",
    "{time}まで余裕がありません。{case}場合に最低限守ることは何ですか？",
)
EVAL_MODIFIERS = (
    "質問にない科目や制度は補わないでください。",
    "大学ごとの違いがある点は確認先も示してください。",
    "短すぎない範囲で、読みやすく答えてください。",
    "具体例を一つだけ添えてください。",
    "複数の対応があるなら、先に行う順番も示してください。",
)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def stable_split(key: str) -> str:
    value = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20
    return "train" if value < 17 else ("validation" if value < 19 else "test")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def chunks(text: str, maximum: int = 260, minimum: int = 40) -> list[str]:
    value = clean_text(text)
    sentences = [piece.strip() for piece in re.split(r"(?<=[。！？])", value) if piece.strip()]
    output: list[str] = []
    current = ""
    for sentence in sentences:
        pieces = [sentence[index:index + maximum] for index in range(0, len(sentence), maximum)]
        for piece in pieces:
            if current and len(current) + len(piece) > maximum:
                if len(current) >= minimum:
                    output.append(current)
                current = ""
            current += piece
    if len(current) >= minimum:
        output.append(current)
    return output


def build_language_rows() -> tuple[list[dict], list[dict], dict]:
    base_rows = []
    for source in read_jsonl(ROOT / "data/campus_v22/knowledge/wikipedia.jsonl"):
        for index, text in enumerate(chunks(source["text"])):
            row_id = f"foundation-v09-base-{source['id']}-{index:02d}"
            base_rows.append({
                "id": row_id, "kind": "text", "text": f"{source['title']}。{text}",
                "domain": "general_japanese_pretraining", "split": stable_split(row_id),
                "source": source["source"], "publisher": source["publisher"],
                "source_url": source["source_url"], "retrieved_at": source["retrieved_at"],
                "license": source["license"], "license_url": source["license_url"],
                "revision_id": source.get("revision_id"), "training_role": "language_pretraining",
            })
    campus_rows = []
    for source in read_jsonl(ROOT / "data/v08/knowledge/documents.jsonl"):
        if source.get("source_type") != "project_authored_standard_knowledge":
            continue
        for index, text in enumerate(chunks(source["text"])):
            row_id = f"foundation-v09-campus-{source['id']}-{index:02d}"
            campus_rows.append({
                "id": row_id, "kind": "text", "text": f"{source['title']}。{text}",
                "domain": source["category"], "split": stable_split(row_id),
                "source": source["source"], "source_url": source.get("source_url"),
                "retrieved_at": source["retrieved_at"], "license": source["license"],
                "training_role": "stable_campus_pretraining",
            })
    rag_only = []
    for path in (ROOT / "data/campus_v22/knowledge/government.jsonl",
                 ROOT / "data/campus_v22/knowledge/university.jsonl"):
        for source in read_jsonl(path):
            rag_only.append({
                "id": source["id"], "category": source["category"], "title": source["title"],
                "source_url": source["source_url"], "publisher": source["publisher"],
                "retrieved_at": source["retrieved_at"], "license": source["license"],
                "scope": source.get("sub_category"), "freshness": source.get("last_verified_at"),
                "training_role": "rag_only_variable_or_institution_specific",
            })
    return base_rows, campus_rows, {
        "rag_only_documents": len(rag_only),
        "rag_only_by_license": dict(Counter(row["license"] for row in rag_only)),
        "rag_only": rag_only,
    }


def known_questions() -> list[str]:
    questions = []
    paths = (
        ROOT / "evaluation/human-comparison-campus-v21.json",
        ROOT / "data/campus_v22/generalization/blind-300.json",
        ROOT / "data/campus_v22/generalization/stress-100.json",
        ROOT / "data/campus_v23/holdouts/blind-500.json",
        ROOT / "data/campus_v23/holdouts/stress-200.json",
        ROOT / "data/standard_50m_short/blind-200.json",
        ROOT / "data/v08/blind/evaluation.json",
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("items", payload) if isinstance(payload, dict) else payload
        questions.extend(str(row.get("question") or row.get("prompt") or "") for row in rows)
    return [question for question in questions if question]


def ngram_index(values: list[str]) -> tuple[list[set[str]], dict[str, set[int]]]:
    grams = [ngrams(value) for value in values]
    inverted: dict[str, set[int]] = {}
    for index, value in enumerate(grams):
        for gram in value:
            inverted.setdefault(gram, set()).add(index)
    return grams, inverted


def maximum_similarity(question: str, grams: list[set[str]], inverted: dict[str, set[int]]) -> float:
    query = ngrams(question)
    possible: set[int] = set()
    for gram in query:
        possible.update(inverted.get(gram, ()))
    return max((len(query & grams[index]) / len(query | grams[index]) for index in possible), default=0.0)


def build_evaluation() -> tuple[list[dict], list[dict], dict]:
    references = known_questions()
    reference_grams, reference_index = ngram_index(references)
    candidates = []
    for category, (label, situations, times) in PROFILES.items():
        for form_index, form in enumerate(EVAL_FORMS):
            for modifier_index, modifier in enumerate(EVAL_MODIFIERS):
                case = situations[(form_index + modifier_index * 3) % len(situations)]
                time_value = times[(form_index * 2 + modifier_index) % len(times)]
                question = f"{form.format(label=label, case=case, time=time_value)}{modifier}"
                candidates.append({
                    "question": question, "expected_category": category,
                    "difficulty": "hard" if form_index in (1, 3, 7, 8, 11) else "medium",
                    "surface": f"foundation-v09-{form_index:02d}-{modifier_index}",
                    "holdout": True, "forbidden_for_training": True,
                })
    grouped = {category: [] for category in PROFILES}
    for row in candidates:
        grouped[row["expected_category"]].append(row)
    selected: list[dict] = []
    internal_grams: list[set[str]] = []
    internal_index: dict[str, set[int]] = {}
    for round_index in range(max(len(rows) for rows in grouped.values())):
        for category in PROFILES:
            row = grouped[category][round_index]
            ref_max = maximum_similarity(row["question"], reference_grams, reference_index)
            int_max = maximum_similarity(row["question"], internal_grams, internal_index) if selected else 0.0
            if ref_max >= .78 or int_max >= .90:
                continue
            row = {**row, "max_reference_similarity": round(ref_max, 4),
                   "max_internal_similarity": round(int_max, 4)}
            selected.append(row)
            grams = ngrams(row["question"])
            internal_grams.append(grams)
            current = len(internal_grams) - 1
            for gram in grams:
                internal_index.setdefault(gram, set()).add(current)
            if len(selected) == 1200:
                break
        if len(selected) == 1200:
            break
    if len(selected) != 1200:
        raise RuntimeError(f"evaluation target 1200, got {len(selected)}")
    random.Random(SEED).shuffle(selected)
    validation, blind = selected[:200], selected[200:]
    for split, rows in (("validation", validation), ("final-blind", blind)):
        for index, row in enumerate(rows, 1):
            row["id"] = f"foundation-v09-{split}-{index:04d}"
            row["evaluation_split"] = split
            row["sealed_before_training"] = True
    metadata = {
        "known_reference_questions": len(references),
        "maximum_reference_similarity": max(row["max_reference_similarity"] for row in selected),
        "maximum_internal_similarity": max(row["max_internal_similarity"] for row in selected),
        "validation_category_counts": dict(Counter(row["expected_category"] for row in validation)),
        "blind_category_counts": dict(Counter(row["expected_category"] for row in blind)),
    }
    return validation, blind, metadata


def human_approved() -> list[dict]:
    source = json.loads((ROOT / "evaluation/campus-ai-quality-20.json").read_text(encoding="utf-8"))
    rows = []
    for row in source["items"]:
        if row.get("human_rating") != "good":
            continue
        rows.append({
            "id": f"foundation-v09-human-{row['item_id']}", "kind": "conversation",
            "user": row["question"], "assistant": row["original_answer"],
            "category": row["category"], "system": SYSTEM, "split": "train",
            "source": "Campus v2.1 human evaluation", "source_url": None,
            "license": "CC0-1.0", "approval": "human_good",
            "human_rating": "good", "training_role": "human_approved_replay",
        })
    return rows


def instruction_candidates(human_rows: list[dict]) -> list[dict]:
    rows = []
    for stage in ("C", "E"):
        for split in ("train", "validation", "test"):
            for row in read_jsonl(ROOT / f"data/v08/curriculum/{stage}/{split}.jsonl"):
                rows.append({**row, "split": split, "approval": "project_authored_quality_checked",
                             "training_role": "instruction_tuning"})
    for row in read_jsonl(ROOT / "data/campus_v2/faq/reviewed.jsonl"):
        if row.get("quality_score", 0) < 5:
            continue
        row_id = f"foundation-v09-faq-{row['id']}"
        rows.append({
            "id": row_id, "kind": "conversation", "user": row["question"],
            "assistant": row["answer"], "category": row["category"], "system": SYSTEM,
            "split": stable_split(row_id), "source": row["source"], "source_url": row.get("source_url"),
            "license": row["license"], "approval": "auto_reviewed_not_human",
            "training_role": "instruction_tuning", "semantic_scenario": row.get("semantic_scenario"),
        })
    rows.extend(human_rows)
    return rows


def filter_instructions(rows: list[dict], evaluation_rows: list[dict]) -> tuple[list[dict], dict]:
    eval_grams, eval_index = ngram_index([row["question"] for row in evaluation_rows])
    seen = set()
    accepted = []
    exact_duplicate = semantic_holdout = invalid = 0
    for row in rows:
        user, assistant = str(row.get("user", "")).strip(), str(row.get("assistant", "")).strip()
        if not user or not assistant or row.get("license") not in ("CC0-1.0", "CC BY-SA 4.0"):
            invalid += 1
            continue
        fingerprint = normalized(user) + "|" + normalized(assistant)
        if fingerprint in seen:
            exact_duplicate += 1
            continue
        if maximum_similarity(user, eval_grams, eval_index) >= .78:
            semantic_holdout += 1
            continue
        seen.add(fingerprint)
        accepted.append(row)
    return accepted, {
        "input_rows": len(rows), "accepted_rows": len(accepted),
        "exact_duplicate_excluded": exact_duplicate,
        "semantic_holdout_overlap_excluded": semantic_holdout,
        "invalid_or_unlicensed_excluded": invalid,
    }


def write_splits(directory: Path, rows: list[dict]) -> dict:
    counts = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        write_jsonl(directory / f"{split}.jsonl", selected)
        counts[split] = len(selected)
    return counts


def inventory_file(path: Path) -> dict:
    rows = read_jsonl(path)
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"), "rows": len(rows),
        "licenses": dict(Counter(str(row.get("license") or "MISSING") for row in rows)),
        "sources": dict(Counter(str(row.get("source") or row.get("publisher") or "MISSING") for row in rows)),
    }


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    baseline_files = [
        ROOT / "pipeline/campus_v23.py", ROOT / "pipeline/campus_tools_v23.py",
        ROOT / "evaluation/campus-v23-summary.json", ROOT / "configs/unipilot-v04.json",
    ]
    baseline = {
        "commit": git_head(), "campus_version": "campus-v2.3", "production_model": "v0.4",
        "protected_file_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                                  for path in baseline_files},
        "answer_logic_frozen_after_baseline": True, "production_changed": False,
        "render_changed": False, "vercel_changed": False, "release_changed": False,
    }
    validation, final_blind, evaluation_meta = build_evaluation()
    human_rows = human_approved()
    instructions, filter_report = filter_instructions(
        instruction_candidates(human_rows), [*validation, *final_blind])
    base_rows, campus_rows, rag_report = build_language_rows()
    base_counts = write_splits(OUT / "base", base_rows)
    campus_counts = write_splits(OUT / "campus", campus_rows)
    instruction_counts = write_splits(OUT / "instruction", instructions)
    write_jsonl(OUT / "human-approved.jsonl", human_rows)
    write_json(OUT / "evaluation/validation-200.json", {
        "schema_version": "foundation-v09-validation-200-v1", "holdout": True,
        "sealed_before_training": True, "used_for_training": False, "items": validation,
    })
    blind_path = OUT / "evaluation/final-blind-1000.json"
    write_json(blind_path, {
        "schema_version": "foundation-v09-final-blind-1000-v1", "holdout": True,
        "sealed_before_training": True, "used_for_training": False,
        "opened_for_this_phase": False, "items": final_blind,
    })
    write_json(OUT / "rag-only-source-catalog.json", rag_report)
    source_paths = [
        ROOT / "data/curated/general_japanese.jsonl",
        ROOT / "data/curated/university_text.jsonl",
        ROOT / "data/curated/university_dialogues.jsonl",
        ROOT / "data/campus_v22/knowledge/wikipedia.jsonl",
        ROOT / "data/campus_v22/knowledge/government.jsonl",
        ROOT / "data/campus_v22/knowledge/university.jsonl",
        ROOT / "data/v08/conversation/conversation.jsonl",
        ROOT / "data/v08/corrected/corrected.jsonl",
        ROOT / "data/campus_v2/faq/reviewed.jsonl",
    ]
    inventory = {
        "schema_version": "foundation-v09-data-inventory-v1", "generated_at": generated_at,
        "baseline": baseline, "source_files": [inventory_file(path) for path in source_paths],
        "selected": {
            "base_language": {"rows": len(base_rows), "splits": base_counts,
                              "licenses": dict(Counter(row["license"] for row in base_rows))},
            "campus_stable_pretraining": {"rows": len(campus_rows), "splits": campus_counts,
                                           "licenses": dict(Counter(row["license"] for row in campus_rows))},
            "instruction": {"rows": len(instructions), "splits": instruction_counts,
                            "approval": dict(Counter(row["approval"] for row in instructions)),
                            **filter_report},
            "human_approved": {"rows": len(human_rows), "policy": "Only explicit human rating=good."},
            "rag_only": {key: value for key, value in rag_report.items() if key != "rag_only"},
        },
        "excluded_legacy": {
            "rows": 50_000,
            "files": ["data/curated/general_japanese.jsonl", "data/curated/university_text.jsonl",
                      "data/curated/university_dialogues.jsonl"],
            "reason": "Legacy rule-based template text contains unnatural Japanese and repeated semantic frames; not used for v0.9.",
        },
        "targets": {"instruction_rows": 30_000, "multi_turn_conversations": 10_000,
                    "corrected_rows": "5,000-10,000", "final_blind": 1_000},
        "gaps": {"instruction_rows": max(0, 30_000 - len(instructions)),
                 "multi_turn_conversations": 10_000,
                 "corrected_rows_minimum": max(0, 5_000 - 3_300)},
        "evaluation": {**evaluation_meta, "validation": 200, "final_blind": 1000,
                       "final_blind_sha256_at_seal": sha256(blind_path)},
        "external_llm_used": False, "external_pretrained_model_used": False,
    }
    write_json(EVAL_OUT / "foundation-v09-data-inventory.json", inventory)
    manifest = {
        "schema_version": "foundation-v09-manifest-v1", "generated_at": generated_at,
        "baseline": baseline, "base": base_counts, "campus": campus_counts,
        "instruction": instruction_counts, "human_approved": len(human_rows),
        "rag_only_documents": rag_report["rag_only_documents"],
        "evaluation": {"validation": "data/foundation_v09/evaluation/validation-200.json",
                       "final_blind": "data/foundation_v09/evaluation/final-blind-1000.json",
                       "final_blind_sha256": sha256(blind_path), "final_blind_opened": False},
        "language_pretraining_separated_from_instruction": True,
        "variable_facts_rag_only": True, "external_ai_api": "OFF",
        "production_enabled": False,
    }
    write_json(OUT / "manifest.json", manifest)
    (EVAL_OUT / "foundation-v09-data-design.md").write_text(
        "# UniPilot Foundation v0.9 Data Design\n\n"
        "- Base: CC BY-SA 4.0の日本語Wikipedia本文のみをlanguage modeling形式で学習する。\n"
        "- Campus: project-authored CC0の安定した大学生活知識だけを追加pretrainingする。\n"
        "- Instruction: project-authored品質確認済みQ&AとHuman ◎を別stageで学習する。\n"
        "- RAG-only: 政府・大学公式の制度、数字、期限はweightへ入れずsource metadata付きで検索する。\n"
        "- Human: 明示的に◎の回答だけをapproved replayへ入れる。AI改善案や△は自動採用しない。\n"
        "- Evaluation: validation 200と未開封final Blind 1,000を学習前に固定し、trainとの近似重複を除外する。\n\n"
        f"Instruction 30,000目標に対する未達: {inventory['gaps']['instruction_rows']}件。"
        "近似文の水増しでは埋めず、Human reviewと新しい意味単位を追加する。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "base": len(base_rows), "campus": len(campus_rows), "instruction": len(instructions),
        "human_approved": len(human_rows), "rag_only": rag_report["rag_only_documents"],
        "validation": len(validation), "final_blind": len(final_blind),
        "duplicates": filter_report, "final_blind_sha256": sha256(blind_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
