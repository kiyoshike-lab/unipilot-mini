from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from scripts.build_campus_v22_generalization import PROFILES, ngrams, similarity


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/standard_50m_short"
SEED = 5082026
STAGE_COUNTS = {"A": 800, "B": 800, "C": 1000, "D": 1000, "E": 200, "F": 200}
FORMS = (
    "{time}までに{case}状態です。最初の十五分で確認することと、その後の進め方を分けてください。",
    "{case}のですが、自己判断で決めつけたくありません。確認先と、問い合わせ前にそろえる情報は何ですか？",
    "{case}。急ぐ対応、待ってよい対応、記録しておく内容の三つに整理してほしいです。",
    "{time}の相談です。{case}とき、避けるべき判断を一つ挙げてから安全な手順を教えてください。",
    "{case}ので、今日できる一歩と、回答が来なかった場合の次の一歩を示してください。",
    "{case}。一般論だけで終わらず、判断条件と実行順を短いチェックリストにできますか？",
    "{time}です。{case}ため、事実・未確認事項・次の行動に分けて整理してください。",
    "{case}。公式情報が見つからない場合も含め、断定せずに進める方法を教えてください。",
    "{case}とき、いま手元にある情報だけで準備できることと、追加で聞く質問を分けてください。",
    "{time}に{case}状況です。優先順位の理由も一文ずつ添えてください。",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def known_questions() -> list[str]:
    questions: list[str] = []
    for path in (
        ROOT / "evaluation/human-comparison-campus-v21.json",
        ROOT / "data/campus_v22/generalization/blind-300.json",
        ROOT / "data/campus_v22/generalization/stress-100.json",
        ROOT / "data/campus_v23/holdouts/blind-500.json",
        ROOT / "data/campus_v23/holdouts/stress-200.json",
        ROOT / "data/v08/blind/evaluation.json",
    ):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("items", payload) if isinstance(payload, dict) else payload
        questions.extend(str(row.get("question") or row.get("prompt") or "") for row in rows)
    for path in (ROOT / "data/v08/curriculum").glob("*/train.jsonl"):
        questions.extend(str(row.get("user") or "") for row in read_jsonl(path))
    return [value for value in questions if value]


def build_blind(references: list[str]) -> tuple[list[dict], dict]:
    reference_grams = [ngrams(value) for value in references]
    inverted: dict[str, set[int]] = {}
    for index, grams in enumerate(reference_grams):
        for gram in grams:
            inverted.setdefault(gram, set()).add(index)

    def reference_similarity(question: str) -> float:
        query = ngrams(question)
        possible: set[int] = set()
        for gram in query:
            possible.update(inverted.get(gram, ()))
        return max((len(query & reference_grams[index]) / len(query | reference_grams[index])
                    for index in possible), default=0.0)

    candidates = []
    for category, (label, situations, times) in PROFILES.items():
        for index, form in enumerate(FORMS):
            case = situations[(index * 3 + 1) % len(situations)]
            time_value = times[(index * 2 + 1) % len(times)]
            question = form.format(label=label, case=case, time=time_value)
            candidates.append({
                "question": question,
                "expected_category": category,
                "difficulty": "hard" if index in (1, 3, 7, 8) else "medium",
                "surface": f"standard-short-independent-{index}",
            })
    selected: list[dict] = []
    counts: Counter[str] = Counter()
    for round_index in range(len(FORMS)):
        for category in PROFILES:
            row = next(item for item in candidates if item["expected_category"] == category
                       and item["surface"].endswith(f"-{round_index}"))
            reference_max = reference_similarity(row["question"])
            internal_max = max((similarity(row["question"], item["question"]) for item in selected), default=0.0)
            if reference_max >= .78 or internal_max >= .88:
                continue
            selected.append({
                **row,
                "holdout": True,
                "sealed_before_training": True,
                "forbidden_for_training": True,
                "max_reference_similarity": round(reference_max, 4),
                "max_internal_similarity": round(internal_max, 4),
            })
            counts[category] += 1
            if len(selected) == 200:
                break
        if len(selected) == 200:
            break
    if len(selected) != 200:
        raise RuntimeError(f"independent Blind target 200, got {len(selected)}")
    for index, row in enumerate(selected, 1):
        row["id"] = f"standard-50m-short-blind-{index:03d}"
    return selected, {
        "reference_questions": len(references),
        "maximum_reference_similarity": max(row["max_reference_similarity"] for row in selected),
        "maximum_internal_similarity": max(row["max_internal_similarity"] for row in selected),
        "category_counts": dict(sorted(counts.items())),
    }


def balanced_curriculum(split: str) -> list[dict]:
    selected = []
    for stage, requested in STAGE_COUNTS.items():
        rows = read_jsonl(ROOT / f"data/v08/curriculum/{stage}/{split}.jsonl")
        random.Random(SEED + ord(stage) + (0 if split == "train" else 10_000)).shuffle(rows)
        take = min(requested if split == "train" else max(20, requested // 10), len(rows))
        selected.extend(rows[:take])
    random.Random(SEED + (0 if split == "train" else 20_000)).shuffle(selected)
    return selected


def main() -> int:
    references = known_questions()
    blind, duplicate_report = build_blind(references)
    train = balanced_curriculum("train")
    validation = balanced_curriculum("validation")
    write_jsonl(OUT / "curriculum/train.jsonl", train)
    write_jsonl(OUT / "curriculum/validation.jsonl", validation)
    blind_path = OUT / "blind-200.json"
    blind_payload = {
        "schema_version": "unipilot-standard-50m-short-blind-200-v1",
        "holdout": True,
        "sealed_before_training": True,
        "used_for_hyperparameter_selection": False,
        "used_for_training": False,
        "items": blind,
    }
    blind_path.parent.mkdir(parents=True, exist_ok=True)
    blind_path.write_text(json.dumps(blind_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha256 = hashlib.sha256(blind_path.read_bytes()).hexdigest()
    manifest = {
        "dataset_version": "unipilot-standard-50m-short-balanced-v1",
        "seed": SEED,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "stage_training_counts": dict(Counter(row.get("stage", "unknown").split("-")[0] for row in train)),
        "blind_questions": len(blind),
        "blind_sha256_at_seal": sha256,
        "blind_deduplication": duplicate_report,
        "external_ai_api": "OFF",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
