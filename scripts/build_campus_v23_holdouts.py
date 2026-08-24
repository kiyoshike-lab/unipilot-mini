#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_campus_v22_generalization import PROFILES


OUT = ROOT / "data/campus_v23/holdouts"

SURFACES = (
    "short", "colloquial", "typo", "incomplete", "specific", "long", "compound", "negation",
    "official_conflict", "followup_like", "direct", "constraint",
)

CONSTRAINTS = (
    "大学サイトがメンテ中", "スマホしか使えない", "担当者の返信待ち", "授業とバイトの間しか時間がない",
    "初めてで用語も曖昧", "期限の時刻だけ分からない", "公式案内が二つある", "学部のルールか全学ルールか不明",
)

ASKS = (
    "最初に何を見ればいい？", "今日できる順にして", "確認先と伝える内容を分けて",
    "決めつけずに動き方を教えて", "見落とし防止のチェック項目がほしい", "待っている間にできることある？",
)

TYPO_MAP = (("レポート", "れぽーと"), ("メール", "メル"), ("履修", "りしゅ"),
            ("確認", "かくにん"), ("提出", "ていしゅつ"), ("試験", "しけん"))


def normalise(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def ngrams(text: str, size: int = 3) -> set[str]:
    value = normalise(text)
    return {value[index:index + size] for index in range(max(1, len(value) - size + 1))}


def similarity(left: str, right: str) -> float:
    a, b = ngrams(left), ngrams(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def typo(text: str) -> str:
    for before, after in TYPO_MAP:
        if before in text:
            return text.replace(before, after, 1)
    return text + " どすればいい"


def references() -> list[str]:
    questions: list[str] = []
    human_100: list[dict] = []
    for path in (
        ROOT / "evaluation/human-comparison-campus-v21.json",
        ROOT / "data/campus_v22/generalization/blind-300.json",
        ROOT / "data/campus_v22/generalization/stress-100.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("items", [])
        if path.name == "human-comparison-campus-v21.json":
            human_100 = rows
        questions.extend(row["question"] for row in rows)
    # The fixed quick 20 are IDs selected from the fixed human 100, so their question
    # text is already in the reference pool. Validate that relationship explicitly.
    quick = json.loads((ROOT / "evaluation/campus-v21-quick-human-ratings-snapshot.json").read_text(encoding="utf-8"))
    human_ids = {row["id"] for row in human_100}
    if any(row["item_id"] not in human_ids for row in quick["items"]):
        raise RuntimeError("quick-20 contains an item outside the fixed human-100 source")
    return list(dict.fromkeys(questions))


def blind_candidates() -> list[dict[str, Any]]:
    categories = list(PROFILES)
    rows: list[dict[str, Any]] = []
    for category_index, (category, (label, situations, contexts)) in enumerate(PROFILES.items()):
        for index in range(192):
            situation = situations[(index + 1) % len(situations)]
            second = situations[(index + 2) % len(situations)]
            context = contexts[(index + 1) % len(contexts)]
            constraint = CONSTRAINTS[(category_index + index) % len(CONSTRAINTS)]
            ask = ASKS[(category_index * 2 + index) % len(ASKS)]
            surface = SURFACES[index % len(SURFACES)]
            if surface == "short":
                question = f"{label}、{constraint}。{ask}"
            elif surface == "colloquial":
                question = f"{situation}っぽくて詰みそう。{ask}"
            elif surface == "typo":
                question = typo(f"{label}で{situation}。{ask}")
            elif surface == "incomplete":
                question = f"{label}の件、まだ{context}くらいしか分からん。今準備できることだけ教えて"
            elif surface == "specific":
                question = f"{situation}。{constraint}なので、見る資料、確認先、伝える情報を順番に整理して"
            elif surface == "long":
                question = (f"{context}までに対応したいです。{situation}うえに、{constraint}状態です。"
                            "確定している事実と未確認事項を分けて、今すぐ・今日・必要ならの行動を説明してください。")
            elif surface == "compound":
                other = categories[(category_index + 11) % len(categories)]
                other_label = PROFILES[other][0]
                question = f"{situation}。同時に{other_label}で{second}。どちらを先にし、両方どう進める？"
            elif surface == "negation":
                question = f"一般論だけじゃなく、{situation}場合に何を記録してどこへ聞くか知りたい"
            elif surface == "official_conflict":
                question = f"{label}で公式案内とLMSの説明が違う。差分をどう確認して、誰に何を伝える？"
            elif surface == "followup_like":
                question = f"さっきの{label}、条件が{constraint}だった。次に変えるべき行動は？"
            elif surface == "direct":
                question = f"{second}ときに、根拠のない制度や数字を作らず具体策を出して"
            else:
                question = f"{constraint}だけど{situation}。できないことを除いて代替手順を作って"
            rows.append({
                "question": question, "expected_category": category, "surface": surface,
                "difficulty": "hard" if surface in ("long", "compound", "official_conflict", "constraint") else "medium",
                "holdout": True, "used_for_improvement": False,
                "forbidden_for_training": True, "forbidden_for_faq_tuning": True,
            })
        rows.append({
            "question": (
                f"{label}で『{situations[0]}』場合と『{situations[-1]}』場合は、"
                "確認先や順番が変わる？共通の準備と分ける対応を整理して"
            ),
            "expected_category": category, "surface": "contrast", "difficulty": "hard",
            "holdout": True, "used_for_improvement": False,
            "forbidden_for_training": True, "forbidden_for_faq_tuning": True,
        })
    return rows


def build_blind(total: int = 500) -> tuple[list[dict], dict]:
    reference = references()
    selected: list[dict] = []
    counts: Counter[str] = Counter()
    grouped = {category: [] for category in PROFILES}
    for row in blind_candidates():
        grouped[row["expected_category"]].append(row)
    candidates = [
        grouped[category][round_index]
        for round_index in range(max(map(len, grouped.values())))
        for category in PROFILES
        if round_index < len(grouped[category])
    ]
    reference_keys = {normalise(value) for value in reference}
    for row in candidates:
        category = row["expected_category"]
        target = 18 if list(PROFILES).index(category) < 7 else 17
        if counts[category] >= target:
            continue
        ref_max = max((similarity(row["question"], value) for value in reference), default=0.0)
        internal_max = max((similarity(row["question"], item["question"]) for item in selected), default=0.0)
        if normalise(row["question"]) in reference_keys or ref_max >= .70 or internal_max >= .90:
            continue
        selected.append({**row, "max_reference_similarity": round(ref_max, 4),
                         "max_internal_similarity": round(internal_max, 4)})
        counts[category] += 1
        if len(selected) == total:
            break
    if len(selected) != total:
        raise RuntimeError(
            f"only {len(selected)} blind questions survived filtering; counts={dict(counts)}"
        )
    for index, row in enumerate(selected, 1):
        row["id"] = f"campus-v23-blind-{index:03d}"
    return selected, {
        "reference_questions": len(reference), "semantic_threshold": .70,
        "internal_near_duplicate_threshold": .90,
        "maximum_reference_similarity": max(row["max_reference_similarity"] for row in selected),
        "maximum_internal_similarity": max(row["max_internal_similarity"] for row in selected),
        "category_counts": dict(sorted(counts.items())),
        "surface_counts": dict(Counter(row["surface"] for row in selected)),
    }


def build_stress() -> list[dict]:
    stress_types = (
        "fictional_policy", "false_premise", "numeric_leading", "university_missing", "vague_deadline",
        "multi_intent", "ultra_short", "typo", "retrieval_collision", "similar_policy_name",
    )
    categories = list(PROFILES)
    rows = []
    for type_index, stress_type in enumerate(stress_types):
        for index in range(20):
            category = categories[(type_index * 5 + index) % len(categories)]
            label, situations, _ = PROFILES[category]
            situation = situations[(index + type_index) % len(situations)]
            if stress_type == "fictional_policy":
                question = f"大学の『全学生自動救済規則2027』なら{label}は申請なしで免除ですよね"
            elif stress_type == "false_premise":
                question = f"{label}は全国共通で取り消せない前提で、抜け道だけ教えて"
            elif stress_type == "numeric_leading":
                question = f"{label}は必ず7回まで、締切は30日後ってことで計算していい？"
            elif stress_type == "university_missing":
                question = f"大学名も年度も伏せるけど、うちの{label}の回数と金額を断定して"
            elif stress_type == "vague_deadline":
                question = f"{situation}、たぶん近いうち期限。今から何を確認する？"
            elif stress_type == "multi_intent":
                other = PROFILES[categories[(type_index + index + 13) % len(categories)]][0]
                question = f"{situation}。さらに{other}も今日中。連絡待ち込みで全部の順番を決めて"
            elif stress_type == "ultra_short":
                question = f"{label}詰んだかも"
            elif stress_type == "typo":
                question = typo(f"{label}で{situation}、かくにん先どこ")
            elif stress_type == "retrieval_collision":
                other = PROFILES[categories[(type_index + index + 7) % len(categories)]][0]
                question = f"{label}の『{other}制度』ではなく、{situation}件だけ答えて"
            else:
                question = f"『{label}特別扱い制度』と『{label}特例手続』は同じ？存在確認からして"
            rows.append({
                "id": f"campus-v23-stress-{len(rows) + 1:03d}", "question": question,
                "expected_category": category, "stress_type": stress_type, "holdout": True,
                "used_for_improvement": False, "forbidden_for_training": True,
                "must_not_assert_unverified_policy": stress_type in {
                    "fictional_policy", "false_premise", "numeric_leading", "university_missing", "similar_policy_name",
                },
            })
    return rows


def comparison_set(blind: list[dict]) -> list[dict]:
    selected = blind[::10][:50]
    axes = (
        "correctness", "relevance", "completeness", "actionability", "specificity",
        "naturalness", "student_usefulness",
    )
    return [{
        "id": f"campus-v23-comparison-{index:02d}", "source_blind_id": row["id"],
        "question": row["question"], "category": row["expected_category"],
        "external_answers_fabricated": False,
        "comparisons": {
            name: {
                "blind_slots": {"A": None, "B": None}, "identity_map": None,
                "manual_scores": {axis: None for axis in axes},
            }
            for name in ("unipilot_vs_chatgpt", "unipilot_vs_gemini")
        },
    } for index, row in enumerate(selected, 1)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    blind, deduplication = build_blind()
    stress = build_stress()
    comparison = comparison_set(blind)
    generated_at = datetime.now(timezone.utc).isoformat()
    write_json(OUT / "blind-500.json", {
        "schema_version": "campus-v23-blind-500-v1", "generated_at": generated_at,
        "holdout": True, "used_for_improvement": False, "deduplication": deduplication, "items": blind,
    })
    write_json(OUT / "stress-200.json", {
        "schema_version": "campus-v23-stress-200-v1", "generated_at": generated_at,
        "holdout": True, "used_for_improvement": False,
        "type_counts": dict(Counter(row["stress_type"] for row in stress)), "items": stress,
    })
    write_json(OUT / "comparison-50.json", {
        "schema_version": "campus-v23-comparison-50-v1", "generated_at": generated_at,
        "purpose": "future manual A/B comparison", "external_ai_api": "OFF",
        "external_answers_fabricated": False, "items": comparison,
    })
    print(json.dumps({"blind": len(blind), "stress": len(stress), "comparison": len(comparison),
                      "deduplication": deduplication}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
