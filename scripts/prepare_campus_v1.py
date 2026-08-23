from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from pipeline.campus_categories import CAMPUS_CATEGORIES, CAMPUS_KEYWORDS, CAMPUS_LABELS, STANDARD_TO_CAMPUS
from pipeline.standard_categories import CATEGORY_PROFILES, STANDARD_CATEGORIES


ROOT = Path("data/campus_v1")
CREATED = "2026-08-23"
LICENSE = "CC0-1.0"
VERSION = "unipilot-campus-v1"


FAQ_CONDITIONS = (
    ("通常", "まず条件と期限を整理してください。"),
    ("今日中に動く必要がある", "今日中に、連絡・確認・提出不能の回避を優先してください。"),
    ("必要な情報がまだそろっていない", "不足情報を決めつけず、確認項目として分けてください。"),
)
QUESTION_FORMS = (
    "{topic}について、今やるべきことを教えてください。状況は{condition}です。",
    "{condition}のですが、{topic}で困っています。何から始めればいい？",
    "{topic}の対応を進めたいです。{condition}場合の手順は？",
)


SPECIAL_FAQ = (
    ("gpa", "GPAを自分で正確に計算したい", "各科目のGPと単位数を入力すると、GP×単位数の合計を対象単位数で割って計算できます。大学ごとのGP換算と除外科目を先に確認してください。"),
    ("grade_simulator", "単位取得にあと何点必要か知りたい", "現在までの獲得点、目標点、残り評価の割合を確認すれば、残りで必要な平均点を計算できます。配点はシラバスで確認してください。"),
    ("absence_email", "教授へ欠席メールを完成させたい", "授業名・日付・欠席理由・必要な確認事項を入力すると、件名と本文を作成できます。理由は事実だけを簡潔に書いてください。"),
    ("lateness_email", "授業への遅刻連絡を送りたい", "到着予定時刻と事実を先に伝え、安全を優先してください。入室可否は授業案内に従います。"),
    ("late_submission_email", "課題提出が遅れる相談メールを作りたい", "課題名、事実、提出可能日時を明記して期限前に相談します。受理されるとは断定しません。"),
    ("study_plan", "試験までの日数から勉強計画を作りたい", "科目・残り日数・一日に使える時間を入力し、範囲確認、演習、誤答復習の順に配分します。"),
    ("assignment_priority", "複数課題の優先順位を決めたい", "締切、所要時間、成績への影響を並べ、提出不能になる期限と短時間で提出可能にできる課題を優先します。"),
    ("deadline_organizer", "課題の締切を一覧にしたい", "課題名、期限、所要時間を入力し、期限順に並べて前日通知を設定します。"),
    ("report_outline", "レポートの章立てを作りたい", "問い、仮の結論、根拠、反対意見、考察、結論の順でアウトラインを作ります。"),
    ("es_outline", "ESの構成を作りたい", "結論、状況、行動、工夫、結果、応募先での再現性の順に事実を整理します。"),
)


BLIND_BASES = {
    category: tuple(dict.fromkeys((
        CAMPUS_LABELS[category] + "やばい",
        CAMPUS_KEYWORDS[category][0] + "どうしよ",
        CAMPUS_KEYWORDS[category][-1] + "のことで困ってる",
        CAMPUS_LABELS[category] + "、何からやる？",
    ))) for category in CAMPUS_CATEGORIES
}
BLIND_BASES.update({
    "gpa": ("gpaってどうやってだす", "成績からGPA計算して", "S 2単位、A 2単位、B 1単位のGPAは", "GPと単位数で平均出したい"),
    "grade_simulator": ("今40点で残り評価30%、合格60点なら何点必要", "単位取るにはあと何点いる", "現在55点、目標60点、残り20%", "期末で必要な点を出して"),
    "absence_email": ("教授になんて欠席連絡すればいい", "今日休むメール作って", "授業を欠席する文面ほしい", "体調不良の欠席メール"),
    "lateness_email": ("遅刻するから先生に送る文作って", "電車遅延のメール", "授業に遅れる連絡どう書く", "到着が遅いことを教授に伝えたい"),
    "late_submission_email": ("課題忘れた謝るメール作って", "提出遅れるって先生に送りたい", "締切に間に合わない相談文", "レポート遅延のメール"),
    "study_plan": ("明日テストむり、数学3時間だけ", "試験まで5日で計画作って", "英語をあと3日、毎日2時間", "テスト勉強の時間割ほしい"),
    "assignment_priority": ("課題3つ、どれからやる", "レポートと発表準備どっち先", "締切近い課題を優先したい", "複数課題の順番決めて"),
    "deadline_organizer": ("課題忘れそうだから締切まとめたい", "提出期限を順番にして", "締切一覧を作りたい", "複数の期限を整理して"),
    "report_outline": ("レポート構成だけ先に作りたい", "章立てが決まらない", "序論から結論までの流れ", "卒論アウトラインを整理して"),
    "citation_check": ("この引用の情報足りてる？", "参考文献の確認したい", "web引用どう書く", "出典抜けをチェックして"),
    "presentation_outline": ("5分プレゼンの流れ作って", "スライド構成を決めたい", "発表何から話す", "プレゼンのアウトライン"),
    "career_schedule": ("就活と授業両立する予定作って", "選考締切を管理したい", "就活いつ何する", "応募計画を整理して"),
    "es_outline": ("es何書けばいい", "自己PRの構成ほしい", "志望動機を整理したい", "エントリーシートの流れ"),
    "toeic_plan": ("toeicまで30日で計画", "英語スコア上げる予定作って", "TOEIC何から勉強", "毎日1時間のTOEIC計画"),
    "university_policy": ("うちの大学って追試ある？", "欠席何回で単位落とす？", "この大学のGPA基準は", "履修変更ってできる？"),
})


DIFFICULTY_CONSTRAINTS = {
    "easy": ("短く教えて", "最初の一歩だけ知りたい", "今すぐ対応したい", "初心者です", "口語で教えて"),
    "medium": ("授業と両立したい", "期限も確認したい", "必要な情報も教えて", "大学ごとの差に注意して", "順番に整理して"),
    "hard": ("情報が一部不明で断定は避けたい", "複数の条件を比較したい", "失敗した場合の次の手も知りたい", "根拠と確認先も必要", "誤字があるけど意図を読んで"),
    "compound": ("試験と課題も重なっている", "欠席連絡と締切対応を同時にしたい", "学業と就活の両方を崩したくない", "期限・負担・成績への影響をまとめて判断したい", "今すぐの対応と今週の計画を分けたい"),
}


def stable(value: str, length: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) % length


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_faq() -> list[dict]:
    rows = []
    for standard_category in STANDARD_CATEGORIES:
        profile = CATEGORY_PROFILES[standard_category]
        category = STANDARD_TO_CAMPUS[standard_category]
        for topic_index, topic in enumerate(profile["concerns"]):
            for condition_index, (condition, condition_action) in enumerate(FAQ_CONDITIONS):
                row_id = f"campus-faq-{standard_category}-{topic_index:02d}-{condition_index}"
                question = QUESTION_FORMS[stable(row_id, len(QUESTION_FORMS))].format(topic=topic, condition=condition)
                answer = (f"結論：{profile['conclusion']}\n今やること：1. {profile['actions'][0]} 2. {profile['actions'][1]} "
                          f"3. {condition_action}\n注意：{profile['caution']}")
                needs_confirmation = standard_category in {
                    "exam", "credit", "gpa", "registration", "attendance", "lateness", "scholarship", "tuition", "study_abroad"
                }
                rows.append({
                    "id": row_id, "category": category, "question": question, "answer": answer,
                    "keywords": [profile["label"], topic, category], "source": "UniPilot project-authored campus guidance",
                    "source_url": None, "license": LICENSE, "retrieved_at": CREATED,
                    "university_specific": needs_confirmation, "needs_confirmation": needs_confirmation,
                    "semantic_scenario": f"{standard_category}|{topic}|{condition}", "dataset_version": VERSION,
                })
    for index, (category, question, answer) in enumerate(SPECIAL_FAQ):
        rows.append({"id": f"campus-faq-tool-{index:02d}", "category": category, "question": question,
                     "answer": "結論：" + answer, "keywords": list(CAMPUS_KEYWORDS[category][:3]),
                     "source": "UniPilot project-authored campus guidance", "source_url": None,
                     "license": LICENSE, "retrieved_at": CREATED, "university_specific": False,
                     "needs_confirmation": category in ("gpa", "grade_simulator"),
                     "semantic_scenario": f"tool|{category}|{question}", "dataset_version": VERSION})
    assert len(rows) == 1000
    return rows


def build_router(faq: list[dict]) -> list[dict]:
    rows = [{"id": "router-" + row["id"], "question": row["question"], "category": row["category"]} for row in faq]
    for category in CAMPUS_CATEGORIES:
        for index, keyword in enumerate(CAMPUS_KEYWORDS[category][:3]):
            rows.append({"id": f"campus-router-{category}-{index}",
                         "question": f"{CAMPUS_LABELS[category]}の相談として、{keyword}を確認したいです。", "category": category})
    return rows


def build_router_dev(faq: list[dict]) -> list[dict]:
    selected = [faq[index * len(faq) // 350] for index in range(350)]
    return [{"id": f"campus-router-dev-{index:03d}",
             "prompt": f"分類してください：{row['question']} 次の対応先を知りたいです。",
             "category": row["category"], "tuning_allowed": True}
            for index, row in enumerate(selected)]


def expected_for(category: str, index: int) -> dict:
    if category == "gpa":
        return {"tool": "gpa", "required_points": ["計算結果", "大学", "規程"], "calculation": "weighted_gpa"}
    if category == "grade_simulator":
        return {"tool": "grade_simulator", "required_points": ["必要", "シラバス"], "calculation": "required_remaining_average"}
    if category in ("professor_email", "absence_email", "lateness_email", "late_submission_email"):
        return {"tool": category, "required_points": ["件名", "先生", "氏名"], "calculation": None}
    if category in {"study_plan", "assignment_priority", "deadline_organizer", "report_outline", "citation_check",
                    "presentation_outline", "career_schedule", "es_outline", "toeic_plan", "registration"}:
        return {"tool": category, "required_points": [CAMPUS_LABELS[category], "確認"], "calculation": None}
    return {"tool": None, "required_points": [CAMPUS_LABELS[category], "確認"], "calculation": None}


def build_blind() -> list[dict]:
    counts = (("easy", 300), ("medium", 300), ("hard", 250), ("compound", 150))
    rows, global_index = [], 0
    for difficulty, count in counts:
        constraints = DIFFICULTY_CONSTRAINTS[difficulty]
        occurrences = Counter()
        for local_index in range(count):
            category = CAMPUS_CATEGORIES[(global_index * 11) % len(CAMPUS_CATEGORIES)]
            bases = BLIND_BASES[category]
            occurrence = occurrences[category]
            occurrences[category] += 1
            base = bases[(occurrence + stable(difficulty + category, len(bases))) % len(bases)]
            constraint = constraints[(occurrence // len(bases) + stable(category, len(constraints))) % len(constraints)]
            prompt = f"{base}。{constraint}"
            expected = expected_for(category, global_index)
            rows.append({
                "id": f"campus-blind-{global_index:04d}", "prompt": prompt, "category": category,
                "difficulty": difficulty, "required_key_points": expected["required_points"],
                "forbidden_claims": ["どの大学でも", "必ず認められる", "全国の大学で"],
                "expected_tool": expected["tool"], "calculation_check": expected["calculation"],
                "university_specific": category == "university_policy", "source_grounding_required": category in {
                    "university_policy", "scholarship", "tuition", "credit", "attendance"},
                "semantic_seed": f"blind-only|{category}|{base}|{constraint}", "blind": True,
                "dataset_version": VERSION + "-blind-1000",
            })
            global_index += 1
    assert len(rows) == 1000 and Counter(row["difficulty"] for row in rows) == {
        "easy": 300, "medium": 300, "hard": 250, "compound": 150}
    return rows


def build_manual(blind: list[dict]) -> list[dict]:
    selected = [blind[index * len(blind) // 100] for index in range(100)]
    return [{
        "id": f"campus-manual-{index:03d}", "question": row["prompt"], "category": row["category"],
        "required_key_points": row["required_key_points"], "forbidden_claims": row["forbidden_claims"],
        "campus_answer": None, "chatgpt_answer": None, "gemini_answer": None,
        "scores": {"campus": None, "chatgpt": None, "gemini": None},
        "winners": {"correct": None, "specific": None, "usable": None, "fast": None, "student_preference": None},
        "notes": "", "manual_only": True, "external_api_calls": False,
        "rubric": {"0": "役に立たない", "1": "一般論だけ", "2": "多少参考になる", "3": "次に何をするか分かる",
                   "4": "そのまま使える", "5": "完成物・計算結果・具体的計画まで出る"},
    } for index, row in enumerate(selected)]


def main() -> None:
    faq = build_faq()
    router = build_router(faq)
    router_dev = build_router_dev(faq)
    blind = build_blind()
    manual = build_manual(blind)
    write_jsonl(ROOT / "faq/faq.jsonl", faq)
    write_jsonl(ROOT / "router/train.jsonl", router)
    (ROOT / "router/dev.json").write_text(json.dumps(router_dev, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "blind").mkdir(parents=True, exist_ok=True)
    (ROOT / "benchmark").mkdir(parents=True, exist_ok=True)
    (ROOT / "blind/evaluation.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "benchmark/chatgpt-gemini-manual-100.json").write_text(json.dumps(manual, ensure_ascii=False, indent=2), encoding="utf-8")

    public = []
    source = Path("data/v07/knowledge/documents.jsonl")
    if source.exists():
        public = [row for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
                  for row in (json.loads(line),) if row.get("source_url") and row.get("license")]
    write_jsonl(ROOT / "knowledge/public.jsonl", public)

    faq_questions = {row["question"].replace(" ", "") for row in faq}
    blind_questions = {row["prompt"].replace(" ", "") for row in blind}
    report = {
        "dataset_version": VERSION, "faq_semantic_scenarios": len(faq), "faq_categories": len(set(row["category"] for row in faq)),
        "router_training_examples": len(router), "router_development_questions": len(router_dev), "blind_questions": len(blind),
        "blind_difficulty": Counter(row["difficulty"] for row in blind),
        "manual_chatgpt_gemini_questions": len(manual), "public_knowledge_documents": len(public),
        "exact_faq_blind_overlap": len(faq_questions & blind_questions),
        "blind_semantic_seed_duplicates": len(blind) - len({row["semantic_seed"] for row in blind}),
        "external_llm_used": False, "external_teacher_generation_used": False,
        "quality_limit": (
            "The 1,000 FAQ rows combine distinct university concerns with operational constraints. They are not paraphrase copies, "
            "but they are programmatically composed and require row-by-row student/expert review before production promotion."
        ),
    }
    Path("evaluation/dataset-quality-campus-v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"version": VERSION, "production_enabled": False, "model": "v0.4", "faq": "data/campus_v1/faq/faq.jsonl",
                "router": "data/campus_v1/router/train.jsonl", "router_development": "data/campus_v1/router/dev.json",
                "blind": "data/campus_v1/blind/evaluation.json",
                "manual_comparison": "data/campus_v1/benchmark/chatgpt-gemini-manual-100.json",
                "official_university_knowledge": "knowledge/universities", "external_ai_api": "OFF"}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
