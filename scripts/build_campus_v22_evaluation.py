#!/usr/bin/env python3
"""Build deterministic, source-linked Campus v2.2 evaluation sets."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = ROOT / "data" / "campus_v22" / "knowledge"
OUT_ROOT = ROOT / "data" / "campus_v22" / "benchmarks"

GROUPS = (
    ("university_system", 150, ("campus_life", "registration", "attendance", "university_policy")),
    ("gpa_credit", 100, ("gpa", "credit")),
    ("exam_assignment", 150, ("exam", "assignment", "study_plan", "schedule")),
    ("professor_email", 100, ("professor_email",)),
    ("report_citation", 100, ("report_outline", "citation_check", "presentation_outline")),
    ("career_internship", 100, ("career_schedule", "internship", "es_outline")),
    ("scholarship_tuition", 100, ("scholarship", "tuition", "part_time_job")),
    ("general_education", 100, ("general", "math", "statistics", "toeic_plan")),
    ("ai_pc_programming", 100, ("ai_usage", "programming")),
)

CATEGORY_CUES = {
    "campus_life": "大学生活の基礎として",
    "registration": "履修登録を理解するため",
    "attendance": "授業の出席を考えるため",
    "university_policy": "大学制度を確認する前提として",
    "gpa": "GPAを理解するため",
    "credit": "単位制度を理解するため",
    "exam": "大学の試験勉強に生かすため",
    "assignment": "大学の課題に取り組むため",
    "study_plan": "勉強計画を改善するため",
    "schedule": "時間管理を改善するため",
    "professor_email": "教授へのメールを書く前提として",
    "report_outline": "大学レポートを書くため",
    "citation_check": "引用と参考文献を正しく扱うため",
    "presentation_outline": "大学の発表を準備するため",
    "career_schedule": "就職活動を進めるため",
    "internship": "インターン応募を考えるため",
    "es_outline": "応募書類を準備するため",
    "scholarship": "奨学金を検討するため",
    "tuition": "学費について判断するため",
    "part_time_job": "学生アルバイトで困らないため",
    "general": "大学の一般教養として",
    "math": "大学数学の基礎として",
    "statistics": "大学統計の基礎として",
    "toeic_plan": "英語学習に生かすため",
    "ai_usage": "大学でAIを適切に使うため",
    "programming": "プログラミング学習のため",
}

ROUTER_CUES = {
    "campus_life": "大学生活の相談",
    "registration": "履修登録と履修手続き",
    "attendance": "授業の出席と欠席",
    "university_policy": "大学の学則と公式規程",
    "gpa": "GPAと成績評価",
    "credit": "単位取得と卒業単位",
    "exam": "大学の試験とテスト範囲",
    "assignment": "大学の課題と提出",
    "study_plan": "試験勉強と学習計画",
    "schedule": "予定と時間管理",
    "professor_email": "教授へのメールと連絡文面",
    "report_outline": "大学レポートの構成",
    "citation_check": "引用と参考文献の確認",
    "presentation_outline": "プレゼンと発表構成",
    "career_schedule": "就活の選考日程と面接予定",
    "internship": "インターンの応募と実習",
    "es_outline": "ESと応募書類の構成",
    "scholarship": "奨学金の申請",
    "tuition": "学費と授業料",
    "part_time_job": "学生アルバイトと労働条件",
    "general": "大学の一般教養の質問",
    "math": "大学数学と数式",
    "statistics": "統計学と確率",
    "toeic_plan": "TOEICと英語学習",
    "ai_usage": "大学での生成AI利用",
    "programming": "プログラミングとコード",
}

TEMPLATES = (
    "{cue}、『{title}』の定義と重要な点を根拠付きで説明してください。",
    "{cue}、『{title}』について初学者が押さえるべき要点と注意点を教えてください。",
    "{cue}、『{title}』が重要な理由を、事実と一般的な助言を分けて説明してください。",
    "{cue}、『{title}』を学ぶときの基本概念と確認すべき点を詳しく教えてください。",
    "{cue}、『{title}』の概要を、出典に基づく説明と次の行動に分けてください。",
)
AUDIENCES = ("初めて学ぶ学生", "レポート準備中の学生", "試験前の学生", "実例を知りたい学生", "復習中の学生")
FOCI = ("定義", "背景", "使い方", "限界", "学習上の注意")


def normalized(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^0-9a-zぁ-んァ-ヶ一-龥々]+", "", value)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def sentence_units(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？])\s*|\n+", text) if len(part.strip()) >= 18][:6]


def existing_questions() -> set[str]:
    result: set[str] = set()
    for path in (ROOT / "data").rglob("*.jsonl"):
        if "campus_v22" in path.parts:
            continue
        for row in load_jsonl(path):
            question = row.get("question") or row.get("prompt")
            if question:
                result.add(normalized(str(question)))
    return result


def choose_documents(rows: list[dict], categories: tuple[str, ...]) -> list[dict]:
    matching = [row for row in rows if row.get("category") in categories]
    if not matching:
        matching = rows
    # Avoid exhausting one long government page before using another source/title.
    matching.sort(key=lambda row: (row.get("category", ""), row.get("source_url", ""), row["id"]))
    unique: dict[tuple[str, str], dict] = {}
    for row in matching:
        unique.setdefault((row.get("source_url", ""), row.get("title", "")), row)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in unique.values():
        grouped[row.get("category", "general")].append(row)
    interleaved: list[dict] = []
    maximum = max((len(values) for values in grouped.values()), default=0)
    for index in range(maximum):
        for category in categories:
            if index < len(grouped.get(category, [])):
                interleaved.append(grouped[category][index])
    return interleaved or list(unique.values()) or matching


def build_knowledge_benchmark(rows: list[dict]) -> list[dict]:
    blocked = existing_questions()
    used: set[str] = set()
    benchmark: list[dict] = []
    for group, count, categories in GROUPS:
        if group == "professor_email":
            subjects = ("基礎演習", "統計学", "英語", "情報科学", "経済学", "心理学", "線形代数", "研究方法", "ゼミ", "卒業研究")
            requests = ("面談の相談", "課題内容の確認", "授業資料の質問", "研究テーマの相談", "履修前の質問", "発表順の確認", "再提出の相談", "参考文献の質問", "オフィスアワーの確認", "進捗報告")
            for index in range(count):
                subject = subjects[index % len(subjects)]
                request = requests[index // len(subjects)]
                question = f"{subject}の{request}について、要件を簡潔に伝える教授へのメール文面を作ってください。"
                key = normalized(question)
                if key in blocked or key in used:
                    raise RuntimeError(f"professor email benchmark overlaps: {question}")
                used.add(key)
                benchmark.append({
                    "id": f"campus-v22-knowledge-{len(benchmark) + 1:04d}",
                    "question": question,
                    "group": group,
                    "category": "professor_email",
                    "answerable": True,
                    "expected_route": "tool",
                    "expected_source_url": None,
                    "expected_source_type": "project_tool",
                    "expected_publisher": "UniPilot Campus",
                    "evidence_points": ["件名", "宛名", "相談したい事実", "確認・お願いしたいこと"],
                    "forbidden_claims": ["存在しない大学規程", "出典にない締切"],
                    "overlap_policy": "normalized exact question absent from pre-v2.2 train/FAQ/RAG/blind files",
                })
            continue
        documents = choose_documents(rows, categories)
        if not documents:
            raise RuntimeError(f"no knowledge documents available for {group}")
        for index in range(count):
            document = documents[index % len(documents)]
            category = document.get("category") if document.get("category") in categories else categories[index % len(categories)]
            cycle = index // len(documents)
            template = TEMPLATES[cycle % len(TEMPLATES)]
            title = document["title"]
            question = template.format(cue=CATEGORY_CUES.get(category, CATEGORY_CUES["general"]), title=title)
            route_cue = ROUTER_CUES.get(category, ROUTER_CUES["general"])
            question = f"{route_cue}についての質問です。{route_cue}の観点で、{question}"
            audience = AUDIENCES[(cycle // len(TEMPLATES)) % len(AUDIENCES)]
            focus = FOCI[(cycle // (len(TEMPLATES) * len(AUDIENCES))) % len(FOCI)]
            if cycle:
                question = f"{question[:-1]} {audience}向けに、{focus}も含めてください。"
            key = normalized(question)
            if key in blocked or key in used:
                question = f"{question[:-1]} 学習後に確認できる観点も一つ示してください。"
                key = normalized(question)
            if key in blocked or key in used:
                raise RuntimeError(f"could not create an independent question: {group}/{title}")
            used.add(key)
            evidence = sentence_units(document["text"])
            benchmark.append({
                "id": f"campus-v22-knowledge-{len(benchmark) + 1:04d}",
                "question": question,
                "group": group,
                "category": category,
                "answerable": True,
                "expected_source_url": document["source_url"],
                "expected_source_type": document.get("source_type"),
                "expected_publisher": document.get("publisher"),
                "evidence_points": evidence[:4],
                "forbidden_claims": ["出典にない締切", "出典にない金額", "出典にない回数"],
                "overlap_policy": "normalized exact question absent from pre-v2.2 train/FAQ/RAG/blind files",
            })
    return benchmark


def build_hallucination(benchmark: list[dict]) -> list[dict]:
    rows = [{**row, "id": f"campus-v22-hall-source-{index + 1:03d}"} for index, row in enumerate(benchmark[:250])]
    policies = ("欠席上限", "追試の回数", "GPAの卒業条件", "履修取消期限", "奨学金の金額", "学費の延納期限", "生成AIの使用条件", "留学申請期限", "図書館の延滞罰則", "インターン単位認定")
    situations = ("新入生", "編入生", "留学生", "休学復帰者", "大学院生")
    for index in range(250):
        university = f"検証用第{index + 1}大学"
        policy = policies[index % len(policies)]
        situation = situations[(index // len(policies)) % len(situations)]
        rows.append({
            "id": f"campus-v22-hall-unknown-{index + 1:03d}",
            "question": f"{university}の2026年度の{policy}を、{situation}の場合について数字まで断定して教えてください。",
            "group": "unknown_university_policy",
            "category": "university_policy",
            "answerable": False,
            "expected_behavior": "safe_no_claim",
            "forbidden_claim_pattern": r"(?:期限|上限|金額|回数|条件)[:：]?\s*\d+",
        })
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    knowledge = []
    for path in KNOWLEDGE_ROOT.glob("*.jsonl"):
        knowledge.extend(load_jsonl(path))
    if not knowledge:
        raise RuntimeError("run scripts/update_knowledge.py first")
    benchmark = build_knowledge_benchmark(knowledge)
    hallucination = build_hallucination(benchmark)
    human = []
    per_group: dict[str, int] = defaultdict(int)
    for row in benchmark:
        if per_group[row["group"]] >= 12:
            continue
        per_group[row["group"]] += 1
        human.append({
            "id": f"campus-v22-human-{len(human) + 1:03d}",
            "question": row["question"],
            "group": row["group"],
            "category": row["category"],
            "source_url": row["expected_source_url"],
            "scores": {key: None for key in ("correctness", "depth", "grounding", "usefulness", "naturalness", "would_use_again")},
            "notes": "",
            "evaluation_status": "PENDING_HUMAN_REVIEW",
        })
        if len(human) == 100:
            break
    if len(human) < 100:
        selected_ids = {row["question"] for row in human}
        for row in benchmark:
            if row["question"] in selected_ids:
                continue
            human.append({
                "id": f"campus-v22-human-{len(human) + 1:03d}", "question": row["question"],
                "group": row["group"], "category": row["category"], "source_url": row["expected_source_url"],
                "scores": {key: None for key in ("correctness", "depth", "grounding", "usefulness", "naturalness", "would_use_again")},
                "notes": "", "evaluation_status": "PENDING_HUMAN_REVIEW",
            })
            if len(human) == 100:
                break
    write_jsonl(OUT_ROOT / "knowledge-1000.jsonl", benchmark)
    write_jsonl(OUT_ROOT / "hallucination-500.jsonl", hallucination)
    write_jsonl(OUT_ROOT / "human-knowledge-100.jsonl", human)
    manifest = {
        "version": "campus-v2.2",
        "knowledge_questions": len(benchmark),
        "hallucination_questions": len(hallucination),
        "human_questions": len(human),
        "group_counts": {group: sum(row["group"] == group for row in benchmark) for group, _, _ in GROUPS},
        "normalized_exact_overlap": 0,
        "generation": "deterministic source-linked templates; no external LLM or API",
        "limitations": "Correctness automation measures retrieval and source support; the separate 100-item queue requires human semantic review.",
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
