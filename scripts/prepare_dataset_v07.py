from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from pipeline.categories import normalize_category
from scripts.prepare_dataset_v05 import SEEDS as V05_SEEDS
from scripts.prepare_dataset_v06 import HARD_CASES


ROOT = Path("data/v07")
VERSION = "unipilot-grounded-routing-v07"
RETRIEVED_AT = "2026-08-23"
LICENSE = "CC0-1.0"

EXTRA_SEEDS = (
    ("exam", "大学の試験に遅刻したらどうすればいい？", "安全を優先して移動し、試験案内で遅刻時の扱いと連絡方法を確認してください。入室可否は大学や試験で異なるため、指定された連絡先へすぐ知らせましょう。"),
    ("assignment", "課題を提出し忘れました", "提出状況と締切後の扱いを確認し、できるだけ早く担当教員へ事実と提出希望を簡潔に連絡してください。受理されるかは断定できません。"),
    ("credit", "単位を落としそうです", "シラバスの評価配分、現在の得点、残りの提出物や試験を確認してください。進級・卒業への影響は履修状況で異なるため、教務窓口にも相談しましょう。"),
    ("gpa", "GPAって何？", "GPAは、大学の成績を数値化した平均指標です。一般に科目のGPや単位数を使いますが、計算方法や用途は大学によって異なるため所属大学の規程を確認してください。"),
    ("attendance", "欠席が続いています", "出席記録とシラバスの評価方法を確認し、事情が続く場合は担当教員や学生相談窓口へ早めに相談してください。欠席の扱いは授業ごとに異なります。"),
    ("registration", "履修登録を忘れました", "現在年度の履修案内と学生ポータルで追加・修正期間を確認し、すぐ教務窓口へ相談してください。登録できるとは断定できません。"),
    ("professor_email", "教授に欠席メールを送りたい", "件名：授業欠席のご連絡\n\n○○先生\nお世話になっております。〈学部・学年〉の〈氏名〉です。〈授業名〉を欠席するためご連絡しました。必要な手続きや確認事項がございましたらご教示いただけますと幸いです。よろしくお願いいたします。"),
    ("citation", "レポートで引用する方法は？", "引用部分を自分の文章と明確に区別し、著者、資料名、出版年、参照箇所などを授業指定の形式で示してください。必要な範囲に限り、引用元も確認しましょう。"),
    ("presentation", "グループ発表の準備が進みません", "発表の結論、必要な作業、担当、期限、次の確認日時を全員で決めて記録してください。意見が割れた場合は評価基準に沿って選びましょう。"),
    ("study", "一週間のテスト勉強計画を作りたい", "最初に試験範囲と残り日数を確認し、前半を理解と問題演習、後半を間違い直し、前日を要点確認に分けてください。毎日進捗を見直しましょう。"),
    ("career", "就活はいつから始めればいい？", "開始時期は卒業年度や応募先で異なります。まず大学のキャリア窓口と企業の公式情報で予定を確認し、経験の整理と応募書類の準備から始めましょう。"),
    ("internship", "インターンは参加した方がいい？", "目的に合うかで判断してください。業務内容、期間、報酬、保険、個人情報の扱いを公式情報で確認し、学業との両立も比較しましょう。"),
    ("scholarship", "奨学金とは何ですか？", "学費や生活費を支援する制度で、返還不要の給付型と返還が必要な貸与型などがあります。条件と期限は制度ごとに異なるため、最新の公式募集要項を確認してください。"),
    ("campus_life", "研究室はどう選べばいい？", "研究テーマ、指導方法、活動頻度、設備、在学生の生活を確認し、公開情報を読んでから複数の研究室を訪問して比較してください。"),
    ("ai_usage", "AIを課題に使ってもいい？", "授業と大学の利用ルールを先に確認してください。許可される範囲でも、事実を一次資料で検証し、自分で理解した内容だけを提出物へ反映しましょう。"),
    ("general", "大学制度の最新情報を知りたい", "大学名、制度名、対象年度を確認し、現在年度の公式サイト、学生ポータル、担当窓口の順に調べてください。手元に最新情報がなければ断定しません。"),
)

FAQ_FORMS = (
    "{}", "大学生活の相談です。{}", "{} 短く教えてください。", "{} まず何をすればいい？",
    "{} 大学ごとの差を決めつけずに答えてください。",
)
DIRECT_FORMS = (
    "{}", "質問に直接答えてください。{}", "結論から答えてください。{}", "余計な科目名を足さずに答えてください。{}",
    "必要な確認先も含めてください。{}",
)

OFFICIAL_SUMMARIES = (
    {"id": "official-mext-syllabus", "title": "シラバスと授業計画", "category": "registration",
     "text": "大学では授業の方法・内容・年間計画を事前に示す取組が求められています。具体的な授業内容と評価方法は各授業の最新シラバスで確認します。",
     "source": "文部科学省", "source_url": "https://www.mext.go.jp/a_menu/koutou/daigaku/04052801/003.htm"},
    {"id": "official-mext-study-support", "title": "高等教育の修学支援新制度", "category": "scholarship",
     "text": "修学支援制度の学業要件や判定方法には制度上の条件があります。対象年度の公式案内と在籍校の窓口で最新条件を確認します。",
     "source": "文部科学省", "source_url": "https://www.mext.go.jp/kyufu/qa/qa_university.html"},
    {"id": "official-mhlw-parttime", "title": "学生アルバイトの労働条件", "category": "campus_life",
     "text": "アルバイトを始める前に、契約期間、仕事内容、勤務場所、労働時間、休日、賃金、退職条件などを書面で確認し保存します。困った場合は公的な労働相談窓口を利用できます。",
     "source": "厚生労働省", "source_url": "https://www.check-roudou.mhlw.go.jp/parttime/"},
    {"id": "official-mhlw-student-campaign", "title": "アルバイトの労働条件確認", "category": "campus_life",
     "text": "学生アルバイトでも労働条件の明示や適切な労働時間管理が重要です。募集内容だけで判断せず、実際の契約条件を確認します。",
     "source": "厚生労働省", "source_url": "https://www.mhlw.go.jp/stf/newpage_54645.html"},
    {"id": "official-jasso-scholarship", "title": "給付奨学金と貸与奨学金", "category": "scholarship",
     "text": "奨学金には給付と貸与などの区分があり、採用後の手続きや返還の有無が異なります。対象年度の案内と自分の採用区分を確認します。",
     "source": "日本学生支援機構", "source_url": "https://www.jasso.go.jp/shogakukin/saiyochu/siori/index.html"},
    {"id": "official-digital-license", "title": "公的情報の利用条件", "category": "citation",
     "text": "公的サイトの情報を再利用するときも、適用される利用規約、出典表示、第三者権利、最新の重要情報を確認します。",
     "source": "デジタル庁", "source_url": "https://www.digital.go.jp/resources/open_data"},
)

BAD_BUILDERS = (
    ("wrong category", lambda row, other: other["answer"]),
    ("invented subject", lambda row, other: "経済学の授業として考え、専門用語を暗記してください。"),
    ("unrelated advice", lambda row, other: "まず友達を増やしてサークルへ参加しましょう。"),
    ("university-specific hallucination", lambda row, other: "どの大学でも同じ規則なので、申請なしで必ず認められます。"),
    ("incomplete answer", lambda row, other: row["answer"][:12]),
    ("excessive generic response", lambda row, other: "落ち着いて頑張りましょう。詳しくは確認してください。"),
)


def split_for_family(family: int) -> str:
    bucket = int(hashlib.sha256(str(family).encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def normalized(text: str) -> str:
    return re.sub(r"[\s、。！？,.!?]", "", text).lower()


def seeds() -> list[tuple[str, str, str]]:
    rows = [(normalize_category(category), question, answer) for category, question, answer in V05_SEEDS]
    rows.extend((normalize_category(category), question, answer) for category, question, answer in HARD_CASES)
    rows.extend(EXTRA_SEEDS)
    if len(rows) != 84:
        raise AssertionError(f"expected 84 semantic seeds, got {len(rows)}")
    return rows


def build_faq() -> list[dict]:
    rows = []
    for family, (category, question, answer) in enumerate(seeds()):
        for variant, form in enumerate(FAQ_FORMS):
            rows.append({
                "id": f"v07-faq-{family:03d}-{variant}", "question": form.format(question), "answer": answer,
                "title": form.format(question), "text": answer, "category": category, "difficulty": "representative",
                "source_type": "project_authored_faq", "quality_score": 5.0, "source": "UniPilot project original",
                "source_url": None, "license": LICENSE, "retrieved_at": RETRIEVED_AT,
                "family": family, "dataset_version": VERSION,
            })
    return rows


def build_direct(faq: list[dict]) -> list[dict]:
    rows = []
    for faq_index, item in enumerate(faq):
        split = split_for_family(item["family"])
        for variant, form in enumerate(DIRECT_FORMS):
            rows.append({
                "id": f"v07-direct-{faq_index:03d}-{variant}", "kind": "conversation",
                "user": form.format(item["question"]), "assistant": item["answer"], "category": item["category"],
                "context": f"関連FAQ：{item['question']}\n{item['answer']}",
                "difficulty": "normal" if variant < 3 else "constraint", "source_type": "project_authored_direct_answer",
                "quality_score": 5.0, "length_type": "short" if item["category"] == "gpa" else "normal",
                "split": split, "family": item["family"], "source": "UniPilot project original", "source_url": None,
                "license": LICENSE, "retrieved_at": RETRIEVED_AT, "dataset_version": VERSION,
            })
    return rows


def build_corrected(faq: list[dict]) -> tuple[list[dict], list[dict]]:
    records, conversations = [], []
    for index, item in enumerate(faq):
        for variant in range(2):
            reason, builder = BAD_BUILDERS[(index * 2 + variant) % len(BAD_BUILDERS)]
            other = faq[(index + 37 + variant * 53) % len(faq)]
            bad_answer = builder(item, other)
            record = {
                "id": f"v07-corrected-{index:03d}-{variant}", "question": item["question"],
                "bad_answer": bad_answer, "bad_reason": reason, "correct_answer": item["answer"],
                "category": item["category"], "source_type": "project_authored_contrastive_correction",
                "quality_score": 5.0, "source": "UniPilot project original", "source_url": None,
                "license": LICENSE, "retrieved_at": RETRIEVED_AT, "dataset_version": VERSION,
            }
            records.append(record)
            conversations.append({
                "id": record["id"], "kind": "conversation",
                "user": f"質問：{item['question']}\n問題のある回答：{bad_answer}\n理由：{reason}\n質問に直接答える内容へ訂正してください。",
                "assistant": item["answer"], "category": item["category"], "difficulty": "hard",
                "context": f"関連FAQ：{item['question']}\n{item['answer']}",
                "source_type": record["source_type"], "quality_score": 5.0, "length_type": "normal",
                "split": "train", "family": f"corrected-{index}", "source": record["source"], "source_url": None,
                "license": LICENSE, "retrieved_at": RETRIEVED_AT, "dataset_version": VERSION,
            })
    return records, conversations


def build_public_knowledge(faq: list[dict]) -> list[dict]:
    rows = []
    for item in faq:
        rows.append({key: item[key] for key in ("id", "title", "text", "category", "source", "source_url", "license", "retrieved_at")})
        rows[-1]["source_type"] = "project_authored_faq"
        rows[-1]["answer"] = item["answer"]
        rows[-1]["question"] = item["question"]
    wikipedia = Path("data/v06/knowledge/wikipedia.jsonl")
    if wikipedia.exists():
        for line in wikipedia.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.append({
                "id": "v07-" + row["id"], "title": row["title"], "text": row["text"],
                "category": normalize_category(row.get("category", "general")), "source": row["source"],
                "source_url": row["source_url"], "license": row["license"], "retrieved_at": row["retrieved_at"],
                "source_type": "public_encyclopedia", "attribution_url": row.get("attribution_url"),
            })
    for item in OFFICIAL_SUMMARIES:
        rows.append({**item, "license": "CC0-1.0 (project-authored summary; source content not copied)",
                     "retrieved_at": RETRIEVED_AT, "source_type": "project_authored_official_source_summary",
                     "source_terms": "Source page terms checked; summary text is project-authored."})
    return rows


def quality_report(faq: list[dict], direct: list[dict], corrected: list[dict], knowledge: list[dict]) -> dict:
    required_knowledge = {"id", "title", "text", "category", "source", "source_url", "license", "retrieved_at"}
    family_splits = {}
    for row in direct:
        family_splits.setdefault(str(row["family"]), set()).add(row["split"])
    return {
        "dataset_version": VERSION, "semantic_seed_count": len(seeds()), "faq_questions": len(faq),
        "direct_answer_rows": len(direct), "corrected_rows": len(corrected), "knowledge_documents": len(knowledge),
        "direct_split_distribution": Counter(row["split"] for row in direct),
        "category_distribution": Counter(row["category"] for row in faq),
        "bad_reason_distribution": Counter(row["bad_reason"] for row in corrected),
        "faq_pair_duplicates": len(faq) - len({normalized(row["question"] + row["answer"]) for row in faq}),
        "faq_answer_duplicates": len(faq) - len({normalized(row["answer"]) for row in faq}),
        "direct_pair_duplicates": len(direct) - len({normalized(row["user"] + row["assistant"]) for row in direct}),
        "split_family_leakage": sum(len(splits) > 1 for splits in family_splits.values()),
        "knowledge_missing_required_fields": sum(not required_knowledge.issubset(row) for row in knowledge),
        "broken_text": sum("�" in row["question"] + row["answer"] for row in faq),
        "external_llm_used": False,
        "quality_note": "420 FAQ phrasings are based on 84 reviewed semantic answer seeds; counts are reported separately to avoid overstating semantic diversity.",
    }


def main() -> None:
    faq = build_faq()
    direct = build_direct(faq)
    corrected, corrected_conversations = build_corrected(faq)
    knowledge = build_public_knowledge(faq)
    for directory in (ROOT / "faq", ROOT / "training", ROOT / "corrected", ROOT / "knowledge", ROOT / "classifier"):
        directory.mkdir(parents=True, exist_ok=True)
    (ROOT / "faq" / "faq.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in faq), encoding="utf-8")
    for split in ("train", "validation", "test"):
        rows = [row for row in direct if row["split"] == split]
        if split == "train":
            rows += corrected_conversations
        (ROOT / "training" / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (ROOT / "corrected" / "corrected.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corrected), encoding="utf-8")
    (ROOT / "knowledge" / "documents.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in knowledge), encoding="utf-8")
    classifier_rows = [{"id": row["id"], "question": row["question"], "category": row["category"]} for row in faq]
    (ROOT / "classifier" / "train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in classifier_rows), encoding="utf-8")
    prompts = json.loads(Path("evaluation/fixed_prompts_v06.json").read_text(encoding="utf-8"))
    for item in prompts:
        item["source_category"] = item["category"]
        item["category"] = normalize_category(item["category"])
        item["evaluation_version"] = "unipilot-eval-v07-300"
    Path("evaluation/fixed_prompts_v07.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    report = quality_report(faq, direct, corrected, knowledge)
    Path("evaluation/dataset-quality-v07.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "dataset_version": VERSION, "faq": "data/v07/faq/faq.jsonl", "direct_training": "data/v07/training",
        "corrected": "data/v07/corrected/corrected.jsonl", "knowledge": "data/v07/knowledge/documents.jsonl",
        "classifier": "data/v07/classifier/train.jsonl", "quality_report": "evaluation/dataset-quality-v07.json",
        "external_llm_used": False, "production_enabled": False,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
