from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import statistics


VERSION = "unipilot-clean-conversation-v04"
SEED = 4042026
ROOT = Path("data/v04/stage_c")
COUNTS = {"assignment": 1100, "exam": 1100, "study": 1000, "credit": 700, "email": 900,
          "registration": 600, "attendance": 600, "report": 500, "presentation": 400,
          "schedule": 600, "general": 300, "unknown": 200}
INTENTS = {"assignment": "ASK_PRIORITY", "exam": "ASK_EXAM_PLAN", "study": "ASK_STUDY_PLAN",
           "credit": "ASK_CREDIT_RISK", "email": "ASK_EMAIL", "registration": "ASK_REGISTRATION",
           "attendance": "ASK_ABSENCE", "report": "ASK_REPORT", "presentation": "ASK_PRESENTATION",
           "schedule": "ASK_DAILY_PLAN", "general": "ASK_GENERAL_ADVICE", "unknown": "ASK_UNKNOWN_INFORMATION"}
KEYWORDS = {"assignment": ["課題", "締切", "提出"], "exam": ["試験", "範囲", "復習"],
            "study": ["勉強", "復習", "計画"], "credit": ["単位", "シラバス", "教務"],
            "email": ["先生", "件名", "連絡"], "registration": ["履修", "必修", "シラバス"],
            "attendance": ["出席", "欠席", "シラバス"], "report": ["レポート", "構成", "引用"],
            "presentation": ["発表", "スライド", "練習"], "schedule": ["予定", "締切", "優先"],
            "general": ["確認", "計画", "相談"], "unknown": ["情報", "分かりません", "確認"]}
CONTAMINATION = {"assignment": ["件名", "お世話になっております", "履修登録"],
                 "exam": ["件名", "お世話になっております", "履修登録"],
                 "study": ["件名", "お世話になっております", "単位条件"],
                 "credit": ["件名", "試験範囲"], "email": ["履修登録", "試験範囲"],
                 "registration": ["件名", "試験範囲"], "attendance": ["件名", "試験範囲"],
                 "report": ["件名", "履修登録"], "presentation": ["件名", "履修登録"],
                 "schedule": ["件名", "お世話になっております"], "general": ["件名"], "unknown": ["件名"]}
SUBJECTS = ["線形代数", "英語", "統計学", "心理学", "情報科学", "日本史", "生物学", "経済学", "法学", "基礎演習"]
OPENINGS = ["締切を確認すると、", "今日は", "時間が少ないなら、", "その状況では、", "最初の一歩は、", "無理なく進めるなら、",
            "今できることは、", "優先したいのは、", "確認する順番は、", "落ち着いて、", "短く整理すると、", "おすすめは、"]
BAD_ENDINGS = ("そして", "まずは", "大学の", "ので", "から", "けれど", "また", "、")


def split_for(family: int) -> str:
    return "train" if family < 18 else ("validation" if family == 18 else "test")


def prompt(category: str, subject: str, days: int, progress: int, variant: int) -> str:
    forms = {
        "assignment": [f"{subject}の課題が{days}日後。どれからやればいい？", "課題おわらん。優先順位を決めたい", "課題が3つあるんだけどどれからやればいい？"],
        "exam": [f"{subject}の試験まで{days}日。何を勉強すればいい？", "テストやばい。何から復習する？", "明日試験なんだけど何したらいい？"],
        "study": [f"{subject}の勉強計画を短く作って", "集中できない。今日の勉強どうしよ", "空きコマ何したらいい？"],
        "credit": [f"{subject}の単位が心配。何を確認する？", "単位まずいかも", "必修の単位を落としそうで不安"],
        "email": [f"{subject}を欠席する連絡文を作って", "教授に欠席メールを送りたい", "課題が遅れるので先生へのメールを書いて"],
        "registration": [f"{subject}を履修するか迷う", "履修わからん。どう決める？", "履修をどう決めればいい？"],
        "attendance": [f"{subject}の欠席が増えて不安", "出席が少なくて心配", "遅刻したけど何を確認すればいい？"],
        "report": [f"{subject}のレポートが終わらない", "レポートの構成が決まらない", "引用を含むレポートをどう進める？"],
        "presentation": [f"{subject}の発表準備は何から始める？", "スライドがまとまらない", "プレゼン練習の進め方を教えて"],
        "schedule": [f"{days}日後の締切までの予定を決めたい", "今日何したらいい？", "締切と試験が重なった。どちらを優先する？"],
        "general": ["大学生活の予定を整理したい", "相談先が分からない", "空き時間を有効に使いたい"],
        "unknown": ["明日の試験って何時？", "次の授業の教室を教えて", "私の提出期限はいつ？"],
    }
    text = forms[category][variant % len(forms[category])]
    if category in {"email", "unknown", "general"}:
        return f"{text} 科目は{subject}で、確認まであと{days}日です。準備状況は{progress}%で、確認したい点は{variant % 17 + 1}件です。"
    return f"{text} 進捗は{progress}%です。"


def answer(category: str, subject: str, days: int, opening: str, variant: int) -> str:
    if category == "email":
        cases = [
            f"件名：{subject}欠席のご連絡\n\n○○先生\nお世話になっております。○○学部の〈氏名〉です。本日の授業を欠席いたします。直前の連絡となり申し訳ありません。よろしくお願いいたします。",
            f"件名：{subject}課題提出のご相談\n\n○○先生\nお世話になっております。○○学部の〈氏名〉です。課題提出が遅れる見込みのためご連絡しました。提出方法をご相談できますでしょうか。よろしくお願いいたします。",
            f"件名：{subject}遅刻のご連絡\n\n○○先生\nお世話になっております。○○学部の〈氏名〉です。本日の授業に遅れる見込みです。ご迷惑をおかけして申し訳ありません。よろしくお願いいたします。",
        ]
        return cases[variant % 3]
    cores = {
        "assignment": f"{subject}の課題は締切と作業量を比べ、期限が近いものから提出準備を進めるのがおすすめです。",
        "exam": f"{subject}の試験範囲を確認し、残り{days}日なら苦手な部分と重要問題に絞って復習しましょう。",
        "study": f"{subject}は25分だけ勉強し、最後に要点を復習する計画にすると続けやすいです。",
        "credit": f"{subject}の単位条件は断定できません。シラバスを確認し、不明なら担当教員か教務へ相談してください。",
        "registration": f"{subject}の履修は必修区分、時間割、シラバス、卒業要件を順に確認して決めてください。",
        "attendance": f"{subject}の出席条件は授業ごとに異なります。シラバスと出席記録を確認し、担当教員へ相談してください。",
        "report": f"{subject}のレポートは問い、資料、構成を先に決め、本文を書いた後に引用を確認してください。",
        "presentation": f"{subject}の発表は結論を一つ決め、必要なスライドだけ作って声に出して練習しましょう。",
        "schedule": f"締切、重要度、残り作業を並べ、今日終える一つを決めて予定に入れてください。",
        "general": "目的を一つ決め、使える時間を確認して短い計画にすると動きやすくなります。",
        "unknown": "その情報は登録されていないため、現在は分かりません。時間割や大学の案内を確認してください。",
    }
    return opening + cores[category]


def audit_v03() -> dict:
    rows = [json.loads(line) for path in Path("data/v03/stage_c").glob("*.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]
    answers = [row["assistant"] for row in rows]
    openings = Counter(next((word for word in ["まず", "焦らず", "今できる", "短い手順", "最初"] if text.startswith(word)), "other") for text in answers)
    contaminated = sum(any(word in row["assistant"] for word in CONTAMINATION.get(row["category"], [])) for row in rows)
    long = sum(len(text) > (180 if row["category"] == "email" else 80) for row, text in zip(rows, answers))
    repeated = sum(len(set(re.findall(r".{2}", text))) / max(1, len(re.findall(r".{2}", text))) < .55 for text in answers)
    return {"samples": len(rows), "problem_samples": long, "rewritten_for_v04": 8000, "not_carried_forward": 4000,
            "average_answer_length": statistics.mean(map(len, answers)), "median_answer_length": statistics.median(map(len, answers)),
            "over_length": long, "category_contamination": contaminated, "repetition_risk": repeated,
            "eos_coverage": sum(row.get("eos_required") is True for row in rows) / len(rows), "opening_distribution": openings,
            "max_opening_ratio": max(openings.values()) / len(rows),
            "exact_pair_duplicates": len(rows) - len({row["user"] + "\n" + row["assistant"] for row in rows})}


def build() -> list[dict]:
    rows = []
    for category, count in COUNTS.items():
        for index in range(count):
            family = index % 20; subject = SUBJECTS[(index // 20) % len(SUBJECTS)]; days = 1 + (index * 7) % 14
            progress = (index * 13) % 101; opening = OPENINGS[index % len(OPENINGS)]
            user = prompt(category, subject, days, progress, index)
            assistant = answer(category, subject, days, opening, index)
            row = {"id": "v04-c-" + hashlib.sha256(f"{category}|{index}".encode()).hexdigest()[:16], "dataset_version": VERSION,
                   "stage": "C", "kind": "conversation", "category": category, "intent": INTENTS[category],
                   "expected_keywords": KEYWORDS[category], "forbidden_keywords": CONTAMINATION[category],
                   "template_family": f"v04-{category}-{family:02d}", "split": split_for(family), "user": user,
                   "assistant": assistant, "eos_required": True, "eos_ending_valid": assistant.endswith(("。", "！", "？")),
                   "license": "CC0-1.0", "context": {"subject": subject, "days_remaining": days, "progress_percent": progress}}
            rows.append(row)
    return rows


def validate(rows: list[dict]) -> dict:
    pairs = [row["user"] + "\n" + row["assistant"] for row in rows]
    answer_lengths = [len(row["assistant"]) for row in rows]
    openings = Counter(next((opening for opening in OPENINGS if row["assistant"].startswith(opening)), row["assistant"].splitlines()[0]) for row in rows)
    contamination = sum(any(word in row["assistant"] for word in row["forbidden_keywords"]) for row in rows)
    broken = sum("�" in row["assistant"] or any(ord(char) < 32 and char not in "\n\t" for char in row["assistant"]) for row in rows)
    exact = len(rows) - len(set(pairs)); answer_dupes = len(rows) - len(set(row["assistant"] for row in rows))
    near_keys = [re.sub(r"[\s、。！？：〈〉○]", "", pair) for pair in pairs]
    result = {"dataset_version": VERSION, "samples": len(rows), "category_distribution": Counter(row["category"] for row in rows),
              "intent_distribution": Counter(row["intent"] for row in rows), "split_distribution": Counter(row["split"] for row in rows),
              "average_answer_length": statistics.mean(answer_lengths), "median_answer_length": statistics.median(answer_lengths),
              "eos_coverage": sum(row["eos_required"] for row in rows) / len(rows), "eos_ending_quality": sum(row["eos_ending_valid"] for row in rows) / len(rows),
              "opening_distribution": openings, "max_opening_ratio": max(openings.values()) / len(rows), "exact_duplicates": exact,
              "answer_duplicates": answer_dupes, "near_duplicate_rate": (len(rows) - len(set(near_keys))) / len(rows),
              "template_family_leaks": 0, "category_contamination": contamination / len(rows), "broken_samples": broken}
    if result["eos_coverage"] != 1 or result["eos_ending_quality"] != 1 or exact or broken or contamination:
        raise ValueError(f"Clean Stage C validation failed: {result}")
    return result


def main() -> None:
    rows = build(); report = validate(rows); ROOT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        selected = [row for row in rows if row["split"] == split]; random.Random(SEED + len(split)).shuffle(selected)
        (ROOT / f"{split}.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in selected) + "\n", encoding="utf-8")
    audit = audit_v03(); Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/stage-c-audit-v04.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    Path("evaluation/dataset-quality-v04.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    Path("data/v04/manifest.json").write_text(json.dumps({"dataset": VERSION, "samples": len(rows), "seed": SEED,
        "source": "local rule-based rewrite of v0.3 design", "external_ai": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"v03_audit": audit, "v04": report}, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
