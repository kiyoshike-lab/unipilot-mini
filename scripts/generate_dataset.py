from __future__ import annotations

import json
from pathlib import Path
import random


CATEGORIES = {
    "university_life": ("大学生活", ["授業", "サークル", "通学", "昼休み", "図書館"]),
    "assignments": ("課題管理", ["課題", "提出物", "演習", "宿題", "制作物"]),
    "exams": ("試験勉強", ["試験", "小テスト", "過去問", "復習", "暗記"]),
    "credits": ("単位", ["単位", "必修科目", "選択科目", "卒業要件", "成績"]),
    "enrollment": ("履修", ["履修登録", "時間割", "シラバス", "抽選科目", "教室"]),
    "professor_email": ("教授メール", ["欠席連絡", "質問メール", "面談依頼", "提出相談", "お礼"]),
    "lateness": ("遅刻", ["電車遅延", "寝坊", "教室移動", "体調不良", "交通渋滞"]),
    "absence": ("欠席", ["発熱", "通院", "家庭の事情", "就職活動", "体調不良"]),
    "reports": ("レポート", ["構成", "参考文献", "推敲", "引用", "考察"]),
    "presentations": ("プレゼン", ["スライド", "発表練習", "質疑応答", "時間配分", "図表"]),
    "gpa": ("GPA", ["成績", "評価", "学習時間", "苦手科目", "目標"]),
    "attendance": ("出席", ["出席回数", "遅刻回数", "出席票", "オンライン出席", "公欠"]),
    "deadlines": ("締切", ["締切", "提出時刻", "優先順位", "予定表", "リマインダー"]),
    "study": ("学習計画", ["復習", "予習", "休憩", "集中時間", "週間計画"]),
}
QUESTIONS = [
    "{item}について、まず何をすればいい？", "{item}がうまく進まなくて困っています。", "今日から{item}を進める計画を作りたい。",
    "{item}を忘れないための方法は？", "{item}に不安があります。", "忙しい日に{item}へ取り組むコツを教えて。",
    "{item}を効率よく終わらせたい。", "{item}の優先順位はどう決める？", "明日までに{item}を準備したい。", "{item}を見直す手順を知りたい。",
]
ANSWERS = [
    "まず必要な情報を一か所に集め、締切と条件を確認しましょう。次に、作業を小さく分けて最初の一つを25分だけ進めてください。",
    "現状を三行で書き出し、できている点と不足している点を分けましょう。迷う点は早めに担当教員や窓口へ確認すると安心です。",
    "今日やることを一つ、今週やることを二つ決めましょう。予定には予備時間も入れ、終わった項目に印を付けてください。",
    "期限から逆算し、確認、実行、見直しの三段階に分けると進めやすいです。無理な計画なら早めに範囲を調整しましょう。",
    "最初の10分で資料と条件を確認し、その後は短い集中と休憩を繰り返しましょう。最後に提出形式や記録を確認してください。",
]


def main():
    random.seed(42)
    root = Path("data")
    conversations = root / "conversations" / "university_train.jsonl"
    conversations.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for category, (label, items) in CATEGORIES.items():
        category_dir = root / category; category_dir.mkdir(parents=True, exist_ok=True)
        category_rows = []
        for index in range(50):
            item = items[index % len(items)]
            user = QUESTIONS[index % len(QUESTIONS)].format(item=item)
            assistant = ANSWERS[(index + len(category)) % len(ANSWERS)]
            row = {"id": f"{category}-{index:03d}", "category": label, "user": user, "assistant": assistant,
                   "source": "UniPilot project original", "license": "CC0-1.0"}
            rows.append(row); category_rows.append(row)
        (category_dir / "examples.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in category_rows) + "\n", encoding="utf-8")
    random.shuffle(rows)
    conversations.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    general = root / "general_japanese"; general.mkdir(parents=True, exist_ok=True)
    sentences = [
        f"大学では、計画を立てて{item}に取り組むことが大切です。分からない点は資料を確認し、必要なら質問します。"
        for _, items in CATEGORIES.values() for item in items
    ] * 3
    (general / "original_sentences.txt").write_text("\n".join(sentences) + "\n", encoding="utf-8")
    for name in ["raw", "processed", "university"]: (root / name).mkdir(parents=True, exist_ok=True)
    (root / "SOURCES.md").write_text("# Data sources\n\nAll bundled examples were authored algorithmically for UniPilot Mini and are dedicated to CC0-1.0. No scraped or copyrighted corpus is included. Add future dataset name, URL/author, license, and retrieval date here before training.\n", encoding="utf-8")
    print(f"created {len(rows)} original conversations and {len(sentences)} pretraining sentences")


if __name__ == "__main__": main()
