from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re


SEED = 20260815
ROOT = Path("data")
CATEGORY_COUNTS = {
    "assignments": 3000, "exams": 3000, "study_planning": 3000, "credits": 2000,
    "professor_email": 2000, "registration": 1600, "attendance": 1400,
    "reports": 1000, "presentations": 1000, "campus_life": 1000, "schedule": 1000,
}
SUBJECTS = ["微積分", "線形代数", "統計学", "英語", "物理学", "化学", "経済学", "心理学", "情報科学", "プログラミング", "日本史", "社会学", "法学", "生物学", "哲学", "教育学", "会計学", "データ分析", "文学", "基礎演習"]
PLACES = ["図書館", "自宅", "自習室", "空き教室", "学習スペース", "研究室前のラウンジ", "オンライン環境", "大学のカフェ", "静かな教室", "共同学習室"]
GOALS = ["今日中に着手したい", "無理なく続けたい", "提出前に見直したい", "苦手部分を減らしたい", "予定の遅れを戻したい", "短時間で要点を確認したい", "質問事項を整理したい", "作業順を決めたい", "休憩も確保したい", "今週中に目途を付けたい"]
USER_PATTERNS = [
    "{subject}の{topic}が{days}日後で、進捗は{progress}%です。何から始めればいい？",
    "{days}日後の{subject}の{topic}が不安です。まだ{progress}%しか進んでいません。",
    "{subject}の{topic}、残り{days}日なのに進捗{progress}%です。どうしよう。",
    "相談です。{subject}の{topic}まで{days}日で、今は{progress}%です。",
    "{topic}の予定を立てたいです。科目は{subject}、期限まで{days}日、進捗{progress}%です。",
    "今から{subject}の{topic}を進めます。{days}日で終えるにはどう分ける？進捗は{progress}%です。",
    "{subject}の{topic}を優先すべきですか？あと{days}日で、{progress}%まで終わっています。",
    "{topic}が重なって焦っています。{subject}は{days}日後、進捗は{progress}%です。",
    "{days}日後に{subject}の{topic}があります。無理のない進め方を教えて。現在{progress}%です。",
    "{subject}の{topic}について短い計画を作って。残り{days}日、進捗{progress}%です。",
    "まだ{progress}%ですが、{days}日後までに{subject}の{topic}を終えたいです。",
    "{subject}の{topic}はどの順番で取り組むといい？期限まで{days}日です。",
    "今日は{subject}の{topic}を進めたいです。進捗{progress}%で、あと{days}日あります。",
    "{subject}の{topic}に手が付いていません。{days}日後に間に合わせる方法は？",
    "{topic}の準備時間が足りません。{subject}は残り{days}日、進捗{progress}%です。",
    "{subject}の{topic}を見直したいです。{days}日後までに何を確認すればいい？",
    "{topic}で迷っています。{subject}、残り{days}日、現在{progress}%という状況です。",
    "{days}日後の{topic}に向けて、{subject}を今日どこまで進めるべき？",
    "{subject}の{topic}を後回しにしてしまいました。あと{days}日です。",
    "落ち着いて{subject}の{topic}を進めたいです。期限まで{days}日、進捗{progress}%です。",
]
TOPICS = {
    "assignments": "課題と締切", "exams": "試験", "study_planning": "勉強計画", "credits": "単位の確認",
    "professor_email": "教授への連絡", "registration": "履修登録", "attendance": "出席状況",
    "reports": "レポート", "presentations": "プレゼン", "campus_life": "大学生活の相談", "schedule": "予定管理",
}
EVAL_KEYWORDS = {"assignments": ["課題", "締切"], "exams": ["試験", "復習"], "credits": ["単位", "シラバス"],
                 "registration": ["履修", "シラバス"], "professor_email": ["件名", "連絡"], "attendance": ["出席", "シラバス"],
                 "reports": ["レポート", "構成"], "schedule": ["予定", "締切"], "study_planning": ["計画", "復習"], "campus_life": ["相談", "予定"]}


def split_for_family(family: int) -> str:
    return "train" if family < 18 else ("validation" if family == 18 else "test")


def stable_id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def assistant_for(category: str, subject: str, days: int, progress: int, required: bool, variant: int) -> str:
    remaining = max(0, 100 - progress)
    openings = ["まず状況を整理しましょう。", "焦らず、できる作業から始めましょう。", "今からでも順番を決めれば進めやすいです。", "最初に条件を確認しましょう。", "短い時間でも着手するのがおすすめです。"]
    close = ["最後に提出条件も確認してください。", "終わった項目に印を付けると進捗が見えます。", "無理な場合は早めに担当者へ相談しましょう。", "休憩時間も予定に含めてください。", "大学の公式情報も確認すると安心です。"]
    start = openings[variant % len(openings)]
    end = close[(variant + days) % len(close)]
    if category == "assignments":
        core = f"{subject}の締切と提出形式を確認し、残り{remaining}%を資料確認・作成・見直しに分けてください。{days}日あるので、今日の作業を一つ決めましょう。"
    elif category == "exams":
        core = f"{subject}の試験範囲を確認し、基礎確認、問題演習、苦手分野の復習の順に進めましょう。残り{days}日に合わせて毎日の範囲を小さく区切ると安全です。"
    elif category == "study_planning":
        core = f"{subject}は25分の集中単位に分け、今日やる内容を一つ、明日以降を二つ決めましょう。進捗{progress}%から無理のない量を設定してください。"
    elif category == "credits":
        core = f"単位条件は大学や授業によって異なります。{subject}のシラバスで評価方法と必要条件を確認し、不明点は担当教員や教務へ問い合わせるのがおすすめです。"
    elif category == "professor_email":
        core = f"件名に授業名と用件を書き、宛名、所属と〈学生氏名〉、簡潔な状況、確認したい点、お詫び、結びの順にまとめましょう。個人情報は送信前に自分で確認してください。"
    elif category == "registration":
        core = f"{subject}のシラバス、必修・選択区分、時間割の重複を確認しましょう。卒業要件は大学ごとに異なるため、履修要項と教務の案内を基準にしてください。"
    elif category == "attendance":
        core = f"出席条件は授業ごとに異なります。{subject}のシラバスと現在の記録を確認し、欠席理由や扱いが不明なら早めに担当教員へ相談しましょう。"
    elif category == "reports":
        core = f"{subject}のレポートを問いの確認、資料集め、構成、執筆、引用確認、推敲に分けましょう。残り{days}日なら、今日は構成まで作ると進めやすいです。"
    elif category == "presentations":
        core = f"{subject}の要点を三つに絞り、導入・説明・まとめの順でスライドを作りましょう。完成後は時間を測って声に出し、質疑応答も一度練習してください。"
    elif category == "campus_life":
        core = f"予定と負担を書き出し、今週優先することを一つに絞りましょう。困りごとが続く場合は、学生相談室や身近な人へ早めに相談して構いません。"
    else:
        core = f"締切を一覧にし、近さ、重要度、必要時間で並べましょう。{subject}は{days}日後なので、今日の予定に具体的な開始時刻を入れてください。"
    return f"{start}{core}{end}"


def make_university_dialogues() -> list[dict]:
    rows = []
    for category, count in CATEGORY_COUNTS.items():
        for index in range(count):
            family = index % 20
            subject = SUBJECTS[(index // 20 + len(category)) % len(SUBJECTS)]
            place = PLACES[(index // 400) % len(PLACES)]
            goal = GOALS[(index // 800 + family) % len(GOALS)]
            days = 1 + (index * 11) % 30
            progress = (index * 13) % 11 * 10
            required = index % 3 != 0
            user = USER_PATTERNS[family].format(subject=subject, topic=TOPICS[category], days=days, progress=progress)
            assistant = assistant_for(category, subject, days, progress, required, index % 5)
            detail = f" 対象は{('必修' if required else '選択')}科目で、{place}を使い、{goal}と思っています。"
            user += detail
            rows.append({
                "id": f"v02-u-{stable_id(category, index)}", "kind": "dialogue", "category": category,
                "template_family": f"{category}-dialogue-{family:02d}", "split": split_for_family(family),
                "context": {"subject": subject, "days_remaining": days, "progress_percent": progress, "required": required, "place": place, "goal": goal},
                "user": user, "assistant": assistant, "source": "UniPilot v0.2 rule-based original", "license": "CC0-1.0",
            })
    return rows


def make_university_text(dialogues: list[dict], count: int = 10000) -> list[dict]:
    rows = []
    per_category = {category: round(count * value / 20000) for category, value in CATEGORY_COUNTS.items()}
    # Correct rounding while keeping the requested total.
    difference = count - sum(per_category.values())
    per_category["assignments"] += difference
    by_category = {category: [row for row in dialogues if row["category"] == category] for category in CATEGORY_COUNTS}
    for category, category_count in per_category.items():
        source_rows = by_category[category]
        for index in range(category_count):
            source = source_rows[index]
            family_number = int(source["template_family"].rsplit("-", 1)[1])
            context = source["context"]
            text = (f"相談例は「{source['user']}」という状況です。{context['subject']}について考えるときは、期限まで{context['days_remaining']}日、進捗{context['progress_percent']}%という条件を整理します。"
                    f"{context['place']}を使い、目標は「{context['goal']}」です。{source['assistant']} 手順は状況に応じて調整します。")
            rows.append({"id": f"v02-t-{stable_id(category, index)}", "kind": "university_text", "category": category,
                         "template_family": f"{category}-text-{family_number:02d}", "split": split_for_family(family_number),
                         "text": text, "source": "UniPilot v0.2 rule-based original", "license": "CC0-1.0"})
    return rows


GENERAL_PATTERNS = [
    "今日は{time}に{activity}をします。終わったら{next_activity}をします。",
    "{thing_a}と{thing_b}を比べると、今回は{thing_a}を先に選びます。理由は予定に合うからです。",
    "まず{activity}を行い、次に{next_activity}を確認し、最後に記録します。",
    "{days}日後の予定に向けて、毎日{minutes}分ずつ準備します。",
    "質問は『{activity}をいつ始めますか』です。答えは『{time}から始めます』です。",
    "予定が変わったため、{activity}を{time}へ移しました。変更した理由もメモします。",
    "昨日は{thing_a}を確認しました。今日は{thing_b}を確認します。",
    "{minutes}分作業したら短く休み、その後で{next_activity}へ進みます。",
    "一つ目は{activity}、二つ目は{next_activity}です。この順序なら迷いにくくなります。",
    "{thing_a}が必要なのは、{activity}を安全に進めるためです。",
    "朝は{activity}、午後は{next_activity}を行う予定です。",
    "今月の目標を小さな週単位の予定に分けると、進み具合を確認しやすくなります。",
    "分からない言葉は調べ、要点を自分の言葉で一文にまとめます。",
    "数字を確認するときは、単位と期限を一緒に見ることが大切です。",
    "急ぐ場合でも、最初に条件を読み、最後に間違いがないか確認します。",
    "{activity}が終わらないときは、作業を小さく分けて一つずつ進めます。",
    "{thing_a}は{thing_b}より先に使いますが、状況によって順番を変えます。",
    "今日は{days}個の項目を確認し、それぞれに{minutes}分を使います。",
    "理由を説明するときは、結論、根拠、具体例の順に話すと伝わりやすいです。",
    "予定には余白を作り、遅れた場合に調整できるようにします。",
]


def make_general(count: int = 20000) -> list[dict]:
    activities = ["資料を読む", "部屋を整える", "買い物へ行く", "料理を作る", "散歩をする", "ノートを書く", "予定を確認する", "本を読む", "練習をする", "連絡を返す"]
    things = ["時間", "方法", "手順", "道具", "予定", "記録", "休憩", "説明", "数字", "結果"]
    times = ["朝7時", "午前9時", "正午", "午後2時", "夕方5時", "夜8時"]
    rows = []
    for index in range(count):
        family = index % 20
        activity = activities[(index // 20) % len(activities)]
        next_activity = activities[(index // 200) % len(activities)]
        thing_a = things[(index // 2000) % len(things)]
        thing_b = things[(index // 4000 + family) % len(things)]
        text = GENERAL_PATTERNS[family].format(
            time=times[(index // 3) % len(times)], activity=activity,
            next_activity=next_activity, thing_a=thing_a,
            thing_b=thing_b, days=1 + (index * 11) % 31, minutes=10 + (index * 13) % 111,
        )
        text += f" 扱う要素は{thing_a}と{thing_b}で、{activity}の後に{next_activity}を行う想定です。"
        rows.append({"id": f"v02-g-{stable_id(index)}", "kind": "general", "category": "general_japanese",
                     "template_family": f"general-{family:02d}", "split": split_for_family(family), "text": text,
                     "source": "UniPilot v0.2 rule-based original", "license": "CC0-1.0"})
    return rows


def fixed_prompts() -> list[dict]:
    categories = ["assignments", "exams", "credits", "registration", "professor_email", "attendance", "reports", "schedule", "study_planning", "campus_life"]
    prompts = []
    for category in categories:
        for index in range(30):
            subject = SUBJECTS[(index * 3 + len(category)) % len(SUBJECTS)]
            days = 1 + (index * 7) % 21
            wording = [
                "{subject}について相談です。{days}日以内に何を優先するといい？",
                "{subject}のことで困っています。残り{days}日ならどう動けばいい？",
                "{days}日後までに{subject}の準備をしたいです。最初の一歩を教えて。",
            ][index % 3].format(subject=subject, days=days)
            prompts.append({"id": f"fixed-v02-{category}-{index:02d}", "category": category, "prompt": f"{TOPICS[category]}の相談です。{wording}",
                            "keywords": EVAL_KEYWORDS[category] + [subject], "held_out": True})
    return prompts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    dialogues = make_university_dialogues()
    university_text = make_university_text(dialogues)
    general = make_general()
    all_rows = dialogues + university_text + general
    # Exact de-duplication on actual learning content.
    unique = {}
    for row in all_rows:
        content = row.get("text") or (row.get("user", "") + "\n" + row.get("assistant", ""))
        unique.setdefault(hashlib.sha256(content.encode()).hexdigest(), row)
    all_rows = list(unique.values())
    for category in CATEGORY_COUNTS:
        write_jsonl(ROOT / "university" / category / "samples.jsonl", [row for row in dialogues + university_text if row["category"] == category])
    write_jsonl(ROOT / "curated" / "university_dialogues.jsonl", dialogues)
    write_jsonl(ROOT / "curated" / "university_text.jsonl", university_text)
    write_jsonl(ROOT / "curated" / "general_japanese.jsonl", general)
    for split in ["train", "validation", "test"]:
        split_rows = [row for row in all_rows if row["split"] == split]
        random.Random(SEED + len(split)).shuffle(split_rows)
        write_jsonl(ROOT / split / "v02.jsonl", split_rows)
    prompts = fixed_prompts()
    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/fixed_prompts_v02.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total": len(all_rows), "university_dialogues": len(dialogues), "university_text": len(university_text),
                      "general_japanese": len(general), "splits": Counter(row["split"] for row in all_rows),
                      "fixed_prompts": len(prompts)}, ensure_ascii=False, default=dict))


if __name__ == "__main__": main()
