from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re


VERSION = "unipilot-dataset-v03"
SEED = 3032026
ROOT = Path("data/v03")
SUBJECTS = ["微積分", "線形代数", "統計学", "英語", "物理学", "化学", "経済学", "心理学", "情報科学", "プログラミング", "日本史", "社会学", "法学", "生物学", "哲学", "教育学", "会計学", "データ分析", "文学", "基礎演習"]
PLACES = ["図書館", "自宅", "自習室", "空き教室", "学習スペース", "大学のカフェ", "共同学習室", "静かな教室", "オンライン環境", "研究室前のラウンジ"]
GOALS = ["今日中に着手する", "要点を整理する", "期限前に見直す", "不明点を質問する", "無理なく継続する"]
INTENTS = {
    "assignment": "ASK_DEADLINE", "exam": "ASK_EXAM_PLAN", "credit": "ASK_CREDIT", "email": "ASK_EMAIL",
    "attendance": "ASK_ATTENDANCE", "registration": "ASK_REGISTRATION", "report": "ASK_REPORT",
    "study": "ASK_STUDY_PLAN", "schedule": "ASK_DAILY_PLAN", "general": "ASK_UNKNOWN_INFO",
}
CATEGORY_KEYWORDS = {
    "assignment": ["課題", "締切", "提出"], "exam": ["試験", "範囲", "復習"], "credit": ["単位", "シラバス", "教務"],
    "email": ["件名", "先生", "連絡"], "attendance": ["出席", "欠席", "シラバス"],
    "registration": ["履修", "必修", "シラバス"], "report": ["レポート", "構成", "引用"],
    "study": ["勉強", "復習", "計画"], "schedule": ["予定", "締切", "優先"], "general": ["確認", "情報", "分かりません"],
}
FORBIDDEN = {
    "assignment": ["欠席メール", "履修登録"], "exam": ["欠席メール", "履修登録"], "credit": ["必ず単位", "出席率70%"],
    "email": ["履修登録", "試験範囲"], "attendance": ["必ず単位", "出席率70%"], "registration": ["欠席メール", "必ず単位"],
    "report": ["履修登録", "欠席メール"], "study": ["欠席メール", "履修登録"], "schedule": ["欠席メール", "必ず単位"],
    "general": ["午前9時です", "午後1時です"],
}
CONVERSATION_COUNTS = {"assignment": 1800, "exam": 1800, "study": 1600, "email": 1200, "credit": 1000,
                       "schedule": 1000, "attendance": 900, "registration": 900, "report": 900, "general": 900}


def split_name(family: int) -> str:
    return "train" if family < 18 else ("validation" if family == 18 else "test")


def uid(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def diversity_score(text: str) -> float:
    compact = re.sub(r"\s", "", text)
    if len(compact) < 2: return 0.0
    grams = [compact[i:i + 2] for i in range(len(compact) - 1)]
    return len(set(grams)) / len(grams)


def quality(row: dict) -> dict:
    answer = row.get("assistant", row.get("text", ""))
    expected = row.get("expected_keywords", [])
    alignment = sum(word in answer for word in expected) / max(1, len(expected))
    return {"length_score": min(1.0, len(answer) / 50), "diversity_score": diversity_score(answer),
            "keyword_alignment": alignment, "format_valid": bool(answer.strip()) and (row.get("kind") != "conversation" or row.get("eos_required") is True)}


def general_rows(count: int = 20000) -> list[dict]:
    activities = ["資料を読む", "予定を確認する", "ノートを整理する", "買い物へ行く", "料理を作る", "散歩をする", "机を片付ける", "連絡を返す", "本を読む", "練習する"]
    next_actions = ["要点を書く", "結果を見直す", "休憩する", "明日の準備をする", "記録を残す", "質問をまとめる", "時間を測る", "順番を変える", "道具を戻す", "家族に伝える"]
    topics = ["時間", "日付", "数字", "比較", "順序", "理由", "方法", "場所", "計画", "結果"]
    connectors = ["そのため", "次に", "一方で", "たとえば", "まず", "最後に", "ただし", "また", "つまり", "このあと"]
    patterns = [
        "今日は{time}から{activity}。{connector}、{next_action}。{topic}も確認する。",
        "{activity}には{minutes}分使う。終わったら{next_action}ので、無理のない順序で進める。",
        "{place}で{activity}予定だ。{connector}、{topic}を一文で記録する。",
        "{days}日後に向けて、まず{activity}。次に{next_action}と進める。",
        "{topic}を比べるときは条件をそろえる。今回は{activity}から始める。",
        "なぜ{activity}のかを考え、理由を短く書く。{connector}、{next_action}。",
        "朝は{activity}、午後は{next_action}予定だ。間に短い休憩を入れる。",
        "予定が変わったため、{activity}を{time}へ移す。変更理由も記録する。",
        "一つ目は{activity}、二つ目は{next_action}。この順番なら迷いにくい。",
        "分からない点を調べてから{activity}。最後に{topic}を確認する。",
        "{minutes}分集中したら休憩し、その後に{next_action}。時間を区切る。",
        "昨日は{activity}、今日は{next_action}。違いを{topic}の観点で比べる。",
        "急ぐときも条件を読み、{activity}。終わったら必ず{next_action}。",
        "{place}は静かなので{activity}に向いている。{connector}、{next_action}。",
        "質問は『いつ{activity}のか』。答えは『{time}から始める』。",
        "大きな作業を分け、最初に{activity}。次の作業は{next_action}。",
        "{days}日間の予定を作り、毎日{minutes}分だけ{activity}。進捗も記録する。",
        "数字を見るときは単位も確認する。{connector}、{topic}を取り違えない。",
        "結論、理由、具体例の順で説明する。例として{activity}を取り上げる。",
        "余裕のある予定を作る。遅れた場合は{next_action}前に調整する。",
    ]
    times = ["朝7時", "午前9時", "正午", "午後2時", "夕方5時", "夜8時", "昼休み", "帰宅後", "朝食後", "夕食前"]
    rows = []
    for index in range(count):
        family = index % 20; activity = activities[(index // 20) % 10]; next_action = next_actions[(index // 200) % 10]
        topic = topics[(index // 2000) % 10]; connector = connectors[(index // 4000 + family) % 10]
        text = patterns[family].format(time=times[(index // 3) % 10], activity=activity, next_action=next_action, topic=topic,
                                       connector=connector, minutes=10 + (index * 7) % 111, days=1 + (index * 11) % 30,
                                       place=PLACES[(index // 7) % 10])
        text += f"この予定では{topic}を意識し、{activity}の後に{next_action}。"
        row = {"id": f"v03-a-{uid(index)}", "dataset_version": VERSION, "stage": "A", "kind": "general",
               "template_family": f"a-{family:02d}", "split": split_name(family), "text": text, "license": "CC0-1.0"}
        row["quality"] = quality(row); rows.append(row)
    return rows


def university_answer(category: str, subject: str, days: int, progress: int, variant: int) -> str:
    starts = ["まず状況を整理しましょう。", "焦らず、確認から始めましょう。", "今できることを順番に進めましょう。", "短い手順に分けると進めやすいです。", "最初の一つを決めましょう。"]
    ends = ["最後に条件をもう一度確認してください。", "分からない点は早めに大学へ確認すると安心です。", "予定には短い休憩も入れてください。", "終わった項目を記録すると進捗が分かります。", "無理な場合は担当者へ早めに相談してください。"]
    if category == "assignment": core = f"{subject}の課題は、提出締切と形式を先に確認します。残り{days}日なら、未完了の作業を分け、今日の提出準備から始めると進めやすいです。"
    elif category == "exam": core = f"{subject}の試験範囲を確認し、基礎、問題演習、苦手部分の復習という順で勉強します。残り{days}日に合わせて、一日ごとの範囲を決めると安心です。"
    elif category == "credit": core = f"{subject}の単位条件は授業や大学によって異なります。シラバスで評価方法を確認し、不明点は担当教員や教務へ相談してください。"
    elif category == "email":
        email_variants = [
            "件名：授業欠席のご連絡\n\n○○先生\nお世話になっております。○○学部の〈学生氏名〉です。本日の授業を欠席いたします。直前の連絡となり申し訳ありません。よろしくお願いいたします。",
            "件名：課題提出についてのご相談\n\n○○先生\nお世話になっております。○○学部の〈学生氏名〉です。課題提出について確認したいことがあり、ご連絡しました。お手数ですが、よろしくお願いいたします。",
            "件名：遅刻のご連絡\n\n○○先生\nお世話になっております。○○学部の〈学生氏名〉です。本日の授業に遅れる見込みです。ご迷惑をおかけして申し訳ありません。よろしくお願いいたします。",
            "件名：面談のお願い\n\n○○先生\nお世話になっております。○○学部の〈学生氏名〉です。授業についてご相談したく、面談可能な時間を伺いたくご連絡しました。よろしくお願いいたします。",
            "件名：授業内容についての質問\n\n○○先生\nお世話になっております。○○学部の〈学生氏名〉です。授業内容について質問があり、ご連絡しました。お時間のあるときにご確認いただけますと幸いです。",
        ]
        return email_variants[variant % 5]
    elif category == "attendance": core = f"{subject}の出席条件は授業によって異なります。シラバスと出席記録を確認し、欠席の扱いが不明なら担当教員へ相談してください。"
    elif category == "registration": core = f"{subject}を履修する前に、必修・選択区分、時間割、シラバスを確認します。卒業要件は大学の履修要項や教務の案内を基準にしてください。"
    elif category == "report": core = f"{subject}のレポートは、問いの確認、資料集め、構成、執筆、引用確認、推敲に分けます。残り{days}日なら、まず構成を作るのがおすすめです。"
    elif category == "study": core = f"{subject}の勉強は、今日の目標を一つに絞り、短い集中と休憩を繰り返します。最後に復習し、次回の計画を一行で残してください。"
    elif category == "schedule": core = f"締切、試験日、重要度、残り作業を一覧にします。期限が近く重要な予定から優先し、{subject}を始める時刻も決めましょう。"
    else:
        unknown_variants = ["試験時間", "教室", "担当者", "提出場所", "個別の予定"]
        return f"{unknown_variants[variant % 5]}の情報が登録されていないため、現在は分かりません。時間割や大学の案内を確認してください。"
    return starts[variant % 5] + core + ends[(variant + days) % 5]


def university_text_rows(count: int = 10000) -> list[dict]:
    categories = list(CONVERSATION_COUNTS)
    rows = []; occurrences = Counter()
    for index in range(count):
        family = index % 20; category = categories[(index // 20) % len(categories)]; occurrence = occurrences[category]; occurrences[category] += 1
        subject = SUBJECTS[occurrence % 20]; place = PLACES[(occurrence // 20) % 10]; goal = GOALS[(occurrence // 200) % 5]
        days = 1 + (occurrence * 7) % 30; progress = (occurrence * 3) % 11 * 10
        advice = university_answer(category, subject, days, progress, index % 5)
        intro = ["大学生活では状況を整理してから行動します。", "期限と条件を先に確認することが大切です。", "無理のない計画は継続しやすくなります。", "不明点は公式情報で確かめます。", "作業は小さな手順に分けられます。"][(index // 4000 + family) % 5]
        text = f"{intro}{advice}{place}で取り組み、目標を「{goal}」とする場合も、現在の進捗{progress}%と残り時間に合わせて手順を調整します。"
        expected = CATEGORY_KEYWORDS[category]
        row = {"id": f"v03-b-{uid(index)}", "dataset_version": VERSION, "stage": "B", "kind": "university_text",
               "category": category, "intent": INTENTS[category], "expected_keywords": expected,
               "template_family": f"b-{category}-{family:02d}", "split": split_name(family), "text": text, "license": "CC0-1.0"}
        row["quality"] = quality(row); rows.append(row)
    return rows


def user_question(category: str, subject: str, days: int, progress: int, place: str, family: int) -> str:
    patterns = {
        "assignment": [f"{subject}の課題が{days}日後締切で、まだ{progress}%です。何から進めればいい？", f"課題が重なっています。{subject}はあと{days}日ですが、どう優先すればいい？"],
        "exam": [f"{subject}の試験が{days}日後なのに、勉強が{progress}%しか進んでいません。何を優先する？", f"{days}日後の{subject}の試験が不安です。復習計画を教えて。"],
        "credit": [f"{subject}の単位が心配です。何を確認すればいい？", f"{subject}の成績と単位条件について、どこへ確認したらいい？"],
        "email": [f"{subject}の授業を欠席するので、教授へメールを送りたいです。", f"{subject}の課題提出が遅れそうです。先生への連絡はどう書けばいい？"],
        "attendance": [f"{subject}の欠席が増えて、出席状況が不安です。", f"{subject}を休みました。出席条件はどう確認すればいい？"],
        "registration": [f"{subject}を履修するか迷っています。何を確認すればいい？", f"履修登録で{subject}を選ぶ前に見る項目を教えて。"],
        "report": [f"{subject}のレポートが{days}日後締切で、進捗{progress}%です。どう進める？", f"{subject}のレポート構成が決まりません。最初に何をする？"],
        "study": [f"{subject}を勉強したいけれど集中できません。今日の計画を作って。", f"{subject}の復習を{place}で進めます。短い勉強計画を教えて。"],
        "schedule": [f"締切と試験が重なりました。{subject}を含め、今日の予定をどう決める？", f"課題が3つあります。{subject}から始めるべきか優先順位を決めたい。"],
        "general": [f"明日の{subject}の試験は何時ですか？", f"私の{subject}の教室を教えてください。情報はまだ渡していません。"],
    }
    suffixes = ["短く教えてください。", "落ち着いて進めたいです。", "最初の一歩を知りたいです。", "無理のない方法がいいです。", "確認する順番も知りたいです。"]
    qualifiers = ["今日から始めます。", "必要なら大学にも確認します。"]
    return patterns[category][family % 2] + suffixes[(family // 2) % 5] + qualifiers[(family // 10) % 2]


def conversation_rows() -> list[dict]:
    rows = []
    for category, count in CONVERSATION_COUNTS.items():
        for index in range(count):
            family = index % 20; subject = SUBJECTS[(index // 20) % 20]; place = PLACES[(index // 400) % 10]
            days = 1 + (index * 7) % 14; progress = (index * 11) % 11 * 10
            user = user_question(category, subject, days, progress, place, family)
            user += f"{place}で進める予定です。"
            assistant = university_answer(category, subject, days, progress, index % 5)
            expected = CATEGORY_KEYWORDS[category]
            row = {"id": f"v03-c-{uid(category, index)}", "dataset_version": VERSION, "stage": "C", "kind": "conversation",
                   "category": category, "intent": INTENTS[category], "expected_keywords": expected, "forbidden_keywords": FORBIDDEN[category],
                   "template_family": f"c-{category}-{family:02d}", "split": split_name(family),
                   "context": {"subject": subject, "days_remaining": days, "progress_percent": progress, "place": place},
                   "user": user, "assistant": assistant, "eos_required": True, "license": "CC0-1.0"}
            row["quality"] = quality(row)
            rows.append(row)
    return rows


def fixed_prompts() -> list[dict]:
    rows = []
    categories = list(INTENTS)
    examples = {
        "assignment": "課題が3つあるんだけど、どれからやればいい？", "exam": "明日試験だけど、まだ何もしていません。",
        "credit": "必修科目の単位が心配です。", "email": "教授に今日休むとメールしたいです。",
        "attendance": "欠席が増えて出席状況が不安です。", "registration": "履修科目をどう選べばいい？",
        "report": "レポートの構成が決まりません。", "study": "試験勉強の計画を立てたいです。",
        "schedule": "締切と試験が重なっています。今日の優先順位は？", "general": "明日の試験は何時ですか？",
    }
    for category in categories:
        for index in range(30):
            subject = SUBJECTS[(index * 3 + len(category)) % 20]; days = 1 + (index * 7) % 14
            prompt = examples[category] if index == 0 else f"{examples[category]} 科目は{subject}で、残り{days}日です。"
            rows.append({"id": f"fixed-v03-{category}-{index:02d}", "prompt": prompt, "category": category, "intent": INTENTS[category],
                         "expected_keywords": CATEGORY_KEYWORDS[category], "forbidden_keywords": FORBIDDEN[category], "held_out": True})
    return rows


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def main():
    stage_rows = {"stage_a": general_rows(), "stage_b": university_text_rows(), "stage_c": conversation_rows()}
    excluded = []
    for stage, rows in stage_rows.items():
        valid = []
        seen = set()
        for row in rows:
            text = row.get("text") or row["user"] + "\n" + row["assistant"]
            signature = hashlib.sha256(text.encode()).hexdigest()
            reasons = []
            if signature in seen: reasons.append("duplicate")
            if not row["quality"]["format_valid"]: reasons.append("format")
            if row["quality"]["diversity_score"] < 0.45: reasons.append("repetition")
            if row["kind"] == "conversation" and (len(row["assistant"]) < 25 or row["quality"]["keyword_alignment"] == 0): reasons.append("alignment")
            if reasons: excluded.append({"id": row["id"], "reasons": reasons})
            else: seen.add(signature); valid.append(row)
        stage_rows[stage] = valid
        for split in ["train", "validation", "test"]:
            selected = [row for row in valid if row["split"] == split]
            random.Random(SEED + len(stage) + len(split)).shuffle(selected)
            write_jsonl(ROOT / stage / f"{split}.jsonl", selected)
    prompts = fixed_prompts(); Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/fixed_prompts_v03.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    human = [{"id": row["id"], "prompt": row["prompt"], "category": row["category"], "model_answer": "", "score": None,
              "score_guide": {"0": "意味不明", "1": "日本語だが無関係", "2": "一部関連", "3": "意味が通る", "4": "良い回答"}, "notes": ""} for row in prompts[:50]]
    Path("evaluation/human-eval-v03.json").write_text(json.dumps(human, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"dataset": VERSION, "source_dataset": "unipilot-dataset-v02", "source_samples": 50000,
                "final_samples": sum(map(len, stage_rows.values())), "excluded_from_v02_target": 50000 - sum(map(len, stage_rows.values())),
                "stages": {key: len(value) for key, value in stage_rows.items()}, "splits": dict(Counter(row["split"] for rows in stage_rows.values() for row in rows)),
                "intents": sorted(set(row.get("intent") for rows in stage_rows.values() for row in rows if row.get("intent"))),
                "license": "CC0-1.0", "generator": "local rule-based", "seed": SEED}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "excluded.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
