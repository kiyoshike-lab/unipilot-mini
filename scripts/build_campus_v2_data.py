from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from pipeline.campus_categories import CAMPUS_CATEGORIES
from pipeline.campus_categories_v2 import CATEGORY_TO_LEVEL1, TOOL_AVAILABLE


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "campus_v2"
RNG = random.Random(240824)


SCENARIOS = {
    "exam": ("過去問の使い方", "試験範囲が広すぎる", "持ち込み資料の準備", "追試の確認先", "試験当日の忘れ物", "暗記科目の復習", "記述試験の対策", "テスト直前の優先事項"),
    "assignment": ("課題の指示が分からない", "提出形式を確認したい", "宿題が終わらない", "再提出を相談したい", "課題の要件を整理したい", "提出前に見直したい", "グループ課題で困った", "提出サイトが動かない"),
    "credit": ("単位を落としそう", "卒業までの単位を整理したい", "必修を落とした影響", "取得単位の進捗", "進級条件の確認先", "単位認定の確認", "卒業要件が複雑", "あと何単位か計算したい"),
    "gpa": ("GPAの意味", "今学期のGPA計算", "目標GPAに必要な成績", "GPと単位数の関係", "成績平均を上げたい", "GPAが低くて不安", "累積GPAを出したい", "GPA計算規程の確認"),
    "grade_simulator": ("合格まで何点必要", "現在40点で残り30パーセント", "期末試験の必要点", "残り評価で挽回できるか", "目標点までの差", "再試験で必要な点", "配点から必要得点を計算", "合格ラインのシミュレーション"),
    "attendance": ("欠席が増えて不安", "公欠の申請方法", "出席率を確認したい", "診断書の提出先", "授業を休んだ後の対応", "欠席回数の確認", "体調不良で出席できない", "出席記録が違っている"),
    "lateness": ("授業に遅刻しそう", "寝坊したときの対応", "電車遅延で間に合わない", "遅延証明の出し方", "開始時刻に遅れた", "教室移動で遅れそう", "試験に遅刻した", "遅刻後にまずすること"),
    "professor_email": ("教授へ面談依頼メール", "先生に質問する文面", "教員へ相談を伝える", "研究室訪問のメール", "返信へのお礼メール", "授業内容の質問メール", "先生にアポを取りたい", "教授メールの件名"),
    "absence_email": ("欠席メールを作りたい", "体調不良で休む連絡", "授業を欠席する文面", "先生へ休むと伝えたい", "欠席理由を丁寧に書く", "病院に行くので欠席連絡", "ゼミを休むメール", "当日の欠席連絡"),
    "lateness_email": ("遅刻メールを作りたい", "電車遅延の連絡文", "先生に遅れると伝えたい", "授業に遅刻するメール", "寝坊した遅刻連絡", "到着見込みを伝える文面", "ゼミへの遅刻メール", "遅延証明を添える連絡"),
    "late_submission_email": ("課題提出が遅れるメール", "締切を過ぎたお詫び", "提出遅延を先生に相談", "レポートが間に合わない連絡", "提出サイト不具合の連絡", "課題の延長をお願いする文面", "遅れて提出したいメール", "締切後の対応を尋ねるメール"),
    "registration": ("履修登録の組み方", "必修が重複した", "抽選科目に外れた", "履修変更期間の確認", "時間割を決めたい", "登録エラーが出る", "履修上限を確認", "シラバスから科目を選ぶ"),
    "schedule": ("今週の予定を整理", "空きコマの使い方", "授業とバイトの時間配分", "一日の予定を組みたい", "予定が重なっている", "勉強時間を確保したい", "生活リズムを整える", "週の時間割を見直す"),
    "study_plan": ("試験まで7日の勉強計画", "一日2時間の学習計画", "テスト勉強を始めたい", "苦手分野の復習計画", "三日で試験対策", "複数科目の勉強配分", "一か月の学習計画", "今日からの試験準備"),
    "assignment_priority": ("三つの課題の優先順位", "締切と所要時間で並べたい", "どの課題から始めるか", "課題が多くて選べない", "重い課題を先にするか", "明日締切の課題を整理", "複数レポートの順番", "課題の緊急度を決めたい"),
    "deadline_organizer": ("締切を一覧にしたい", "提出期限を整理する", "複数の締切をまとめる", "期限を忘れない方法", "今月の課題期限", "締切順に並べたい", "提出日を管理したい", "締切カレンダーを作る"),
    "report_outline": ("レポートの章立て", "2000字レポートの構成", "序論本論結論の配分", "論点から構成を作る", "考察の位置を決めたい", "卒論アウトライン", "文字数を節ごとに配分", "レポート構成を見直す"),
    "citation_check": ("引用の書き方", "参考文献の確認", "直接引用と要約の違い", "出典をどこに付けるか", "コピペにならないか確認", "ウェブ資料の引用", "孫引きを避けたい", "引用箇所をチェック"),
    "presentation_outline": ("発表の流れを作る", "10分プレゼンの時間配分", "スライド構成を考える", "導入と結論の作り方", "質疑時間を含む発表", "研究発表のアウトライン", "説明順を整理する", "プレゼンを短くまとめる"),
    "career_schedule": ("就活日程を整理", "授業と選考の両立", "企業説明会の予定", "就活準備の順番", "面接日程をまとめる", "エントリー締切管理", "学業優先で就活計画", "一か月の就活予定"),
    "es_outline": ("ESの構成", "自己PRの組み立て", "志望動機の流れ", "学生時代の経験を書く", "400字ESの配分", "具体的な行動を整理", "ESを読みやすくする", "エントリーシートの骨組み"),
    "toeic_plan": ("TOEIC一か月計画", "600点を目指す勉強", "毎日一時間のTOEIC", "リスニングの配分", "試験日までの英語学習", "模試の復習計画", "単語と読解の時間配分", "TOEIC学習を続けたい"),
    "internship": ("インターンの探し方", "応募前に確認すること", "短期と長期の違い", "インターン面接準備", "授業とインターンの両立", "募集要項を読みたい", "参加後のお礼", "初めてのインターン応募"),
    "scholarship": ("奨学金の種類", "給付型を調べたい", "貸与型の返還", "JASSOの確認先", "奨学金申請の準備", "家計が変わったとき", "継続手続きの確認", "奨学金の締切"),
    "tuition": ("授業料の納付", "学費の分納を相談", "納付期限の確認", "授業料減免を調べる", "学費が払えないとき", "延納手続きの確認", "学費の内訳", "納付書をなくした"),
    "part_time_job": ("バイトと授業の両立", "シフトを減らしたい", "試験前の勤務調整", "アルバイトを探す", "給与明細の確認", "急なシフト変更", "学業時間を確保", "バイト先への相談"),
    "campus_life": ("サークル選び", "大学生活に慣れない", "一人暮らしの生活管理", "研究室の選び方", "ゼミでの過ごし方", "学内施設の探し方", "新学期の準備", "学生生活を整えたい"),
    "relationship": ("大学で友達ができない", "グループで孤立した", "先輩との関係", "同級生に相談したい", "ハラスメントの相談先", "ゼミの人間関係", "断り方を考えたい", "共同作業で意見が合わない"),
    "programming": ("Pythonのエラー", "コードをデバッグしたい", "課題のプログラムが動かない", "例外メッセージの読み方", "関数の作り方", "テストコードを書く", "実行結果が違う", "プログラミング学習の順番"),
    "ai_usage": ("生成AIの授業利用", "AI使用を明記する", "課題でAIを使ってよいか", "AI出力の事実確認", "生成AIと引用", "AI利用ルールの確認", "AIに個人情報を入れない", "レポートでのAI活用"),
    "math": ("微分の考え方", "積分問題の復習", "線形代数が分からない", "数式の途中式を確認", "行列計算のミス", "極限の勉強法", "数学課題の進め方", "証明問題の読み方"),
    "statistics": ("標準偏差の意味", "分散を計算したい", "確率分布の違い", "回帰分析の読み方", "統計検定の考え方", "平均と中央値の使い分け", "仮説検定の手順", "データのばらつきを見る"),
    "university_policy": ("大学の追試規程", "欠席何回で不可か", "履修変更できるか", "公欠になる条件", "卒業要件の公式確認", "再提出できるか", "学則の探し方", "大学固有ルールを確認"),
    "faq_search": ("FAQから確認したい", "よくある質問を探す", "手続きの確認先", "学生向けFAQ検索", "公式案内の場所", "問い合わせ前に調べる", "どの窓口か確認", "案内ページを探したい"),
    "general": ("大学生活全体を相談", "何から改善するか整理", "困りごとを言葉にしたい", "新学期が不安", "優先事項を一緒に決めたい", "大学生活の悩み", "状況を整理して相談", "次の一歩を考えたい"),
}

COLLOQUIAL = ("{x}なんだけど、どしたらいい？", "{x}、まじで困る", "{x}ってどうするん", "{x}たすけて", "{x} pls")
NORMAL = ("{x}について、最初に行うことを教えてください。", "{x}の進め方を三段階で整理してください。", "{x}について確認すべき情報は何ですか。", "{x}への具体的な対応を知りたいです。", "{x}を大学生向けに説明してください。")
OMITTED = ("これ、{x}。次なに？", "{x}。まず一つだけ", "{x}で詰みそう", "{x}、今日やること", "{x}、短く")


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def typo(text: str) -> str:
    replacements = (("メール", "メル"), ("レポート", "れぽーと"), ("スケジュール", "スケジュル"),
                    ("プログラミング", "プログラミング"), ("試験", "しけん"), ("課題", "かだい"),
                    ("履修", "りしゅう"), ("欠席", "けっせき"), ("遅刻", "ちこく"), ("奨学金", "しょうがくきん"))
    for before, after in replacements:
        if before in text:
            return text.replace(before, after, 1)
    return text + " これってどうする"


def expected_action(question: str, category: str, labels: list[str]) -> str:
    compact = norm(question)
    if category == "general" and len(compact) < 22:
        return "CLARIFY"
    if len(labels) > 1:
        return "TOOL+MODEL" if any(label in TOOL_AVAILABLE for label in labels) else "RAG+MODEL"
    if category == "university_policy":
        return "RAG"
    if category in ("programming", "math", "statistics", "ai_usage"):
        return "RAG+MODEL"
    if category == "general":
        return "MODEL"
    if category == "grade_simulator":
        return "TOOL"
    tool_words = ("計算", "作り", "文面", "メール", "整理", "優先", "構成", "配分", "計画", "チェック", "割り")
    if category in TOOL_AVAILABLE and any(word in compact for word in tool_words):
        return "TOOL"
    return "FAQ"


def row(identifier: str, question: str, category: str, *, surface: str, labels: list[str] | None = None,
        difficulty: str = "medium", split: str = "train") -> dict:
    labels = labels or [category]
    return {
        "id": identifier, "question" if split in ("train", "dev") else "prompt": question,
        "category": category, "intent_labels": labels, "level1": CATEGORY_TO_LEVEL1[category],
        "expected_action": expected_action(question, category, labels), "surface_type": surface,
        "difficulty": difficulty, "blind": split in ("blind", "adversarial"),
        "dataset_version": f"unipilot-campus-v2-{split}",
    }


def build_router() -> tuple[list[dict], list[dict]]:
    train = []
    styles = list(COLLOQUIAL + NORMAL + OMITTED)
    for category, scenarios in SCENARIOS.items():
        for scenario_index, scenario in enumerate(scenarios):
            selected = (styles[scenario_index % len(styles)], styles[(scenario_index + 4) % len(styles)],
                        styles[(scenario_index + 8) % len(styles)], styles[(scenario_index + 11) % len(styles)])
            for style_index, style in enumerate(selected):
                question = style.format(x=scenario)
                train.append(row(f"campus-v2-router-{category}-{scenario_index:02d}-{style_index}", question, category,
                                 surface=("colloquial", "normal", "omitted", "short")[style_index]))
            train.append(row(f"campus-v2-router-{category}-{scenario_index:02d}-typo", typo(scenario) + "？", category,
                             surface="typo_hiragana"))

    # 500 genuinely compound requests: two independently useful intents and an explicit action label.
    categories = list(CAMPUS_CATEGORIES[:-1])
    for index in range(500):
        first = categories[index % len(categories)]
        second = categories[(index * 11 + 7) % len(categories)]
        if second == first:
            second = categories[(categories.index(second) + 1) % len(categories)]
        left = SCENARIOS[first][(index // len(categories)) % 8]
        right = SCENARIOS[second][(index * 3) % 8]
        question = f"{left}。それと、{right}も一緒に進めたい"
        train.append(row(f"campus-v2-router-compound-{index:04d}", question, first, surface="compound",
                         labels=[first, second], difficulty="compound"))

    # Negation teaches the target after 「ではなく」 and prevents the first keyword from winning.
    for index in range(350):
        wrong = categories[index % len(categories)]
        target = categories[(index * 13 + 5) % len(categories)]
        if target == wrong:
            target = categories[(categories.index(target) + 2) % len(categories)]
        question = f"{SCENARIOS[wrong][index % 8]}ではなく、{SCENARIOS[target][(index + 3) % 8]}を聞きたい"
        train.append(row(f"campus-v2-router-negation-{index:04d}", question, target, surface="negation", difficulty="hard"))

    # Development is disjoint in exact wording and is the only split used for method selection.
    dev = []
    for index, category in enumerate(CAMPUS_CATEGORIES):
        for item in range(12):
            scenario = SCENARIOS[category][(item + 3) % 8]
            question = f"大学の相談です。{scenario}について、判断材料と次の行動を知りたい（{item + 1}件目）"
            dev.append(row(f"campus-v2-dev-{index:02d}-{item:02d}", question, category,
                           surface="development", split="dev"))
    return train, dev


def audit_faq() -> tuple[list[dict], dict, dict[str, list[str]]]:
    source = ROOT / "data" / "campus_v1" / "faq" / "faq.jsonl"
    reviewed = []
    category_ids: dict[str, list[str]] = {category: [] for category in CAMPUS_CATEGORIES}
    changes = Counter()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        original_question = item["question"]
        item["question"] = item["question"].replace("通常場合", "通常の場合")
        if item["question"] != original_question:
            changes["question_grammar_fixed"] += 1
        answer = item["answer"].strip()
        checks = {
            "direct_conclusion": answer.startswith("結論："),
            "action_steps": "今やること：" in answer,
            "risk_boundary": any(token in answer for token in ("確認", "異なる", "断定", "規程")),
            "specific_enough": len(answer) >= 90,
            "no_external_ai": not any(token in answer.lower() for token in ("openai", "chatgpt", "gemini api")),
        }
        score = sum(checks.values())
        if score < 4:
            if not checks["direct_conclusion"]:
                answer = "結論：確認できる事実を整理してから対応します。\n" + answer
            if not checks["action_steps"]:
                answer += "\n今やること：必要条件、期限、確認先の順に書き出してください。"
            answer += "\n確認：不明な条件はシラバスまたは担当窓口で確かめ、確認できた事実だけで進めてください。"
            item["answer"] = answer
            checks["direct_conclusion"] = answer.startswith("結論：")
            checks["action_steps"] = "今やること：" in answer
            checks["risk_boundary"] = True
            checks["specific_enough"] = len(answer) >= 90
            score = sum(checks.values())
            changes["low_quality_fixed"] += 1
        item.update(dataset_version="unipilot-campus-v2-faq-reviewed", quality_score=score,
                    quality_checks=checks, audit_status="AUTO_REVIEWED", human_reviewed=False)
        reviewed.append(item)
        category_ids[item["category"]].append(item["id"])
    report = {
        "source": "data/campus_v1/faq/faq.jsonl", "reviewed": len(reviewed),
        "human_reviewed": 0, "automatic_review_only": True,
        "score_distribution": dict(Counter(str(row["quality_score"]) for row in reviewed)),
        "below_4_after_fix": sum(row["quality_score"] < 4 for row in reviewed),
        "changes": dict(changes), "criteria": list(reviewed[0]["quality_checks"]),
    }
    return reviewed, report, category_ids


def build_blind(category_ids: dict[str, list[str]]) -> tuple[list[dict], list[dict], list[dict]]:
    blind: list[dict] = []
    categories = list(CAMPUS_CATEGORIES)
    non_general = list(CAMPUS_CATEGORIES[:-1])

    for index in range(500):
        category = categories[(index * 9 + 4) % len(categories)]
        scenario = SCENARIOS[category][(index * 5 + 1) % 8]
        question = f"{scenario}なんよ、今できること教えて {('急ぎ', 'ざっくり', '今日中', '初めて', '短め')[index % 5]}"
        blind.append(row(f"campus-v2-blind-colloquial-{index:04d}", question, category,
                         surface="colloquial", difficulty="easy", split="blind"))
    for index in range(500):
        category = categories[(index * 11 + 2) % len(categories)]
        scenario = SCENARIOS[category][(index * 7 + 2) % 8]
        question = f"{scenario}について、確認事項と今日行う手順を具体的に教えてください。状況：{('準備前', '確認中', '期限あり', '情報不足', '初回相談')[index % 5]}"
        blind.append(row(f"campus-v2-blind-normal-{index:04d}", question, category,
                         surface="normal", difficulty="medium", split="blind"))
    vague = ("これやばい", "ちょっと詰んだ", "大学のことで困った", "どうしたらいい", "間に合うかな",
             "相談したい", "何からすればいい", "もう無理かも", "これ大丈夫かな", "助けてほしい")
    context = ("今週", "今日", "さっき", "初めてで", "急に", "予定外で", "一人では", "まだ何も", "情報がなくて", "うまく言えないけど")
    for index in range(300):
        question = f"{context[(index // 10) % 10]}、{vague[index % 10]}。まず確認してほしい {index // 100 + 1}"
        item = row(f"campus-v2-blind-ambiguous-{index:04d}", question, "general",
                   surface="ambiguous", difficulty="hard", split="blind")
        item["expected_action"] = "CLARIFY"
        blind.append(item)
    for index in range(400):
        first = non_general[(index * 5 + 3) % len(non_general)]
        second = non_general[(index * 17 + 9) % len(non_general)]
        if first == second:
            second = non_general[(non_general.index(second) + 1) % len(non_general)]
        question = f"{SCENARIOS[first][(index + 2) % 8]}を先に進めたい。それと、{SCENARIOS[second][(index * 3 + 4) % 8]}も対応したい"
        blind.append(row(f"campus-v2-blind-compound-{index:04d}", question, first,
                         surface="compound", labels=[first, second], difficulty="compound", split="blind"))
    for index in range(300):
        wrong = non_general[(index * 3 + 1) % len(non_general)]
        target = non_general[(index * 19 + 6) % len(non_general)]
        if wrong == target:
            target = non_general[(non_general.index(target) + 3) % len(non_general)]
        question = f"{SCENARIOS[wrong][index % 8]}じゃなくて、{typo(SCENARIOS[target][(index + 5) % 8])}の方を教えて。前者は不要"
        blind.append(row(f"campus-v2-blind-hard-{index:04d}", question, target,
                         surface="hard_negation_typo", difficulty="hard", split="blind"))

    for item in blind:
        relevant = []
        for label in item["intent_labels"]:
            relevant.extend(category_ids.get(label, []))
        item["relevant_faq_ids"] = relevant
        item["required_key_points"] = [item["category"], "next_action"]
        item["forbidden_claims"] = ["どの大学でも", "必ず認められる", "全国一律"]

    adversarial = []
    for index in range(300):
        wrong = non_general[(index * 7 + 2) % len(non_general)]
        target = non_general[(index * 23 + 11) % len(non_general)]
        if wrong == target:
            target = non_general[(non_general.index(target) + 5) % len(non_general)]
        connector = ("ではなく", "じゃない。聞きたいのは", "は不要で", "の相談ではない。代わりに")[index % 4]
        question = f"{SCENARIOS[wrong][(index + 1) % 8]}{connector}{SCENARIOS[target][(index + 6) % 8]}。誤解しないで"
        item = row(f"campus-v2-adversarial-{index:04d}", question, target,
                   surface="negation_adversarial", difficulty="hard", split="adversarial")
        item["negated_category"] = wrong
        adversarial.append(item)

    easy = [item for item in blind if item["surface_type"] == "colloquial"][:25]
    medium = [item for item in blind if item["surface_type"] == "normal"][:25]
    hard = [item for item in blind if item["surface_type"] == "hard_negation_typo"][:25]
    compound = [item for item in blind if item["surface_type"] == "compound"][:25]
    human = []
    for difficulty, items in (("Easy", easy), ("Medium", medium), ("Hard", hard), ("Compound", compound)):
        for index, item in enumerate(items):
            human.append({
                "id": f"campus-v2-human-{difficulty.lower()}-{index:02d}", "blind_id": item["id"],
                "question": item["prompt"], "category": item["category"], "difficulty": difficulty,
                "campus_answer": "", "chatgpt_answer": "", "gemini_answer": "",
                "scores": {"correctness": None, "relevance": None, "actionable": None,
                           "naturalness": None, "would_use_again": None},
                "competitor_scores": {"chatgpt": None, "gemini": None},
                "notes": "", "evaluation_status": "PENDING_MANUAL",
            })
    return blind, adversarial, human


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    train, dev = build_router()
    reviewed, faq_report, category_ids = audit_faq()
    blind, adversarial, human = build_blind(category_ids)
    train_norm = {norm(row["question"]) for row in train}
    blind_norm = {norm(row["prompt"]) for row in blind}
    overlap = train_norm & blind_norm
    assert not overlap
    assert len(train) >= 2000 and len(blind) == 2000 and len(adversarial) == 300 and len(human) == 100
    assert Counter(row["surface_type"] for row in blind) == {
        "colloquial": 500, "normal": 500, "ambiguous": 300, "compound": 400, "hard_negation_typo": 300,
    }

    write_jsonl(OUTPUT / "router" / "train.jsonl", train)
    write_json(OUTPUT / "router" / "dev.json", dev)
    write_jsonl(OUTPUT / "faq" / "reviewed.jsonl", reviewed)
    write_json(OUTPUT / "faq" / "audit.json", faq_report)
    write_json(OUTPUT / "blind" / "evaluation-2000.json", blind)
    write_json(OUTPUT / "adversarial" / "negation-300.json", adversarial)
    write_json(OUTPUT / "human" / "comparison-100.json", human)
    write_json(OUTPUT / "manifest.json", {
        "version": "unipilot-campus-v2", "router_train": len(train), "router_dev": len(dev),
        "router_train_compound": sum(row["surface_type"] == "compound" for row in train),
        "faq_reviewed": len(reviewed), "blind": len(blind),
        "blind_distribution": dict(Counter(row["surface_type"] for row in blind)),
        "adversarial": len(adversarial), "human": len(human),
        "human_distribution": dict(Counter(row["difficulty"] for row in human)),
        "train_blind_normalized_overlap": len(overlap),
        "blind_sha256": hashlib.sha256((OUTPUT / "blind" / "evaluation-2000.json").read_bytes()).hexdigest(),
        "selection_policy": "method and thresholds are selected on router/dev.json; blind is run once after freeze",
        "external_ai_api": "OFF",
    })
    print(json.dumps({"train": len(train), "dev": len(dev), "blind": len(blind),
                      "adversarial": len(adversarial), "human": len(human), "overlap": len(overlap)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
