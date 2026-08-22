from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

try:
    from scripts.prepare_dataset_v05 import SEEDS as V05_SEEDS
except ModuleNotFoundError:  # direct `python scripts/prepare_dataset_v06.py`
    from prepare_dataset_v05 import SEEDS as V05_SEEDS


VERSION = "unipilot-quality-preserving-curriculum-v06"
ROOT = Path("data/v06")
LICENSE = "CC0-1.0"
SUBJECT_WORDS = ("法学", "経済学", "心理学", "情報科学", "日本史", "統計学", "英語", "数学", "物理学")

PROMPT_FORMS = (
    "大学生活の相談です。{}",
    "{} まず何をすべきか教えてください。",
    "{} 結論から簡潔に答えてください。",
    "{} 自分で確認すべき点も含めてください。",
    "困っています。{}",
    "{} 一般的な対応を教えてください。",
    "{} 今からできる行動を知りたいです。",
    "{} 判断の順番を教えてください。",
    "{} 理由も短く説明してください。",
    "{} 大学固有の情報は決めつけずに答えてください。",
    "{} 何を確認すればよいですか。",
    "{} 実行しやすい形で教えてください。",
    "{} 要点を整理してください。",
    "{} 迷わないように手順を示してください。",
    "{} 注意点も一つ挙げてください。",
    "{} 具体的な次の一歩を教えてください。",
    "{} 事実と一般的な助言を分けて答えてください。",
    "{} 分からない条件は推測しないでください。",
    "{} 今日できることから説明してください。",
    "{} 優先順位を付けてください。",
    "{} 確認先があれば示してください。",
    "{} 短期的な対応と今後の対策を教えてください。",
    "{} 前提が足りなければ、その点も明示してください。",
    "{} 結論、理由、行動の順で答えてください。",
)

DETAIL_SUFFIXES = (
    "確認した内容と次に行うことを一つずつメモすると、落ち着いて進められます。",
    "条件が分からない部分は推測で埋めず、公式案内か担当窓口で確かめてください。",
    "最初の行動を小さく決め、終わったら状況を見直すと進めやすくなります。",
    "期限や指定方法がある場合は、一般的な助言よりその公式情報を優先してください。",
)

CORRECTIONS = (
    ("科目が不明なのに法学の試験だと決めつけた回答", "科目は特定できません。まず試験範囲と配点を確認し、重要項目の理解、問題演習、間違い直しの順で進めてください。"),
    ("履修登録は全国の大学でいつでも変更できると断定した回答", "履修登録の変更可否と期限は大学によって異なります。履修案内と学生ポータルを確認し、不明なら教務窓口へ問い合わせてください。"),
    ("GPAは必ず同じ計算式だと説明した回答", "GPAは成績を数値化して平均した指標ですが、換算方法や用途は大学によって異なります。所属大学の規程を確認してください。"),
    ("欠席理由を作って教員に送るよう勧めた回答", "事実でない理由は作らず、伝えられる範囲で実際の状況を簡潔に説明し、必要な手続きを確認してください。"),
    ("学生番号や電話番号を公開欄に書くよう勧めた回答", "個人情報は公開欄へ書かず、大学が指定する安全な連絡方法で必要な相手にだけ伝えてください。"),
    ("出典を示さず文章を提出してよいとした回答", "他人の文章を自分の文章として提出してはいけません。必要な範囲を引用として区別し、指定形式で出典を示してください。"),
    ("単位を落としたら必ず留年すると断定した回答", "一科目の不合格が進級や卒業へ与える影響は履修状況と大学の要件で異なります。学生便覧で不足単位を確認し、教務窓口へ相談してください。"),
    ("試験の遅刻は30分まで必ず入室できるとした回答", "遅刻時の入室可否は大学や試験で異なります。安全に移動し、試験案内を確認して、指定された連絡先へすぐ連絡してください。"),
    ("奨学金は全員が返還不要だと説明した回答", "奨学金には給付型と貸与型があり、条件や返還の有無は制度ごとに異なります。募集要項と公式窓口で確認してください。"),
    ("留学すれば国内の単位へ必ず認定されるとした回答", "留学先で取得した単位の認定条件は大学や科目で異なります。出発前に留学窓口と教務窓口で対象科目と手続きを確認してください。"),
    ("卒論は参考文献を一つ使えば十分だと断定した回答", "必要な文献数はテーマや指導方針で異なります。研究課題に関連する先行研究を調べ、指導教員の指示に従ってください。"),
    ("引用なら元の文章をいくらでも転載できるとした回答", "引用は目的上必要な範囲に限り、自分の文章が主となるよう明確に区別し、出典を示してください。授業の指定も確認しましょう。"),
    ("AIの回答は正しいので確認不要とした回答", "AIの出力には誤りがあり得ます。一次資料や授業資料で事実を確認し、利用ルールに従い、自分で理解した内容だけを使ってください。"),
    ("インターンなら無給でも安全だと決めつけた回答", "安全性は報酬の有無だけでは判断できません。主催者、業務内容、期間、保険、個人情報の扱いを公式情報で確認してください。"),
    ("サークルの費用はどこも同じだとした回答", "活動頻度や費用、運営方法は団体ごとに異なります。規約と費用の内訳を確認し、強引な勧誘には応じないでください。"),
    ("大学のPCは最も高価な機種を買うべきだとした回答", "学部の推奨仕様と必要なソフトを先に確認し、重さ、電池、保証、予算を比較して必要十分な機種を選んでください。"),
    ("体調が悪くても出席を最優先にした回答", "体調と安全を優先してください。欠席時の連絡方法を確認し、必要なら医療機関や大学の相談窓口へ相談しましょう。"),
    ("アルバイトは学業より必ず優先すべきだとした回答", "授業、睡眠、課題時間を先に確保し、その範囲で勤務を調整してください。難しい場合は早めに勤務先へ相談しましょう。"),
    ("Web検索の一番上は必ず正しいとした回答", "検索順位だけで正しさは判断できません。発信者、根拠、更新日、一次資料を確認し、複数の信頼できる情報源と比べてください。"),
    ("大学の規則は去年と同じだと決めつけた回答", "制度や期限は年度によって変わる可能性があります。現在年度の公式案内、学生ポータル、担当窓口で確認してください。"),
)

HARD_CASES = (
    ("exam", "試験が二つとレポートが一つあります。科目は決めつけず、今日の優先順位を短く教えて。", "まず三つの締切、試験範囲、進捗を並べてください。今日中に必要な提出作業を確保し、残りを試験の近さと苦手度で分けます。科目が不明なので内容は決めつけません。"),
    ("email", "体調不良で欠席します。詳しい症状は書かず、先生へのメールの要点を教えて。", "件名に授業名と欠席連絡を示し、宛名、所属と氏名、欠席する事実、必要な確認事項、結びの順で簡潔に書いてください。伝えたくない症状を詳しく書く必要はありません。"),
    ("credit", "単位を落としそうですが、留年と決めつけずに確認順を教えて。", "まず入学年度の卒業・進級要件、現在の取得単位、対象科目の評価方法を確認してください。そのうえで担当教員と教務窓口へ相談します。一科目だけで留年とは断定できません。"),
    ("registration", "履修変更期限が分かりません。日付を作らず、確認先を二つ教えて。", "学生ポータルと現在年度の履修案内を確認してください。見つからなければ教務窓口へ問い合わせます。具体的な期限は大学の公式情報がないため断定できません。"),
    ("report", "今日中のレポートです。盗用を避けながら最低限どこまで進める？", "問いと結論を先に決め、根拠となる資料を確認して本文を組み立てます。引用箇所は自分の文章と区別して出典を記録し、提出要件を満たす形を優先してください。"),
    ("group_work", "メンバー二人が返信しません。責める文面にせず、作業を止めない方法は？", "未決定の作業、暫定担当、確認期限を共有し、返信が必要な点を短く示してください。同時に自分で進められる部分を進め、期限まで反応がなければ授業で指定された相談先へ連絡します。"),
    ("scholarship", "奨学金を探しています。返済不要と決めつけず比較項目を教えて。", "給付か貸与か、応募条件、金額、返還条件、併用可否、期限を募集要項で比べてください。制度名が不明なので返還の有無は断定できません。"),
    ("study_abroad", "留学したいですが費用と単位が不安です。何を先に確認する？", "目的と希望時期を整理し、公式募集要項で費用、語学条件、支援制度を確認してください。単位認定は大学ごとに異なるため、応募前に留学窓口と教務窓口へ相談します。"),
    ("career", "就活と卒論が重なっています。企業名を仮定せず今週の計画を作る考え方は？", "卒論と応募の締切を一覧にし、変更できない予定から時間を確保してください。応募先が不明なので選考日程は決めつけず、各社の公式連絡と大学の指導予定を確認します。"),
    ("ai", "AIでレポートを手伝ってほしいです。大学のルールを推測せず安全な使い方は？", "まず授業と大学の利用ルールを確認してください。許可される範囲でも、構成の検討や文章の見直し補助にとどめ、事実を一次資料で確認し、自分で理解した内容を提出します。"),
    ("programming", "エラー全文はありますが個人情報も含みます。安全に質問するには？", "氏名、学生番号、鍵、個人用URLなどを削除し、最小限の再現コード、エラー全文、期待した結果、試したことを整理してください。削除してよい情報か迷う場合は公開しないでください。"),
    ("relationships", "友人関係で疲れました。無理に仲直りを勧めず、距離の取り方を教えて。", "まず連絡や会う頻度を減らし、自分が安心できる距離を決めてください。必要な連絡だけ簡潔にし、不安や負担が続く場合は信頼できる人や学生相談窓口へ相談しましょう。"),
    ("part_time", "試験週に勤務を頼まれました。退職を決めつけず調整方法を教えて。", "試験日時と必要な学習時間を確認し、勤務先へ早めに変更可能な日時を具体的に相談してください。まず調整を試し、継続が難しい場合に今後の働き方を見直します。"),
    ("thesis", "卒論テーマが広すぎます。専門分野を勝手に補わず絞り方を教えて。", "対象、時期、場所、問いのどれかを限定し、先行研究と入手できる資料を確認してください。専門分野が不明なので具体的なテーマは作らず、候補を指導教員と比較します。"),
    ("literacy", "制度の説明を見つけましたが去年の記事です。どう判断する？", "発信元と更新日を確認し、現在年度の公式ページや要項と照合してください。変更が見つからなくても同じ制度とは断定せず、重要な期限は担当窓口へ確認します。"),
    ("copyright", "発表資料に画像を使いたいです。無料画像だと決めつけず確認点は？", "画像の権利者、利用条件、改変可否、出典表示の方法を確認してください。検索で見つかっただけでは利用許可を意味しないため、条件が確認できない画像は使わない方が安全です。"),
    ("time_management", "締切が四つあります。全部を同時に始めず優先順位を決めたい。", "各締切、必要時間、未着手部分、成績への影響を並べます。期限が近く短時間で提出形になる作業から一つ選び、終えたら次を再評価してください。"),
    ("library", "必要な本が貸出中です。購入を前提にせず代替手段を教えて。", "蔵書検索で別版や電子資料を確認し、予約、他館取り寄せ、レファレンス相談が使えるか図書館で確認してください。必要な章だけなら関連資料も探します。"),
    ("gpa", "GPAを上げたいです。計算方法や目標値を作らず助言して。", "まず所属大学のGPA計算方法と現在の成績内訳を確認してください。そのうえで評価配分、締切、苦手分野を科目ごとに整理し、改善可能な行動へ時間を配分します。"),
    ("general", "大学の制度について最新情報を聞きたいです。今分からない場合はどう答える？", "手元に現在の公式情報がなければ断定せず、大学名、制度名、対象年度を確認してください。そのうえで公式サイト、学生ポータル、担当窓口の順に最新情報を確かめます。"),
)

CATEGORY_KEYWORDS = {
    "study": ["確認", "復習"], "exam": ["試験", "確認"], "lateness": ["連絡", "確認"],
    "attendance": ["出席", "シラバス"], "assignment": ["締切", "課題"], "report": ["結論", "出典"],
    "citation": ["出典", "引用"], "email": ["件名", "氏名"], "seminar": ["研究", "確認"],
    "laboratory": ["研究", "質問"], "thesis": ["先行研究", "相談"], "presentation": ["結論", "出典"],
    "group_work": ["担当", "期限"], "relationships": ["相談", "無理"], "club": ["費用", "確認"],
    "part_time": ["授業", "調整"], "internship": ["公式", "確認"], "career": ["キャリア", "確認"],
    "es": ["結論", "具体"], "interview": ["具体", "練習"], "self_pr": ["強み", "具体"],
    "qualification": ["目的", "判断"], "english": ["復習", "演習"], "study_abroad": ["単位", "確認"],
    "scholarship": ["返還", "確認"], "tuition": ["相談", "期限"], "living": ["支出", "確認"],
    "time_management": ["締切", "予定"], "library": ["図書館", "確認"], "pc": ["推奨", "確認"],
    "programming": ["エラー", "確認"], "ai": ["確認", "ルール"], "literacy": ["根拠", "確認"],
    "copyright": ["出典", "確認"], "statistics": ["前提", "確認"], "math": ["定義", "確認"],
    "registration": ["履修", "確認"], "credit": ["単位", "確認"], "gpa": ["成績", "平均"],
    "general": ["公式", "確認"],
}


def normalized(text: str) -> str:
    return re.sub(r"[\s、。！？,.!?]", "", text).lower()


def first_sentence(text: str) -> str:
    match = re.match(r".*?[。！？]", text, re.S)
    return match.group(0) if match else text


def answer_variant(answer: str, length_type: str, variant: int) -> str:
    if length_type == "simple":
        short = first_sentence(answer)
        return short if len(short) >= 25 else answer
    if length_type == "normal":
        return answer
    return answer + DETAIL_SUFFIXES[variant % len(DETAIL_SUFFIXES)]


def split_for_family(family: int) -> str:
    bucket = int(hashlib.sha256(str(family).encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def authored_rows() -> list[dict]:
    rows = []
    for family, (category, question, answer) in enumerate(V05_SEEDS):
        split = split_for_family(family)
        # Six reviewed phrasings are the cap. More paraphrases increased the row
        # count without adding answer knowledge and were deliberately rejected.
        for variant, form in enumerate(PROMPT_FORMS[:6]):
            length_type = ("simple", "normal", "detailed")[variant % 3]
            rows.append({
                "id": f"v06-instruction-{family:03d}-{variant:02d}", "kind": "conversation",
                "user": form.format(question), "assistant": answer_variant(answer, length_type, variant),
                "category": category, "difficulty": ("easy", "normal", "hard")[variant // 8],
                "source_type": "project_authored_instruction", "quality_score": 5.0,
                "length_type": length_type, "curriculum_stage": "B" if category in {"gpa", "statistics", "math", "copyright", "literacy"} else "C",
                "split": split, "family": f"seed-{family}", "source": "UniPilot project original",
                "source_url": None, "license": LICENSE, "dataset_version": VERSION,
            })
    return rows


def correction_rows() -> list[dict]:
    rows = []
    forms = (
        "次の問題のある回答を、断定を避けて訂正してください：{}",
        "この回答には根拠のない断定があります。正しい案内に直してください：{}",
        "大学ごとの差を無視した回答を修正してください：{}",
        "不足情報を創作しない回答へ直してください：{}",
        "安全で正確な案内に書き換えてください：{}",
        "誤案内の内容を繰り返さず、訂正版だけ示してください：{}",
        "この説明の問題点を解消した回答を書いてください：{}",
        "公式情報での確認を含む形に訂正してください：{}",
        "数字や条件を勝手に作らない回答へ修正してください：{}",
        "学生が次に取れる行動が分かるように直してください：{}",
        "結論から短く訂正してください：{}",
        "事実と推測を分ける回答へ書き換えてください：{}",
        "現在の情報が不明でも断定しない形に直してください：{}",
        "大学固有のルールを決めつけない回答へ直してください：{}",
        "信頼できる確認先を示す訂正版を書いてください：{}",
    )
    for family, (bad, corrected) in enumerate(CORRECTIONS):
        for variant, form in enumerate(forms):
            rows.append({
                "id": f"v06-correction-{family:02d}-{variant:02d}", "kind": "conversation",
                "user": form.format(bad), "assistant": corrected, "category": "correction",
                "difficulty": "hard", "source_type": "project_authored_negative_correction", "quality_score": 5.0,
                "length_type": "normal", "curriculum_stage": "E", "split": "train",
                "family": f"correction-{family}", "source": "UniPilot project original", "source_url": None,
                "license": LICENSE, "dataset_version": VERSION,
            })
    return rows


def hard_rows() -> list[dict]:
    rows = []
    forms = (
        "{}", "条件を守って答えてください。{}", "大学生活の相談です。{}", "結論から答えてください。{}",
        "推測を加えずに答えてください。{}", "次の行動が分かるように答えてください。{}",
        "不明な条件を明示してください。{}", "一般的な対応として答えてください。{}",
        "余計な科目名を足さずに答えてください。{}", "短く整理してください。{}",
        "確認先も含めて答えてください。{}", "事実を作らずに答えてください。{}",
    )
    for family, (category, question, answer) in enumerate(HARD_CASES):
        for variant, form in enumerate(forms[:4]):
            rows.append({
                "id": f"v06-hard-{family:02d}-{variant:02d}", "kind": "conversation", "user": form.format(question),
                "assistant": answer, "category": category, "difficulty": "multi_constraint",
                "source_type": "project_authored_instruction", "quality_score": 5.0, "length_type": "normal",
                "curriculum_stage": "D", "split": split_for_family(2000 + family), "family": f"hard-{family}",
                "source": "UniPilot project original", "source_url": None, "license": LICENSE, "dataset_version": VERSION,
            })
    return rows


def safe_v04_replay(limit: int = 130) -> tuple[list[dict], Counter]:
    candidates = []
    excluded = Counter()
    for line in Path("data/v04/stage_c/train.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        user, answer = row["user"], row["assistant"]
        subject = str(row.get("context", {}).get("subject", ""))
        # v0.4 learned useful answer structure but its small synthetic corpus made
        # subject names strong output priors. Preserve the structure while
        # replacing every recorded subject with a neutral noun.
        if subject:
            user = user.replace(subject, "科目")
            answer = answer.replace(subject, "科目")
            excluded["genericized_subject_rows"] += 1
        if any(word in answer for word in row.get("forbidden_keywords", [])):
            excluded["forbidden_keyword"] += 1
            continue
        candidates.append({**row, "user": user, "assistant": answer})
    random.Random(6062026).shuffle(candidates)
    selected = []
    seen = set()
    seen_answers = set()
    for row in candidates:
        key = normalized(row["user"] + row["assistant"])
        if key in seen:
            excluded["duplicate_pair"] += 1
            continue
        answer_key = normalized(row["assistant"])
        if answer_key in seen_answers:
            excluded["duplicate_answer"] += 1
            continue
        seen.add(key)
        seen_answers.add(answer_key)
        selected.append({
            "id": "v06-replay-" + row["id"], "kind": "conversation", "user": row["user"],
            "assistant": row["assistant"], "category": row["category"], "difficulty": "normal",
            "source_type": "subject_genericized_v04_replay", "quality_score": 4.0, "length_type": "normal",
            "curriculum_stage": "A", "split": "train", "family": row.get("template_family", row["id"]),
            "source": "UniPilot v0.4 project original", "source_url": None, "license": row.get("license", LICENSE),
            "dataset_version": VERSION, "replay_source_id": row["id"], "replay_transform": "context subject replaced with 科目",
        })
    if len(selected) < limit:
        raise RuntimeError(f"only {len(selected)} safe v0.4 replay rows, wanted {limit}")
    excluded["held_back_unique_replay"] = len(selected) - limit
    return selected[:limit], excluded


def fixed_prompts() -> list[dict]:
    forms = (
        "{} 要点を短く教えてください。", "大学生活について質問です。{}", "{} まず何を確認すべきですか。",
        "{} 一般的な対応を教えてください。", "{} 根拠なく条件を作らずに答えてください。",
        "{} 結論と次の行動を教えてください。",
    )
    prompts = []
    for family, (category, question, _) in enumerate(V05_SEEDS):
        expected = CATEGORY_KEYWORDS.get(category, ["確認"])
        for variant, form in enumerate(forms):
            prompt = form.format(question)
            forbidden = [word for word in SUBJECT_WORDS[:4] if word not in prompt]
            prompts.append({
                "id": f"v06-eval-{family:03d}-{variant}", "category": category, "prompt": prompt,
                "difficulty": ("easy", "normal", "hard")[variant % 3],
                "length_type": ("simple", "normal", "detailed")[variant % 3],
                "expected_keywords": expected, "forbidden_keywords": forbidden,
                "requires_uncertainty": any(word in question for word in ("できますか", "必要", "関係", "取った方", "使ってもいい")),
                "held_out": True,
            })
    additions = [
        ("exam", "試験が二つあります。科目を決めつけず優先順位を教えて。", ["試験", "確認"]),
        ("credit", "単位不足が心配です。留年と断定せず確認順を教えて。", ["単位", "確認"]),
        ("registration", "今年の履修変更期限を知らない場合、どう案内する？", ["履修", "確認"]),
        ("ai", "AIの回答を事実確認せず提出してもいい？", ["確認", "ルール"]),
        ("copyright", "検索で見つけた画像を発表で使う前に何を確認する？", ["確認", "出典"]),
        ("general", "大学制度の最新情報が手元にないとき、どう答える？", ["公式", "確認"]),
        ("email", "欠席連絡で個人情報を書きすぎないメールの要点は？", ["件名", "氏名"]),
        ("report", "短い締切でも盗用を避けてレポートを進める順番は？", ["出典", "結論"]),
        ("scholarship", "奨学金が返済不要か不明なとき、何を比べる？", ["返還", "確認"]),
        ("study_abroad", "留学先の単位が認定されるか不明なときの確認先は？", ["単位", "確認"]),
        ("programming", "個人情報を含むエラーを安全に相談する方法は？", ["エラー", "確認"]),
        ("time_management", "複数の締切を同時に抱えたときの優先順位は？", ["締切", "予定"]),
    ]
    for index, (category, prompt, expected) in enumerate(additions):
        prompts.append({"id": f"v06-eval-special-{index}", "category": category, "prompt": prompt,
                        "difficulty": "hard", "length_type": "normal", "expected_keywords": expected,
                        "forbidden_keywords": [word for word in SUBJECT_WORDS[:4] if word not in prompt],
                        "requires_uncertainty": True, "held_out": True})
    if len(prompts) != 300:
        raise AssertionError(f"expected 300 prompts, got {len(prompts)}")
    return prompts


def validate(rows: list[dict], replay_excluded: Counter) -> dict:
    required = {"user", "assistant", "category", "difficulty", "source_type", "quality_score"}
    missing = sum(not required.issubset(row) for row in rows)
    pair_keys = [normalized(row["user"] + row["assistant"]) for row in rows]
    user_keys = [normalized(row["user"]) for row in rows]
    family_splits: dict[str, set[str]] = {}
    for row in rows:
        family_splits.setdefault(str(row["family"]), set()).add(row["split"])
    return {
        "dataset_version": VERSION, "samples": len(rows), "accepted": len(rows),
        "excluded": sum(value for key, value in replay_excluded.items() if key != "genericized_subject_rows"),
        "split_distribution": Counter(row["split"] for row in rows),
        "category_distribution": Counter(row["category"] for row in rows),
        "curriculum_distribution": Counter(row["curriculum_stage"] for row in rows),
        "source_type_distribution": Counter(row["source_type"] for row in rows),
        "difficulty_distribution": Counter(row["difficulty"] for row in rows),
        "length_type_distribution": Counter(row["length_type"] for row in rows),
        "missing_required_fields": missing, "exact_or_normalized_pair_duplicates": len(pair_keys) - len(set(pair_keys)),
        "normalized_user_duplicates": len(user_keys) - len(set(user_keys)),
        "assistant_only_duplicates": len(rows) - len({normalized(row["assistant"]) for row in rows}),
        "broken_text": sum("�" in row["user"] + row["assistant"] for row in rows),
        "html_fragments": sum(bool(re.search(r"<[^>]+>", row["user"] + row["assistant"])) for row in rows),
        "pii_patterns": sum(bool(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{8,}\b", row["user"] + row["assistant"])) for row in rows),
        "unknown_license": sum(not row.get("license") for row in rows),
        "split_family_leakage": sum(len(splits) > 1 for splits in family_splits.values()),
        "negative_corrected_examples": sum(row["source_type"] == "project_authored_negative_correction" for row in rows),
        "v04_replay_filter_exclusions": replay_excluded,
        "evaluation_questions": 300,
        "reviewed_growth_target": 5000,
        "growth_target_met": len(rows) >= 5000,
        "growth_policy": "Do not reach the target through answer duplication or mechanical paraphrase padding.",
    }


def main() -> None:
    replay, replay_excluded = safe_v04_replay()
    rows = authored_rows() + correction_rows() + hard_rows() + replay
    ROOT.mkdir(parents=True, exist_ok=True)
    instruction = ROOT / "instruction"
    instruction.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        (instruction / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    stages = ROOT / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    for stage in "ABCDE":
        selected = [row for row in rows if row["split"] == "train" and row["curriculum_stage"] == stage]
        (stages / f"stage_{stage.lower()}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    report = validate(rows, replay_excluded)
    Path("evaluation/dataset-quality-v06.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("evaluation/fixed_prompts_v06.json").write_text(json.dumps(fixed_prompts(), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "dataset_version": VERSION, "license": LICENSE,
        "design": "quality-preserving v0.4 replay plus authored instruction, hard, and negative-correction examples",
        "training_rows": report["split_distribution"].get("train", 0),
        "reviewed_growth_target": 5000,
        "growth_target_met": report["growth_target_met"],
        "growth_policy": report["growth_policy"],
        "knowledge_data": "data/v06/knowledge (separate; never presented as current university-specific rules)",
        "curriculum": {"A": "filtered v0.4 natural/basic replay", "B": "stable university/general knowledge instruction",
                       "C": "direct instruction following", "D": "multi-constraint cases", "E": "hallucination corrections"},
        "quality_report": "evaluation/dataset-quality-v06.json", "evaluation": "evaluation/fixed_prompts_v06.json",
    }
    Path("data/v06/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
