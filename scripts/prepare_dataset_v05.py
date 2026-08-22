from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


VERSION = "unipilot-high-quality-conversation-v05"
ROOT = Path("data/v05/conversation")
LICENSE = "CC0-1.0"

# Project-authored seed answers. They deliberately avoid university-specific assertions.
SEEDS = [
    ("study", "試験勉強を効率よく進めるには？", "まず試験範囲と残り日数を確認し、重要な項目を問題演習と復習に分けて進めましょう。科目が不明なので、特定科目の内容は決めつけません。"),
    ("study", "勉強しても覚えられません", "読むだけで終えず、何も見ずに説明する、問題を解く、翌日にもう一度確認する、の順で思い出す練習を増やしましょう。"),
    ("study", "集中できないときはどうする？", "机に必要な物だけ置き、通知を切って25分だけ一つの課題に取り組みましょう。終わったら5分休み、続けるか判断してください。"),
    ("exam", "試験まで一週間です。どう計画する？", "最初に範囲と配点を確認し、前半は理解と演習、後半は間違い直し、前日は要点確認に使う計画が基本です。"),
    ("exam", "試験を欠席しそうです", "欠席時の扱いは大学や授業で異なります。シラバスと学生便覧を確認し、できるだけ早く担当教員と教務窓口へ連絡してください。"),
    ("exam", "追試や再試験は受けられますか？", "追試・再試験の条件は大学や科目で異なるため断定できません。学生便覧、シラバス、教務窓口で対象条件と申請期限を確認してください。"),
    ("lateness", "授業に遅刻しそうです", "安全を優先して移動し、連絡方法が指定されていれば担当教員へ簡潔に知らせてください。遅刻の扱いはシラバスで確認しましょう。"),
    ("attendance", "欠席が増えて不安です", "まず自分の出席記録とシラバスの評価方法を確認してください。体調や事情が続く場合は、担当教員、教務窓口、学生相談窓口へ早めに相談しましょう。"),
    ("assignment", "課題が多くて終わりません", "締切、所要時間、成績への影響を一覧にし、締切が近く短時間で進む課題から着手しましょう。各課題を30分単位に分けると始めやすくなります。"),
    ("assignment", "課題の締切に間に合わないかもしれません", "提出要件を確認して完成度より提出可能な形を優先し、難しい場合は締切前に担当教員へ現状と相談内容を簡潔に連絡してください。"),
    ("report", "レポートは何から書けばいい？", "問いを一文で決め、結論、根拠、具体例の順に見出しを作ってください。資料を読んだら出典をその場で記録すると引用漏れを防げます。"),
    ("report", "参考文献リストの作り方を教えて", "著者、題名、出版情報、年、参照箇所など、指定された書式の必要項目をそろえます。授業の指示や大学のガイドを優先してください。"),
    ("citation", "引用とコピペの違いは？", "引用は出典を明示し、必要な範囲を自分の文章と区別して使います。出典を示さないコピーは盗用になり得るため避けてください。"),
    ("citation", "Webサイトを引用するときの注意は？", "著者・組織名、ページ名、URL、公開日や閲覧日を記録し、授業指定の引用形式に合わせてください。情報の発信元と更新日も確認しましょう。"),
    ("email", "先生に質問するメールを書きたい", "件名で授業名と用件を示し、宛名、所属と氏名、質問、結びの順に簡潔に書きます。質問は確認した資料と分からない点を具体的にしてください。"),
    ("email", "課題提出が遅れる連絡文を作って", "件名：課題提出についてのご相談\n\n○○先生\nお世話になっております。○○学部の〈氏名〉です。課題の提出が遅れる見込みのためご連絡しました。現在の状況は〈状況〉です。提出方法についてご相談できますでしょうか。よろしくお願いいたします。"),
    ("seminar", "ゼミはどう選べばいい？", "研究テーマ、指導方法、活動頻度、選考条件を確認し、説明会や教員・在学生への質問で自分の目的と合うか比較しましょう。"),
    ("laboratory", "研究室訪問では何を聞く？", "研究テーマ、日常の進め方、指導体制、設備、在学生の生活を質問しましょう。事前に研究室の公開情報を読み、面談日時を丁寧に依頼してください。"),
    ("thesis", "卒論のテーマが決まりません", "興味のある範囲を挙げ、先行研究があるか、資料を入手できるか、期限内に調べられるかで絞り、指導教員に相談しましょう。"),
    ("presentation", "プレゼン資料を見やすくしたい", "一枚一メッセージに絞り、文字を減らし、結論を先に示してください。図表には題名と出典を付け、実際の画面で読める大きさか確認しましょう。"),
    ("group_work", "グループワークが進みません", "目的、担当、期限、次の確認日時を全員で言葉にしてください。意見が割れたら評価基準を決め、記録を残して合意点から進めましょう。"),
    ("relationships", "大学で友達ができません", "無理に人数を増やす必要はありません。同じ授業で挨拶する、課題について短く話す、興味のある活動を一度見学するなど、小さな接点から始めましょう。"),
    ("club", "サークル選びで迷っています", "活動内容、頻度、費用、雰囲気、学業との両立を確認し、複数を見学してから決めましょう。強引な勧誘や不明瞭な費用には注意してください。"),
    ("part_time", "アルバイトと学業を両立したい", "授業、睡眠、課題時間を先に確保し、その残りに勤務を入れましょう。試験期間のシフト調整が可能か、応募前に確認すると安心です。"),
    ("internship", "インターンはどう探す？", "目的を決め、大学のキャリア窓口、企業公式サイト、信頼できる募集情報を比較してください。業務内容、期間、報酬、保険、個人情報の扱いを確認しましょう。"),
    ("career", "就職活動は何から始める？", "まず経験と関心を書き出し、業界・職種を調べ、大学のキャリア窓口で予定と応募書類を確認しましょう。開始時期は卒業年度や応募先で異なります。"),
    ("es", "ESがうまく書けません", "設問の結論を先に書き、具体的な行動、その理由、結果、学びを一つの経験でつなげましょう。企業が尋ねている点に直接答えているか最後に確認してください。"),
    ("interview", "面接の練習方法を教えて", "志望理由と経験を丸暗記せず、結論、具体例、学びの順で一分程度にまとめ、録音や模擬面接で伝わりにくい部分を直しましょう。"),
    ("self_pr", "自己PRには何を書く？", "自分の強みを一つ選び、それが表れた状況、行動、結果、応募先での生かし方を具体的につなげてください。事実を誇張しないことも大切です。"),
    ("qualification", "資格は取った方がいい？", "目的に必要か、学習時間に見合うかで判断しましょう。資格取得だけを目的にせず、授業や実践で使う機会まで考えると選びやすくなります。"),
    ("english", "TOEICの勉強方法を教えて", "現在の得点と目標を確認し、語彙と文法を短時間で毎日復習しつつ、時間を測った問題演習と間違い分析を続けましょう。"),
    ("study_abroad", "留学を考えています", "目的、時期、費用、必要な語学力、単位認定、安全情報を整理し、大学の留学窓口と公式募集要項で確認してください。"),
    ("scholarship", "奨学金について知りたい", "給付か貸与か、応募条件、金額、返還条件、期限を公式情報で確認してください。制度は年度や大学で異なるため、学生支援窓口にも相談しましょう。"),
    ("tuition", "学費の支払いが厳しいです", "一人で抱えず、大学の学生支援・会計窓口へ早めに相談してください。分納、延納、減免、奨学金などの有無と申請期限を公式案内で確認しましょう。"),
    ("living", "一人暮らしの生活費を管理したい", "家賃などの固定費、食費、交通費、予備費に分け、週一回支出を確認しましょう。最初に生活費を確保してから自由に使える額を決めると安全です。"),
    ("time_management", "時間管理が苦手です", "締切を一か所に集め、今日やることを三つ以内に絞り、開始時刻を予定に入れてください。予定には遅れを吸収する空白も残しましょう。"),
    ("library", "大学図書館を活用したい", "蔵書検索、データベース、レファレンス、取り寄せ、学習スペースを確認しましょう。資料探しに迷ったら司書へテーマと必要期限を伝えて相談できます。"),
    ("pc", "大学用PCはどう選ぶ？", "学部の推奨仕様を最優先にし、必要なソフト、重さ、電池、保証、予算を比較してください。高性能が必要か分からない場合は購入前に大学へ確認しましょう。"),
    ("programming", "プログラミングのエラーが直せません", "エラーメッセージを最初から読み、再現手順を小さくし、入力と期待結果を確認してください。質問するときはコード、エラー全文、試したことを整理しましょう。"),
    ("ai", "AIを勉強にどう活用すればいい？", "説明の比較、練習問題の案、文章の見直し補助には使えますが、出力を事実とみなさず一次資料で確認し、自分で理解してから提出物に反映してください。"),
    ("literacy", "ネットの情報が正しいか見分けたい", "発信者、根拠、公開日、一次資料へのリンクを確認し、独立した複数の信頼できる情報源と比べましょう。検索上位だけで判断しないことが大切です。"),
    ("copyright", "著作権で気を付けることは？", "他人の文章や画像は利用条件を確認し、必要な許可と出典表示を行ってください。引用は必要な範囲に限り、自分の文章と明確に区別しましょう。"),
    ("statistics", "統計の勉強は何から始める？", "平均や分散などの意味を具体例で理解し、グラフを読み、手計算とソフトで同じ結果を確認しましょう。公式を覚えるだけでなく前提条件も確認してください。"),
    ("math", "大学数学が難しいです", "定義を自分の言葉で説明し、簡単な例を手で計算してから標準問題へ進みましょう。分からない箇所を式の一行単位で特定すると質問しやすくなります。"),
    ("registration", "履修科目はどう決めればいい？", "卒業要件と必修科目を確認し、時間割、シラバス、課題量を比べて無理のない組み合わせにしてください。制度は大学の公式情報を優先しましょう。"),
    ("registration", "履修登録を変更できますか？", "変更できる期間や条件は大学によって異なります。履修案内と学生ポータルを確認し、期限が不明なら教務窓口へすぐ問い合わせてください。"),
    ("credit", "卒業に必要な単位を確認したい", "入学年度の学生便覧で卒業要件を確認し、取得済み・履修中・不足を区分ごとに整理してください。不明点は教務窓口で確認しましょう。"),
    ("gpa", "GPAは成績にどう関係しますか？", "GPAは科目の成績を一定の点数に換算して平均した指標です。計算方法や用途は大学によって異なるため、所属大学の規程を確認してください。"),
]

HELD_OUT = [
    ("study", "テストを効率よく勉強する方法を教えてください", ["試験範囲", "復習"], ["法学", "経済学"]),
    ("lateness", "大学のテストに遅刻したらどうすればいい？", ["連絡", "確認"], []),
    ("gpa", "GPAって何？", ["成績", "平均"], []),
    ("email", "教授に欠席メールを送りたい", ["件名", "氏名"], []),
    ("credit", "単位を落としそうです", ["シラバス", "確認"], []),
    ("citation", "レポートの引用方法を教えて", ["出典", "引用"], []),
    ("registration", "履修登録を忘れました", ["教務", "確認"], []),
    ("general", "大学1年生は何をした方がいい？", ["授業", "計画"], []),
    ("career", "就活はいつから始めればいい？", ["キャリア", "確認"], []),
    ("ai", "生成AIをレポートに使ってもいい？", ["ルール", "確認"], []),
]

BAD_CORRECTIONS = [
    ("科目が不明なのに『法学の試験範囲を覚えましょう』と答えています。直してください。", "科目が分からないため、まず試験範囲を確認し、重要項目の理解、問題演習、間違い直しの順で進める一般的な方法を案内します。"),
    ("履修登録は全国どの大学でも追加できます、と断定した回答を直してください。", "履修登録の追加・変更可否は大学によって異なります。履修案内と学生ポータルを確認し、期限が不明なら教務窓口へ問い合わせてください。"),
    ("出典なしの文章をそのまま提出してよい、という回答を直してください。", "他人の文章をそのまま提出してはいけません。必要な範囲を引用として区別し、指定形式で出典を示してください。"),
    ("欠席理由を作って先生へ送る回答を直してください。", "事実でない理由は作らず、伝えられる範囲で実際の状況を簡潔に説明し、必要な手続きを確認してください。"),
    ("学生番号や電話番号を公開欄へ書かせる回答を直してください。", "個人情報は公開欄へ書かず、大学が指定する安全な連絡方法を使ってください。必要な情報だけを担当窓口へ伝えましょう。"),
]


def normalize(text: str) -> str:
    return re.sub(r"[\s、。！？,.!?]", "", text).lower()


def main() -> None:
    rows = []
    prompt_forms = ["{}", "{} 短く教えてください。", "大学生活の相談です。{}"]
    for family, (category, question, answer) in enumerate(SEEDS):
        split = "train" if family % 10 < 8 else ("validation" if family % 10 == 8 else "test")
        for variant, form in enumerate(prompt_forms):
            rows.append({"id": f"v05-{family:03d}-{variant}", "kind": "conversation", "category": category,
                         "intent": "DIRECT_HELP", "user": form.format(question), "assistant": answer,
                         "split": split, "family": family, "source": "UniPilot project original",
                         "source_url": None, "license": LICENSE, "dataset_version": VERSION,
                         "quality_principles": ["direct", "no_unstated_subject", "official_source_when_variable"]})
    for index, (bad, corrected) in enumerate(BAD_CORRECTIONS):
        rows.append({"id": f"v05-correction-{index:02d}", "kind": "conversation", "category": "correction",
                     "intent": "CORRECT_BAD_ANSWER", "user": bad, "assistant": corrected, "split": "train",
                     "family": 1000 + index, "source": "UniPilot project original", "source_url": None,
                     "license": LICENSE, "dataset_version": VERSION,
                     "quality_principles": ["bad_to_corrected", "no_fabrication"]})

    ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        (ROOT / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")

    normalized_pairs = [normalize(row["user"] + row["assistant"]) for row in rows]
    exact_duplicates = len(normalized_pairs) - len(set(normalized_pairs))
    broken = sum("�" in row["user"] + row["assistant"] for row in rows)
    html = sum(bool(re.search(r"<[^>]+>", row["user"] + row["assistant"])) for row in rows)
    url_only = sum(bool(re.fullmatch(r"https?://\S+", row["assistant"])) for row in rows)
    pii = sum(bool(re.search(r"\b\d{8,}\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", row["user"] + row["assistant"])) for row in rows)
    report = {
        "dataset_version": VERSION, "samples": len(rows), "accepted": len(rows), "excluded": 0,
        "split_distribution": Counter(row["split"] for row in rows),
        "category_distribution": Counter(row["category"] for row in rows),
        "exact_or_normalized_duplicates": exact_duplicates, "broken_text": broken, "html_fragments": html,
        "url_only": url_only, "pii_patterns": pii, "unknown_license": sum(not row["license"] for row in rows),
        "too_short": sum(len(row["assistant"]) < 15 for row in rows),
        "too_long": sum(len(row["assistant"]) > 500 for row in rows),
        "held_out_evaluation_items": len(HELD_OUT), "bad_to_corrected_items": len(BAD_CORRECTIONS),
    }
    Path("evaluation/dataset-quality-v05.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    prompts = [{"id": f"required-v05-{index:02d}", "category": category, "prompt": prompt,
                "expected_keywords": expected, "forbidden_keywords": forbidden, "held_out": True}
               for index, (category, prompt, expected, forbidden) in enumerate(HELD_OUT)]
    for index, row in enumerate([row for row in rows if row["split"] == "test"]):
        prompts.append({"id": f"broad-v05-{index:03d}", "category": row["category"], "prompt": row["user"],
                        "expected_keywords": [], "forbidden_keywords": [], "held_out": True})
    # Reach 60 prompts with deterministic, non-training paraphrases of the held-out questions.
    cycle = 0
    while len(prompts) < 60:
        category, prompt, expected, forbidden = HELD_OUT[cycle % len(HELD_OUT)]
        prompts.append({"id": f"paraphrase-v05-{cycle:03d}", "category": category,
                        "prompt": f"大学生活の相談です。{prompt} 簡潔に答えてください。",
                        "expected_keywords": expected, "forbidden_keywords": forbidden, "held_out": True})
        cycle += 1
    Path("evaluation/fixed_prompts_v05.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("data/v05/manifest.json").write_text(json.dumps({"dataset_version": VERSION, "license": LICENSE,
        "source": "UniPilot project original", "knowledge_data": "data/v05/knowledge (separate; not used in conversation fine-tuning)",
        "quality_report": "evaluation/dataset-quality-v05.json"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
