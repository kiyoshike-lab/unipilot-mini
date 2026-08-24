from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from pipeline.campus_categories import CAMPUS_CATEGORIES
from pipeline.campus_categories_v2 import CATEGORY_TO_LEVEL1
from scripts.build_campus_v2_data import expected_action, norm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "campus_v21"
RNG = random.Random(210824)


# These concerns are intentionally different from the v2 blind/adversarial semantic seeds.
V21_CONCEPTS = {
    "exam": ("試験会場を取り違えそう", "解答時間の配分を決めたい", "受験票を忘れた場合"),
    "assignment": ("課題ファイルを誤って提出した", "共同課題の担当が進んでいない", "指定拡張子へ直したい"),
    "credit": ("選択必修の区分が足りない", "認定済み科目を数え直したい", "卒業見込の単位確認"),
    "gpa": ("交換留学申請用のGPAを確認", "再履修を含むGPAの扱い", "今後の成績で平均を上げる"),
    "grade_simulator": ("小テスト配点から期末必要点を出す", "レポート点を含めた合格可能性", "評価割合から目標点を逆算"),
    "attendance": ("出席端末が反応しなかった", "オンライン授業の出席記録", "忌引きの欠席手続き"),
    "lateness": ("教室を間違えて開始に遅れた", "バス遅延で試験開始に遅れる", "途中入室するときの対応"),
    "professor_email": ("教員へ資料閲覧をお願いする文", "オフィスアワー予約の連絡", "質問回答への返信文"),
    "absence_email": ("実習を休む連絡文", "家族事情による欠席連絡", "オンライン授業欠席の連絡"),
    "lateness_email": ("到着時刻未定の遅刻連絡", "実験開始に遅れる連絡", "面談へ遅れるお詫び文"),
    "late_submission_email": ("違うファイルを提出した連絡", "通信障害で提出できなかった文", "修正版提出をお願いするメール"),
    "registration": ("前提科目を未修得の履修相談", "集中講義を時間割へ追加", "他学部科目の登録確認"),
    "schedule": ("通学時間込みで一週間を組む", "朝型へ予定を移したい", "空き時間を曜日別に集計"),
    "study_plan": ("演習中心の二週間計画", "復習間隔を空けた学習予定", "模試結果から弱点計画を作る"),
    "assignment_priority": ("配点と締切を合わせて着手順を決める", "共同課題を含む作業順", "提出可能な最低ラインから優先する"),
    "deadline_organizer": ("仮締切と正式締切を分ける", "応募と課題の期限を一表にする", "時刻付きの提出期限を管理"),
    "report_outline": ("比較型レポートの骨組み", "調査結果を中心に章を組む", "反対意見を含む構成"),
    "citation_check": ("図表の出典表記を確認", "講義資料を参考文献にする", "著者不明ページの書誌情報"),
    "presentation_outline": ("三人発表の担当分け", "デモを含む発表順", "ポスター発表の説明構成"),
    "career_schedule": ("複数社の面接間隔を整理", "資格勉強と選考予定を両立", "内定後手続きの日程確認"),
    "es_outline": ("研究経験をESへまとめる", "失敗経験を設問へ書く", "短い文字数で強みを示す"),
    "toeic_plan": ("パート別正答率から計画を作る", "通学中だけで英語対策", "受験二回分の学習予定"),
    "internship": ("授業期間中の長期実習を検討", "選考課題の提出準備", "実習先へ日程を確認"),
    "scholarship": ("併給できる制度を確認", "成績基準の継続条件", "振込停止時の確認先"),
    "tuition": ("引落口座を変更したい", "領収証の再発行を確認", "休学復学時の請求を確認"),
    "part_time_job": ("実習期間だけ勤務を減らす", "深夜勤務と授業を調整", "扶養条件を確認してシフト相談"),
    "campus_life": ("学内ロッカーの利用先", "通学経路を変えたい", "新しいゼミ環境へ慣れる"),
    "relationship": ("共同研究で役割が偏っている", "誘いを断る言い方", "指導時の不適切な発言を相談"),
    "programming": ("依存パッケージの競合エラー", "入力境界値でコードが落ちる", "処理速度を測って改善する"),
    "ai_usage": ("AI生成部分の申告方法", "講義ごとのAI許可範囲を確認", "AI回答の根拠を検証する"),
    "math": ("固有値計算を復習", "数列の収束を確認", "ベクトル空間の証明を読む"),
    "statistics": ("信頼区間を解釈したい", "外れ値を含む要約統計", "標本サイズと検定力の関係"),
    "university_policy": ("転学部条件の公式規程", "長期履修制度の対象確認", "試験受験資格の公式条文"),
    "faq_search": ("証明書手続きの案内を検索", "相談窓口一覧から探す", "学生ポータルの案内場所"),
    "general": ("学期途中で目標を見直したい", "複数の悩みを整理したい", "相談内容の優先度を決めたい"),
}

REAL_CONCEPTS = {
    category: tuple(f"{concept}の実際の進め方" for concept in concepts)
    for category, concepts in V21_CONCEPTS.items()
}

VAGUE = ("やばい", "これやばい", "どうしよう", "どうしたらいい", "間に合わない", "間に合うかな", "大学むり", "大学のことで困った", "これ落とす？", "先生どうしよう",
         "もうだめ", "詰んだかも", "何か忘れた", "これで平気？", "助けて", "不安すぎる",
         "今日むり", "怒られそう", "終わった", "どう連絡するの", "これ必要？", "無理そう",
         "判断できない", "いま何する", "相談したい", "何からすればいい", "もう無理かも",
         "これ大丈夫かな", "助けてほしい")


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def adversarial_row(identifier: str, question: str, category: str, kind: str, negative: str | None = None) -> dict:
    return {"id": identifier, "question": question, "category": category,
            "level1": CATEGORY_TO_LEVEL1[category], "adversarial_type": kind,
            "negative_category": negative, "expected_action": expected_action(question, category, [category]),
            "dataset_version": "unipilot-campus-v21-adversarial-train"}


def build_adversarial_train() -> list[dict]:
    rows = []
    categories = list(CAMPUS_CATEGORIES[:-1])
    connectors = {
        "NEGATION": ("ではなく", "じゃなくて", "じゃない。聞きたいのは"),
        "CORRECTION": ("。違う、", "。訂正すると、", "ではない。正しくは"),
        "CONTRAST": ("より、知りたいのは", "の話は不要で、", "と比べたいのではなく、"),
    }
    for kind in ("NEGATION", "CORRECTION", "CONTRAST"):
        for index in range(300):
            negative = categories[(index * 7 + len(kind)) % len(categories)]
            positive = categories[(index * 19 + 5) % len(categories)]
            if positive == negative:
                positive = categories[(categories.index(positive) + 4) % len(categories)]
            left = V21_CONCEPTS[negative][index % 3]
            right = V21_CONCEPTS[positive][(index // 3) % 3]
            connector = connectors[kind][index % 3]
            rows.append(adversarial_row(f"campus-v21-{kind.lower()}-{index:04d}", f"{left}{connector}{right}を聞きたい",
                                        positive, kind, negative))
    for index in range(300):
        target = categories[(index * 13 + 2) % len(categories)]
        collision = categories[(index * 17 + 9) % len(categories)]
        if CATEGORY_TO_LEVEL1[target] != CATEGORY_TO_LEVEL1[collision]:
            same_group = [item for item in categories if CATEGORY_TO_LEVEL1[item] == CATEGORY_TO_LEVEL1[target] and item != target]
            collision = same_group[index % len(same_group)] if same_group else collision
        question = f"{V21_CONCEPTS[collision][index % 3]}に見えるけど、必要な出力は{V21_CONCEPTS[target][(index + 1) % 3]}"
        rows.append(adversarial_row(f"campus-v21-collision-{index:04d}", question, target, "CATEGORY_COLLISION", collision))
    for index in range(300):
        target = categories[(index * 11 + 1) % len(categories)]
        concept = V21_CONCEPTS[target][index % 3]
        short = re.sub(r"(?:したい|を確認|の公式|場合|進める|作る)", "", concept)[:18]
        rows.append(adversarial_row(f"campus-v21-short-clear-{index:04d}", short, target, "SHORT_BUT_CLEAR"))
    assert len(rows) == 1500
    return rows


def build_adversarial_validation() -> list[dict]:
    """Held-out meanings/forms; these rows are never added to the router fit examples."""
    rows = []
    categories = list(CAMPUS_CATEGORIES[:-1])
    forms = {
        "NEGATION": ("{a}を頼んだわけではなく、{b}が必要", "{a}ではありません。対象は{b}"),
        "CORRECTION": ("{a}と言ったが訂正する。正しくは{b}", "最初の説明は違う。{b}について聞きたい"),
        "CONTRAST": ("{a}との比較ではなく、焦点は{b}", "{a}より優先したいのは{b}"),
    }
    for kind, templates in forms.items():
        for index in range(60):
            negative = categories[(index * 11 + len(kind)) % len(categories)]
            positive = categories[(index * 23 + 7) % len(categories)]
            if positive == negative:
                positive = categories[(categories.index(positive) + 5) % len(categories)]
            question = templates[index % len(templates)].format(
                a=V21_CONCEPTS[negative][(index + 1) % 3], b=V21_CONCEPTS[positive][(index + 2) % 3])
            row = adversarial_row(f"campus-v21-validation-{kind.lower()}-{index:03d}", question,
                                  positive, kind, negative)
            row["dataset_version"] = "unipilot-campus-v21-adversarial-validation"
            rows.append(row)
    collision_forms = ("{a}と似ているが、分類対象は{b}", "{a}の語がある。ただし求める処理は{b}")
    for index in range(60):
        target = categories[(index * 17 + 3) % len(categories)]
        neighbors = [item for item in categories if CATEGORY_TO_LEVEL1[item] == CATEGORY_TO_LEVEL1[target]
                     and item != target]
        collision = neighbors[index % len(neighbors)] if neighbors else categories[(index + 1) % len(categories)]
        question = collision_forms[index % 2].format(a=V21_CONCEPTS[collision][(index + 2) % 3],
                                                      b=V21_CONCEPTS[target][index % 3])
        row = adversarial_row(f"campus-v21-validation-collision-{index:03d}", question, target,
                              "CATEGORY_COLLISION", collision)
        row["dataset_version"] = "unipilot-campus-v21-adversarial-validation"; rows.append(row)
    short_forms = ("{x}", "{x}だけ確認", "{x}の手順")
    for index in range(60):
        target = categories[(index * 29 + 4) % len(categories)]
        concept = re.sub(r"(?:したい|を確認|の公式|場合|進める|作る)", "", V21_CONCEPTS[target][(index + 1) % 3])[:16]
        question = short_forms[index % 3].format(x=concept)
        row = adversarial_row(f"campus-v21-validation-short-{index:03d}", question, target, "SHORT_BUT_CLEAR")
        row["dataset_version"] = "unipilot-campus-v21-adversarial-validation"; rows.append(row)
    assert len(rows) == 300
    return rows


def build_clarification_validation() -> list[dict]:
    rows = []
    contexts = ("今", "今日", "急に", "大学で", "さっき", "明日までに", "初めてで", "正直", "ちょっと", "かなり")
    tails = ("", "。まず確認してほしい", "。何を言えばいいかも分からない", "。対象をまだ説明できていない")
    for index in range(400):
        question = f"{contexts[(index // 20) % len(contexts)]}{VAGUE[index % len(VAGUE)]}{tails[(index // 100) % len(tails)]}"
        rows.append({"id": f"campus-v21-clarify-amb-{index:04d}", "question": question,
                     "category": "general", "ambiguous": True})
    categories = list(CAMPUS_CATEGORIES)
    templates = ("{x}", "{x}、まず何をする", "明日までに{x}", "{x}で困っている", "{x}を短く教えて")
    for index in range(800):
        category = categories[(index * 9 + 3) % len(categories)]
        concept = V21_CONCEPTS[category][(index * 5) % 3]
        question = templates[index % len(templates)].format(x=concept)
        rows.append({"id": f"campus-v21-clarify-clear-{index:04d}", "question": question,
                     "category": category, "ambiguous": False})
    RNG.shuffle(rows)
    return rows


def real_row(identifier: str, prompt: str, category: str, surface: str, labels: list[str] | None = None) -> dict:
    labels = labels or [category]
    return {"id": identifier, "prompt": prompt, "category": category, "intent_labels": labels,
            "level1": CATEGORY_TO_LEVEL1[category], "expected_action": expected_action(prompt, category, labels),
            "surface_type": surface, "ambiguous": False, "blind": True,
            "dataset_version": "unipilot-campus-v21-real-student"}


def build_real_student() -> list[dict]:
    rows = []
    categories = list(CAMPUS_CATEGORIES)
    non_general = list(CAMPUS_CATEGORIES[:-1])
    for index in range(100):
        category = categories[(index * 13 + 2) % len(categories)]
        concept = REAL_CONCEPTS[category][index % 3]
        prompt = re.sub(r"(?:の実際の進め方|したい|を確認)", "", concept)[:20]
        rows.append(real_row(f"campus-v21-real-short-{index:03d}", prompt, category, "very_short"))
    for index in range(100):
        category = categories[(index * 17 + 4) % len(categories)]
        prompt = f"{REAL_CONCEPTS[category][index % 3]}なんだけど、いま何すればいいん"
        rows.append(real_row(f"campus-v21-real-colloquial-{index:03d}", prompt, category, "colloquial"))
    for index in range(100):
        wrong = non_general[(index * 5 + 1) % len(non_general)]
        category = non_general[(index * 23 + 8) % len(non_general)]
        if wrong == category:
            category = non_general[(non_general.index(category) + 6) % len(non_general)]
        prompt = f"{REAL_CONCEPTS[wrong][index % 3]}じゃない、{REAL_CONCEPTS[category][(index + 1) % 3]}の方"
        rows.append(real_row(f"campus-v21-real-correction-{index:03d}", prompt, category, "correction"))
    for index in range(100):
        category = categories[(index * 19 + 6) % len(categories)]
        prompt = f"{REAL_CONCEPTS[category][index % 3]}について、必要情報と次の手順を教えてください"
        rows.append(real_row(f"campus-v21-real-normal-{index:03d}", prompt, category, "normal"))
    for index in range(100):
        first = non_general[(index * 7 + 3) % len(non_general)]
        second = non_general[(index * 29 + 10) % len(non_general)]
        if first == second:
            second = non_general[(non_general.index(second) + 2) % len(non_general)]
        prompt = f"{REAL_CONCEPTS[first][index % 3]}。それと、{REAL_CONCEPTS[second][(index + 2) % 3]}も"
        rows.append(real_row(f"campus-v21-real-compound-{index:03d}", prompt, first, "compound", [first, second]))
    assert len(rows) == 500 and len({norm(row["prompt"]) for row in rows}) == 500
    return rows


def build_retrieval_sets() -> tuple[list[dict], list[dict]]:
    faq = [json.loads(line) for line in (ROOT / "data" / "campus_v2" / "faq" / "reviewed.jsonl").read_text(encoding="utf-8").splitlines() if line]
    groups = defaultdict(list)
    for item in faq:
        parts = item["semantic_scenario"].split("|", 2)
        groups[(item["category"], parts[1])].append(item["id"])
    concepts = sorted(groups)
    RNG.shuffle(concepts)
    split = int(len(concepts) * .3)

    def make(selected: list[tuple[str, str]], prefix: str) -> list[dict]:
        rows = []
        templates = ("{topic}、確認すること", "{topic}ってどう進める", "{topic}の次の行動")
        for index, (category, topic) in enumerate(selected):
            query = templates[index % len(templates)].format(topic=topic)
            rows.append({"id": f"campus-v21-retrieval-{prefix}-match-{index:04d}", "query": query,
                         "category": category, "relevant_ids": groups[(category, topic)], "has_match": True})
        no_match = ("学食の今日限定メニュー", "キャンパスの落とし物在庫番号", "体育館の現在の混雑人数",
                    "図書館の座席空き状況", "大学バスの現在位置", "売店の商品価格", "教室の現在温度",
                    "サークル部室の鍵番号", "学内プリンタの残量", "食堂のアレルギー個別対応")
        count = 50 if prefix == "validation" else 100
        for index in range(count):
            query = f"{no_match[index % len(no_match)]}を今すぐ知りたい {index // len(no_match) + 1}"
            rows.append({"id": f"campus-v21-retrieval-{prefix}-nomatch-{index:03d}", "query": query,
                         "category": "general", "relevant_ids": [], "has_match": False})
        RNG.shuffle(rows)
        return rows

    return make(concepts[:split], "validation"), make(concepts[split:], "test")


def main() -> None:
    adversarial = build_adversarial_train()
    adversarial_validation = build_adversarial_validation()
    clarification = build_clarification_validation()
    real = build_real_student()
    retrieval_validation, retrieval_test = build_retrieval_sets()

    existing_text = "\n".join(path.read_text(encoding="utf-8") for path in (
        ROOT / "data" / "campus_v2" / "router" / "train.jsonl",
        ROOT / "data" / "campus_v2" / "blind" / "evaluation-2000.json",
        ROOT / "data" / "campus_v2" / "adversarial" / "negation-300.json",
    ))
    existing_norm = {norm(value) for value in re.findall(r'"(?:question|prompt)"\s*:\s*"([^"]+)"', existing_text)}
    real_norm = {norm(row["prompt"]) for row in real}
    overlap = existing_norm & real_norm
    assert not overlap

    save_jsonl(OUTPUT / "router" / "adversarial-train-1500.jsonl", adversarial)
    save_json(OUTPUT / "router" / "adversarial-validation-300.json", adversarial_validation)
    save_json(OUTPUT / "router" / "clarification-validation-1200.json", clarification)
    save_json(OUTPUT / "real-student" / "evaluation-500.json", real)
    save_json(OUTPUT / "retrieval" / "validation.json", retrieval_validation)
    save_json(OUTPUT / "retrieval" / "test.json", retrieval_test)
    save_json(OUTPUT / "manifest.json", {
        "version": "unipilot-campus-v2.1", "adversarial_train": len(adversarial),
        "adversarial_validation": len(adversarial_validation),
        "adversarial_types": dict(Counter(row["adversarial_type"] for row in adversarial)),
        "clarification_validation": len(clarification),
        "clarification_distribution": dict(Counter("ambiguous" if row["ambiguous"] else "determinate" for row in clarification)),
        "real_student": len(real), "real_student_distribution": dict(Counter(row["surface_type"] for row in real)),
        "real_existing_normalized_overlap": len(overlap), "retrieval_validation": len(retrieval_validation),
        "retrieval_test": len(retrieval_test),
        "real_student_sha256": hashlib.sha256((OUTPUT / "real-student" / "evaluation-500.json").read_bytes()).hexdigest(),
        "existing_adversarial_300_role": "test-only; never included in v2.1 training",
        "external_ai_api": "OFF",
    })
    print(json.dumps({"adversarial": len(adversarial), "adversarial_validation": len(adversarial_validation), "clarification": len(clarification),
                      "real": len(real), "retrieval_validation": len(retrieval_validation),
                      "retrieval_test": len(retrieval_test), "overlap": len(overlap)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
