from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re

from pipeline.standard_categories import CATEGORY_PROFILES, STANDARD_CATEGORIES


ROOT = Path("data/v08")
VERSION = "unipilot-standard-v08-structured-scenarios"
LICENSE = "CC0-1.0"
CREATED = "2026-08-23"
SYSTEM = (
    "あなたは大学生活支援に特化した完全ローカルのUniPilot Standardです。"
    "検索文脈に根拠がある場合はそれを使い、不足する情報は推測せず確認方法を案内します。"
)

CONDITIONS = (
    ("まず基本だけ知りたい", "最初に結論を示し、今すぐ確認できる一項目から始めます。"),
    ("期限が近く時間が少ない", "残り時間を確認し、必須作業、連絡、追加改善の順に優先します。"),
    ("大学ごとの規則が分からない", "現在年度の公式案内を確認し、大学固有の条件は担当窓口へ尋ねます。"),
    ("すでに問題が起きている", "事実と時刻を記録し、影響を小さくする連絡を先に行います。"),
    ("初めてで手順が分からない", "必要な情報、行動、確認先の三つに分けて順番に進めます。"),
    ("複数の選択肢を比較したい", "目的、期限、負担、リスクを同じ基準で並べて比較します。"),
    ("体調や生活との両立が不安", "安全と休息を確保し、無理のない範囲と相談先を同時に考えます。"),
    ("相手への連絡も必要", "用件、事実、希望する確認を簡潔にまとめ、早めに指定先へ連絡します。"),
    ("再発を防ぎたい", "今回の原因を一つ特定し、期限通知や事前確認を仕組みにします。"),
    ("理由も含めて詳しく知りたい", "判断理由と例外条件を分け、根拠を確認できる場所も示します。"),
)

QUESTION_FORMS = (
    "{topic}ことで困っています。{condition}です。どう進めればいいですか？",
    "{topic}場合について、{condition}ので対応を教えてください。",
    "大学生活の相談です。{topic}状況で、{condition}です。優先順位は？",
    "{topic}ときに何を確認すべきですか。{condition}です。",
)

GENERAL_SUBJECTS = (
    "新しい制度の説明", "二つの選択肢", "長い文章", "意見の対立", "作業手順", "原因と結果", "条件の違い", "調査結果",
    "失敗からの改善", "初学者向け解説", "安全上の注意", "予定変更", "資料の要点", "複数の根拠", "例外のある規則", "不確かな情報",
    "時間の見積り", "比較表", "質問への返答", "相談内容", "短い報告", "丁寧な依頼", "反対意見", "段階的な計画",
    "データの読み方", "用語の定義", "具体例", "優先順位", "要約文", "注意書き", "振り返り", "改善提案",
    "初心者の疑問", "複合条件", "意思決定", "確認方法", "根拠の示し方", "順序の説明", "誤解の訂正", "結論の伝え方",
)

GENERAL_TASKS = (
    ("要点を二文でまとめる", "結論を先に置き、重要な理由を一つ続けます。細部は要点を変えない範囲で省きます。"),
    ("理由と結果をつなぐ", "原因と観察された結果を分け、推測にすぎない部分は断定しません。"),
    ("二案を比較する", "目的、利点、負担、条件を同じ順序で並べ、どの条件ならどちらを選ぶか示します。"),
    ("手順を順番に説明する", "準備、実行、確認の順に分け、各段階の終了条件を明確にします。"),
    ("曖昧さを残して答える", "分かっている範囲を先に答え、不足情報と確認方法を続けます。"),
    ("具体例を一つ加える", "原則を説明してから短い例を示し、例を一般規則と混同しないよう補足します。"),
    ("反対意見も扱う", "主張、根拠、反対意見、判断条件の順に整理します。"),
    ("長文を読みやすく直す", "一文を短くし、接続関係を明示して、段落ごとに一つの要点へ絞ります。"),
    ("誤りを丁寧に訂正する", "誤っている箇所を特定し、正しい内容と確認根拠を攻撃的でない表現で示します。"),
    ("次の行動を提案する", "状況を確認したうえで、今日できる小さな行動と後で確認する事項を分けます。"),
)

BAD_REASONS = (
    "wrong category", "invented subject", "unrelated advice", "university-specific hallucination",
    "incomplete answer", "excessive generic response", "unsupported date or fee", "ignores compound constraint",
)


def normalized(text: str) -> str:
    return re.sub(r"[\s、。！？,.!?・:：]", "", text).lower()


def stable_index(value: str, size: int) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % size


def split_for_scenario(scenario_id: str) -> str:
    bucket = stable_index(scenario_id, 20)
    return "train" if bucket < 17 else ("validation" if bucket < 19 else "test")


def answer_for(category: str, condition_index: int, mode: str, key: str) -> str:
    profile = CATEGORY_PROFILES[category]
    conclusion = profile["conclusion"]
    actions = list(profile["actions"])
    shift = stable_index(key, len(actions))
    actions = actions[shift:] + actions[:shift]
    condition_action = CONDITIONS[condition_index][1]
    caution = profile["caution"]
    if mode == "short":
        variants = (
            f"{conclusion}{actions[0]}",
            f"まず、{conclusion}{condition_action}",
            f"{actions[0]}そのうえで、{conclusion}",
        )
    elif mode == "normal":
        variants = (
            f"{conclusion}理由は、確認条件を曖昧にしたまま進めると手戻りが増えるためです。{actions[0]}次に、{condition_action}{caution}",
            f"まず{actions[0]}続いて{actions[1]}{condition_action}なお、{caution}",
            f"結論として、{conclusion}{actions[0]}状況に合わせて、{condition_action}最後に、{caution}",
        )
    else:
        variants = (
            f"{conclusion}\n\n進め方は三段階です。第一に、{actions[0]}第二に、{actions[1]}第三に、{actions[2]}"
            f"今回の条件では、{condition_action}\n\n判断できない情報は推測せず、公式案内や担当窓口で確認してください。{caution}",
            f"最初に状況と期限を整理してください。{actions[0]}その結果を基に、{actions[1]}さらに、{actions[2]}"
            f"特に、{condition_action}\n\nこの順番なら、急ぐ行動と後から改善する行動を分けられます。{caution}",
            f"おすすめは、確認・実行・再確認を分けることです。\n1. {actions[0]}\n2. {actions[1]}\n3. {actions[2]}\n"
            f"加えて、{condition_action}不明点は確認事項として残してください。{caution}",
        )
    return variants[stable_index(key + mode, len(variants))]


def conversation_row(row_id: str, user: str, assistant: str, category: str, stage: str,
                     context: str | None = None, length_mode: str = "normal") -> dict:
    row = {
        "id": row_id, "kind": "conversation", "user": user, "assistant": assistant,
        "category": category, "stage": stage, "length_mode": length_mode,
        "system": SYSTEM,
        "source": "UniPilot project original", "source_url": None, "license": LICENSE,
        "created_at": CREATED, "dataset_version": VERSION,
    }
    if context is not None:
        row["context"] = context
    return row


def build_scenarios() -> tuple[list[dict], list[dict]]:
    scenarios, knowledge = [], []
    for category in STANDARD_CATEGORIES:
        profile = CATEGORY_PROFILES[category]
        for topic_index, topic in enumerate(profile["concerns"]):
            doc_id = f"v08-knowledge-{category}-{topic_index:02d}"
            knowledge.append({
                "id": doc_id, "title": f"{profile['label']}：{topic}",
                "text": answer_for(category, topic_index, "detailed", doc_id), "category": category,
                "source": "UniPilot project original", "source_url": None, "license": LICENSE,
                "retrieved_at": CREATED, "source_type": "project_authored_standard_knowledge",
            })
            for condition_index, (condition, _) in enumerate(CONDITIONS):
                scenario_id = f"v08-scenario-{category}-{topic_index:02d}-{condition_index:02d}"
                form = QUESTION_FORMS[stable_index(scenario_id, len(QUESTION_FORMS))]
                question = form.format(topic=topic, condition=condition)
                scenarios.append({
                    "id": scenario_id, "question": question, "category": category,
                    "topic": topic, "topic_index": topic_index, "condition": condition,
                    "condition_index": condition_index, "knowledge_id": doc_id,
                    "semantic_key": f"{category}|{topic}|{condition}", "split": split_for_scenario(scenario_id),
                })
    return scenarios, knowledge


def build_general() -> list[dict]:
    rows = []
    styles = ("簡潔に", "初学者向けに", "理由を明確に", "自然な接続詞を使って", "条件を分けて")
    for subject_index, subject in enumerate(GENERAL_SUBJECTS):
        for task_index, (task, answer) in enumerate(GENERAL_TASKS):
            for style_index, style in enumerate(styles):
                row_id = f"v08-general-{subject_index:02d}-{task_index:02d}-{style_index}"
                user = f"「{subject}」について、{task}文章を{style}作ってください。"
                prefix = ("要点から述べます。", "整理すると、", "まず結論です。", "順序立てると、", "確認できる範囲では、")[style_index]
                row = conversation_row(row_id, user, prefix + answer, "general_education", "A-general")
                row["split"] = split_for_scenario(f"general-{subject_index}-{task_index}")
                rows.append(row)
    return rows


def build_training(scenarios: list[dict], knowledge: list[dict]) -> dict[str, list[dict]]:
    by_doc = {row["id"]: row for row in knowledge}
    stage_b, stage_c, stage_d, corrected, corrections = [], [], [], [], []
    for scenario in scenarios:
        category, key = scenario["category"], scenario["id"]
        normal = answer_for(category, scenario["condition_index"], "normal", key)
        stage_b_row = conversation_row("v08-b-" + key[13:], scenario["question"], normal, category, "B-university")
        stage_b_row["split"] = scenario["split"]
        stage_b.append(stage_b_row)
        for mode in ("short", "normal", "detailed"):
            request = {"short": "短く答えてください。", "normal": "結論、理由、行動を含めて答えてください。",
                       "detailed": "条件と注意点まで詳しく答えてください。"}[mode]
            stage_c_row = conversation_row(f"v08-c-{key[13:]}-{mode}", scenario["question"] + request,
                                           answer_for(category, scenario["condition_index"], mode, key),
                                           category, "C-instruction", length_mode=mode)
            stage_c_row["split"] = scenario["split"]
            stage_c.append(stage_c_row)
        document = by_doc[scenario["knowledge_id"]]
        context = f"[{document['title']}]\n{document['text']}"
        stage_d_row = conversation_row("v08-d-" + key[13:], scenario["question"], normal, category,
                                       "D-rag-grounded", context=context)
        stage_d_row["split"] = scenario["split"]
        stage_d.append(stage_d_row)
        reason = BAD_REASONS[stable_index(key, len(BAD_REASONS))]
        other_category = STANDARD_CATEGORIES[(STANDARD_CATEGORIES.index(category) + 7) % len(STANDARD_CATEGORIES)]
        if reason == "wrong category":
            bad = CATEGORY_PROFILES[other_category]["conclusion"]
        elif reason == "invented subject":
            bad = "法学の授業だと決めて、法学の専門用語だけを暗記してください。"
        elif reason == "unrelated advice":
            bad = "サークルへ参加して友達を増やせば解決します。"
        elif reason == "university-specific hallucination":
            bad = "どの大学でも申請なしで必ず認められます。"
        elif reason == "incomplete answer":
            bad = "確認してください。"
        elif reason == "unsupported date or fee":
            bad = "申請は2026年9月1日までで、料金は必ず3000円です。"
        elif reason == "ignores compound constraint":
            bad = "一つだけ進めればよく、ほかの期限は考えなくて大丈夫です。"
        else:
            bad = "状況によります。落ち着いて頑張ってください。"
        correction = {
            "id": "v08-correction-" + key[13:], "question": scenario["question"], "bad_answer": bad,
            "bad_reason": reason, "correct_answer": normal, "category": category,
            "source": "UniPilot project original", "source_url": None, "license": LICENSE,
            "created_at": CREATED, "dataset_version": VERSION,
        }
        corrections.append(correction)
        corrected_row = conversation_row(
            "v08-f-" + key[13:],
            f"質問：{scenario['question']}\n問題のある回答：{bad}\n問題点：{reason}\n根拠のない補完を除いて訂正してください。",
            normal, category, "F-correction", context=context,
        )
        corrected_row["split"] = scenario["split"]
        corrected.append(corrected_row)
    return {"stage_b": stage_b, "stage_c": stage_c, "stage_d": stage_d,
            "stage_f": corrected, "corrections": corrections}


def build_compound() -> list[dict]:
    rows = []
    pairs = list(itertools.combinations(STANDARD_CATEGORIES, 2))[:500]
    for pair_index, (left, right) in enumerate(pairs):
        for variant in range(2):
            left_profile, right_profile = CATEGORY_PROFILES[left], CATEGORY_PROFILES[right]
            left_topic = left_profile["concerns"][(pair_index + variant) % 10]
            right_topic = right_profile["concerns"][(pair_index * 3 + variant + 1) % 10]
            constraint = ("両方の期限が近い", "体調を崩さない範囲で進めたい")[variant]
            question = f"{left_topic}うえに、{right_topic}状況です。{constraint}とき、どちらから何をすればいいですか？"
            answer = (
                f"まず両方の締切、影響、連絡期限を確認してください。{left_profile['actions'][0]}"
                f"同時に、{right_profile['actions'][0]}取り返しにくい期限や安全に関わる対応を先に行い、"
                f"残りを30〜60分の作業へ分けます。{left_profile['caution']}{right_profile['caution']}"
            )
            row = conversation_row(f"v08-e-{pair_index:03d}-{variant}", question, answer,
                                   f"{left}+{right}", "E-compound", length_mode="detailed")
            row["split"] = split_for_scenario(f"compound-{pair_index}")
            rows.append(row)
    return rows


def build_blind(knowledge: list[dict]) -> list[dict]:
    docs_by_category = {}
    for row in knowledge:
        docs_by_category.setdefault(row["category"], []).append(row["id"])
    modifiers = (
        "結論と最初の行動を教えてください。", "判断に必要な条件も示してください。", "大学固有の点は断定しないでください。",
        "今日できる対応を順番に教えてください。", "理由を含めて説明してください。", "注意点を一つ加えてください。",
        "選択肢があるなら比較してください。", "不足情報があれば確認方法も示してください。",
    )
    rows = []
    for category in STANDARD_CATEGORIES:
        profile = CATEGORY_PROFILES[category]
        for topic_index, topic in enumerate(profile["blind"]):
            for modifier_index, modifier in enumerate(modifiers):
                item_id = f"v08-blind-{category}-{topic_index}-{modifier_index}"
                rows.append({
                    "id": item_id, "prompt": f"{topic}ことで相談です。{modifier}", "category": category,
                    "difficulty": ("simple", "medium", "hard", "compound")[modifier_index % 4],
                    "expected_key_points": [profile["conclusion"], profile["actions"][0], profile["caution"]],
                    "relevant_document_ids": docs_by_category[category],
                    "family": f"blind-only-{category}-{topic_index}", "blind": True,
                    "source": "UniPilot project original blind benchmark", "license": LICENSE,
                    "dataset_version": "unipilot-standard-v08-blind-528",
                })
    return rows


def build_chatgpt_comparison(blind: list[dict]) -> list[dict]:
    selected = [blind[index] for index in range(0, len(blind), 5)][:100]
    return [{
        "id": f"v08-comparison-{index:03d}", "question": row["prompt"], "category": row["category"],
        "expected_key_points": row["expected_key_points"],
        "scoring_rubric": {
            "0": "質問に答えていない、または危険な断定がある", "1": "ほぼ使えない", "2": "一部だけ役立つ",
            "3": "普通に使える", "4": "正確で具体的", "5": "正確・具体的で条件と次の行動が明確",
        },
        "manual_only": True, "external_api_calls": False,
    } for index, row in enumerate(selected)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    scenarios, standard_knowledge = build_scenarios()
    general = build_general()
    training = build_training(scenarios, standard_knowledge)
    compound = build_compound()
    blind = build_blind(standard_knowledge)
    comparison = build_chatgpt_comparison(blind)

    public_knowledge = []
    v07_knowledge = Path("data/v07/knowledge/documents.jsonl")
    if v07_knowledge.exists():
        public_knowledge = [json.loads(line) for line in v07_knowledge.read_text(encoding="utf-8").splitlines() if line]
    knowledge = standard_knowledge + public_knowledge
    classifier = [{"id": row["id"], "question": row["question"], "category": row["category"]}
                  for row in scenarios if row["split"] == "train"]
    retrieval_dev = [{
        "id": "v08-retrieval-dev-" + row["id"][13:], "prompt": row["question"],
        "category": row["category"], "relevant_document_ids": [row["knowledge_id"]],
        "source_split": "validation", "tuning_allowed": True,
    } for row in scenarios if row["split"] == "validation"]
    conversation = (training["stage_c"] + training["stage_d"])[:6000]

    files = {
        "conversation/conversation.jsonl": conversation,
        "corrected/corrected.jsonl": training["corrections"],
        "knowledge/documents.jsonl": knowledge,
        "classifier/train.jsonl": classifier,
    }
    for relative, rows in files.items():
        write_jsonl(ROOT / relative, rows)
    curriculum = {
        "A": general, "B": training["stage_b"], "C": training["stage_c"],
        "D": training["stage_d"], "E": compound, "F": training["stage_f"],
    }
    for stage, rows in curriculum.items():
        for split in ("train", "validation", "test"):
            write_jsonl(ROOT / "curriculum" / stage / f"{split}.jsonl",
                        [row for row in rows if row.get("split") == split])
    (ROOT / "blind").mkdir(parents=True, exist_ok=True)
    (ROOT / "benchmarks").mkdir(parents=True, exist_ok=True)
    (ROOT / "blind/evaluation.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "retrieval").mkdir(parents=True, exist_ok=True)
    (ROOT / "retrieval/dev.json").write_text(json.dumps(retrieval_dev, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "benchmarks/chatgpt-gemini-manual.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    train_questions = {normalized(row["question"]) for row in scenarios}
    blind_questions = {normalized(row["prompt"]) for row in blind}
    knowledge_titles = {normalized(row["title"]) for row in knowledge}
    direct = training["stage_c"]
    report = {
        "dataset_version": VERSION,
        "categories": len(STANDARD_CATEGORIES),
        "semantic_scenario_cells": len(scenarios),
        "semantic_structure": "33 categories x 10 distinct concerns x 10 distinct constraints",
        "high_quality_instruction_rows": len(direct),
        "corrected_rows": len(training["corrections"]),
        "conversation_rows": len(conversation),
        "compound_rows": len(compound),
        "general_japanese_rows": len(general),
        "knowledge_documents": len(knowledge),
        "standard_project_knowledge_documents": len(standard_knowledge),
        "blind_questions": len(blind),
        "retrieval_development_questions": len(retrieval_dev),
        "blind_difficulty": Counter(row["difficulty"] for row in blind),
        "chatgpt_gemini_manual_questions": len(comparison),
        "exact_train_blind_question_overlap": len(train_questions & blind_questions),
        "exact_knowledge_title_blind_overlap": len(knowledge_titles & blind_questions),
        "instruction_pair_duplicates": len(direct) - len({normalized(row["user"] + row["assistant"]) for row in direct}),
        "corrected_reason_distribution": Counter(row["bad_reason"] for row in training["corrections"]),
        "external_llm_used": False,
        "external_pretrained_model_used": False,
        "quality_limit": (
            "Scenario cells combine independently authored concern and constraint inventories. They are semantically broader than paraphrase-only rows, "
            "but are still programmatically composed and have not received row-by-row human review. Counts must not be described as 3,300 independently authored answers."
        ),
    }
    Path("evaluation/dataset-quality-v08.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "dataset_version": VERSION, "system": SYSTEM, "files": {key: len(value) for key, value in files.items()},
        "curriculum": {stage: len(rows) for stage, rows in curriculum.items()},
        "blind": "data/v08/blind/evaluation.json", "retrieval_development": "data/v08/retrieval/dev.json",
        "manual_comparison": "data/v08/benchmarks/chatgpt-gemini-manual.json",
        "quality_report": "evaluation/dataset-quality-v08.json", "production_enabled": False,
        "external_ai_api": "OFF", "license_policy": "Project-authored rows are CC0; inherited public knowledge retains per-row source and license.",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
