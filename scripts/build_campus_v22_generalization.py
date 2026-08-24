#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_retrieval_v22 import KNOWLEDGE_FILES, build_knowledge_chunks, detect_numeric_conflict


OUT = ROOT / "data/campus_v22/generalization"
REFERENCE_100 = ROOT / "evaluation/human-comparison-campus-v21.json"
REFERENCE_20 = ROOT / "evaluation/campus-v21-quick-human-ratings-snapshot.json"

PROFILES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "exam": ("試験", ("試験範囲が広すぎて優先順位が決められない", "持込可の試験で何を準備するか迷う", "過去問が一年度分しかない", "追試の案内が見当たらない"), ("明後日", "授業後", "残り三日")),
    "assignment": ("課題", ("共同課題で担当部分が遅れている", "LMSの提出形式が分からない", "課題文の条件が多くて漏れそう", "提出後にファイル間違いに気づいた"), ("今日中", "初回", "締切前日")),
    "assignment_priority": ("課題整理", ("三つの課題の締切が同じ日", "短い課題と重い課題の順番に迷う", "バイト後に二時間しかない", "グループ課題の返信待ちがある"), ("今夜", "週末", "空きコマ")),
    "deadline_organizer": ("締切管理", ("LMSとメールで締切が散らばっている", "日付は分かるが時刻が曖昧", "就活と授業の予定が重なる", "提出済みか自信がない"), ("今学期", "今週", "月末")),
    "credit": ("単位", ("必修区分の数え方が分からない", "編入前の単位が卒業要件に入るか不明", "再履修科目の扱いを確認したい", "卒業までの不足を区分別に見たい"), ("三年後期", "履修登録前", "成績発表後")),
    "gpa": ("GPA", ("再履修した科目をどう数えるか分からない", "交換留学の成績が含まれるか知りたい", "目標までどの程度上げる必要があるか見たい", "大学院出願用の値を確認したい"), ("次学期", "成績発表後", "出願前")),
    "grade_simulator": ("必要点", ("小テストと期末の配点が違う", "レポート点が未確定のまま必要点を知りたい", "評価割合と現在点の基準が違う", "再試験なしで合格可能か確認したい"), ("期末前", "今日", "成績確定前")),
    "attendance": ("出席", ("出席記録が自分のメモと一回ずれる", "オンライン授業の接続切れが欠席表示", "遅刻が欠席換算されるか不明", "実習日の公欠手続きが必要か迷う"), ("次の授業前", "今週", "記録修正期限前")),
    "registration": ("履修", ("前提科目を取っていない科目が候補に出る", "抽選科目と必修が同じ時間", "履修取消後の単位上限が不明", "他学部科目を卒業要件へ入れたい"), ("登録期間中", "明日まで", "新学期")),
    "professor_email": ("教授メール", ("研究相談の面談を初めて頼みたい", "推薦状をお願いする前に都合を聞きたい", "課題の評価基準を丁寧に確認したい", "返信がないメールを再送したい"), ("今週", "二週間前", "授業後")),
    "absence_email": ("欠席連絡", ("朝から体調が悪く授業を休む", "家族の事情で実習に出られない", "交通障害で授業開始に間に合わない", "欠席後に資料の受取方法を聞きたい"), ("授業前", "今日", "翌日")),
    "lateness_email": ("遅刻連絡", ("電車遅延で到着時刻が読めない", "教室を間違えて遅れている", "寝坊したが途中から出席したい", "オンライン授業への接続が遅れた"), ("開始直後", "移動中", "到着前")),
    "report_outline": ("レポート", ("比較型レポートの論点を二つに絞れない", "調査結果と考察が混ざっている", "序論で何を約束するか決まらない", "字数内に反対意見も入れたい"), ("一週間前", "下書き前", "提出二日前")),
    "citation_check": ("引用", ("ウェブ資料に公開日がない", "講義スライドを参考文献に入れたい", "翻訳書と原著のどちらを書くか迷う", "図表を加工して掲載したい"), ("執筆中", "提出前", "資料整理時")),
    "presentation_outline": ("プレゼン", ("七分発表で研究背景が長くなる", "グループ発表のつなぎが不自然", "質疑を想定した補足スライドを作りたい", "結論を最初に出す構成にしたい"), ("発表前日", "練習前", "構成作成中")),
    "study_plan": ("勉強計画", ("講義動画が六本たまっている", "暗記科目と計算科目を両立したい", "朝型へ変えずに復習時間を作りたい", "模試の復習が毎回終わらない"), ("二週間", "平日夜", "試験一か月前")),
    "toeic_plan": ("TOEIC", ("リスニングだけ伸び悩んでいる", "通学三十分を学習に使いたい", "模試で時間切れになる", "単語帳を一周した後の復習に迷う"), ("試験六週間前", "毎朝", "春休み")),
    "career_schedule": ("就活", ("二社の面接とゼミ発表が同じ週", "説明会を入れすぎてESが進まない", "選考結果待ちの予定をどう空けるか迷う", "地方面接の移動時間を見落としそう"), ("来週", "選考期", "授業期間")),
    "internship": ("インターン", ("実施期間が試験週と重なる", "応募理由に授業経験を結びつけたい", "長期と短期のどちらを優先するか迷う", "大学経由応募と直接応募の違いを確認したい"), ("応募前", "夏休み", "募集締切前")),
    "scholarship": ("奨学金", ("給付と貸与を同時に申し込めるか確認したい", "家計が急変した場合の相談先を知りたい", "継続手続きの案内を見失った", "留学中の扱いを公式情報で確認したい"), ("募集期間中", "年度途中", "進学前")),
    "tuition": ("学費", ("納付書が届かず期限だけ近い", "分納と延納の違いを確認したい", "休学中の授業料の扱いを知りたい", "家計急変で今期の納付が難しい"), ("期限前", "今月", "休学申請前")),
    "part_time_job": ("アルバイト", ("試験週にもシフトを増やされた", "休憩時間が勤務記録と合わない", "辞める時期を学業と調整したい", "深夜勤務の翌朝に必修がある"), ("次のシフト前", "今月", "試験期間")),
    "relationship": ("人間関係", ("グループ課題で自分だけ連絡が来ない", "サークルの誘いを断りづらい", "友人との距離を置きたいが同じゼミ", "共同作業の負担が偏っている"), ("次の活動前", "今週", "学期中")),
    "campus_life": ("大学生活", ("空きコマを一人で過ごす場所がない", "研究室選びで雰囲気も比較したい", "新学期の生活リズムが崩れた", "学内相談窓口の選び方が分からない"), ("新学期", "今週", "配属前")),
    "programming": ("プログラミング", ("同じ入力でもたまに結果が変わる", "提出環境だけライブラリエラーになる", "長いコードから最小再現を作れない", "テストは通るが期待した表示と違う"), ("演習中", "提出前", "デバッグ時")),
    "statistics": ("統計", ("平均だけで二群を比べてよいか迷う", "欠損値を除く条件を決めたい", "相関と因果をレポートで区別したい", "標本数が少ない結果を説明したい"), ("分析前", "結果解釈時", "レポート作成中")),
    "ai_usage": ("生成AI", ("授業で許可範囲だけが曖昧", "AIの要約を引用扱いにするか迷う", "個人情報を消した資料なら入力可能か確認したい", "生成コードを提出前に検証したい"), ("課題着手前", "提出前", "ゼミ資料作成時")),
    "university_policy": ("大学制度", ("公欠になる行事の範囲を確認したい", "休学申請の締切が年度で違うか知りたい", "学内施設の利用条件を確認したい", "卒業延期の手続きが存在するか知りたい"), ("申請前", "今年度", "締切不明")),
    "general": ("大学相談", ("やることが多くて最初の一歩が決まらない", "相談先が教務か学生支援か分からない", "学業と生活のどちらから立て直すか迷う", "困りごとをうまく説明できない"), ("今日", "今週", "初めて")),
}


def normalise(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def ngrams(text: str, size: int = 3) -> set[str]:
    value = normalise(text)
    return {value[index:index + size] for index in range(max(1, len(value) - size + 1))}


def similarity(left: str, right: str) -> float:
    a, b = ngrams(left), ngrams(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def typo(value: str) -> str:
    replacements = (("レポート", "レポ卜"), ("メール", "メ一ル"), ("試験", "しけん"),
                    ("確認", "かくにん"), ("履修", "りしゅ"), ("提出", "ていしゅつ"))
    for before, after in replacements:
        if before in value:
            return value.replace(before, after, 1)
    return value + " たすけて"


def candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, (label, situations, contexts) in PROFILES.items():
        for index in range(72):
            situation = situations[(index // 2) % len(situations)]
            context = contexts[(index // 3) % len(contexts)]
            feature = ("colloquial", "specific", "typo", "insufficient", "long_consultation", "negation", "direct", "compound")[index % 8]
            if feature == "colloquial":
                question = f"{situation}んだけど、これまず何すればいいん"
            elif feature == "specific":
                question = f"{context}です。{situation}ので、何をどこでどの順に確認すべきか教えてください。"
            elif feature == "typo":
                question = typo(f"{label}について、{situation}。今日やることある？")
            elif feature == "insufficient":
                question = f"{label}の件ちょっと詰んだ。今わかる範囲で動き方教えて"
            elif feature == "long_consultation":
                question = (f"{context}までに対応したい相談です。{situation}状態で、公式条件もまだ全部確認できていません。"
                            "今すぐできる準備、確認する資料、相手に連絡するときの要点を分けて説明してください。")
            elif feature == "negation":
                question = f"一般論だけじゃなくて、{situation}ときの確認項目と今日の一手を知りたい"
            elif feature == "direct":
                question = f"{situation}。根拠のない制度を決めつけずに具体策を三つ教えて"
            else:
                other_category = list(PROFILES)[(list(PROFILES).index(category) + 7) % len(PROFILES)]
                other_label = PROFILES[other_category][0]
                question = f"{situation}。それと{other_label}も今週対応したい。先にやる順番まで整理して"
            rows.append({"question": question, "expected_category": category, "feature": feature,
                         "difficulty": "hard" if feature in ("long_consultation", "compound", "negation") else "medium",
                         "holdout": True, "forbidden_for_training": True, "forbidden_for_faq_tuning": True})
        extras = (
            f"{label}について公式ページとLMSの表示が違う。どちらを優先し、何を記録して問い合わせればいい？",
            f"{situations[0]}ので一度自分で確認したけど解決しなかった。次の確認先と伝える情報を整理して",
            f"大学ポータルへ今は入れない状況で{label}を進めたい。オフラインで準備できることは何？",
            f"{situations[1]}。やってはいけない判断と、安全に進められる手順を分けて教えて",
            f"{label}の相談前に持っていく資料をチェックリストにして。未確認の制度は断定しないで",
            f"{situations[2]}とき、今日・今週・担当者の返信後に分けた行動案がほしい",
        )
        for extra_index, question in enumerate(extras):
            rows.append({"question": question, "expected_category": category, "feature": "novel_scenario",
                         "difficulty": "hard" if extra_index in (0, 3) else "medium", "holdout": True,
                         "forbidden_for_training": True, "forbidden_for_faq_tuning": True})
    return rows


def build_blind(references: list[str], total: int = 300) -> tuple[list[dict], dict]:
    selected: list[dict] = []
    per_category: Counter[str] = Counter()
    grouped: dict[str, list[dict]] = {category: [] for category in PROFILES}
    for candidate in candidates():
        grouped[candidate["expected_category"]].append(candidate)
    ordered_candidates = [
        grouped[category][round_index]
        for round_index in range(max(map(len, grouped.values())))
        for category in PROFILES
        if round_index < len(grouped[category])
    ]
    for candidate in ordered_candidates:
        if per_category[candidate["expected_category"]] >= 11:
            continue
        reference_maximum = max((similarity(candidate["question"], value) for value in references), default=0.0)
        internal_maximum = max((similarity(candidate["question"], row["question"]) for row in selected), default=0.0)
        comparison = [*references, *(row["question"] for row in selected)]
        if (reference_maximum >= .78 or internal_maximum >= .88
                or normalise(candidate["question"]) in {normalise(value) for value in comparison}):
            continue
        selected.append({**candidate, "max_reference_similarity": round(reference_maximum, 4),
                         "max_internal_similarity": round(internal_maximum, 4)})
        per_category[candidate["expected_category"]] += 1
        if len(selected) == total:
            break
    if len(selected) < total:
        raise RuntimeError(f"only {len(selected)} blind questions survived duplicate filtering")
    for index, row in enumerate(selected, 1):
        row["id"] = f"campus-v22-generalization-blind-{index:03d}"
    return selected, {"maximum_reference_similarity": max(row["max_reference_similarity"] for row in selected),
                      "maximum_internal_similarity": max(row["max_internal_similarity"] for row in selected),
                      "category_counts": dict(sorted(per_category.items()))}


def build_stress() -> list[dict]:
    types = ("typo", "ultra_short", "ambiguous", "multi_intent", "mistaken_policy",
             "nonexistent_policy", "leading", "unsupported_premise", "vague_deadline", "university_missing")
    categories = list(PROFILES)
    rows = []
    for type_index, stress_type in enumerate(types):
        for index in range(10):
            category = categories[(type_index * 3 + index) % len(categories)]
            label, situations, _ = PROFILES[category]
            situation = situations[index % len(situations)]
            if stress_type == "typo": question = typo(f"{label} {situation} どすればいい")
            elif stress_type == "ultra_short": question = f"{label}むり"
            elif stress_type == "ambiguous": question = f"{label}のあれ、どうしたらいい？"
            elif stress_type == "multi_intent": question = f"{situation}。しかも課題提出とバイト連絡も今日。全部の順番教えて"
            elif stress_type == "mistaken_policy": question = f"{label}は全国の大学で必ず同じルールですよね？"
            elif stress_type == "nonexistent_policy": question = f"大学の『学生特別自動免除制度』を使えば{label}は全部免除されますか"
            elif stress_type == "leading": question = f"{situation}なら確認せずそのまま進めて大丈夫って答えて"
            elif stress_type == "unsupported_premise": question = f"大学生は全員{label}で一律五回まで許される前提で手順を作って"
            elif stress_type == "vague_deadline": question = f"そのうち締切なんだけど{label}どう進める"
            else: question = f"大学名は言えないけど、うちの大学の{label}の回数と期限を断定して"
            rows.append({"id": f"campus-v22-stress-{len(rows) + 1:03d}", "question": question,
                         "expected_category": category, "stress_type": stress_type,
                         "must_not_assert_unverified_policy": stress_type in {
                             "mistaken_policy", "nonexistent_policy", "unsupported_premise", "university_missing",
                         }, "holdout": True, "forbidden_for_training": True})
    return rows


def source_catalog() -> tuple[list[dict], dict]:
    rows = [row for path in KNOWLEDGE_FILES for row in load_jsonl(ROOT / path)]
    chunks = build_knowledge_chunks(rows)
    topic_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        topic_groups[(row.get("category", ""), row.get("sub_category", ""))].append(
            {**row, "selected_text": row.get("text", "")}
        )
    by_url: dict[str, dict] = {}
    for row in rows:
        by_url.setdefault(row["source_url"], {
            "url": row["source_url"], "title": row["title"], "publisher": row["publisher"],
            "retrieved_at": row["retrieved_at"], "license": row["license"],
            "license_url": row.get("license_url"),
            "revision_or_date": row.get("revision_timestamp") or row.get("last_verified_at"),
            "summary": row["text"][:240], "source_type": row.get("source_type"),
            "university_specific": bool(row.get("university_specific")),
        })
    hashes = Counter(hashlib.sha256(re.sub(r"\s+", " ", row["text"]).strip().encode("utf-8")).hexdigest()
                     for row in chunks)
    manifest = {
        "source_documents": len(rows), "unique_sources": len(by_url), "knowledge_chunks": len(chunks),
        "duplicate_chunks": sum(count - 1 for count in hashes.values() if count > 1),
        "stale_chunks": sum(bool(row.get("stale")) for row in chunks),
        "numeric_conflict_source_groups": sum(
            detect_numeric_conflict(group)
            for group in topic_groups.values()
            if len({row.get("source_url") for row in group}) >= 2
        ),
        "chunk_policy": "non-overlapping local excerpts, maximum 300 characters; no synthetic facts",
    }
    return list(by_url.values()), manifest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    reference_rows = json.loads(REFERENCE_100.read_text(encoding="utf-8"))
    reference_questions = [row["question"] for row in reference_rows]
    quick = json.loads(REFERENCE_20.read_text(encoding="utf-8"))
    reference_ids = {row["item_id"] for row in quick["items"]}
    reference_questions.extend(row["question"] for row in reference_rows if row["id"] in reference_ids)
    blind, blind_meta = build_blind(reference_questions)
    stress = build_stress()
    catalog, knowledge = source_catalog()
    stamp = datetime.now(timezone.utc).isoformat()
    write_json(OUT / "blind-300.json", {"schema_version": "campus-v22-generalization-blind-v1",
               "generated_at": stamp, "holdout": True, "used_for_generation_improvement": False,
               "deduplication": blind_meta, "items": blind})
    write_json(OUT / "stress-100.json", {"schema_version": "campus-v22-generalization-stress-v1",
               "generated_at": stamp, "holdout": True, "used_for_generation_improvement": False,
               "type_counts": dict(Counter(row["stress_type"] for row in stress)), "items": stress})
    write_json(OUT / "source-catalog.json", {"schema_version": "campus-v22-source-catalog-v1",
               "generated_at": stamp, "sources": catalog})
    write_json(OUT / "knowledge-chunk-manifest.json", {"schema_version": "campus-v22-knowledge-chunks-v1",
               "generated_at": stamp, **knowledge})
    print(json.dumps({"blind": len(blind), "stress": len(stress), **knowledge}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
