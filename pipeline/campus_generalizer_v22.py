from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from pipeline.campus_planner_v22 import AnswerPlan, INTENT_SIGNALS


DEPTH_LIMITS = {
    "simple": (150, 300),
    "normal": (350, 700),
    "complex": (600, 1200),
}

INTENT_LABELS = {
    "exam": "試験", "assignment": "課題", "assignment_priority": "課題の優先順位",
    "deadline_organizer": "締切管理", "credit": "単位", "gpa": "GPA",
    "grade_simulator": "必要点", "attendance": "出席・欠席", "absence_email": "欠席連絡",
    "lateness": "遅刻", "lateness_email": "遅刻連絡", "registration": "履修",
    "professor_email": "教授へのメール", "late_submission_email": "提出遅延の連絡",
    "report_outline": "レポート", "citation_check": "引用・出典",
    "presentation_outline": "プレゼン", "study_plan": "勉強計画", "toeic_plan": "TOEIC学習",
    "career_schedule": "就活", "internship": "インターン", "scholarship": "奨学金",
    "tuition": "学費", "part_time_job": "アルバイト", "relationship": "人間関係",
    "campus_life": "大学生活", "programming": "プログラミング", "statistics": "統計",
    "ai_usage": "生成AI利用", "university_policy": "大学固有制度", "general": "相談内容",
}

ACTION_STEPS: dict[str, tuple[str, str, str]] = {
    "exam": ("試験日・範囲・形式・配点を一か所に書く", "問題を一度解き、知識不足と時間不足を分ける", "弱点から解き直し、最後に持込条件と教室を確認する"),
    "assignment": ("課題文から成果物・締切・提出方法・評価条件を抜き出す", "必要作業を調査・作成・見直し・提出に分ける", "提出後の受付表示や送信記録を保存する"),
    "assignment_priority": ("各課題の締切・配点・所要時間を並べる", "期限が近く短時間で進む作業から着手する", "残り時間を再確認して次の課題へ移る"),
    "deadline_organizer": ("LMS・シラバス・連絡メールから締切を集める", "日付・時刻・提出先・所要時間を同じ表にする", "前日確認と提出完了確認を予定に入れる"),
    "credit": ("成績表と入学年度の履修要項を開く", "取得済み単位を必修・選択など区分別に集計する", "不足や読み替えを科目名付きで教務へ確認する"),
    "gpa": ("大学のGPA計算式とGP上限を確認する", "現在GPA・取得単位・目標・今後の単位をそろえる", "必要平均を計算し、履修計画と照合する"),
    "grade_simulator": ("現在点・合格点・残り配点を同じ基準にそろえる", "残り評価で必要な得点を計算する", "シラバスの評価割合と再試験条件を確認する"),
    "attendance": ("LMSの出席記録と自分の記録を授業ごとに照合する", "シラバスで欠席・遅刻・公欠の扱いを読む", "差があれば授業名と日付を添えて担当教員へ確認する"),
    "absence_email": ("件名に授業名・欠席連絡・氏名を書く", "欠席日と確認できた理由を簡潔に伝える", "課題や補講の確認方法を一つ質問して送信する"),
    "lateness": ("到着見込みと遅れている理由を確認する", "授業案内で遅刻時の連絡方法を確認する", "入室後の扱いを自己判断せず担当教員へ確認する"),
    "lateness_email": ("件名に授業名・遅刻連絡・氏名を書く", "理由と到着見込みを事実だけで書く", "送信前に宛先・授業名・時刻を確認する"),
    "registration": ("入学年度の履修要項と最新時間割を開く", "必修・前提科目・上限・重複を確認する", "登録画面との差や例外は締切前に教務へ確認する"),
    "professor_email": ("件名に授業名・用件・氏名を書く", "本文で要件と希望する対応を一文ずつ示す", "宛名・学籍番号・添付・候補日時を確認して送る"),
    "late_submission_email": ("提出物・締切・現在の完成状況を確認する", "遅れる事実と提出見込みを簡潔に書く", "代替提出の可否を決めつけず指示をお願いする"),
    "report_outline": ("課題の問い・条件・評価基準を抜き出す", "仮の結論と根拠を二つ以上対応させる", "各段落の主張・根拠・説明を置いてから本文を書く"),
    "citation_check": ("原典を開いて引用箇所と文脈を確認する", "著者・題名・年・ページ・URL・参照日を記録する", "引用と自分の説明を分けて指定形式に整える"),
    "presentation_outline": ("発表目的と制限時間を確認する", "結論・根拠・具体例・まとめの順に配置する", "声に出して時間を測り、超過部分を削る"),
    "study_plan": ("試験日・範囲・使える時間を一覧にする", "理解・演習・復習へ時間を分ける", "終了時に間違いを記録し、次回の最初に解き直す"),
    "toeic_plan": ("現在スコア・目標・試験日を確認する", "語彙・リスニング・読解を毎週すべて練習する", "模試の誤答を原因別に記録して配分を更新する"),
    "career_schedule": ("企業・選考段階・期限・候補日時を一覧にする", "授業や試験の動かせない時間を先に固定する", "ES・企業研究・移動・振り返りまで予定に入れる"),
    "internship": ("募集要項で対象・期間・締切を確認する", "応募書類と面接準備を逆算する", "授業や単位認定との両立条件を大学へ確認する"),
    "scholarship": ("制度名・対象年度・給付か貸与かを確認する", "学内締切・要件・必要書類・提出先を一覧にする", "不明点を締切前に学生支援窓口へ確認する"),
    "tuition": ("対象年度の納付案内と期限を確認する", "延納・分納・減免の正式な案内を探す", "期限前に会計または学生支援窓口へ相談する"),
    "part_time_job": ("労働条件通知書・シフト・勤務記録を保存する", "困っている条件を日時と事実に分けて整理する", "勤務先へ確認し、必要なら公的相談先や大学窓口へ相談する"),
    "relationship": ("起きた事実と自分の受け取り方を分けて書く", "相手へ伝えたい要望を一つに絞る", "安全が保てない場合は一人で対応せず相談窓口につながる"),
    "campus_life": ("困りごとを学業・生活・人間関係・健康に分ける", "期限と影響が大きい一件を選ぶ", "学内の担当窓口か信頼できる人へ具体的に相談する"),
    "programming": ("エラー全文・入力・期待結果・実際の結果を保存する", "再現する最小コードまで減らす", "一つずつ変更して結果を記録する"),
    "statistics": ("変数・標本・目的を確認する", "手法の前提と出力する指標を決める", "数値だけでなく不確実性と限界を説明する"),
    "ai_usage": ("授業と大学の生成AIルールを確認する", "個人情報・未公開資料・他人の文章を入力しない", "出力の事実と引用元を一次資料で検証して利用範囲を記録する"),
    "university_policy": ("大学名・学部・入学年度・対象授業を確認する", "学生便覧・履修要項・シラバスで該当項目を探す", "見つからなければ資料名を添えて公式窓口へ確認する"),
    "general": ("相談内容を一文にし、期限と希望する結果を書く", "分かっている事実と未確認事項を分ける", "今日できる連絡・確認・作業を一つ進める"),
}

CAUTIONS = {
    "university_policy": "大学固有の回数・期限・可否は、大学名と対象年度が分からない状態で断定しません。",
    "attendance": "出席扱いは授業ごとに異なるため、一般論だけで可否を決めないでください。",
    "credit": "卒業可否や単位区分は、入学年度の公式要項との照合が必要です。",
    "registration": "登録上限や例外は年度・学部で変わるため、最新画面と公式要項を優先してください。",
    "scholarship": "採用・支給額・併用可否・締切は制度ごとに異なり、公式募集要項の確認が必要です。",
    "tuition": "延納や減免が認められるとは事前に断定せず、納付期限前に公式窓口へ相談してください。",
    "ai_usage": "AI出力を正しいと仮定せず、授業ルールと一次資料を優先してください。",
    "general": "安全・健康・ハラスメントに関わる場合は、作業整理より先に信頼できる人や学内外の相談先へつながってください。",
}

GENERIC_CONCLUSIONS = {
    intent: f"{label}は、条件を整理して公式情報または担当者を確認し、今日できる一手まで決めると進めやすくなります。"
    for intent, label in INTENT_LABELS.items()
}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？])\s*|\n+", text) if part.strip()]


def _deduplicate(text: str) -> str:
    kept: list[str] = []
    keys: list[str] = []
    for sentence in _sentences(text):
        key = re.sub(r"[\s\W_]+", "", sentence.lower())
        if not key or key in keys or any(len(key) >= 18 and key in previous for previous in keys):
            continue
        kept.append(sentence)
        keys.append(key)
    return "\n".join(kept)


def _clip(text: str, maximum: int) -> str:
    if len(text) <= maximum:
        return text
    selected: list[str] = []
    for sentence in _sentences(text):
        proposed = "\n".join([*selected, sentence])
        if len(proposed) > maximum:
            continue
        selected.append(sentence)
    return "\n".join(selected) if selected else text[:maximum].rstrip()


@dataclass(frozen=True)
class GeneralizationResult:
    text: str
    checks: dict[str, Any]
    revision_count: int
    card: dict[str, Any] | None


class CampusResponseGeneralizerV22:
    """Planner-driven, one-revision answer completion without a generative or external model."""

    @staticmethod
    def _steps(intent: str) -> tuple[str, str, str]:
        return ACTION_STEPS.get(intent, ACTION_STEPS["general"])

    @staticmethod
    def _caution(intent: str) -> str:
        return CAUTIONS.get(intent, "確認できていない数字・制度・期限は作らず、公式資料または担当者の回答を優先してください。")

    @staticmethod
    def _intent_covered(intent: str, text: str) -> bool:
        signals = INTENT_SIGNALS.get(intent, ())
        if intent == "general" or not signals:
            return True
        return any(signal.lower() in text.lower() for signal in signals)

    def checks(self, question: str, text: str, plan: AnswerPlan) -> dict[str, Any]:
        minimum, maximum = DEPTH_LIMITS[plan.answer_depth]
        generic_only = any(value in text for value in (
            "まず質問の対象を入力", "一致度の高いFAQを確認できなかった", "大学によります。",
            "確認してください。", "勉強しましょう。",
        )) and len(text) < minimum
        covered = {intent: self._intent_covered(intent, text) for intent in plan.sub_intents}
        action_present = any(token in text for token in (
            "今やること", "次に", "まず", "1.", "確認する", "送る", "入力して", "問い合わせ",
        ))
        required_conditions = not plan.unknown_facts or any(token in text for token in (
            "確認", "必要", "教えて", "大学名", "期限", "対象年度", "入力",
        ))
        return {
            "character_count": len(text),
            "target_min": minimum,
            "target_max": maximum,
            "length_ok": minimum <= len(text) <= maximum,
            "all_question_elements_answered": all(covered.values()),
            "covered_intents": covered,
            "conditions_present": required_conditions,
            "next_action_present": action_present,
            "specific_not_generic": not generic_only,
            "needs_revision": not (
                minimum <= len(text) <= maximum and all(covered.values()) and required_conditions
                and action_present and not generic_only
            ),
        }

    def _intent_block(self, intent: str, *, numbered_from: int = 1) -> str:
        label = INTENT_LABELS.get(intent, intent)
        steps = self._steps(intent)
        return (f"【{label}】{GENERIC_CONCLUSIONS.get(intent, GENERIC_CONCLUSIONS['general'])}\n"
                f"今やること：{numbered_from}. {steps[0]} {numbered_from + 1}. {steps[1]} "
                f"{numbered_from + 2}. {steps[2]}\n"
                f"注意：{self._caution(intent)}")

    def _rewrite(self, question: str, original: str, plan: AnswerPlan, result: dict[str, Any]) -> str:
        route = str(result.get("route") or "")
        tool = result.get("tool")
        intents = tuple(dict.fromkeys([*(plan.sub_intents or ()), result.get("category") or plan.intent]))
        intents = tuple(intent for intent in intents if intent) or ("general",)
        generic_original = any(value in original for value in (
            "一致度の高いFAQを確認できなかった", "まず質問の対象を入力",
        ))

        if route == "clarify" or plan.need_clarification:
            steps = self._steps(plan.intent)
            text = (f"結論：情報が少ないため断定はできませんが、止まらずにできる準備から進められます。\n"
                    f"今やること：1. {steps[0]} 2. {steps[1]}\n"
                    f"確認したいこと：{original.strip()} 期限がある場合は、その日時も一緒に教えてください。\n"
                    f"注意：{self._caution(plan.intent)}")
        elif len(intents) >= 2 and not (tool or route == "tool"):
            blocks = [self._intent_block(intent) for intent in intents[:3]]
            text = ("結論：複数の相談を分け、締切が近いもの・放置した影響が大きいものから進めます。\n"
                    + "\n".join(blocks)
                    + "\n優先順位：締切時刻、提出や連絡に必要な時間、相手の返信待ち時間を比べ、"
                      "先に連絡が必要なものを送ってから、自分だけで進められる作業に着手してください。")
        elif tool or route == "tool":
            steps = self._steps(plan.intent)
            text = (f"{original.strip()}\n"
                    f"結果の使い方：この結果を目標や期限と比べ、無理がある条件は早めに調整します。\n"
                    f"次にやること：1. {steps[0]} 2. {steps[1]} 3. {steps[2]}\n"
                    f"確認：{self._caution(plan.intent)}")
        elif route in ("rag", "faq", "official") and not generic_original:
            steps = self._steps(plan.intent)
            if plan.answer_depth == "simple":
                first = _sentences(original.strip())[0] if _sentences(original.strip()) else original.strip()
                text = _clip(first, 205)
            else:
                text = original.strip()
            if "今やること" not in text:
                action_steps = steps[:1] if plan.answer_depth == "simple" else steps
                text += "\n今やること：" + " ".join(
                    f"{index}. {step}" for index, step in enumerate(action_steps, 1)
                )
            if "注意" not in text and plan.answer_depth != "simple":
                text += f"\n注意：{self._caution(plan.intent)}"
        else:
            text = self._intent_block(plan.intent).replace(f"【{INTENT_LABELS.get(plan.intent, plan.intent)}】", "結論：", 1)

        minimum, maximum = DEPTH_LIMITS[plan.answer_depth]
        supplements = [
            "理由：期限・条件・確認先を分けると、推測に頼らず、今できる作業と相手の回答を待つ作業を区別できるためです。",
            "確認メモ：確認した資料名・対象年度・担当者・確認日を残し、まだ不明な点は一つずつ質問してください。",
            "実行後：完了したこと、残っていること、次の期限を三行で記録し、状況が変わったら優先順位を更新してください。",
            "迷う場合は、今日中に必要な連絡を先に送り、その待ち時間に資料確認や下書きなど自分だけで進められる作業を行ってください。",
        ]
        for supplement in supplements:
            if len(text) >= minimum:
                break
            text += "\n" + supplement
        text = _deduplicate(text)
        for supplement in supplements:
            if len(text) >= minimum:
                break
            if supplement not in text:
                text += "\n" + supplement
        return _clip(text, maximum)

    def improve(self, question: str, original: str, plan: AnswerPlan,
                result: dict[str, Any]) -> GeneralizationResult:
        before = self.checks(question, original, plan)
        if before["needs_revision"]:
            text = self._rewrite(question, original, plan, result)
            revision_count = 1
        else:
            text = original
            revision_count = 0
        after = self.checks(question, text, plan)
        card = None
        if result.get("tool") or len(plan.sub_intents) >= 2 or plan.intent in {
            "study_plan", "assignment_priority", "deadline_organizer", "career_schedule",
        }:
            steps = self._steps(plan.intent)
            card = {
                "kind": "action_plan",
                "title": f"{INTENT_LABELS.get(plan.intent, '大学生活')}：今やること",
                "summary": "上から順に進め、終わった項目を確認してください。",
                "action_label": "手順をコピー",
                "copy_text": "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1)),
                "fields": [],
                "data": {"steps": list(steps)},
            }
        return GeneralizationResult(text=text, checks={"before": before, "after": after},
                                    revision_count=revision_count, card=card)
