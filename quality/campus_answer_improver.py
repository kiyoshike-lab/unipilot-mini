from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from quality.campus_ai_judge import CampusAIJudge, _normalise, _sentences


CATEGORY_GUIDANCE: dict[str, dict[str, Any]] = {
    "exam": {"conclusion": "過去問は答えを覚えるためではなく、出題形式と弱点を把握するために使うのが効果的です。",
             "reason": "本番と同じ時間で解き、採点後に間違いの原因を分けると、残り時間を得点に結び付けやすくなります。",
             "steps": ("シラバスで範囲・形式・持込条件を確認する", "一度時間を測って解き、迷った問題にも印を付ける",
                       "知識不足・読み違い・時間不足に分け、同じ型を解き直す"),
             "caution": "過去問と同じ問題が出るとは限らないため、周辺の基本事項も確認してください。"},
    "attendance": {"conclusion": "欠席回数は、記憶ではなく授業ごとの出席記録とシラバスを照合して確認してください。",
                   "reason": "欠席・遅刻の扱いや成績への影響は、大学全体ではなく授業単位で異なることがあります。",
                   "steps": ("LMSや学生ポータルの出席記録を確認する", "シラバスの出席・遅刻・公欠の扱いを読む",
                             "記録が合わなければ、授業名・日付を添えて担当教員へ確認する"),
                   "caution": "『何回まで大丈夫』とは一律に断定せず、最新の公式情報を優先してください。"},
    "citation_check": {"conclusion": "孫引きを避けるには、可能な限り元の一次資料までたどって、その内容を自分で確認して引用します。",
                       "reason": "二次資料だけを頼ると、文脈や表現の正確さを検証できず、引用元も不明確になりやすいためです。",
                       "steps": ("二次資料の参考文献から元資料を特定する", "元資料の該当箇所を読み、ページやURLを記録する",
                                 "引用部分と自分の解釈を分け、指定形式で出典を書く"),
                       "caution": "元資料を入手できない場合は、二次資料からの引用であることを明示し、授業の引用規則を確認してください。"},
    "scholarship": {"conclusion": "奨学金の締切は制度ごとに異なるため、募集要項と在籍大学の最新案内を今日確認するのが確実です。",
                    "reason": "同じ名称でも給付・貸与、学内締切、必要書類、継続手続きが異なる場合があります。",
                    "steps": ("大学ポータルと奨学金窓口の案内を開く", "制度名・学内締切・提出方法・必要書類をメモする",
                              "不明点を締切前に学生支援窓口へ問い合わせる"),
                    "caution": "採用、金額、併用可否は募集要項を確認する前に断定しないでください。"},
    "campus_life": {"conclusion": "研究室は研究テーマだけでなく、指導方法・活動時間・学生の雰囲気まで確認して選ぶと失敗を減らせます。",
                    "reason": "同じ分野でも、教員との相談頻度や研究の進め方によって日々の相性が大きく変わるためです。",
                    "steps": ("教員ページと最近の研究成果を読む", "説明会や訪問で指導頻度・在室時間・配属条件を質問する",
                              "複数の研究室を同じ観点で比較し、希望理由を書く"),
                    "caution": "外部の評判だけで決めず、可能なら所属学生と教員の両方から事実を確認してください。"},
    "study_plan": {"conclusion": "一日2時間なら、理解・演習・復習に時間を分け、毎日小さく確認する計画が続けやすいです。",
                   "reason": "読むだけで終えず、問題を解いて弱点を翌日に戻す流れを作ると、限られた時間を使いやすくなります。",
                   "steps": ("科目と試験日、範囲を一覧にする", "120分を理解30分・演習70分・復習20分の目安で仮置きする",
                             "終了時に間違いを3つまで記録し、翌日の最初に解き直す"),
                   "caution": "苦手度や試験形式に合わせて配分を変えるため、科目・残り日数・使える曜日も確認してください。"},
    "assignment": {"conclusion": "提出サイトが動かない場合は、復旧を待つだけでなく、証拠を残して代替提出の確認を同時に進めてください。",
                   "reason": "締切直前の障害では、完成ファイルと発生時刻を示せることが重要だからです。",
                   "steps": ("エラー画面・時刻・操作内容をスクリーンショットで保存する", "ファイルを完成状態で保存し、別端末や推奨ブラウザを一度だけ試す",
                             "担当教員または指定窓口へ、証拠を添えて代替提出方法を確認する"),
                   "caution": "メール提出が正式に受理されるとは断定せず、授業案内と教員の指示を優先してください。"},
    "registration": {"conclusion": "履修上限は、現在年度の履修要項と学生ポータルで確認し、例外条件があれば教務窓口へ確認します。",
                     "reason": "学年・学部・成績・資格課程などで上限や算入対象が変わる可能性があります。",
                     "steps": ("履修要項で上限と対象外科目の扱いを確認する", "登録済み単位と追加予定単位を分けて合計する",
                               "上限超過や表示差があれば、画面を保存して教務へ問い合わせる"),
                     "caution": "追加登録や上限超過が認められるとは事前に断定しないでください。"},
    "ai_usage": {"conclusion": "生成AIには個人を特定できる情報や未公開資料を入力せず、授業ルールを確認してから使ってください。",
                 "reason": "入力内容の保存・再利用条件がサービスごとに異なり、課題では利用申告や禁止事項が設定される場合があるためです。",
                 "steps": ("氏名・学籍番号・連絡先・成績・他人の文章を削除または匿名化する", "課題指示と大学の生成AIガイドラインを確認する",
                           "出力の事実と引用元を一次資料で検証し、必要なら利用範囲を記録する"),
                 "caution": "AIの出力や引用候補を正しいと仮定せず、機密情報は匿名化しても入力しない判断を優先してください。"},
    "general": {"conclusion": "まず締切と影響が大きいものを一つ特定し、今日中に止血できる行動から始めましょう。",
                "reason": "困りごとを全部同時に解こうとすると優先順位が曖昧になるため、期限・影響・所要時間で分けると整理しやすくなります。",
                "steps": ("困っていることを一行ずつ書き、締切を付ける", "今日放置すると困るものを一つ選ぶ",
                          "提出・連絡・確認のどれか一つを15分だけ進める"),
                "caution": "安全や健康に関わる場合は、課題整理より先に家族・友人・学内相談窓口などへつながってください。"},
    "professor_email": {"conclusion": "面談依頼は、用件・候補日時・必要時間を短く示すと先生が返答しやすくなります。",
                        "reason": "背景を長く書くより、何を相談したいかと調整可能な時間を明確にする方が予定を確認しやすいためです。",
                        "steps": ("件名に授業名・面談依頼・氏名を書く", "本文で相談内容を一文、候補日時を2〜3個示す",
                                  "送信前に宛名・学籍番号・添付・候補日時を確認する"),
                        "caution": "返信期限を一方的に決めず、急ぎの場合は理由を簡潔に添えてください。"},
    "lateness_email": {"conclusion": "遅刻連絡は、授業名・到着見込み・理由を事実だけで簡潔に伝えます。",
                       "reason": "先生が出欠や入室後の対応を判断するために必要な情報を先に示すと伝わりやすいためです。",
                       "steps": ("件名と宛名を授業に合わせて直す", "到着予定時刻と理由を確認できた事実だけで書く",
                                 "送信前に学籍番号・氏名・誤字を確認する"),
                       "caution": "入室や出席扱いを自己判断せず、授業案内または先生の指示に従ってください。"},
    "university_policy": {"conclusion": "公欠の条件は大学・学部・授業で異なるため、学生便覧や履修要項の正式名称と手続期限を確認してください。",
                          "reason": "対象事由だけでなく、証明書・申請先・申請期限・授業ごとの扱いが定められている場合があります。",
                          "steps": ("大学サイト内で『公欠』『欠席届』『追試』を検索する", "対象事由・必要書類・期限・提出先を抜き出す",
                                    "該当するか不明なら教務窓口と担当教員へ同じ事実を伝えて確認する"),
                          "caution": "公式根拠がない状態で公欠になる、出席扱いになるとは断定しません。"},
    "credit": {"conclusion": "単位認定は、科目の成績だけでなく卒業要件や区分への算入方法まで公式資料で確認します。",
               "reason": "同じ単位数でも、必修・選択・自由科目などの区分によって卒業要件への数え方が異なる場合があるためです。",
               "steps": ("成績表と履修要項を開き、科目名・単位数・区分を照合する", "不足単位を区分別に整理する",
                         "不明な認定や読み替えは、科目情報を添えて教務窓口へ確認する"),
               "caution": "個別の単位認定や卒業可否は、大学の公式確認なしに断定しないでください。"},
    "toeic_plan": {"conclusion": "単語と読解は、単語だけを長時間続けず、毎日両方に触れて間違いを翌日に戻す配分が実行しやすいです。",
                   "reason": "語彙を読解の中で使い直すことで、意味を覚えるだけでなく読む速度と正確さも確認できるためです。",
                   "steps": ("最初の25分で頻出単語を復習する", "次の25分で時間を測って読解問題を解く",
                             "最後の10分で間違えた語と根拠箇所を記録する"),
                   "caution": "現在スコア・目標・試験日・一日の学習時間が分かれば、配分をより具体的に調整できます。"},
    "gpa": {"conclusion": "目標GPAに必要な成績は、現在GPA・取得済み単位・目標GPA・今後の単位数をそろえて計算します。",
            "reason": "今後必要な平均GPは、残り単位数と大学のGP上限によって変わるためです。",
            "steps": ("成績表から現在GPAと取得済み単位を確認する", "目標GPAと今後取得予定の単位数を入力する",
                      "必要平均がGP上限を超える場合は、目標時期や履修計画を見直す"),
            "caution": "GPAの計算式や再履修科目の扱いは大学ごとに異なるため、履修要項も確認してください。"},
    "grade_simulator": {"conclusion": "必要点を出すには、現在の獲得点・合格または目標点・残り評価の配点割合を同じ基準でそろえます。",
                        "reason": "『現在40点』が総合点なのか既実施部分の得点率なのかで計算結果が変わるためです。",
                        "steps": ("現在40点の意味と合格点を確認する", "残り30%が何点満点かを確認する",
                                  "条件を入力して必要点を計算し、上限を超える場合は再試験や評価条件を確認する"),
                        "caution": "合格点・追試・救済措置は授業ごとに異なるため、シラバスや担当教員の案内を優先してください。"},
    "career_schedule": {"conclusion": "面接日程は企業ごとの締切と準備日を一つの表にし、授業時間を先に固定してから配置します。",
                        "reason": "面接日だけでなく、ES提出・企業研究・移動・振り返りの時間も確保すると重複を防ぎやすくなります。",
                        "steps": ("企業名・選考段階・期限・候補日時を一覧にする", "授業・試験・課題の動かせない時間を先に入れる",
                                  "面接の前日準備と終了後の記録時間まで予定に入れる"),
                        "caution": "企業ごとの正式な締切は採用ページや連絡メールで確認し、推測の日付を入れないでください。"},
    "report_outline": {"conclusion": "2000字レポートは、最初に問いへの仮の答えを決め、各段落の根拠を対応させてから書き始めます。",
                       "reason": "字数だけを先に割り振るより、主張と根拠の関係を決める方が本論の重複や脱線を減らせるためです。",
                       "steps": ("課題文から問い・条件・評価基準を抜き出す", "序論で示す仮結論と本論の根拠を2〜3個決める",
                                 "各段落に主張・根拠・説明を置き、最後に問いへの回答を確認する"),
                       "caution": "引用形式と生成AI利用条件は授業の指示を確認し、出典のない事実を追加しないでください。"},
}

GENERIC_GUIDANCE = {"conclusion": "質問の目的と期限を確認し、今日できる最小の行動から進めます。",
                    "reason": "必要な結果を先に決めると、確認先と作業順を選びやすくなります。",
                    "steps": ("必要な結果を一文で書く", "期限と未確認事項を分ける", "公式情報または担当者へ確認して一つ行動する"),
                    "caution": "確認できていない数字や制度は断定せず、事実と推測を分けてください。"}


class FAQSourceStore:
    def __init__(self, path: str | Path = "data/campus_v2/faq/reviewed.jsonl"):
        source_path = Path(path)
        self.rows = {}
        if source_path.exists():
            for line in source_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["id"]] = row

    def documents(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        return [self.rows[item_id] for item_id in CampusAIJudge.source_ids(metadata) if item_id in self.rows]


class CampusAnswerImprover:
    """One-pass deterministic answer improvement. Existing Campus routing is never called or mutated."""

    def __init__(self, judge: CampusAIJudge | None = None, faq_store: FAQSourceStore | None = None):
        self.judge = judge or CampusAIJudge()
        self.faq_store = faq_store or FAQSourceStore()

    @staticmethod
    def _deduplicate(text: str) -> str:
        kept: list[str] = []
        normalised: list[str] = []
        for sentence in _sentences(text):
            value = _normalise(sentence)
            if not value:
                continue
            if any(value == previous or (len(value) >= 18 and value in previous) for previous in normalised):
                continue
            kept.append(sentence)
            normalised.append(value)
        return "\n".join(kept)

    @staticmethod
    def _format_guidance(guide: dict[str, Any], *, compound: bool = False) -> str:
        steps = " ".join(f"{index}. {step}" for index, step in enumerate(guide["steps"], 1))
        lead = "まず優先すること：" if compound else "結論："
        return (f"{lead}{guide['conclusion']}\n"
                f"理由：{guide['reason']}\n"
                f"今やること：{steps}\n"
                f"注意：{guide['caution']}")

    def _rewrite_once(self, question: str, original: str, metadata: dict[str, Any],
                      source_documents: list[dict[str, Any]]) -> str:
        category = str(metadata.get("category") or "general")
        action = str(metadata.get("action") or "")
        route = str(metadata.get("route") or "")
        guide = CATEGORY_GUIDANCE.get(category, GENERIC_GUIDANCE)
        compound = "それと" in question or "両方" in question
        mode = self.judge.response_mode(question, metadata)
        min_chars = self.judge.rubric["length_guidance"][mode]["min_chars"]

        if action == "CLARIFY":
            improved = ("まず、いちばん期限が近い相談を確認させてください。\n"
                        "課題・試験・単位・出席・教授への連絡・アルバイトのうち、最も近いものはどれですか？ "
                        "あわせて期限を一言（例：今日中、今週、来週）教えてください。\n"
                        "安全や体調に関わる場合は、その点を先に書いてください。内容が分かれば、次に行う手順を具体化します。")
        elif source_documents and route == "faq":
            improved = source_documents[0]["answer"].strip()
            if len(improved) < min_chars:
                steps = " ".join(f"{index}. {step}" for index, step in enumerate(guide["steps"], 1))
                improved += (f"\n\n理由：{guide['reason']}\n"
                             f"確認ポイント：{steps}\n"
                             f"補足：{guide['caution']}")
        elif "TOOL" in action and any(card.get("fields") for card in metadata.get("cards") or []):
            fields = [field.get("label", field.get("name", "必要項目")) for card in metadata.get("cards") or []
                      for field in card.get("fields") or []]
            field_text = "・".join(dict.fromkeys(fields))
            improved = (f"{guide['conclusion']}\n"
                        f"入力してほしい項目：{field_text}。\n"
                        f"理由：{guide['reason']}\n"
                        f"入力後の進め方：" + " ".join(f"{index}. {step}" for index, step in enumerate(guide["steps"], 1)) +
                        f"\n確認：{guide['caution']}")
        elif route in ("safe", "safety") or "一致度の高いFAQ" in original:
            improved = self._format_guidance(guide, compound=compound)
        else:
            improved = original.strip()
            addition = self._format_guidance(guide, compound=compound)
            if len(improved) < min_chars or compound:
                improved = f"{improved}\n\n{addition}"

        if compound and "もう一つ" not in improved and "それと" not in improved:
            second = question.split("それと", 1)[1].strip(" 、。") if "それと" in question else "もう一方の相談"
            improved += (f"\n\n次に、もう一方の相談（{second}）も別項目として扱います。"
                         "期限が近い方を先にし、それぞれの確認先と今日の一手を分けてください。")

        if len(improved) < min_chars and mode == "normal":
            supplements = [
                ("進めた後は、確認できた事実・まだ不明な点・次の期限を3行で残してください。"
                 "途中で条件が違うと分かった場合は、推測で埋めず、公式案内または担当者の回答に合わせて手順を更新します。"),
                ("確認したページ名・担当者・確認日も一緒にメモしておくと、後から条件が変わったときに見直しやすくなります。"
                 "今日の行動が終わったら、次に確認する一項目だけを決めてください。"),
            ]
            for supplement in supplements:
                if len(improved) >= min_chars:
                    break
                improved += f"\n\n{supplement}"
        return self._deduplicate(improved)

    def self_critique(self, judge_result: dict[str, Any]) -> dict[str, Any]:
        checks = judge_result["checks"]
        issues = judge_result["issues"]
        result = {
            "answered_question": judge_result["scores_0_to_5"]["relevance"] >= 3.8 and "WRONG_PRIORITY" not in issues,
            "not_too_short": not checks["too_short"],
            "specific_enough": judge_result["scores_0_to_5"]["specificity"] >= 4.2,
            "next_action_present": checks["action_present"],
            "no_unsupported_assertion": not judge_result["unsupported_claims"],
            "source_consistent": not judge_result["unsupported_claims"],
            "issues": issues,
        }
        result["needs_revision"] = not all(value for key, value in result.items()
                                                  if key not in ("issues", "needs_revision"))
        return result

    def improve(self, question: str, original: str, metadata: dict[str, Any] | None = None,
                *, force: bool = False) -> dict[str, Any]:
        metadata = metadata or {}
        documents = self.faq_store.documents(metadata)
        source_texts = [document["answer"] for document in documents]
        before = self.judge.evaluate(question, original, metadata, source_texts)
        critique = self.self_critique(before)
        should_rewrite = force or critique["needs_revision"] or before["overall_score"] < 90
        improved = self._rewrite_once(question, original, metadata, documents) if should_rewrite else original
        after = self.judge.evaluate(question, improved, metadata, source_texts)
        return {"original": original, "critique": critique, "improved_answer": improved,
                "rewrite_performed": should_rewrite, "rewrite_count": 1 if should_rewrite else 0,
                "before_judge": before, "after_judge": after,
                "source_ids": [document["id"] for document in documents],
                "external_ai_api": "OFF", "production_eligible_automatically": False}
