from __future__ import annotations


CAMPUS_CATEGORIES = (
    "exam", "assignment", "credit", "gpa", "grade_simulator", "attendance", "lateness",
    "professor_email", "absence_email", "lateness_email", "late_submission_email", "registration",
    "schedule", "study_plan", "assignment_priority", "deadline_organizer", "report_outline",
    "citation_check", "presentation_outline", "career_schedule", "es_outline", "toeic_plan",
    "internship", "scholarship", "tuition", "part_time_job", "campus_life", "relationship",
    "programming", "ai_usage", "math", "statistics", "university_policy", "faq_search", "general",
)


CAMPUS_LABELS = {
    "exam": "試験", "assignment": "課題", "credit": "単位", "gpa": "GPA計算",
    "grade_simulator": "必要点数", "attendance": "欠席", "lateness": "遅刻",
    "professor_email": "教授メール", "absence_email": "欠席メール", "lateness_email": "遅刻メール",
    "late_submission_email": "課題提出遅延メール", "registration": "履修相談", "schedule": "予定整理",
    "study_plan": "試験勉強計画", "assignment_priority": "課題優先順位",
    "deadline_organizer": "締切整理", "report_outline": "レポート構成", "citation_check": "引用確認",
    "presentation_outline": "プレゼン構成", "career_schedule": "就活スケジュール",
    "es_outline": "ES構成", "toeic_plan": "TOEIC計画", "internship": "インターン",
    "scholarship": "奨学金", "tuition": "学費", "part_time_job": "アルバイト",
    "campus_life": "大学生活", "relationship": "人間関係", "programming": "プログラミング",
    "ai_usage": "生成AI", "math": "数学", "statistics": "統計",
    "university_policy": "大学固有制度", "faq_search": "FAQ検索", "general": "一般相談",
}


CAMPUS_KEYWORDS = {
    "exam": ("試験", "テスト", "追試", "再試験", "持ち込み"),
    "assignment": ("課題", "提出", "宿題"),
    "credit": ("単位", "落単", "卒業要件", "進級"),
    "gpa": ("gpa", "成績平均", "gpを", "gpの"),
    "grade_simulator": ("必要点", "何点必要", "合格点", "残り評価"),
    "attendance": ("欠席", "公欠", "出席率", "診断書"),
    "lateness": ("遅刻", "遅れそう", "間に合わない", "遅延証明"),
    "professor_email": ("教授にメール", "先生にメール", "メール文", "件名", "面談依頼"),
    "absence_email": ("欠席メール", "欠席連絡", "休むメール"),
    "lateness_email": ("遅刻メール", "遅刻連絡", "遅れるメール"),
    "late_submission_email": ("提出遅延メール", "課題遅延", "締切に遅れ", "提出が遅れ"),
    "registration": ("履修", "時間割", "必修", "登録期間", "抽選科目"),
    "schedule": ("予定", "スケジュール", "日程", "空き時間"),
    "study_plan": ("勉強計画", "学習計画", "テストまで", "試験まで", "何を勉強"),
    "assignment_priority": ("課題の優先", "どの課題", "どれから", "課題が複数"),
    "deadline_organizer": ("締切整理", "期限整理", "締切一覧", "締切を忘れ"),
    "report_outline": ("レポート構成", "章立て", "アウトライン", "序論"),
    "citation_check": ("引用", "出典", "参考文献", "コピペ", "著作権"),
    "presentation_outline": ("プレゼン構成", "スライド構成", "発表構成", "プレゼン"),
    "career_schedule": ("就活スケジュール", "就活計画", "選考日程", "就活と授業"),
    "es_outline": ("es構成", "エントリーシート", "志望動機", "自己pr"),
    "toeic_plan": ("toeic", "英語試験計画"),
    "internship": ("インターン", "職業体験"),
    "scholarship": ("奨学金", "給付型", "貸与型", "jasso"),
    "tuition": ("学費", "授業料", "分納", "延納", "減免"),
    "part_time_job": ("アルバイト", "バイト", "シフト", "賃金"),
    "campus_life": ("大学生活", "サークル", "一人暮らし", "学生生活", "研究室", "ゼミ"),
    "relationship": ("人間関係", "友達", "孤立", "ハラスメント", "同級生"),
    "programming": ("プログラミング", "コード", "エラー", "デバッグ", "python"),
    "ai_usage": ("生成ai", "aiを", "ai利用", "人工知能"),
    "math": ("数学", "微分", "積分", "線形代数", "数式"),
    "statistics": ("統計", "確率", "標準偏差", "分散", "回帰"),
    "university_policy": ("うちの大学", "この大学", "大学では", "学則", "大学のルール"),
    "faq_search": ("よくある質問", "faq", "どこを確認", "確認先"),
    "general": ("相談", "どうしよう", "困った"),
}


TOOL_INTENTS = {
    "gpa", "grade_simulator", "professor_email", "absence_email", "lateness_email",
    "late_submission_email", "registration", "study_plan", "assignment_priority",
    "deadline_organizer", "report_outline", "citation_check", "presentation_outline",
    "career_schedule", "es_outline", "toeic_plan",
}


STANDARD_TO_CAMPUS = {
    "exam": "exam", "assignment": "assignment", "credit": "credit", "gpa": "gpa",
    "registration": "registration", "attendance": "attendance", "lateness": "lateness",
    "professor_email": "professor_email", "report": "report_outline", "citation": "citation_check",
    "presentation": "presentation_outline", "seminar": "campus_life", "laboratory": "campus_life",
    "thesis": "report_outline", "career": "career_schedule", "internship": "internship",
    "qualification": "career_schedule", "toeic": "toeic_plan", "study_abroad": "campus_life",
    "scholarship": "scholarship", "tuition": "tuition", "part_time": "part_time_job",
    "campus_life": "campus_life", "relationships": "relationship", "time_management": "schedule",
    "study": "study_plan", "pc": "campus_life", "programming": "programming",
    "ai_usage": "ai_usage", "information_literacy": "citation_check", "statistics": "statistics",
    "math": "math", "general_education": "general",
}


UNIVERSITY_SPECIFIC_PHRASES = (
    "追試ある", "欠席何回", "何回休", "gpa何点", "進級でき", "卒業でき", "公欠になる",
    "単位になる", "再提出でき", "履修変更でき", "学費はいくら", "返金され",
)
