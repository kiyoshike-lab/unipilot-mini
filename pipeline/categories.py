from __future__ import annotations


CATEGORIES = (
    "exam", "assignment", "credit", "gpa", "attendance", "lateness", "registration",
    "professor_email", "report", "citation", "presentation", "study", "career", "internship",
    "scholarship", "campus_life", "programming", "ai_usage", "math", "statistics", "general",
)

CATEGORY_ALIASES = {
    "study": "study", "exam": "exam", "lateness": "lateness", "attendance": "attendance",
    "assignment": "assignment", "report": "report", "citation": "citation", "email": "professor_email",
    "professor_email": "professor_email", "seminar": "campus_life", "laboratory": "campus_life",
    "thesis": "report", "presentation": "presentation", "group_work": "presentation",
    "relationships": "campus_life", "club": "campus_life", "part_time": "campus_life",
    "internship": "internship", "career": "career", "es": "career", "interview": "career",
    "self_pr": "career", "qualification": "career", "english": "career", "study_abroad": "career",
    "scholarship": "scholarship", "tuition": "scholarship", "living": "campus_life",
    "time_management": "study", "library": "campus_life", "pc": "campus_life",
    "programming": "programming", "ai": "ai_usage", "ai_usage": "ai_usage",
    "literacy": "citation", "copyright": "citation", "statistics": "statistics", "math": "math",
    "registration": "registration", "credit": "credit", "gpa": "gpa", "schedule": "general",
    "general": "general", "unknown": "general", "correction": "general",
}

CATEGORY_KEYWORDS = {
    "exam": ("試験", "テスト", "追試", "再試験", "試験範囲"),
    "assignment": ("課題", "提出", "締切", "提出期限"),
    "credit": ("単位", "卒業要件", "進級", "落単", "必修単位"),
    "gpa": ("gpa", "成績平均", "gp"),
    "attendance": ("出席", "欠席", "公欠", "出席率"),
    "lateness": ("遅刻", "間に合わ", "遅れそう"),
    "registration": ("履修登録", "履修変更", "履修科目", "必修科目", "時間割", "履修"),
    "professor_email": ("メール", "文面", "件名", "教授に", "先生に", "連絡文"),
    "report": ("レポート", "卒論", "論文", "研究テーマ", "参考文献リスト"),
    "citation": ("引用", "出典", "盗用", "コピペ", "著作権", "参考文献", "情報源"),
    "presentation": ("プレゼン", "発表", "スライド", "グループワーク", "共同作業"),
    "study": ("勉強", "学習", "復習", "覚え", "集中", "時間管理", "計画"),
    "career": ("就活", "就職", "キャリア", "面接", "es", "履歴書", "自己pr", "資格", "toeic", "留学"),
    "internship": ("インターン", "職業体験"),
    "scholarship": ("奨学金", "学費", "授業料", "給付", "貸与", "返還", "減免"),
    "campus_life": ("友達", "サークル", "大学生活", "一人暮らし", "アルバイト", "図書館", "研究室", "ゼミ", "大学用pc"),
    "programming": ("プログラミング", "コード", "エラー", "デバッグ", "python"),
    "ai_usage": ("生成ai", "aiを", "aiの", "人工知能", "チャットai"),
    "math": ("数学", "微分", "積分", "線形代数", "数式", "定理"),
    "statistics": ("統計", "確率", "分散", "標準偏差", "回帰分析"),
    "general": ("大学", "学生", "相談"),
}

SAFE_FALLBACKS = {
    "exam": "試験の案内とシラバスを確認し、不明な点は担当教員または教務窓口へ早めに相談してください。",
    "assignment": "提出条件と締切を確認し、間に合わない可能性があれば締切前に担当教員へ相談してください。",
    "credit": "取得済み単位と所属大学の進級・卒業要件を確認し、不明点は教務窓口へ相談してください。",
    "gpa": "GPAは成績を数値化した平均指標です。計算方法や用途は大学によって異なるため、所属大学の規程も確認してください。",
    "attendance": "出席の扱いは授業ごとに異なるため、シラバスと出席記録を確認し、必要なら担当教員へ相談してください。",
    "lateness": "安全を優先して移動し、試験や授業の案内を確認して、指定された方法で担当者へ連絡してください。",
    "registration": "履修登録の期間と変更条件は大学によって異なります。現在年度の履修案内と学生ポータルを確認してください。",
    "professor_email": "件名に授業名と用件を書き、宛名、所属・氏名、事実、確認したいこと、結びの順で簡潔に伝えてください。",
    "report": "問いと結論を先に決め、根拠となる資料、本文、引用と出典の確認という順で進めてください。",
    "citation": "引用部分を自分の文章と区別し、授業で指定された形式で出典を示してください。利用条件も確認しましょう。",
    "presentation": "結論を先に示し、一枚一メッセージに絞って、文字量と図表の出典を確認してください。",
    "study": "目的と期限を確認し、今日行う作業を小さく分け、問題演習と復習を組み合わせて進めてください。",
    "career": "目的と期限を整理し、企業の公式情報と大学のキャリア窓口を使って次の行動を確認してください。",
    "internship": "目的、業務内容、期間、報酬、保険、個人情報の扱いを公式情報で確認して比較してください。",
    "scholarship": "給付か貸与か、条件、金額、返還の有無、期限を最新の公式募集要項で確認してください。",
    "campus_life": "状況を一つずつ整理し、大学の公式案内や学生支援窓口など適切な相談先を確認してください。",
    "programming": "エラー全文を確認し、再現手順を小さくして、入力・期待結果・試したことを整理してください。",
    "ai_usage": "授業と大学の利用ルールを確認し、AIの出力を一次資料で検証して、自分で理解した内容だけを使ってください。",
    "math": "定義を確認し、簡単な具体例を手で確かめてから標準問題へ進んでください。",
    "statistics": "指標の意味と前提条件を確認し、具体例とグラフで理解してから計算やソフトの結果を確かめてください。",
    "general": "大学によって条件が異なる場合があります。現在年度の公式案内や担当窓口で最新情報を確認してください。",
}

LENGTH_POLICY = {
    "gpa": "short", "credit": "normal", "professor_email": "detailed", "exam": "normal",
    "assignment": "normal", "report": "normal", "citation": "normal", "study": "detailed",
}


def normalize_category(category: str) -> str:
    return CATEGORY_ALIASES.get(category, "general")
