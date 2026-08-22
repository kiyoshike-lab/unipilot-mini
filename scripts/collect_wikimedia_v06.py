"""Collect redistributable public knowledge; never treat it as current university rules."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://ja.wikipedia.org/w/api.php"
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
TOPICS = (
    "大学", "学位", "単位 (大学)", "GPA", "シラバス", "卒業論文", "ゼミナール", "研究室",
    "学習", "記憶", "試験", "統計学", "確率論", "微分積分学", "線型代数学", "情報科学",
    "計算機科学", "プログラミング", "人工知能", "情報リテラシー", "著作権", "引用", "盗用",
    "就職活動", "履歴書", "面接", "インターンシップ", "資格", "英語", "留学", "奨学金",
    "プレゼンテーション", "時間管理", "図書館", "電子図書館", "レポート", "電子メール",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/v06/knowledge/wikipedia.jsonl")
    args = parser.parse_args()
    params = {"action": "query", "format": "json", "formatversion": "2", "prop": "extracts|info|revisions",
              "exintro": "1", "explaintext": "1", "redirects": "1", "inprop": "url", "rvprop": "ids|timestamp",
              "titles": "|".join(TOPICS)}
    request = Request(f"{API}?{urlencode(params)}", headers={
        "User-Agent": "UniPilot-Mini-dataset-builder/0.6 (local educational model; no external AI)"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    retrieved = datetime.now(timezone.utc).isoformat()
    rows, excluded = [], []
    for page in payload.get("query", {}).get("pages", []):
        text = re.sub(r"\n{3,}", "\n\n", page.get("extract", "")).strip()
        if page.get("missing") or len(text) < 80 or "曖昧さ回避" in text:
            excluded.append({"title": page.get("title"), "reason": "missing, disambiguation, or too short"})
            continue
        url = page.get("fullurl", "")
        revision = (page.get("revisions") or [{}])[0]
        rows.append({
            "id": f"wikipedia-ja-{page['pageid']}", "kind": "knowledge", "category": "public_general_knowledge",
            "title": page["title"], "text": text, "source_type": "public_encyclopedia", "quality_score": 3.5,
            "source": "Wikipedia contributors", "source_url": url, "attribution_url": url + "?action=history",
            "retrieved_at": retrieved, "license": LICENSE, "license_url": LICENSE_URL, "page_id": page["pageid"],
            "revision_id": revision.get("revid"), "revision_timestamp": revision.get("timestamp"), "modified": False,
            "usage_guard": "Not authoritative for current, university-specific rules; verify those with official current sources.",
            "training_stage": "knowledge_only_not_instruction_finetune",
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "source": "Japanese Wikipedia via official MediaWiki Action API", "api": API, "retrieved_at": retrieved,
        "license": LICENSE, "license_url": LICENSE_URL, "requested": len(TOPICS), "accepted": len(rows),
        "excluded": excluded, "output": output.as_posix(),
        "usage": "Separate knowledge corpus. Current university rules and deadlines always require official verification.",
    }
    Path("evaluation/wikimedia-collection-v06.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
