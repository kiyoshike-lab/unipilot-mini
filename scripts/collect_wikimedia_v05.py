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
TOPICS = ["大学", "単位 (大学)", "GPA", "学習", "記憶", "試験", "統計学", "確率論", "微分積分学", "線型代数学",
          "物理学", "経済学", "経営学", "情報科学", "計算機科学", "人工知能", "機械学習", "プログラミング",
          "著作権", "情報リテラシー", "就職活動", "履歴書", "面接", "英語", "プレゼンテーション"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect selected Japanese Wikipedia introductions with provenance.")
    parser.add_argument("--output", default="data/v05/knowledge/wikipedia.jsonl")
    args = parser.parse_args()
    params = {"action": "query", "format": "json", "formatversion": "2", "prop": "extracts|info|revisions",
              "exintro": "1", "explaintext": "1", "redirects": "1", "inprop": "url", "rvprop": "ids|timestamp",
              "titles": "|".join(TOPICS)}
    request = Request(f"{API}?{urlencode(params)}", headers={"User-Agent": "UniPilot-Mini-dataset-builder/0.5 (educational local model)"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    retrieved = datetime.now(timezone.utc).isoformat()
    rows = []
    excluded = []
    for page in payload.get("query", {}).get("pages", []):
        text = re.sub(r"\n{3,}", "\n\n", page.get("extract", "")).strip()
        if page.get("missing") or len(text) < 80 or "曖昧さ回避" in text:
            excluded.append({"title": page.get("title"), "reason": "missing, disambiguation, or too short"})
            continue
        url = page.get("fullurl", f"https://ja.wikipedia.org/wiki/{page.get('title', '')}")
        revision = (page.get("revisions") or [{}])[0]
        rows.append({"id": f"wikipedia-ja-{page['pageid']}", "kind": "knowledge", "category": "university_general_knowledge",
                     "title": page["title"], "text": text, "source": "Wikipedia contributors",
                     "source_url": url, "attribution_url": url + "?action=history", "retrieved_at": retrieved,
                     "license": LICENSE, "license_url": LICENSE_URL, "page_id": page["pageid"],
                     "revision_id": revision.get("revid"), "revision_timestamp": revision.get("timestamp"),
                     "modified": False, "training_stage": "knowledge_only_not_v05_chat_finetune"})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {"source": "Japanese Wikipedia via official MediaWiki Action API", "api": API,
              "retrieved_at": retrieved, "license": LICENSE, "license_url": LICENSE_URL,
              "requested": len(TOPICS), "accepted": len(rows), "excluded": excluded,
              "output": str(output).replace("\\", "/"), "usage": "knowledge data kept separate from instruction/chat data"}
    Path("evaluation/wikimedia-collection-v05.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
