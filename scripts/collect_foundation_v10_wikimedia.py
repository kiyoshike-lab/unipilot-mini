"""Collect licensed Japanese text from official Wikimedia Action APIs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SOURCES = {
    "wikipedia": {
        "api": "https://ja.wikipedia.org/w/api.php",
        "site": "Japanese Wikipedia",
        "source": "日本語版ウィキペディアの執筆者",
        "publisher": "Wikimedia Foundation",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    },
    "wikibooks": {
        "api": "https://ja.wikibooks.org/w/api.php",
        "site": "Japanese Wikibooks",
        "source": "日本語版ウィキブックスの執筆者",
        "publisher": "Wikimedia Foundation",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    },
}
USER_AGENT = (
    "UniPilot-Foundation/1.0 (offline Japanese model corpus research; "
    "https://github.com/kiyoshike-lab/unipilot-mini)"
)
STOP_SECTIONS = {
    "脚注", "注釈", "出典", "参考文献", "関連項目", "外部リンク", "参考資料",
    "リンク", "ギャラリー", "画像", "一覧", "典拠管理",
}
TITLE_EXCLUSIONS = (
    "一覧", "曖昧さ回避", "索引", "年表", "の登場人物", "のエピソード一覧",
)


def request_json(api: str, params: dict, retries: int = 5) -> dict:
    query = urlencode({"format": "json", "formatversion": "2", **params})
    request = Request(f"{api}?{query}", headers={"User-Agent": USER_AGENT})
    error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # network retries are recorded by the caller
            error = exc
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"Wikimedia request failed after {retries} attempts") from error


def clean_extract(text: str) -> tuple[str, dict]:
    original = text
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\[(?:編集|注釈|要出典|誰によって)\]", "", text)
    lines: list[str] = []
    stopped_sections = 0
    noise_lines = 0
    stop_depth: int | None = None
    for raw in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        heading = re.fullmatch(r"={2,6}\s*(.*?)\s*={2,6}", line)
        if heading:
            depth = len(line) - len(line.lstrip("="))
            title = heading.group(1).strip()
            if title in STOP_SECTIONS:
                stop_depth = depth
                stopped_sections += 1
                continue
            if stop_depth is not None and depth <= stop_depth:
                stop_depth = None
            if stop_depth is None:
                lines.append(title + "。")
            continue
        if stop_depth is not None:
            continue
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if (
            line.startswith(("Category:", "カテゴリ:", "{{", "|", "* 外部"))
            or re.fullmatch(r"[-–—_=*・#\s]+", line)
            or "この節の加筆が望まれています" in line
        ):
            noise_lines += 1
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned = re.sub(r"([。！？])\1{2,}", r"\1", cleaned)
    return cleaned, {
        "original_characters": len(original),
        "cleaned_characters": len(cleaned),
        "stopped_sections": stopped_sections,
        "noise_lines": noise_lines,
        "modified": cleaned != original.strip(),
    }


def quality_reason(title: str, text: str) -> str | None:
    if any(value in title for value in TITLE_EXCLUSIONS):
        return "excluded_title_type"
    if len(text) < 500:
        return "too_short"
    if len(text) > 80_000:
        return "extreme_length"
    japanese = len(re.findall(r"[ぁ-んァ-ヶ一-龥々]", text))
    if japanese / max(1, len(text)) < 0.35:
        return "low_japanese_ratio"
    if "曖昧さ回避" in text[:500] or "この項目では" in text[:200] and "区別" in text[:500]:
        return "disambiguation"
    nonblank = [line.strip() for line in text.splitlines() if line.strip()]
    list_lines = sum(line.startswith(("*", "#", ";", ":")) for line in nonblank)
    if nonblank and list_lines / len(nonblank) > 0.40:
        return "list_heavy"
    sentence_marks = len(re.findall(r"[。！？]", text))
    if sentence_marks < max(2, len(text) // 800):
        return "low_sentence_density"
    if sum(text.count(marker) for marker in ("{{", "}}", "[[", "]]", "{|", "|}")) > 4:
        return "residual_markup"
    return None


def existing_ids(path: Path) -> tuple[set[int], set[str], int, int]:
    page_ids: set[int] = set()
    fingerprints: set[str] = set()
    documents = characters = 0
    if not path.exists():
        return page_ids, fingerprints, documents, characters
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            page_ids.add(int(row["page_id"]))
            fingerprints.add(row["content_sha256"])
            documents += 1
            characters += len(row["text"])
    return page_ids, fingerprints, documents, characters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=tuple(SOURCES), required=True)
    parser.add_argument("--target-characters", type=int, required=True)
    parser.add_argument("--max-documents", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument("--output")
    parser.add_argument("--report")
    args = parser.parse_args()
    source = SOURCES[args.source]
    output = Path(args.output or f"data/foundation_v10/raw/{args.source}-ja.jsonl.gz")
    report_path = Path(args.report or f"evaluation/foundation-v10-{args.source}-collection.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ids, fingerprints, accepted, characters = existing_ids(output)
    excluded: dict[str, int] = {}
    duplicate_pages = duplicate_texts = requests = 0
    retrieved_at = datetime.now(timezone.utc).isoformat()

    with gzip.open(output, "at", encoding="utf-8", newline="\n") as file:
        while characters < args.target_characters and accepted < args.max_documents:
            random_payload = request_json(source["api"], {
                "action": "query", "list": "random", "rnnamespace": 0,
                "rnfilterredir": "nonredirects", "rnminsize": 1000,
                "rnlimit": args.batch_size,
            })
            requests += 1
            page_ids = [int(row["id"]) for row in random_payload.get("query", {}).get("random", [])]
            new_ids = [page_id for page_id in page_ids if page_id not in ids]
            duplicate_pages += len(page_ids) - len(new_ids)
            if not new_ids:
                continue
            def fetch_page(page_id: int) -> dict:
                payload = request_json(source["api"], {
                    "action": "query", "pageids": str(page_id),
                    "prop": "extracts|info|revisions|categories", "explaintext": 1,
                    "inprop": "url", "rvprop": "ids|timestamp", "cllimit": "max",
                })
                return (payload.get("query", {}).get("pages") or [{}])[0]

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                pages = list(executor.map(fetch_page, new_ids))
            requests += len(new_ids)
            for page in pages:
                page_id = int(page.get("pageid", 0))
                if not page_id or page_id in ids or page.get("missing"):
                    duplicate_pages += 1
                    continue
                ids.add(page_id)
                cleaned, cleaning = clean_extract(page.get("extract", ""))
                reason = quality_reason(page.get("title", ""), cleaned)
                if reason:
                    excluded[reason] = excluded.get(reason, 0) + 1
                    continue
                fingerprint = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
                if fingerprint in fingerprints:
                    duplicate_texts += 1
                    continue
                fingerprints.add(fingerprint)
                revision = (page.get("revisions") or [{}])[0]
                categories = [row["title"].removeprefix("Category:")
                              for row in page.get("categories", [])]
                url = page.get("fullurl", "")
                row = {
                    "id": f"{args.source}-ja-{page_id}", "kind": "text",
                    "title": page["title"], "text": cleaned, "categories": categories,
                    "source_type": f"wikimedia_{args.source}", "source": source["source"],
                    "publisher": source["publisher"], "source_url": url,
                    "attribution_url": url + "?action=history", "retrieved_at": retrieved_at,
                    "license": source["license"], "license_url": source["license_url"],
                    "page_id": page_id, "revision_id": revision.get("revid"),
                    "revision_timestamp": revision.get("timestamp"),
                    "content_sha256": fingerprint, "cleaning": cleaning,
                    "training_role": "general_japanese_pretraining_candidate",
                    "external_ai_used": False,
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                accepted += 1
                characters += len(cleaned)
            file.flush()
            print(json.dumps({"source": args.source, "documents": accepted,
                              "characters": characters}, ensure_ascii=False), flush=True)
            time.sleep(args.pause)

    report = {
        "schema_version": "foundation-v10-wikimedia-collection-v1",
        "source": source["site"], "api": source["api"], "method": "official Action API",
        "retrieved_at": retrieved_at, "license": source["license"],
        "license_url": source["license_url"], "output": output.as_posix(),
        "accepted_documents": accepted, "accepted_characters": characters,
        "target_characters": args.target_characters, "requests": requests,
        "excluded": excluded, "duplicate_pages": duplicate_pages,
        "duplicate_texts": duplicate_texts, "cleaning": [
            "plain-text extracts", "reference/navigation sections removed",
            "HTML/template/noise lines removed", "short/disambiguation/list pages removed",
        ],
        "attribution_preserved_per_article": True, "external_ai_api": "OFF",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
