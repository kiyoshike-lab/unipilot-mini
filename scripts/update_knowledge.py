#!/usr/bin/env python3
"""Refresh the opt-in Campus v2.2 local knowledge corpus.

Only sources explicitly enabled in data/knowledge/source_registry.json are
downloaded.  The generated corpus is deterministic at the document level and
keeps the attribution/freshness metadata needed by the local retriever.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
TOPICS_PATH = ROOT / "data" / "knowledge" / "wiki_topics.json"
REGISTRY_PATH = ROOT / "data" / "knowledge" / "source_registry.json"
OUT_ROOT = ROOT / "data" / "campus_v22"
KNOWLEDGE_ROOT = OUT_ROOT / "knowledge"
RAG_ROOT = OUT_ROOT / "rag_index"
REPORT_PATH = ROOT / "evaluation" / "campus-v22-knowledge-report.json"
WIKI_API = "https://ja.wikipedia.org/w/api.php"
USER_AGENT = "UniPilot-Campus-v2.2/1.0 (local educational RAG; contact: repository maintainers)"
REQUIRED_FIELDS = (
    "id",
    "title",
    "text",
    "category",
    "sub_category",
    "source",
    "source_url",
    "retrieved_at",
    "license",
    "publisher",
    "university_name",
    "university_specific",
    "last_verified_at",
)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(value or ""))
    return re.sub(r"\s+", " ", value).strip()


def content_hash(title: str, text: str) -> str:
    canonical = normalize(f"{title}\n{text}").lower()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _verification_age(value: str | None) -> int:
    if not value:
        return 10_000
    try:
        verified = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return 10_000
    return max(0, (utc_now().date() - verified).days)


def request_bytes(url: str, *, retries: int = 5, timeout: int = 30) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.5"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code != 429:
                break
            if attempt + 1 < retries:
                retry_after = None
                if isinstance(exc, HTTPError) and exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 5.0 * (attempt + 1)
                time.sleep(wait)
    raise RuntimeError(f"request failed: {url}: {last_error}")


def request_json(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url).decode("utf-8"))


def api_url(params: dict[str, Any]) -> str:
    common = {"format": "json", "formatversion": 2, "utf8": 1, "maxlag": 5}
    return f"{WIKI_API}?{urlencode({**common, **params})}"


def wiki_candidates(topics: list[dict[str, str]], limit: int, delay: float) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    per_topic = max(10, min(20, (limit // max(1, len(topics))) + 6))
    for topic_number, topic in enumerate(topics, 1):
        payload = request_json(
            api_url(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": topic["query"],
                    "srnamespace": 0,
                    "srlimit": per_topic,
                }
            )
        )
        for item in payload.get("query", {}).get("search", []):
            page_id = int(item["pageid"])
            candidates.setdefault(
                page_id,
                {
                    "pageid": page_id,
                    "title": item["title"],
                    "category": topic["category"],
                    "sub_category": topic["sub_category"],
                },
            )
        if topic_number % 10 == 0 or topic_number == len(topics):
            print(f"Wikipedia search {topic_number}/{len(topics)}: {len(candidates)} candidates", flush=True)
        time.sleep(delay)
    return list(candidates.values())


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def collect_wikipedia(limit: int, delay: float, stamp: datetime) -> tuple[list[dict[str, Any]], list[str]]:
    topics = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))["topics"]
    candidates = wiki_candidates(topics, limit, delay)
    by_title = {candidate["title"]: candidate for candidate in candidates}
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    candidate_batches = list(chunks(candidates, 20))
    for batch_number, batch in enumerate(candidate_batches, 1):
        if len(records) >= limit:
            break
        titles = "|".join(item["title"] for item in batch)
        try:
            payload = request_json(
                api_url(
                    {
                        "action": "query",
                        "prop": "extracts|info|pageprops|revisions",
                        "titles": titles,
                        "redirects": 1,
                        "exintro": 1,
                        "explaintext": 1,
                        "inprop": "url",
                        "rvprop": "ids|timestamp",
                    }
                )
            )
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        redirects = {
            redirect.get("to", ""): redirect.get("from", "")
            for redirect in payload.get("query", {}).get("redirects", [])
        }
        for page in payload.get("query", {}).get("pages", []):
            if len(records) >= limit:
                break
            title = normalize(page.get("title", ""))
            original = redirects.get(title, title)
            meta = by_title.get(original) or by_title.get(title)
            text = normalize(page.get("extract", ""))
            if not meta or page.get("missing") or "disambiguation" in page.get("pageprops", {}):
                continue
            if len(text) < 160 or re.search(r"(?:曖昧さ回避|一覧)$", title):
                continue
            text = text[:2600].rsplit("。", 1)[0] + "。" if len(text) > 2600 and "。" in text[:2600] else text[:2600]
            revision = (page.get("revisions") or [{}])[0]
            revision_id = revision.get("revid")
            page_id = int(page.get("pageid") or meta["pageid"])
            page_url = page.get("fullurl") or f"https://ja.wikipedia.org/wiki/{title}"
            record = {
                "id": f"wikipedia-ja-{page_id}",
                "title": title,
                "text": text,
                "category": meta["category"],
                "sub_category": meta["sub_category"],
                "source": "日本語版ウィキペディアの執筆者",
                "source_url": page_url,
                "retrieved_at": stamp.isoformat(),
                "license": "CC BY-SA 4.0",
                "publisher": "Wikimedia Foundation",
                "university_name": None,
                "university_specific": False,
                "last_verified_at": stamp.date().isoformat(),
                "source_type": "wikipedia",
                "source_priority": 30,
                "confidence": "medium-high",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution_url": f"{page_url}?action=history",
                "revision_id": revision_id,
                "revision_timestamp": revision.get("timestamp"),
            }
            record["content_sha256"] = content_hash(title, text)
            records.append(record)
        time.sleep(delay)
        if batch_number % 3 == 0 or len(records) >= limit:
            print(f"Wikipedia pages {batch_number}/{len(candidate_batches)}: {len(records)} documents", flush=True)
    return records, failures


class VisibleTextParser(HTMLParser):
    ALLOWED = {"h1", "h2", "h3", "p", "li", "td"}
    SKIP = {"script", "style", "svg", "nav", "header", "footer", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.tag: str | None = None
        self.buffer: list[str] = []
        self.blocks: list[tuple[str, str]] = []
        self.title = ""
        self.in_title = False
        self.title_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP:
            self.depth += 1
        if self.depth:
            return
        if tag == "title":
            self.in_title = True
        if tag in self.ALLOWED:
            self.tag = tag
            self.buffer = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP and self.depth:
            self.depth -= 1
            return
        if self.depth:
            return
        if tag == "title":
            self.in_title = False
            self.title = normalize("".join(self.title_buffer))
        if self.tag == tag:
            value = normalize("".join(self.buffer))
            if value:
                self.blocks.append((tag, value))
            self.tag = None
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.depth:
            return
        if self.in_title:
            self.title_buffer.append(data)
        if self.tag:
            self.buffer.append(data)


NOISE = re.compile(r"^(?:本文へ|メニュー|検索|サイトマップ|English|ページトップ|トップページ|このページを|印刷用|SNS|関連リンク)$", re.I)


def page_chunks(blocks: list[tuple[str, str]], target: int = 700) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    heading = ""
    parts: list[str] = []
    for tag, value in blocks:
        if NOISE.search(value) or len(value) < 8:
            continue
        if tag.startswith("h"):
            if parts:
                output.append((heading, normalize(" ".join(parts))))
                parts = []
            heading = value
            continue
        if sum(len(part) for part in parts) + len(value) > target and parts:
            output.append((heading, normalize(" ".join(parts))))
            parts = []
        parts.append(value)
    if parts:
        output.append((heading, normalize(" ".join(parts))))
    return [
        (title, text) for title, text in output
        if len(text) >= 80 and not (text.count("PDF:") >= 3 and len(text) < 500)
    ]


def collect_registered(stamp: datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["sources"]
    government: list[dict[str, Any]] = []
    universities: list[dict[str, Any]] = []
    failures: list[str] = []
    unsupported: list[str] = []
    for source in registry:
        if not source.get("enabled"):
            unsupported.append(source["id"])
            continue
        print(f"Official source: {source['id']}", flush=True)
        if not source.get("license") or not source.get("terms_url") or not source.get("url"):
            unsupported.append(source["id"])
            continue
        try:
            raw = request_bytes(source["url"])
            parser = VisibleTextParser()
            parser.feed(raw.decode("utf-8", errors="replace"))
            extracted = page_chunks(parser.blocks)[: int(source.get("max_chunks", 20))]
            if not extracted:
                failures.append(f"{source['id']}: no reusable text extracted")
                continue
        except (RuntimeError, UnicodeError) as exc:
            failures.append(f"{source['id']}: {exc}")
            continue
        destination = universities if source["source_type"] == "official_university" else government
        for index, (heading, text) in enumerate(extracted, 1):
            title = normalize(f"{parser.title or source['publisher']} — {heading}" if heading else parser.title or source["publisher"])
            record = {
                "id": f"{source['id']}-{index:03d}",
                "title": title[:240],
                "text": text[:2200],
                "category": source["category"],
                "sub_category": source["sub_category"],
                "source": source["publisher"],
                "source_url": source["url"],
                "retrieved_at": stamp.isoformat(),
                "license": source["license"],
                "publisher": source["publisher"],
                "university_name": source.get("university_name"),
                "university_slug": source.get("university_slug"),
                "university_specific": bool(source.get("university_specific", False)),
                "last_verified_at": stamp.date().isoformat(),
                "source_type": source["source_type"],
                "source_priority": 50 if source["source_type"] == "official_government" else 40,
                "confidence": source.get("confidence", "high"),
                "license_url": source["terms_url"],
            }
            record["content_sha256"] = content_hash(record["title"], record["text"])
            destination.append(record)
    return government, universities, failures, unsupported


def validate_and_deduplicate(groups: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    seen: set[str] = set()
    duplicate_count = 0
    clean: dict[str, list[dict[str, Any]]] = {}
    for name, records in groups.items():
        clean[name] = []
        for record in records:
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(f"{record.get('id', '<unknown>')} missing fields: {missing}")
            if not record["license"] or not record["source_url"]:
                raise ValueError(f"{record['id']} has no reusable source/license")
            fingerprint = record["content_sha256"]
            if fingerprint in seen:
                duplicate_count += 1
                continue
            seen.add(fingerprint)
            clean[name].append(record)
    return clean, duplicate_count


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    path.write_text(body, encoding="utf-8")


def write_university_layout(records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        slug = record.get("university_slug")
        if slug:
            grouped.setdefault(slug, []).append(record)
    root = KNOWLEDGE_ROOT / "universities"
    for slug, rows in grouped.items():
        write_jsonl(root / slug / "general.jsonl", rows)
        manifest = {
            "university_name": rows[0]["university_name"],
            "university_slug": slug,
            "records": len(rows),
            "session_match_required": True,
            "files": {
                "general": "general.jsonl",
                "official_rules": None,
                "attendance": None,
                "grading": None,
                "registration": None,
                "scholarship": None,
                "career": None,
            },
            "policy": "Only explicit-license records are present; null categories are not fabricated.",
        }
        (root / slug / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_existing() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for name in ("wikipedia", "government", "university"):
        path = KNOWLEDGE_ROOT / f"{name}.jsonl"
        groups[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    return groups


def build_report(
    groups: dict[str, list[dict[str, Any]]],
    duplicate_count: int,
    failures: list[str],
    unsupported: list[str],
    stamp: datetime,
) -> dict[str, Any]:
    records = [record for values in groups.values() for record in values]
    faq_path = ROOT / "data" / "campus_v2" / "faq" / "reviewed.jsonl"
    faq_count = sum(1 for line in faq_path.read_text(encoding="utf-8").splitlines() if line.strip()) if faq_path.exists() else 0
    stale = sum(_verification_age(record.get("last_verified_at")) > (180 if record.get("category") in {
        "scholarship", "tuition", "part_time_job", "career_schedule", "internship", "registration", "gpa", "credit"
    } else 730) for record in records)
    return {
        "version": "campus-v2.2",
        "generated_at": stamp.isoformat(),
        "policy": "explicit-license-only; disabled/unknown-license source bodies are not stored",
        "required_schema": list(REQUIRED_FIELDS),
        "counts": {
            "total": len(records),
            "external_knowledge": len(records),
            "reviewed_faq": faq_count,
            "total_available_to_pipeline": len(records) + faq_count,
            "wikipedia": len(groups.get("wikipedia", [])),
            "government": len(groups.get("government", [])),
            "university": len(groups.get("university", [])),
            "duplicates_removed": duplicate_count,
            "unsupported_sources_excluded": len(unsupported),
            "fetch_failures": len(failures),
            "stale_documents": stale,
        },
        "targets_are_quality_guidelines_not_padding_requirements": {
            "wikipedia": "500-2000",
            "government": "200-500",
            "university": "300-1000 (only explicit reusable licenses)",
            "faq_and_self_authored": "2000",
        },
        "by_category": dict(sorted(Counter(record["category"] for record in records).items())),
        "by_publisher": dict(sorted(Counter(record["publisher"] for record in records).items())),
        "by_license": dict(sorted(Counter(record["license"] for record in records).items())),
        "failed_sources": failures,
        "unsupported_sources": unsupported,
        "limitations": [
            "University pages without an explicit reusable license are excluded even if publicly viewable.",
            "Official web text may change; dynamic administrative facts must be treated as stale after the configured freshness window.",
            "Wikipedia is secondary knowledge and does not override official government or matching university sources.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-limit", type=int, default=500)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--offline", action="store_true", help="validate and report the existing local corpus without network access")
    parser.add_argument("--official-only", action="store_true", help="refresh registry sources while retaining the existing Wikipedia corpus")
    args = parser.parse_args()
    stamp = utc_now()
    existing = load_existing()
    failures: list[str] = []
    unsupported: list[str] = []
    if args.offline:
        groups = load_existing()
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["sources"]
        unsupported = [source["id"] for source in registry if not source.get("enabled")]
    elif args.official_only:
        groups = load_existing()
        government, university, source_failures, unsupported = collect_registered(stamp)
        groups["government"] = government
        groups["university"] = university
        failures.extend(source_failures)
    else:
        wikipedia, wiki_failures = collect_wikipedia(max(1, args.wiki_limit), max(0.0, args.delay), stamp)
        government, university, source_failures, unsupported = collect_registered(stamp)
        groups = {"wikipedia": wikipedia, "government": government, "university": university}
        failures.extend(wiki_failures)
        failures.extend(source_failures)
    groups, duplicate_count = validate_and_deduplicate(groups)
    for name, records in groups.items():
        write_jsonl(KNOWLEDGE_ROOT / f"{name}.jsonl", records)
    write_university_layout(groups.get("university", []))
    RAG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "campus-v2.2",
        "generated_at": stamp.isoformat(),
        "documents": sum(len(records) for records in groups.values()),
        "files": {name: str((KNOWLEDGE_ROOT / f"{name}.jsonl").relative_to(ROOT)).replace("\\", "/") for name in groups},
        "index": "built in memory by campus_retrieval_v22.py (BM25 + word/character TF-IDF)",
    }
    (RAG_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = build_report(groups, duplicate_count, failures, unsupported, stamp)
    old_by_id = {row["id"]: row for values in existing.values() for row in values}
    new_by_id = {row["id"]: row for values in groups.values() for row in values}
    report["changes"] = {
        "added": sum(identifier not in old_by_id for identifier in new_by_id),
        "updated": sum(
            identifier in old_by_id and row.get("content_sha256") != old_by_id[identifier].get("content_sha256")
            for identifier, row in new_by_id.items()
        ),
        "unchanged": sum(
            identifier in old_by_id and row.get("content_sha256") == old_by_id[identifier].get("content_sha256")
            for identifier, row in new_by_id.items()
        ),
        "removed": sum(identifier not in new_by_id for identifier in old_by_id),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False))
    if failures:
        print(json.dumps({"failed_sources": failures}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
