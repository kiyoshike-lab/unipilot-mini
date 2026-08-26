"""Extract clean, attributable Japanese text from an official Wikimedia XML dump."""
from __future__ import annotations

import argparse
import bz2
from datetime import datetime, timezone
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_foundation_v10_wikimedia import clean_extract, quality_reason


PROJECTS = {
    "wikipedia": {
        "site": "Japanese Wikipedia", "source": "日本語版ウィキペディアの執筆者",
        "article_base": "https://ja.wikipedia.org/wiki/",
    },
    "wikibooks": {
        "site": "Japanese Wikibooks", "source": "日本語版ウィキブックスの執筆者",
        "article_base": "https://ja.wikibooks.org/wiki/",
    },
}
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def remove_nested(text: str, opening: str, closing: str) -> str:
    output: list[str] = []
    depth = 0
    index = 0
    while index < len(text):
        if text.startswith(opening, index):
            depth += 1
            index += len(opening)
            continue
        if depth and text.startswith(closing, index):
            depth -= 1
            index += len(closing)
            continue
        if depth == 0:
            output.append(text[index])
        index += 1
    return "".join(output)


def clean_wikitext(value: str) -> tuple[str, dict]:
    original = value
    text = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
    text = re.sub(r"<ref\b[^>/]*?>.*?</ref\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref\b[^>]*/\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<(gallery|timeline|imagemap|score)\b.*?</\1\s*>", "", text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = remove_nested(text, "{|", "|}")
    text = remove_nested(text, "{{", "}}")
    text = re.sub(r"\[\[(?:File|Image|ファイル|画像|Category|カテゴリ):[^\]]*\]\]", "",
                  text, flags=re.IGNORECASE)
    for _ in range(3):
        text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
        text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[(?:https?|ftp)://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[(?:https?|ftp)://[^\]]+\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"''+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text, extract_metrics = clean_extract(text)
    return text, {
        "original_characters": len(original),
        "cleaned_characters": len(text),
        "modified": text != original.strip(),
        "reference_and_template_markup_removed": True,
        **{key: value for key, value in extract_metrics.items()
           if key not in {"original_characters", "cleaned_characters", "modified"}},
    }


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element, name: str) -> str | None:
    for child in element:
        if local_name(child.tag) == name:
            return child.text
    return None


def revision_field(page: ET.Element, name: str) -> str | None:
    for child in page:
        if local_name(child.tag) != "revision":
            continue
        for field in child:
            if local_name(field.tag) == name:
                return field.text
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=tuple(PROJECTS), required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--dump-url", required=True)
    parser.add_argument("--target-characters", type=int, default=0,
                        help="Stop after this many clean characters; zero reads the whole dump.")
    parser.add_argument("--output")
    parser.add_argument("--report")
    args = parser.parse_args()
    project = PROJECTS[args.project]
    dump_path = Path(args.dump)
    output = Path(args.output or f"data/foundation_v10/raw/{args.project}-dump-ja.jsonl.gz")
    report_path = Path(args.report or f"evaluation/foundation-v10-{args.project}-dump.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    accepted = parsed = redirects = non_main = 0
    characters = 0
    excluded: dict[str, int] = {}
    fingerprints: set[str] = set()

    with bz2.open(dump_path, "rb") as source, gzip.open(output, "wt", encoding="utf-8",
                                                         newline="\n") as target:
        for _, element in ET.iterparse(source, events=("end",)):
            if local_name(element.tag) != "page":
                continue
            parsed += 1
            title = child_text(element, "title") or ""
            namespace = child_text(element, "ns") or ""
            page_id = child_text(element, "id") or ""
            redirect = any(local_name(child.tag) == "redirect" for child in element)
            if namespace != "0":
                non_main += 1
                element.clear()
                continue
            if redirect:
                redirects += 1
                element.clear()
                continue
            raw_text = revision_field(element, "text") or ""
            categories = re.findall(
                r"\[\[(?:Category|カテゴリ):([^\]|]+)", raw_text, flags=re.IGNORECASE
            )
            revision_id = revision_field(element, "id")
            timestamp = revision_field(element, "timestamp")
            cleaned, cleaning = clean_wikitext(raw_text)
            reason = quality_reason(title, cleaned)
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
                element.clear()
                continue
            fingerprint = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            if fingerprint in fingerprints:
                excluded["exact_duplicate"] = excluded.get("exact_duplicate", 0) + 1
                element.clear()
                continue
            fingerprints.add(fingerprint)
            url = project["article_base"] + quote(title.replace(" ", "_"), safe="()/:_")
            row = {
                "id": f"{args.project}-dump-ja-{page_id}", "kind": "text",
                "title": title, "text": cleaned, "categories": categories,
                "source_type": f"wikimedia_{args.project}_official_dump",
                "source": project["source"], "publisher": "Wikimedia Foundation",
                "source_url": url, "attribution_url": url + "?action=history",
                "retrieved_at": retrieved_at, "license": LICENSE,
                "license_url": LICENSE_URL, "page_id": int(page_id),
                "revision_id": int(revision_id) if revision_id else None,
                "revision_timestamp": timestamp, "content_sha256": fingerprint,
                "cleaning": cleaning, "dump_url": args.dump_url,
                "training_role": "general_japanese_pretraining_candidate",
                "external_ai_used": False,
            }
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            accepted += 1
            characters += len(cleaned)
            element.clear()
            if accepted % 500 == 0:
                print(json.dumps({"project": args.project, "parsed": parsed,
                                  "accepted": accepted, "characters": characters}), flush=True)
            if args.target_characters and characters >= args.target_characters:
                break

    digest = hashlib.sha256()
    with dump_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    report = {
        "schema_version": "foundation-v10-wikimedia-dump-v1",
        "project": project["site"], "dump_url": args.dump_url,
        "dump_path": dump_path.as_posix(), "dump_bytes": dump_path.stat().st_size,
        "dump_sha256": digest.hexdigest(), "retrieved_at": retrieved_at,
        "license": LICENSE, "license_url": LICENSE_URL,
        "parsed_pages": parsed, "accepted_documents": accepted,
        "accepted_characters": characters, "redirects": redirects,
        "non_main_namespace": non_main, "excluded": excluded,
        "output": output.as_posix(), "attribution_preserved_per_article": True,
        "modifications_marked_per_article": True, "external_ai_api": "OFF",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
