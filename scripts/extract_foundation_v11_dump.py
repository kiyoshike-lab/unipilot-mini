"""Extract a strict, attributable Foundation v1.1 corpus from Wikimedia XML dumps."""
from __future__ import annotations

import argparse
import bz2
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.mediawiki_cleaner import clean_mediawiki, strict_quality_reason
from scripts.collect_foundation_v10_wikimedia import clean_extract
from scripts.extract_foundation_v10_dump import (
    LICENSE,
    LICENSE_URL,
    PROJECTS,
    child_text,
    local_name,
    revision_field,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=tuple(PROJECTS), required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--dump-url", required=True)
    parser.add_argument("--target-characters", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--report")
    args = parser.parse_args()
    project = PROJECTS[args.project]
    dump_path = Path(args.dump)
    output = Path(args.output or f"data/foundation_v11/raw/{args.project}-dump-ja.jsonl.gz")
    report_path = Path(args.report or f"evaluation/foundation-v11-{args.project}-dump.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    accepted = parsed = redirects = non_main = characters = modified = empty_after_clean = 0
    excluded: dict[str, int] = {}
    fingerprints: set[str] = set()

    with bz2.open(dump_path, "rb") as source, gzip.open(
        output, "wt", encoding="utf-8", newline="\n"
    ) as target:
        for _, element in ET.iterparse(source, events=("end",)):
            if local_name(element.tag) != "page":
                continue
            parsed += 1
            title = child_text(element, "title") or ""
            namespace = child_text(element, "ns") or ""
            page_id = child_text(element, "id") or "0"
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
            cleaned, cleaning = clean_mediawiki(raw_text)
            cleaned, line_metrics = clean_extract(cleaned)
            if not cleaned:
                empty_after_clean += 1
                element.clear()
                continue
            reason = strict_quality_reason(title, cleaned)
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
            revision_id = revision_field(element, "id")
            timestamp = revision_field(element, "timestamp")
            url = project["article_base"] + quote(title.replace(" ", "_"), safe="()/:_")
            row = {
                "id": f"{args.project}-dump-ja-{page_id}", "kind": "text", "title": title,
                "text": cleaned, "categories": categories,
                "source_type": f"wikimedia_{args.project}_official_dump",
                "source": project["source"], "publisher": "Wikimedia Foundation",
                "source_url": url, "attribution_url": url + "?action=history",
                "retrieved_at": retrieved_at, "license": LICENSE, "license_url": LICENSE_URL,
                "page_id": int(page_id),
                "revision_id": int(revision_id) if revision_id else None,
                "revision_timestamp": timestamp, "content_sha256": fingerprint,
                "cleaning": {**cleaning, "line_cleanup": line_metrics,
                             "cleaner": "foundation-v11-stack-v1"},
                "dump_url": args.dump_url,
                "training_role": "general_japanese_pretraining_candidate",
                "external_ai_used": False,
            }
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            accepted += 1
            characters += len(cleaned)
            modified += int(cleaning["characters_removed"] != 0)
            element.clear()
            if accepted % 500 == 0:
                print(json.dumps({"project": args.project, "parsed": parsed,
                                  "accepted": accepted, "characters": characters}), flush=True)
            if args.target_characters and characters >= args.target_characters:
                break

    report = {
        "schema_version": "foundation-v11-wikimedia-dump-v1",
        "project": project["site"], "dump_url": args.dump_url,
        "dump_path": dump_path.as_posix(), "dump_bytes": dump_path.stat().st_size,
        "dump_sha256": sha256(dump_path), "retrieved_at": retrieved_at,
        "license": LICENSE, "license_url": LICENSE_URL,
        "parsed_pages": parsed, "accepted_documents": accepted,
        "accepted_characters": characters, "modified_documents": modified,
        "empty_after_clean": empty_after_clean, "redirects": redirects,
        "non_main_namespace": non_main, "excluded": excluded,
        "output": output.as_posix(), "cleaner": "stack-based nested MediaWiki cleaner",
        "strict_zero_residue_gate": True, "attribution_preserved_per_article": True,
        "modifications_marked_per_article": True, "external_ai_api": "OFF",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
