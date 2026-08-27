from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import re


DROP_LINK_NAMESPACES = {
    "file", "image", "category", "media", "special", "portal", "help", "template",
    "ファイル", "画像", "カテゴリ", "メディア", "特別", "ポータル", "ヘルプ", "テンプレート",
}
DROP_CONTENT_TAGS = {
    "ref", "references", "script", "style", "gallery", "timeline", "imagemap", "score",
}
BLOCK_TAGS = {"br", "p", "div", "li", "tr", "section", "h1", "h2", "h3", "h4", "h5", "h6"}


class _HTMLTextCleaner(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.skip_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        name = tag.lower()
        if self.skip_stack:
            if name in DROP_CONTENT_TAGS:
                self.skip_stack.append(name)
            return
        if name in DROP_CONTENT_TAGS:
            self.skip_stack.append(name)
        elif name in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        if not self.skip_stack and tag.lower() in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if self.skip_stack:
            if name == self.skip_stack[-1]:
                self.skip_stack.pop()
            return
        if name in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_stack:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if not self.skip_stack:
            self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if not self.skip_stack:
            self.parts.append(f"&#{name};")


def strip_html_and_references(text: str) -> str:
    parser = _HTMLTextCleaner()
    parser.feed(text)
    parser.close()
    return unescape("".join(parser.parts))


def remove_balanced(text: str, opening: str, closing: str) -> tuple[str, int]:
    output: list[str] = []
    index = 0
    removed = 0
    while index < len(text):
        if not text.startswith(opening, index):
            output.append(text[index])
            index += 1
            continue
        start = index
        depth = 1
        index += len(opening)
        while index < len(text) and depth:
            if text.startswith(opening, index):
                depth += 1
                index += len(opening)
            elif text.startswith(closing, index):
                depth -= 1
                index += len(closing)
            else:
                index += 1
        if depth:
            output.append(text[start:])
            break
        removed += 1
    return "".join(output), removed


def _matching_link_end(text: str, start: int) -> int | None:
    depth = 1
    index = start + 2
    while index < len(text):
        if text.startswith("[[", index):
            depth += 1
            index += 2
        elif text.startswith("]]", index):
            depth -= 1
            index += 2
            if depth == 0:
                return index
        else:
            index += 1
    return None


def _split_link_parts(content: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(content):
        if content.startswith("[[", index):
            depth += 1
            index += 2
        elif content.startswith("]]", index) and depth:
            depth -= 1
            index += 2
        elif content[index] == "|" and depth == 0:
            parts.append(content[start:index])
            start = index + 1
            index += 1
        else:
            index += 1
    parts.append(content[start:])
    return parts


def _is_dropped_link(target: str) -> bool:
    normalized = target.strip().lstrip(":")
    if ":" not in normalized:
        return False
    prefix = normalized.split(":", 1)[0].strip().lower()
    return prefix in DROP_LINK_NAMESPACES or bool(re.fullmatch(r"[a-z]{2,3}(?:-[a-z]+)?", prefix))


def clean_internal_links(text: str) -> tuple[str, dict]:
    output: list[str] = []
    index = 0
    kept = dropped = 0
    while index < len(text):
        start = text.find("[[", index)
        if start < 0:
            output.append(text[index:])
            break
        output.append(text[index:start])
        end = _matching_link_end(text, start)
        if end is None:
            output.append(text[start:])
            break
        content = text[start + 2:end - 2]
        parts = _split_link_parts(content)
        target = parts[0].strip()
        if _is_dropped_link(target):
            dropped += 1
        else:
            display = parts[-1].strip() if len(parts) > 1 else target.split("#", 1)[0].strip()
            nested, nested_metrics = clean_internal_links(display)
            output.append(nested)
            kept += 1 + nested_metrics["kept"]
            dropped += nested_metrics["dropped"]
        index = end
    return "".join(output), {"kept": kept, "dropped": dropped}


def residue_signals(text: str) -> dict[str, int]:
    return {
        "wiki_open": text.count("[["),
        "wiki_close": text.count("]]"),
        "template_open": text.count("{{"),
        "template_close": text.count("}}"),
        "table_open": text.count("{|"),
        "table_close": text.count("|}"),
        "html_tag": len(re.findall(r"</?[A-Za-z][^>]*>", text)),
        "file_image": len(re.findall(r"(?i)(?:File|Image|ファイル|画像)\s*:", text)),
        "reference_tag": len(re.findall(r"(?i)<references?\b", text)),
    }


def clean_mediawiki(text: str) -> tuple[str, dict]:
    original = text
    value = strip_html_and_references(text)
    value, tables_removed = remove_balanced(value, "{|", "|}")
    value, templates_removed = remove_balanced(value, "{{", "}}")
    value, link_metrics = clean_internal_links(value)
    value = re.sub(r"\[(?:https?|ftp)://[^\s\]]+\s+([^\]]+)\]", r"\1", value)
    value = re.sub(r"\[(?:https?|ftp)://[^\]]+\]", "", value)
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"__(?:TOC|NOTOC|FORCETOC|NOEDITSECTION|NEWSECTIONLINK)__", "", value,
                   flags=re.IGNORECASE)
    value = re.sub(r"''+", "", value)
    value = value.replace("\u00a0", " ").replace("\u3000", " ")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    signals = residue_signals(value)
    return value, {
        "original_characters": len(original),
        "cleaned_characters": len(value),
        "characters_removed": len(original) - len(value),
        "tables_removed": tables_removed,
        "templates_removed": templates_removed,
        "internal_links_kept": link_metrics["kept"],
        "file_image_or_namespace_links_removed": link_metrics["dropped"],
        "residue_signals": signals,
        "markup_free": not any(signals.values()),
    }


def strict_quality_reason(title: str, text: str) -> str | None:
    signals = residue_signals(text)
    if any(signals.values()):
        return "residual_markup"
    if "\ufffd" in text:
        return "broken_text"
    if len(text) < 500:
        return "too_short"
    if len(text) > 80_000:
        return "extreme_length"
    japanese = len(re.findall(r"[ぁ-んァ-ヶ一-龥々]", text))
    if japanese / max(1, len(text)) < .35:
        return "low_japanese_ratio"
    if "曖昧さ回避" in text[:500] or ("この項目では" in text[:200] and "区別" in text[:500]):
        return "disambiguation"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    list_lines = sum(line.startswith(("*", "#", ";", ":", "|-", "|+")) for line in lines)
    if lines and list_lines / len(lines) > .30:
        return "list_or_navigation_heavy"
    url_count = len(re.findall(r"(?:https?://|www\.)", text, flags=re.IGNORECASE))
    if url_count:
        return "url_residue"
    symbol_runs = len(re.findall(r"[^0-9A-Za-zぁ-んァ-ヶ一-龥々\s]{12,}", text))
    if symbol_runs > max(2, len(text) // 5000):
        return "unnatural_symbol_runs"
    sentence_marks = len(re.findall(r"[。！？]", text))
    if sentence_marks < max(2, len(text) // 800):
        return "low_sentence_density"
    return None
