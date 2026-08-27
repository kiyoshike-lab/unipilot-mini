from __future__ import annotations

import pytest

from foundation.mediawiki_cleaner import clean_mediawiki, residue_signals, strict_quality_reason


CASES = [
    ("[[人工知能]]", "人工知能"),
    ("[[人工知能|AI]]", "AI"),
    ("[[東京大学|東大]]", "東大"),
    ("前[[東京]]後", "前東京後"),
    ("[[東京#歴史]]", "東京"),
    ("[[A|B [[C|D]] E]]", "B D E"),
    ("[[A|[[B|表示]]]]", "表示"),
    ("[[A|一|二]]", "二"),
    ("[[A B]]", "A B"),
    ("[[A|日本語。]]", "日本語。"),
    ("[[File:x.jpg|thumb|説明]]本文", "本文"),
    ("[[Image:x.png|説明 [[東京]]]]本文", "本文"),
    ("[[ファイル:x.svg|thumb|[[数学]]の図]]本文", "本文"),
    ("[[画像:x.jpg|説明]]", ""),
    ("[[:Category:数学]]本文", "本文"),
    ("[[カテゴリ:数学]]本文", "本文"),
    ("[[en:Tokyo]]東京", "東京"),
    ("[[de:Japan|Japan]]日本", "日本"),
    ("[[Portal:科学]]科学", "科学"),
    ("[[Help:目次]]本文", "本文"),
    ("{{A}}本文", "本文"),
    ("{{A|{{B|値}}}}本文", "本文"),
    ("前{{A|x={{B}}}}後", "前後"),
    ("{{Infobox\n|a=b\n}}本文", "本文"),
    ("本文{{lang|ja|日本語}}です", "本文です"),
    ("<ref>出典</ref>本文", "本文"),
    ("<ref name='a'>出典</ref>本文", "本文"),
    ("<ref name='a' />本文", "本文"),
    ("本文<references />", "本文"),
    ("本文<ref>[[出典]] {{cite}}</ref>後", "本文後"),
    ("<b>太字</b>", "太字"),
    ("<div>本文</div>", "本文"),
    ("本文<br />次", "本文\n次"),
    ("<!-- コメント -->本文", "本文"),
    ("<script>alert(1)</script>本文", "本文"),
    ("<style>.x{}</style>本文", "本文"),
    ("<gallery>File:x.jpg|説明</gallery>本文", "本文"),
    ("<span lang='ja'>日本語</span>", "日本語"),
    ("A &amp; B", "A & B"),
    ("{| class='wikitable'\n|a\n|}本文", "本文"),
    ("前{|\n| {{A}}\n{|\n|b\n|}\n|}後", "前後"),
    ("{|\n|-\n|+題\n|a||b\n|}", ""),
    ("本文\n{|\n|a\n|}\n次", "本文\n\n次"),
    ("[https://example.com 表示]本文", "表示本文"),
    ("[https://example.com]本文", "本文"),
    ("https://example.com 本文", "本文"),
    ("'''太字'''と''斜体''", "太字と斜体"),
    ("__TOC__本文", "本文"),
    ("__NOTOC__本文", "本文"),
    ("日本語の「括弧」と【記号】は保持する。", "日本語の「括弧」と【記号】は保持する。"),
    ("配列array[i]と辞書map[key]を使う。", "配列array[i]と辞書map[key]を使う。"),
    ("1 < 2 であり、3 > 2 である。", "1 < 2 であり、3 > 2 である。"),
    ("C++とC#はプログラミング言語である。", "C++とC#はプログラミング言語である。"),
    ("メールはa@example.comへ送る。", "メールはa@example.comへ送る。"),
    ("改行を保持する。\n\n次の段落。", "改行を保持する。\n\n次の段落。"),
    ("全角　空白を整える。", "全角 空白を整える。"),
    ("連続   空白を整える。", "連続 空白を整える。"),
    ("[[日本|日本国]]は{{基礎情報|x={{y}}}}東アジアにある。", "日本国は東アジアにある。"),
    ("[[File:x|thumb|[[東京大学|東大]]の[[校舎]]]]東大は大学である。", "東大は大学である。"),
    ("<ref>{{cite|[[本]]}}</ref>知識は検証される。", "知識は検証される。"),
]


@pytest.mark.parametrize(("source", "expected"), CASES)
def test_clean_mediawiki_cases(source: str, expected: str):
    cleaned, metrics = clean_mediawiki(source)
    assert cleaned == expected
    assert not any(residue_signals(cleaned).values())
    assert metrics["markup_free"] is True


def test_long_japanese_paragraph_is_not_destroyed():
    paragraph = (
        "大学では授業を受けるだけでなく、自分で資料を読み、考えを整理し、"
        "根拠を示しながら文章を書くことが求められる。"
    ) * 20
    cleaned, _ = clean_mediawiki(paragraph)
    assert cleaned == paragraph
    assert strict_quality_reason("大学での学習", cleaned) is None


def test_unmatched_markup_is_left_for_strict_quality_rejection():
    cleaned, _ = clean_mediawiki("本文[[壊れたリンク")
    assert "[[" in cleaned
    assert strict_quality_reason("壊れた文書", cleaned * 100) == "residual_markup"


def test_strict_quality_rejects_every_residue_marker():
    base = "これは十分に長い日本語本文であり、内容を明確に説明する文章である。" * 30
    for marker in ("[[", "]]", "{{", "}}", "{|", "|}", "<b>", "File:"):
        assert strict_quality_reason("品質検査", base + marker) == "residual_markup"
