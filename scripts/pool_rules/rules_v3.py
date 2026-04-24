"""Rule set v3 — final tightening based on round_4-10 survivor inspection.

Additions over v2:
- Arrow symbols (⇩⇧⇪⇨⇦…) anywhere (not just lead)
- Circled digits ①-⑳ etc. in surface (round 3 "③金銭感覚...")
- `>` and `|` as breadcrumb separators anywhere in surface

Changes to existing rules:
- lead_punct now explicit about ★☆ etc. was already added in v2
"""
from __future__ import annotations
import re

_LEAD_REJECT = re.compile(
    r"^["
    r"\s　"
    r"、。！？,.!?"
    r"『』（）()\[\]【】〔〕〈〉《》"
    r"・ー…‥々〇〻※"
    r":;\-_/|@#*\\"
    r"\"'”“‘’"
    r"«»‹›★☆◇◆○●■□▲△▼▽♪♫♬♭♯§¶†‡→←↑↓↔⇒⇐⇑⇓⇨⇦⇧⇩⇪"
    r"]"
)

_MARKDOWN_LEAD = re.compile(r"^(\*\*|##|:::|;|\*|#|:)")
_KATA = re.compile(r"[ァ-ヺ]")
_KANJI = re.compile(r"[一-鿿]")
_NONJP = re.compile(
    r"[Ѐ-ӿԀ-ԯ֐-׿؀-ۿݐ-ݿऀ-ॿ฀-๿ﭐ-﷿ﹰ-﻿]"
)
_MEDIA_LINK = re.compile(r"(\.jpg|\.png|\.jpeg|\.gif|\.svg|\.webp)\|", re.IGNORECASE)
_WIKI_TEMPLATE = re.compile(r"^(Category:|redirect\s|\[\[|\{\{)", re.IGNORECASE)
_PURE_NUMERIC = re.compile(r"^[\d,\s\.\-]+$")
_WIKI_CATEGORY_ANY = re.compile(r"(カテゴリ:|Category:|ファイル:|File:)", re.IGNORECASE)
_BRACE = re.compile(r"[\{\}]")

# v3 additions
_ARROW_ANYWHERE = re.compile(r"[⇩⇧⇪⇨⇦⇤⇥↶↷↺↻]")
_CIRCLED_DIGIT = re.compile(r"[①-⑳㊀-㊉]")
_BREADCRUMB = re.compile(r">\S")  # bare ">" followed by non-space (navigation)


def rule_lead_punct(row):
    r = row.get("reading", "") or ""
    s = row.get("surface", "") or ""
    if r and _LEAD_REJECT.match(r):
        return f"reading_lead='{r[:2]}'"
    if s and _LEAD_REJECT.match(s):
        return f"surface_lead='{s[:2]}'"
    return None


def rule_markdown_lead(row):
    r = row.get("reading", "") or ""
    s = row.get("surface", "") or ""
    for name, val in (("reading", r), ("surface", s)):
        if val and _MARKDOWN_LEAD.match(val):
            return f"{name}_md='{val[:3]}'"
    return None


def rule_kigou_bug(row):
    for field in ("reading", "left_context_reading"):
        v = row.get(field, "") or ""
        if "きごう" in v:
            return f"{field}_has_kigou"
    return None


def rule_reading_kanji(row):
    r = row.get("reading", "") or ""
    if _KANJI.search(r):
        return "reading_has_kanji"
    return None


def rule_reading_katakana(row):
    r = row.get("reading", "") or ""
    if _KATA.search(r):
        return "reading_has_katakana"
    return None


def rule_length(row):
    r = row.get("reading", "") or ""
    s = row.get("surface", "") or ""
    if len(r) < 2:
        return f"reading_too_short({len(r)})"
    if len(r) > 200:
        return f"reading_too_long({len(r)})"
    if len(s) < 1:
        return "surface_empty"
    if len(s) > 200:
        return f"surface_too_long({len(s)})"
    return None


def rule_length_ratio(row):
    r = row.get("reading", "") or ""
    s = row.get("surface", "") or ""
    if not r or not s:
        return None
    ratio = len(r) / len(s)
    if ratio < 0.4 or ratio > 3.0:
        return f"length_ratio={ratio:.2f}"
    return None


def rule_nonjp_script(row):
    s = row.get("surface", "") or ""
    if _NONJP.search(s):
        return "nonjp_script_in_surface"
    return None


def rule_media_link(row):
    s = row.get("surface", "") or ""
    if _MEDIA_LINK.search(s):
        return "media_link_in_surface"
    return None


def rule_wiki_template(row):
    s = row.get("surface", "") or ""
    if _WIKI_TEMPLATE.search(s):
        return "wiki_template_in_surface"
    return None


def rule_pure_numeric(row):
    s = (row.get("surface", "") or "").strip()
    r = (row.get("reading", "") or "").strip()
    if s and _PURE_NUMERIC.match(s) and len(s) > 0:
        return "pure_numeric_surface"
    if r and _PURE_NUMERIC.match(r):
        return "pure_numeric_reading"
    return None


def rule_wiki_category_anywhere(row):
    s = row.get("surface", "") or ""
    m = _WIKI_CATEGORY_ANY.search(s)
    if m:
        return f"wiki_category={m.group(0)}"
    return None


def rule_brace(row):
    s = row.get("surface", "") or ""
    if _BRACE.search(s):
        return "brace_in_surface"
    return None


# v3 new

def rule_arrow_anywhere(row):
    s = row.get("surface", "") or ""
    if _ARROW_ANYWHERE.search(s):
        return "arrow_in_surface"
    return None


def rule_circled_digit(row):
    s = row.get("surface", "") or ""
    if _CIRCLED_DIGIT.search(s):
        return "circled_digit_in_surface"
    return None


def rule_breadcrumb(row):
    s = row.get("surface", "") or ""
    if _BREADCRUMB.search(s):
        return "breadcrumb_gt_in_surface"
    return None


RULES = [
    ("lead_punct", rule_lead_punct),
    ("markdown_lead", rule_markdown_lead),
    ("kigou_bug", rule_kigou_bug),
    ("reading_kanji", rule_reading_kanji),
    ("reading_katakana", rule_reading_katakana),
    ("length", rule_length),
    ("length_ratio", rule_length_ratio),
    ("nonjp_script", rule_nonjp_script),
    ("media_link", rule_media_link),
    ("wiki_template", rule_wiki_template),
    ("wiki_category_anywhere", rule_wiki_category_anywhere),
    ("brace", rule_brace),
    ("arrow_anywhere", rule_arrow_anywhere),
    ("circled_digit", rule_circled_digit),
    ("breadcrumb", rule_breadcrumb),
    ("pure_numeric", rule_pure_numeric),
]
