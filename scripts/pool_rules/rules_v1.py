"""Rule set v1 — derived from round_1 manual reading.

Each rule is a (name, fn) where fn(row: dict) returns a short reason
string if the row should be rejected, None to pass.

Iteration policy:
- user read samples, identified bad patterns
- these rules operationalize the identified patterns
- left_context_reading / left_context_surface are inspected when the
  row is bunsetsu-schema (they carry extra artifacts like the
  きごう bug).
"""
from __future__ import annotations
import re

# Exceptions: 「 」 stay as acceptable leads per user.
# All other symbols/punct/whitespace leads are rejected.
_LEAD_REJECT = re.compile(
    r"^["
    r"\s　"                 # whitespace + full-width
    r"、。！？,.!?"
    r"『』（）()\[\]【】〔〕〈〉《》"
    r"・ー…‥々〇〻※"
    r":;\-_/|@#*\\"
    r"\"'”“‘’"
    r"]"
)

_MARKDOWN_LEAD = re.compile(r"^(\*\*|##|:::|;|\*|#|:)")

_HIRA = re.compile(r"[ぁ-ゖ]")
_KATA = re.compile(r"[ァ-ヺ]")
_KANJI = re.compile(r"[一-鿿]")
# Non-JP scripts (outside ASCII, latin-1, CJK, kana, common punct).
# Arabic / Hebrew / Devanagari / Thai / Cyrillic.
_NONJP = re.compile(
    r"["
    r"Ѐ-ӿ"   # Cyrillic
    r"Ԁ-ԯ"   # Cyrillic Supplement
    r"֐-׿"   # Hebrew
    r"؀-ۿ"   # Arabic
    r"ݐ-ݿ"   # Arabic Supplement
    r"ऀ-ॿ"   # Devanagari
    r"฀-๿"   # Thai
    r"ﭐ-﷿"   # Arabic Presentation Forms-A
    r"ﹰ-﻿"   # Arabic Presentation Forms-B
    r"]"
)

_MEDIA_LINK = re.compile(r"(\.jpg|\.png|\.jpeg|\.gif|\.svg|\.webp)\|", re.IGNORECASE)
_WIKI_TEMPLATE = re.compile(r"^(Category:|redirect\s|\[\[|\{\{)", re.IGNORECASE)
_PURE_NUMERIC = re.compile(r"^[\d,\s\.\-]+$")


def _r(name, reason):
    return f"{name}={reason}"


# -- rules --------------------------------------------------------------

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
    # bunsetsu pipeline v2 inserted literal "きごう" (=記号) when the
    # source sentence contained punctuation/brackets. Pollutes reading
    # or left_context_reading.
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
    # Reading-only would be weird
    if r and _PURE_NUMERIC.match(r):
        return "pure_numeric_reading"
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
    ("pure_numeric", rule_pure_numeric),
]
