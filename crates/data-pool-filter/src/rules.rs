//! Rust port of `scripts/pool_rules/rules_v3.py`.
//!
//! Schema-agnostic: each rule inspects the fields it cares about
//! (reading, surface, left_context_reading, left_context_surface)
//! via `serde_json::Value`, returning `None` (pass) or
//! `Some(reason)` to reject.
//!
//! Kept 1:1 with the Python version so that iteration feedback from
//! scripts/pool_filter.py maps directly to this binary's behaviour.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

// ---- regex sources ----------------------------------------------------

static LEAD_REJECT: Lazy<Regex> = Lazy::new(|| {
    // Chars to reject as leading. 「」 intentionally omitted per user rule.
    let pattern = concat!(
        "^[",
        r"\s　",
        "、。！？,.!?",
        "『』（）()",
        r"\[\]",
        "【】〔〕〈〉《》",
        "・ー…‥々〇〻※",
        r":;\-_/|@#*\\",
        "\"'”“‘’",
        "«»‹›★☆◇◆○●■□▲△▼▽",
        "♪♫♬♭♯",
        "§¶†‡",
        "→←↑↓↔⇒⇐⇑⇓⇨⇦⇧⇩⇪",
        "]",
    );
    Regex::new(pattern).expect("LEAD_REJECT regex")
});

static MARKDOWN_LEAD: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(\*\*|##|:::|;|\*|#|:)").unwrap());

static KANJI: Lazy<Regex> = Lazy::new(|| Regex::new(r"[一-鿿]").unwrap());
static KATAKANA: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ァ-ヺ]").unwrap());

static NONJP_SCRIPT: Lazy<Regex> = Lazy::new(|| {
    // Non-Japanese scripts: Cyrillic, Hebrew, Arabic, Devanagari, Thai, and
    // Arabic Presentation Forms. Latin / CJK / kana deliberately allowed.
    Regex::new(concat!(
        "[",
        "Ѐ-ӿ",                 // U+0400..U+04FF Cyrillic
        "Ԁ-ԯ",                 // U+0500..U+052F Cyrillic Supplement
        "֐-\u{05ff}",          // U+0590..U+05FF Hebrew
        "\u{0600}-ۿ",          // U+0600..U+06FF Arabic
        "\u{0750}-\u{077f}",   // Arabic Supplement
        "\u{0900}-\u{097f}",   // Devanagari
        "\u{0e00}-\u{0e7f}",   // Thai
        "\u{fb50}-\u{fdff}",   // Arabic Presentation Forms-A
        "\u{fe70}-\u{feff}",   // Arabic Presentation Forms-B
        "]",
    ))
    .unwrap()
});

static MEDIA_LINK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(\.jpg|\.png|\.jpeg|\.gif|\.svg|\.webp)\|").unwrap());

static WIKI_TEMPLATE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)^(Category:|redirect\s|\[\[|\{\{)").unwrap());

static WIKI_CATEGORY_ANY: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(カテゴリ:|Category:|ファイル:|File:)").unwrap());

static BRACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\{\}]").unwrap());

static ARROW_ANYWHERE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[⇩⇧⇪⇨⇦⇤⇥↶↷↺↻]").unwrap());

static CIRCLED_DIGIT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[①-⑳㊀-㊉]").unwrap());

static BREADCRUMB: Lazy<Regex> = Lazy::new(|| Regex::new(r">\S").unwrap());

static PURE_NUMERIC: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[\d,\s\.\-]+$").unwrap());

// ---- helpers ----------------------------------------------------------

fn as_str<'a>(row: &'a Value, field: &str) -> &'a str {
    row.get(field).and_then(Value::as_str).unwrap_or("")
}

fn char_count(s: &str) -> usize {
    s.chars().count()
}

// ---- rules ------------------------------------------------------------

pub type RuleFn = fn(&Value) -> Option<String>;

pub const RULES: &[(&str, RuleFn)] = &[
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
];

fn rule_lead_punct(row: &Value) -> Option<String> {
    let r = as_str(row, "reading");
    let s = as_str(row, "surface");
    if !r.is_empty() && LEAD_REJECT.is_match(r) {
        return Some(format!("reading_lead={}", first_two(r)));
    }
    if !s.is_empty() && LEAD_REJECT.is_match(s) {
        return Some(format!("surface_lead={}", first_two(s)));
    }
    None
}

fn rule_markdown_lead(row: &Value) -> Option<String> {
    for field in ["reading", "surface"] {
        let v = as_str(row, field);
        if !v.is_empty() && MARKDOWN_LEAD.is_match(v) {
            return Some(format!("{}_md={}", field, first_three(v)));
        }
    }
    None
}

fn rule_kigou_bug(row: &Value) -> Option<String> {
    for field in ["reading", "left_context_reading"] {
        let v = as_str(row, field);
        if v.contains("きごう") {
            return Some(format!("{}_has_kigou", field));
        }
    }
    None
}

fn rule_reading_kanji(row: &Value) -> Option<String> {
    let r = as_str(row, "reading");
    if KANJI.is_match(r) {
        return Some("reading_has_kanji".into());
    }
    None
}

fn rule_reading_katakana(row: &Value) -> Option<String> {
    let r = as_str(row, "reading");
    if KATAKANA.is_match(r) {
        return Some("reading_has_katakana".into());
    }
    None
}

fn rule_length(row: &Value) -> Option<String> {
    let r = as_str(row, "reading");
    let s = as_str(row, "surface");
    let rlen = char_count(r);
    let slen = char_count(s);
    if rlen < 2 {
        return Some(format!("reading_too_short({})", rlen));
    }
    if rlen > 200 {
        return Some(format!("reading_too_long({})", rlen));
    }
    if slen == 0 {
        return Some("surface_empty".into());
    }
    if slen > 200 {
        return Some(format!("surface_too_long({})", slen));
    }
    None
}

fn rule_length_ratio(row: &Value) -> Option<String> {
    let r = as_str(row, "reading");
    let s = as_str(row, "surface");
    if r.is_empty() || s.is_empty() {
        return None;
    }
    let ratio = char_count(r) as f64 / char_count(s) as f64;
    if ratio < 0.4 || ratio > 3.0 {
        return Some(format!("length_ratio={:.2}", ratio));
    }
    None
}

fn rule_nonjp_script(row: &Value) -> Option<String> {
    let s = as_str(row, "surface");
    if NONJP_SCRIPT.is_match(s) {
        return Some("nonjp_script_in_surface".into());
    }
    None
}

fn rule_media_link(row: &Value) -> Option<String> {
    let s = as_str(row, "surface");
    if MEDIA_LINK.is_match(s) {
        return Some("media_link_in_surface".into());
    }
    None
}

fn rule_wiki_template(row: &Value) -> Option<String> {
    let s = as_str(row, "surface");
    if WIKI_TEMPLATE.is_match(s) {
        return Some("wiki_template_in_surface".into());
    }
    None
}

fn rule_wiki_category_anywhere(row: &Value) -> Option<String> {
    let s = as_str(row, "surface");
    if let Some(m) = WIKI_CATEGORY_ANY.find(s) {
        return Some(format!("wiki_category={}", m.as_str()));
    }
    None
}

fn rule_brace(row: &Value) -> Option<String> {
    let s = as_str(row, "surface");
    if BRACE.is_match(s) {
        return Some("brace_in_surface".into());
    }
    None
}

fn rule_arrow_anywhere(row: &Value) -> Option<String> {
    let s = as_str(row, "surface");
    if ARROW_ANYWHERE.is_match(s) {
        return Some("arrow_in_surface".into());
    }
    None
}

fn rule_circled_digit(row: &Value) -> Option<String> {
    let s = as_str(row, "surface");
    if CIRCLED_DIGIT.is_match(s) {
        return Some("circled_digit_in_surface".into());
    }
    None
}

fn rule_breadcrumb(row: &Value) -> Option<String> {
    let s = as_str(row, "surface");
    if BREADCRUMB.is_match(s) {
        return Some("breadcrumb_gt_in_surface".into());
    }
    None
}

fn rule_pure_numeric(row: &Value) -> Option<String> {
    let s = as_str(row, "surface").trim();
    let r = as_str(row, "reading").trim();
    if !s.is_empty() && PURE_NUMERIC.is_match(s) {
        return Some("pure_numeric_surface".into());
    }
    if !r.is_empty() && PURE_NUMERIC.is_match(r) {
        return Some("pure_numeric_reading".into());
    }
    None
}

fn first_two(s: &str) -> String {
    s.chars().take(2).collect()
}

fn first_three(s: &str) -> String {
    s.chars().take(3).collect()
}

pub fn evaluate(row: &Value) -> Option<String> {
    for (name, fun) in RULES {
        if let Some(reason) = fun(row) {
            return Some(format!("{}={}", name, reason));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ok(row: &Value) -> bool {
        evaluate(row).is_none()
    }

    #[test]
    fn passes_normal_row() {
        assert!(ok(&json!({
            "reading": "わたしはりんごを",
            "surface": "私はリンゴを",
            "context": ""
        })));
    }

    #[test]
    fn rejects_punct_lead() {
        let r = evaluate(&json!({"reading":"、あ","surface":"、あ"}));
        assert!(r.unwrap().starts_with("lead_punct"));
    }

    #[test]
    fn keeps_kagikakko_lead() {
        assert!(ok(&json!({
            "reading": "「あめがふる",
            "surface": "「雨が降る",
            "context": ""
        })));
    }

    #[test]
    fn rejects_kigou_bug() {
        let r = evaluate(&json!({
            "reading": "ごぎきごうだい",
            "surface": "語義)",
            "context": ""
        }));
        assert!(r.unwrap().starts_with("kigou_bug"));
    }

    #[test]
    fn rejects_kanji_in_reading() {
        let r = evaluate(&json!({
            "reading": "わたし私",
            "surface": "私",
            "context": ""
        }));
        assert!(r.unwrap().starts_with("reading_kanji"));
    }
}
