"""Reclaim rows that rules_v3 rejected purely for a leading wikitext
prefix (`:`, `*`, `#`, `;`, `**`, `##`, `:::`, `#:`, `*:`, etc.).

After stripping the prefix + any leading whitespace, the body is often
a perfectly clean Japanese sentence. We strip once, re-run the rule
set, and (only if it now passes) append to the cleaned pool.

Conservative: we do NOT try to fix things like `(...)` annotations or
`[[...]]` links — those touch content, not just markers.

Usage (per pool):
    python scripts/salvage_markdown_prefix.py \\
        --rejects datasets/audits/cleaned/sentence-wikibooks.rejects.jsonl \\
        --cleaned datasets/corpus/cleaned/sentence/wikibooks.jsonl
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Load rules v3 (same module the Rust filter mirrors).
def load_rules():
    spec = importlib.util.spec_from_file_location(
        "rules_v3",
        str(REPO / "scripts" / "pool_rules" / "rules_v3.py"),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.RULES


# Strip only wiki-markdown marker characters + following ASCII whitespace.
# Crucially, we DON'T strip `(`, `[`, `{`, `（`, `【`, `『`, etc., because
# those frame content and removing only the opener produces unbalanced text.
_PREFIX_RE = re.compile(r"^([:#;*]+)\s*")


def strip_prefix(text: str) -> str | None:
    m = _PREFIX_RE.match(text)
    if not m:
        return None
    return text[m.end():]


def apply_rules(row, rules):
    for name, fn in rules:
        r = fn(row)
        if r:
            return f"{name}={r}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rejects", required=True)
    ap.add_argument("--cleaned", required=True, help="output survivor pool; rows appended")
    ap.add_argument("--salvage-log", help="where to record salvaged rows (reason=salvaged)")
    ap.add_argument("--still-bad", help="where to record rows that still reject after strip")
    args = ap.parse_args()

    rules = load_rules()
    n_in = n_prefix_ok = n_re_pass = 0

    cleaned_path = Path(args.cleaned)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_fh = open(cleaned_path, "a", encoding="utf-8")

    salvage_fh = None
    if args.salvage_log:
        p = Path(args.salvage_log)
        p.parent.mkdir(parents=True, exist_ok=True)
        salvage_fh = open(p, "w", encoding="utf-8")

    still_bad_fh = None
    if args.still_bad:
        p = Path(args.still_bad)
        p.parent.mkdir(parents=True, exist_ok=True)
        still_bad_fh = open(p, "w", encoding="utf-8")

    with open(args.rejects, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            n_in += 1
            row = d.get("row", {})
            reading = row.get("reading", "") or ""
            surface = row.get("surface", "") or ""

            new_r = strip_prefix(reading)
            new_s = strip_prefix(surface)
            if new_r is None and new_s is None:
                # No prefix to strip; rule rejected for other reason.
                if still_bad_fh:
                    still_bad_fh.write(line + "\n")
                continue
            n_prefix_ok += 1

            fixed = dict(row)
            if new_r is not None:
                fixed["reading"] = new_r
            if new_s is not None:
                fixed["surface"] = new_s

            reason = apply_rules(fixed, rules)
            if reason is None:
                cleaned_fh.write(json.dumps(fixed, ensure_ascii=False) + "\n")
                if salvage_fh:
                    salvage_fh.write(json.dumps({"from": d.get("reason"), "row": fixed}, ensure_ascii=False) + "\n")
                n_re_pass += 1
            else:
                if still_bad_fh:
                    still_bad_fh.write(json.dumps({"reason": reason, "row": fixed}, ensure_ascii=False) + "\n")

    cleaned_fh.close()
    if salvage_fh:
        salvage_fh.close()
    if still_bad_fh:
        still_bad_fh.close()

    print(
        f"[salvage] rejects_in={n_in:,}  had_prefix={n_prefix_ok:,}  re_passed={n_re_pass:,}"
        f"  yield={n_re_pass/max(n_in,1)*100:.1f}%"
    )


if __name__ == "__main__":
    main()
