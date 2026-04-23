"""Audit JSONL dataset quality by detecting known error patterns.

Runs multiple heuristic checks on a sample and reports:
- Counts of each issue type
- Example lines for each issue
- Overall quality score

Usage:
    uv run python scripts/audit_data.py --input /tmp/aozora_audit.jsonl --examples 5
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict


# ---- Audit checks ----
# Each returns (issue_name, detail) or None

def check_old_orthography(r: str, s: str) -> tuple[str, str] | None:
    """Residual old orthography that slipped through filters."""
    # ぢ/づ in reading where じ/ず expected (context-dependent, hard to filter)
    # But some ぢ/づ are valid (e.g. 鼻血=はなぢ). Skip this.

    # Old-style verb endings in surface: ～ぬ (archaic negative)
    if re.search(r"[らかさたなまわ]ぬ[。、」\s]", s):
        # Could be valid (死ぬ, etc.), only flag if reading also has archaic form
        pass

    # せう/ませう that wasn't caught
    if re.search(r"でせう|ませう", s):
        return ("old_ortho_residual", f"せう/ませう in: {s[:50]}")

    # つた/つて not caught (check surface more carefully)
    if re.search(r"[かきくけこさしすせそたちつてとなにぬねの"
                  r"はひふへほまみむめもやゆよらりるれろわ]つ[たてだで]", s):
        return ("old_ortho_residual", f"historical sokuon in: {s[:50]}")

    return None


def check_reading_surface_mismatch(r: str, s: str) -> tuple[str, str] | None:
    """Reading doesn't plausibly correspond to surface."""
    # Surface has katakana but reading has no corresponding hiragana
    kata_in_surface = re.findall(r"[\u30a1-\u30fa]+", s)
    if kata_in_surface:
        longest_kata = max(kata_in_surface, key=len)
        if len(longest_kata) >= 3:
            # The hiragana equivalent should appear in reading
            import jaconv
            hira_equiv = jaconv.kata2hira(longest_kata)
            if hira_equiv not in r:
                return ("reading_mismatch", f"kata '{longest_kata}' not in reading: {r[:40]}")

    return None


def check_incomplete_sentence(r: str, s: str) -> tuple[str, str] | None:
    """Sentence fragment that isn't a real sentence."""
    # Ends with comma or no punctuation and is short
    if len(s) < 15 and not re.search(r"[。！？」）]$", s):
        return ("incomplete_fragment", s)

    # Starts with particle/conjunction that suggests a fragment
    if re.match(r"^[をにへとでがはもの、]", s) and len(s) < 20:
        return ("fragment_starts_with_particle", s)

    return None


def check_garbled_reading(r: str, s: str) -> tuple[str, str] | None:
    """Reading contains patterns that suggest MeCab errors."""
    # Very long runs without any kanji conversion in surface
    # (reading and surface nearly identical = no kanji, pointless pair)
    if len(s) > 20:
        kanji_count = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
        kata_count = sum(1 for c in s if "\u30a1" <= c <= "\u30fa")
        if kanji_count == 0 and kata_count == 0 and len(s) > 30:
            return ("no_kanji_long", f"all-hiragana surface ({len(s)} chars): {s[:40]}")

    return None


def check_punctuation_only(r: str, s: str) -> tuple[str, str] | None:
    """Mostly punctuation/symbols."""
    content = re.sub(r"[、。！？「」『』（）・…―─　\s「」\n]", "", s)
    if len(content) < 3:
        return ("mostly_punctuation", s)
    return None


def check_context_quality(ctx: str) -> tuple[str, str] | None:
    """Context field has issues."""
    if ctx and len(ctx) < 5:
        return ("short_context", f"context='{ctx}'")
    return None


ALL_CHECKS = [
    check_old_orthography,
    check_reading_surface_mismatch,
    check_incomplete_sentence,
    check_garbled_reading,
    check_punctuation_only,
]


def audit(input_path: str, num_examples: int = 5) -> None:
    issue_counts: Counter = Counter()
    issue_examples: defaultdict[str, list[str]] = defaultdict(list)
    total = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            total += 1
            pair = json.loads(line)
            r = pair["reading"]
            s = pair["surface"]
            ctx = pair.get("context", "")

            for check_fn in ALL_CHECKS:
                result = check_fn(r, s)
                if result:
                    issue_name, detail = result
                    issue_counts[issue_name] += 1
                    if len(issue_examples[issue_name]) < num_examples:
                        issue_examples[issue_name].append(detail)

            # Context check
            ctx_result = check_context_quality(ctx)
            if ctx_result:
                issue_counts[ctx_result[0]] += 1
                if len(issue_examples[ctx_result[0]]) < num_examples:
                    issue_examples[ctx_result[0]].append(ctx_result[1])

    print(f"=== Audit Report: {input_path} ===")
    print(f"Total samples: {total:,}")
    print()

    if not issue_counts:
        print("No issues found!")
    else:
        print("Issues found:")
        for issue, count in issue_counts.most_common():
            pct = count / total * 100
            print(f"\n  [{issue}] {count:,} ({pct:.2f}%)")
            for ex in issue_examples[issue]:
                print(f"    - {ex}")

    clean_count = total - sum(issue_counts.values())
    print(f"\nClean: {clean_count:,} / {total:,} ({clean_count/total*100:.1f}%)")
    print(f"Issues: {sum(issue_counts.values()):,} / {total:,} "
          f"({sum(issue_counts.values())/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Audit dataset quality")
    parser.add_argument("--input", required=True, help="JSONL file to audit")
    parser.add_argument("--examples", type=int, default=5, help="Examples per issue")
    args = parser.parse_args()
    audit(args.input, args.examples)


if __name__ == "__main__":
    main()
