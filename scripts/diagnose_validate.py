"""Quick diagnostic: how many probe_v3 references fail validate_candidate?

For each item, loads (input=reading, expected_output=references) and asks the
Rust engine's `validate_candidate` (via PyO3? we don't have a binding, so we
re-implement the rule in Python here matching validate.rs semantics).

Reports per-category and total fail rate so we know how aggressively the
filter rejects ground-truth candidates.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def char_kind(c: str) -> str:
    o = ord(c)
    if 0x3041 <= o <= 0x3096:
        return "hira"
    if 0x30A1 <= o <= 0x30FA:
        return "kata"
    if o == 0x30FC:
        return "neutral"
    if (
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0xF900 <= o <= 0xFAFF
        or 0x20000 <= o <= 0x2A6DF
    ):
        return "kanji"
    return "other"


def kata_to_hira(s: str) -> str:
    out = []
    for c in s:
        o = ord(c)
        if 0x30A1 <= o <= 0x30F6:
            out.append(chr(o - 0x60))
        else:
            out.append(c)
    return "".join(out)


def split_runs(s: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    buf = ""
    group = None
    for c in s:
        k = char_kind(c)
        ng = (
            "kanji" if k == "kanji"
            else "kana" if k in ("hira", "kata", "neutral")
            else "other"
        )
        if group is not None and group != ng:
            runs.append((group, buf))
            buf = ""
        group = ng
        buf += c
    if group is not None and buf:
        runs.append((group, buf))
    return runs


def find_subseq(haystack: list[str], needle: list[str]) -> int | None:
    if not needle:
        return 0
    n = len(needle)
    if n > len(haystack):
        return None
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def validate_from(runs, i, input_chars, cursor) -> tuple[bool, str]:
    if i >= len(runs):
        if cursor == len(input_chars):
            return True, "ok"
        return False, f"unconsumed input: cursor={cursor} len={len(input_chars)}"
    kind, text = runs[i]
    if kind == "kana":
        normalized = list(kata_to_hira(text))
        n = len(normalized)
        if cursor + n > len(input_chars):
            return False, f"kana run {text!r} overflows input"
        if input_chars[cursor : cursor + n] != normalized:
            return (
                False,
                f"kana run {text!r} != input[{cursor}:{cursor+n}]={input_chars[cursor:cursor+n]}",
            )
        return validate_from(runs, i + 1, input_chars, cursor + n)
    # kanji or other
    j = next((k for k in range(i + 1, len(runs)) if runs[k][0] == "kana"), None)
    if j is not None:
        normalized = list(kata_to_hira(runs[j][1]))
        n = len(normalized)
        block_has_kanji = any(runs[k][0] == "kanji" for k in range(i, j))
        min_p = 1 if block_has_kanji else 0
        last_reason = "no anchor pos worked"
        start = cursor + min_p
        while start + n <= len(input_chars):
            if input_chars[start : start + n] == normalized:
                ok, reason = validate_from(runs, j + 1, input_chars, start + n)
                if ok:
                    return True, "ok"
                last_reason = reason
            start += 1
        return (
            False,
            f"block {''.join(r[1] for r in runs[i:j])!r} → next_kana {runs[j][1]!r}: {last_reason}",
        )
    # trailing
    block_has_kanji = any(runs[k][0] == "kanji" for k in range(i, len(runs)))
    if block_has_kanji:
        if cursor >= len(input_chars):
            return False, f"trailing kanji block has no input chars left"
        return True, "ok"
    if cursor == len(input_chars):
        return True, "ok"
    return False, f"trailing pure-other but {len(input_chars)-cursor} input chars unconsumed"


def validate(reading: str, candidate: str) -> tuple[bool, str]:
    runs = split_runs(candidate)
    input_chars = list(kata_to_hira(reading))
    return validate_from(runs, 0, input_chars, 0)


def main():
    probe_path = Path("datasets/eval/probe/probe.json")
    items = json.loads(probe_path.read_text(encoding="utf-8"))

    cat_total: dict[str, int] = defaultdict(int)
    cat_fail: dict[str, int] = defaultdict(int)
    failures: list[tuple[str, str, str, str]] = []

    for item in items:
        cat = item.get("category", "?")
        reading = item["input"]
        for ref in item["expected_output"]:
            cat_total[cat] += 1
            ok, reason = validate(reading, ref)
            if not ok:
                cat_fail[cat] += 1
                if len(failures) < 30:
                    failures.append((cat, reading, ref, reason))

    print("=== probe_v3 reference validation rate (by category) ===")
    print(f"{'category':<12} {'total':>6} {'fail':>6} {'fail_rate':>10}")
    total_t = total_f = 0
    for cat in sorted(cat_total):
        t = cat_total[cat]
        f = cat_fail[cat]
        total_t += t
        total_f += f
        print(f"{cat:<12} {t:>6} {f:>6} {f/max(t,1):>10.3f}")
    print(f"{'OVERALL':<12} {total_t:>6} {total_f:>6} {total_f/max(total_t,1):>10.3f}")

    print("\n=== sample failures (first 30) ===")
    for cat, reading, ref, reason in failures:
        print(f"  [{cat}] reading={reading!r} ref={ref!r}\n    -> {reason}")


if __name__ == "__main__":
    main()
