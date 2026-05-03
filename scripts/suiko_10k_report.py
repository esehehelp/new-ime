"""Summarize Suiko train logs against the 10k dev EM target."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EVAL_RE = re.compile(
    r"^\[eval (?P<step>\d+)\]\s+loss=(?P<loss>[0-9.]+)\s+"
    r"EM=(?P<em>[0-9.]+)\s+CharAcc=(?P<char>[0-9.]+)"
)


def parse_log(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    if not path.exists():
        return rows
    text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    for line in text.splitlines():
        match = EVAL_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "em": float(match.group("em")),
                "char": float(match.group("char")),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="models/checkpoints",
        help="Checkpoint root containing Suiko-* run directories.",
    )
    parser.add_argument("--target-step", type=int, default=10000)
    parser.add_argument("--target-em", type=float, default=0.15)
    args = parser.parse_args()

    root = Path(args.root)
    summaries = []
    for log in sorted(root.glob("Suiko-*/train.log")):
        rows = parse_log(log)
        if not rows:
            continue
        target_rows = [row for row in rows if row["step"] == args.target_step]
        best_to_target = max(
            (row for row in rows if row["step"] <= args.target_step),
            key=lambda row: row["em"],
            default=None,
        )
        latest = rows[-1]
        summaries.append(
            {
                "name": log.parent.name,
                "at_target": target_rows[-1] if target_rows else None,
                "best_to_target": best_to_target,
                "latest": latest,
            }
        )

    summaries.sort(
        key=lambda item: (
            item["at_target"]["em"] if item["at_target"] else -1.0,
            item["best_to_target"]["em"] if item["best_to_target"] else -1.0,
        ),
        reverse=True,
    )

    print(f"{'run':<34} {'EM@10k':>8} {'best<=10k':>10} {'latest':>14} status")
    for item in summaries:
        at = item["at_target"]
        best = item["best_to_target"]
        latest = item["latest"]
        em10 = at["em"] if at else None
        best_s = f"{best['em']:.4f}@{best['step']}" if best else "-"
        latest_s = f"{latest['em']:.4f}@{latest['step']}"
        status = "PASS" if em10 is not None and em10 > args.target_em else "FAIL"
        em10_s = f"{em10:.4f}" if em10 is not None else "-"
        print(f"{item['name']:<34} {em10_s:>8} {best_s:>10} {latest_s:>14} {status}")


if __name__ == "__main__":
    main()
