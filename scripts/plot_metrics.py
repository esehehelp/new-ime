"""Plot training metrics curves from one or more `metrics.jsonl` files or
raw `train.log` files (post-hoc parse).

Usage:
    uv run --with matplotlib python scripts/plot_metrics.py \
        --run "v2.1=checkpoints/suiko-v2.1-small/metrics.jsonl" \
        --logfile "v2=checkpoints/suiko-v2-small/train.log.local" \
        --logfile "v1.17=logs/suiko-v1.17-small/train.log" \
        --out results/learning_curve.png

Each run is plotted in 4 subplots (probe_em1, eval_em1, eval_loss, eval_blank).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

EVAL_RE = re.compile(
    r"\[eval\] step=(\d+) loss=([\d.eE+-]+) EM1=([\d.eE+-]+) "
    r"charAcc=([\d.eE+-]+) blank=([\d.eE+-]+) n=(\d+)"
)
PROBE_RE = re.compile(r"\[probe\] step=(\d+) EM1=([\d.eE+-]+) n=(\d+)")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda r: r["step"])
    return rows


def parse_log(path: Path) -> list[dict]:
    rows_by_step: dict[int, dict] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for m in EVAL_RE.finditer(text):
        step = int(m.group(1))
        rows_by_step.setdefault(step, {"step": step})
        rows_by_step[step].update(
            {
                "eval_loss": float(m.group(2)),
                "eval_em1": float(m.group(3)),
                "eval_charAcc": float(m.group(4)),
                "eval_blank": float(m.group(5)),
                "eval_n": int(m.group(6)),
            }
        )
    for m in PROBE_RE.finditer(text):
        step = int(m.group(1))
        rows_by_step.setdefault(step, {"step": step})
        rows_by_step[step]["probe_em1"] = float(m.group(2))
    return sorted(rows_by_step.values(), key=lambda r: r["step"])


def parse_run_arg(arg: str) -> tuple[str, Path]:
    name, _, path_s = arg.partition("=")
    if not path_s:
        raise SystemExit(f"--run/--logfile expects NAME=PATH, got {arg!r}")
    return name, Path(path_s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", default=[], help="NAME=PATH (jsonl)")
    p.add_argument(
        "--logfile", action="append", default=[], help="NAME=PATH (raw train.log)"
    )
    p.add_argument("--out", default="results/learning_curve.png", type=Path)
    p.add_argument("--xaxis", choices=["step", "wall_s"], default="step")
    args = p.parse_args(argv)

    runs: list[tuple[str, list[dict]]] = []
    for arg in args.run:
        name, path = parse_run_arg(arg)
        if not path.exists():
            print(f"[warn] missing: {path}", file=sys.stderr)
            continue
        runs.append((name, load_jsonl(path)))
    for arg in args.logfile:
        name, path = parse_run_arg(arg)
        if not path.exists():
            print(f"[warn] missing: {path}", file=sys.stderr)
            continue
        runs.append((name, parse_log(path)))

    if not runs:
        print("no runs given", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    panels = [
        ("probe_em1", "probe EM1 (n=348)"),
        ("eval_em1", "eval EM1 (n=1024)"),
        ("eval_loss", "eval loss"),
        ("eval_blank", "eval blank fraction"),
    ]
    for ax, (key, title) in zip(axes.flatten(), panels):
        for name, rows in runs:
            xs = [r.get(args.xaxis, r.get("step")) for r in rows if key in r]
            ys = [r[key] for r in rows if key in r]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.0, label=name)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("wall_s" if args.xaxis == "wall_s" else "step")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
