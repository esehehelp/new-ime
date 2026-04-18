"""Auto-sweep α/β for interactive_ctc.exe across a domain-diverse probe
set and report the globally-best configuration.

The probe TSV at datasets/probe_v1/probe.tsv carries
    category<TAB>reading<TAB>expected
lines across general / tech / medical / legal / names / homophone / edge
categories. For each (α, β) combination in the sweep, we drive
interactive_ctc.exe as a subprocess — probe inputs piped to stdin, the
top-3 candidates and per-line latency parsed from stdout — then
compute:

    top1_em:  top-1 surface == expected (strict EM, per category)
    top3_em:  expected in top-3
    p50_ms / p95_ms: per-probe latency

A markdown + JSON report is emitted under results/sweep_{timestamp}/.
The "best" is picked by weighted overall: mean of per-category top1_em
(so a tiny edge-case category doesn't dominate), then ties broken by
top3_em, then by -p95_ms.

Usage:
    uv run python -m scripts.sweep_interactive_ctc \
        --exe build/win32/interactive_ctc.exe \
        --onnx models/ctc_nat_30m_best_latest.onnx \
        --lm   models/kenlm_eval_v3_train_4gram_probing.bin \
        --dict "models/user_dict.tsv,models/fixed_dict_mozc.tsv,models/fixed_dict_mozc_ut.tsv" \
        --probe datasets/probe_v1/probe.tsv \
        --alphas 0.0,0.2,0.4,0.6,0.8 \
        --betas  0.0,0.3,0.6,1.0 \
        --beam 8
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


CAND_RE = re.compile(r"^\s+\[(\d)\]\s+(.+?)\s+\(([-\d\.e]+)\)\s*$")
TIME_RE = re.compile(r"^\s+\[time\]\s+total=([\d\.]+)ms")


def load_probe(path: Path) -> list[dict]:
    """Parse category<TAB>reading<TAB>expected lines; skip # comments."""
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 3:
            continue
        items.append({"category": parts[0], "reading": parts[1], "expected": parts[2]})
    return items


def run_probe_batch(args, alpha: float, beta: float, probe: list[dict]) -> list[dict]:
    """Run interactive_ctc.exe over the whole probe set in one subprocess.

    Each probe input is written as a line to stdin; the subprocess emits
    the standard interactive UI. We then split stdout on the '>' prompt
    separator and pull the candidate / latency lines per segment. The
    trailing empty line in stdin closes the loop.
    """
    cmd = [
        args.exe,
        "--onnx", args.onnx,
        "--lm", args.lm,
        "--dict", args.dict,
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--beam", str(args.beam),
    ]
    stdin_blob = "\n".join(item["reading"] for item in probe) + "\n\n"
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=stdin_blob,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"  [warn] interactive_ctc exit={proc.returncode}")
        if proc.stderr:
            print(f"  [warn] stderr: {proc.stderr[:200]}")

    # Parse: segments delimited by "> " prompt. First segment = header.
    segments = proc.stdout.split("\n>")
    # Ensure we have at least len(probe) candidate segments; the first
    # one is preamble before the first "> " prompt.
    results = []
    # Iterate the probe in order; each iteration consumes one segment.
    for i, item in enumerate(probe):
        seg = segments[i + 1] if i + 1 < len(segments) else ""
        cands: list[tuple[int, str, float]] = []
        latency = 0.0
        for line in seg.splitlines():
            m = CAND_RE.match(line)
            if m:
                cands.append((int(m.group(1)), m.group(2), float(m.group(3))))
                continue
            m2 = TIME_RE.match(line)
            if m2:
                latency = float(m2.group(1))
        cands.sort(key=lambda c: c[0])
        top_texts = [c[1] for c in cands]
        hit1 = bool(top_texts) and top_texts[0] == item["expected"]
        hit3 = item["expected"] in top_texts[:3]
        results.append({
            "category": item["category"],
            "reading": item["reading"],
            "expected": item["expected"],
            "top1": top_texts[0] if top_texts else "",
            "top3": top_texts[:3],
            "hit1": hit1,
            "hit3": hit3,
            "latency_ms": latency,
        })
    return results


def summarise(results: list[dict]) -> dict:
    """Per-category and overall aggregate."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    def agg(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0, "top1_em": 0.0, "top3_em": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        n = len(rows)
        lat = sorted(r["latency_ms"] for r in rows)
        return {
            "n": n,
            "top1_em": sum(r["hit1"] for r in rows) / n,
            "top3_em": sum(r["hit3"] for r in rows) / n,
            "p50_ms": lat[n // 2],
            "p95_ms": lat[min(n - 1, int(0.95 * n))],
        }

    out = {"overall": agg(results)}
    out["per_category"] = {cat: agg(rows) for cat, rows in by_cat.items()}
    # Weighted-by-category overall: mean of per-cat top1_em so small
    # categories don't get buried under the dominant one.
    cat_top1s = [v["top1_em"] for v in out["per_category"].values()]
    out["macro_top1_em"] = sum(cat_top1s) / len(cat_top1s) if cat_top1s else 0.0
    cat_top3s = [v["top3_em"] for v in out["per_category"].values()]
    out["macro_top3_em"] = sum(cat_top3s) / len(cat_top3s) if cat_top3s else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default="build/win32/interactive_ctc.exe")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--lm", required=True)
    parser.add_argument("--dict", default="")
    parser.add_argument("--probe", default="datasets/probe_v1/probe.tsv")
    parser.add_argument("--alphas", default="0.0,0.2,0.4,0.6,0.8")
    parser.add_argument("--betas", default="0.0,0.3,0.6,1.0")
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    probe = load_probe(Path(args.probe))
    print(f"probe: {len(probe)} items, {len({p['category'] for p in probe})} categories")
    print(f"alphas: {alphas}")
    print(f"betas:  {betas}")
    print(f"grid:   {len(alphas) * len(betas)} configs")
    print()

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"results/sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for alpha in alphas:
        for beta in betas:
            t0 = time.perf_counter()
            results = run_probe_batch(args, alpha, beta, probe)
            s = summarise(results)
            elapsed = time.perf_counter() - t0
            print(
                f"a={alpha:.2f} b={beta:.2f}  "
                f"macro_top1={s['macro_top1_em']:.3f} "
                f"macro_top3={s['macro_top3_em']:.3f} "
                f"p50={s['overall']['p50_ms']:.0f}ms "
                f"p95={s['overall']['p95_ms']:.0f}ms  "
                f"took {elapsed:.1f}s",
                flush=True,
            )
            row = {"alpha": alpha, "beta": beta, **s, "results": results}
            all_rows.append(row)
            # Save per-config detail for offline inspection.
            per_path = out_dir / f"alpha{alpha:.2f}_beta{beta:.2f}.json"
            per_path.write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # Pick best by (macro_top1, macro_top3, -p95).
    best = max(
        all_rows,
        key=lambda r: (
            r["macro_top1_em"],
            r["macro_top3_em"],
            -r["overall"]["p95_ms"],
        ),
    )
    print()
    print(
        f"BEST: a={best['alpha']:.2f} b={best['beta']:.2f}  "
        f"macro_top1={best['macro_top1_em']:.3f} "
        f"macro_top3={best['macro_top3_em']:.3f} "
        f"p50={best['overall']['p50_ms']:.0f}ms "
        f"p95={best['overall']['p95_ms']:.0f}ms"
    )

    # Markdown report.
    md_lines = []
    md_lines.append(f"# α/β sweep — {time.strftime('%Y-%m-%d %H:%M')}")
    md_lines.append("")
    md_lines.append(f"- Probe: `{args.probe}` ({len(probe)} items)")
    md_lines.append(f"- Model: `{args.onnx}`")
    md_lines.append(f"- LM:    `{args.lm}`")
    md_lines.append(f"- Dict:  `{args.dict}`")
    md_lines.append(f"- Beam:  {args.beam}")
    md_lines.append("")
    md_lines.append("## Grid (macro top-1 EM — mean across categories)")
    md_lines.append("")
    header = "| β \\\\ α | " + " | ".join(f"{a:.2f}" for a in alphas) + " |"
    sep = "|---" * (len(alphas) + 1) + "|"
    md_lines.append(header)
    md_lines.append(sep)
    grid: dict[tuple[float, float], dict] = {(r["alpha"], r["beta"]): r for r in all_rows}
    for b in betas:
        cells = [f"{b:.2f}"]
        for a in alphas:
            r = grid[(a, b)]
            mark = "**" if r is best else ""
            cells.append(f"{mark}{r['macro_top1_em']:.3f}{mark}")
        md_lines.append("| " + " | ".join(cells) + " |")

    md_lines.append("")
    md_lines.append("## Best config detail")
    md_lines.append("")
    md_lines.append(
        f"α={best['alpha']:.2f}, β={best['beta']:.2f}, "
        f"macro top-1 {best['macro_top1_em']:.3f}, "
        f"macro top-3 {best['macro_top3_em']:.3f}"
    )
    md_lines.append("")
    md_lines.append("| Category | n | top-1 EM | top-3 EM | p50 ms | p95 ms |")
    md_lines.append("|---|---|---|---|---|---|")
    for cat, s in sorted(best["per_category"].items()):
        md_lines.append(
            f"| {cat} | {s['n']} | {s['top1_em']:.3f} | {s['top3_em']:.3f} | "
            f"{s['p50_ms']:.0f} | {s['p95_ms']:.0f} |"
        )

    # Per-category grid to see where α/β hurts vs helps.
    md_lines.append("")
    md_lines.append("## Per-category top-1 EM across grid")
    md_lines.append("")
    cats = sorted({c for r in all_rows for c in r["per_category"].keys()})
    for cat in cats:
        md_lines.append(f"### {cat}")
        md_lines.append("")
        md_lines.append(header)
        md_lines.append(sep)
        for b in betas:
            cells = [f"{b:.2f}"]
            for a in alphas:
                v = grid[(a, b)]["per_category"][cat]["top1_em"]
                cells.append(f"{v:.2f}")
            md_lines.append("| " + " | ".join(cells) + " |")
        md_lines.append("")

    # Top failures from best config — surface which inputs still miss.
    md_lines.append("## Best-config failures (top-1 miss) by category")
    md_lines.append("")
    fails_by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in best["results"]:
        if not r["hit1"]:
            fails_by_cat[r["category"]].append(r)
    for cat, rows in sorted(fails_by_cat.items()):
        md_lines.append(f"### {cat} ({len(rows)} misses)")
        md_lines.append("")
        md_lines.append("| reading | expected | top-1 | top-3 hit |")
        md_lines.append("|---|---|---|---|")
        for r in rows[:20]:
            top3 = " / ".join(r["top3"][:3])
            hit = "yes" if r["hit3"] else "no"
            md_lines.append(f"| {r['reading']} | {r['expected']} | {r['top1']} | {hit} |")
        md_lines.append("")

    (out_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")
    summary_json = {
        "grid": [
            {k: v for k, v in r.items() if k != "results"}
            for r in all_rows
        ],
        "best": {
            "alpha": best["alpha"],
            "beta": best["beta"],
            "macro_top1_em": best["macro_top1_em"],
            "macro_top3_em": best["macro_top3_em"],
            "overall": best["overall"],
            "per_category": best["per_category"],
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nreport: {out_dir / 'report.md'}")
    print(f"json:   {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
