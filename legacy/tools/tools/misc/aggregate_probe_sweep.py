"""Aggregate 30m-student probe sweep results.

Outputs:
  1) Flat table: every (step, config) row
  2) Best config per step
  3) No-LM vs best-LM delta
  4) Per-category analysis: which categories benefit from KenLM, which regress
  5) Step x category trajectory (greedy only) — training dynamics per category
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("results/probe_v3_30m_student_120_140k")
STEPS = [120000, 125000, 130000, 135000, 140000, 160000]
CONFIGS = [
    ("greedy", "greedy"),
    ("beam5_nolm", "beam5_nolm"),
    ("a0.2_b0.3", "LM a=0.2 b=0.3"),
    ("a0.2_b0.6", "LM a=0.2 b=0.6"),
    ("a0.4_b0.3", "LM a=0.4 b=0.3"),
    ("a0.4_b0.6", "LM a=0.4 b=0.6"),
    ("a0.6_b0.3", "LM a=0.6 b=0.3"),
    ("a0.6_b0.6", "LM a=0.6 b=0.6"),
]
CATS = ["edge", "general", "homophone", "names", "numeric", "particle", "tech"]
LM_CONFIGS = [c for c, _ in CONFIGS if c.startswith("a")]


def load(step: int, cfg_dir: str) -> dict | None:
    p = ROOT / f"step{step}" / cfg_dir / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if not d:
        return None
    k = next(iter(d))
    return d[k]


def cat_em1(r: dict, cat: str) -> float | None:
    v = r.get("per_category", {}).get(cat, {}).get("exact_match_top1", None)
    return v if isinstance(v, (int, float)) else None


def flat_table(rows_out):
    header = ["step", "config", "EM1", "EM5", "CharAcc"] + CATS + ["p50ms"]
    rows = [header]
    for step in STEPS:
        for cfg_dir, label in CONFIGS:
            r = load(step, cfg_dir)
            if r is None:
                rows.append([str(step), label, "-"] + [""] * (len(header) - 3))
                continue
            row = [
                str(step), label,
                f"{r.get('em1', 0):.3f}",
                f"{r.get('em5', 0):.3f}",
                f"{r.get('char_acc', 0):.3f}",
            ]
            for c in CATS:
                v = cat_em1(r, c)
                row.append(f"{v:.3f}" if v is not None else "-")
            row.append(str(r.get("latency_ms", {}).get("p50", "-")))
            rows.append(row)

    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for i, r in enumerate(rows):
        line = " | ".join(c.ljust(widths[j]) for j, c in enumerate(r))
        rows_out.append(line)
        if i == 0:
            rows_out.append("-+-".join("-" * w for w in widths))


def best_per_step():
    print("\n=== BEST CONFIG PER STEP (overall EM1) ===")
    for step in STEPS:
        best = None
        for cfg_dir, label in CONFIGS:
            r = load(step, cfg_dir)
            if r is None:
                continue
            em1 = r.get("em1", 0)
            if best is None or em1 > best[0]:
                best = (em1, label, r)
        if best:
            em1, label, r = best
            print(f"step{step}: {label:<20} EM1={em1:.3f} EM5={r['em5']:.3f} CharAcc={r['char_acc']:.3f} p50={r['latency_ms']['p50']}ms")


def nolm_vs_lm():
    print("\n=== No-LM (greedy) vs best KenLM per step ===")
    print(f"{'step':<8} {'greedy':<8} {'best-LM':<8} {'delta':<8} {'best cfg'}")
    for step in STEPS:
        g = load(step, "greedy")
        best_lm = None
        for cfg_dir in LM_CONFIGS:
            r = load(step, cfg_dir)
            if r is None:
                continue
            if best_lm is None or r["em1"] > best_lm[0]:
                best_lm = (r["em1"], cfg_dir, r)
        if g and best_lm:
            delta = best_lm[0] - g["em1"]
            print(f"{step:<8} {g['em1']:<8.3f} {best_lm[0]:<8.3f} {delta:+.3f}  {best_lm[1]}")


def per_category_lm_effect():
    """For each category, show greedy vs best-LM (per step). Then avg delta by category."""
    print("\n=== PER-CATEGORY: greedy EM1 vs best-LM EM1 (avg over steps) ===")
    print(f"{'category':<10} {'n':<5} {'greedy μ':<10} {'bestLM μ':<10} {'delta μ':<10} {'max hurt':<18} {'max help':<18}")
    for cat in CATS:
        g_vals, lm_vals, deltas, n = [], [], [], None
        worst_hurt = (0, None)   # (delta, step) — most negative
        best_help = (0, None)    # most positive
        for step in STEPS:
            g = load(step, "greedy")
            if g is None:
                continue
            g_em = cat_em1(g, cat)
            if g_em is None:
                continue
            # best LM for this cat at this step
            best_lm = None
            for cfg_dir in LM_CONFIGS:
                r = load(step, cfg_dir)
                if r is None:
                    continue
                v = cat_em1(r, cat)
                if v is None:
                    continue
                if best_lm is None or v > best_lm[0]:
                    best_lm = (v, cfg_dir)
            if best_lm is None:
                continue
            d = best_lm[0] - g_em
            g_vals.append(g_em)
            lm_vals.append(best_lm[0])
            deltas.append(d)
            if d < worst_hurt[0]:
                worst_hurt = (d, step)
            if d > best_help[0]:
                best_help = (d, step)
            n = g.get("per_category", {}).get(cat, {}).get("n", "?")
        if not deltas:
            continue
        gm = sum(g_vals) / len(g_vals)
        lm = sum(lm_vals) / len(lm_vals)
        dm = sum(deltas) / len(deltas)
        print(f"{cat:<10} {str(n):<5} {gm:<10.3f} {lm:<10.3f} {dm:<+10.3f} "
              f"{str(worst_hurt[1] or '-')+' ('+f'{worst_hurt[0]:+.3f}'+')':<18} "
              f"{str(best_help[1] or '-')+' ('+f'{best_help[0]:+.3f}'+')':<18}")


def per_category_alpha_sensitivity():
    """For each category, show how α ∈ {0.2, 0.4, 0.6} affects EM1 (avg over steps, β=0.6)."""
    print("\n=== PER-CATEGORY × α sensitivity (β=0.6, avg EM1 across steps) ===")
    alphas = ["0.2", "0.4", "0.6"]
    header = f"{'category':<10} " + "".join(f"greedy   α={a}    " for a in [""] + alphas)
    print(f"{'category':<10} {'greedy':<8} " + " ".join(f"α={a}:β=0.6':<10" for a in alphas))
    print(f"{'category':<10} {'greedy':<8} " + "".join(f"{'α='+a+' β=0.6':<14}" for a in alphas) + "trend")
    for cat in CATS:
        g_vals, a_vals = [], {a: [] for a in alphas}
        for step in STEPS:
            g = load(step, "greedy")
            if g:
                v = cat_em1(g, cat)
                if v is not None:
                    g_vals.append(v)
            for a in alphas:
                r = load(step, f"a{a}_b0.6")
                if r:
                    v = cat_em1(r, cat)
                    if v is not None:
                        a_vals[a].append(v)
        if not g_vals:
            continue
        gm = sum(g_vals) / len(g_vals)
        ams = {a: sum(v)/len(v) for a, v in a_vals.items() if v}
        trend_parts = [f"{ams[a]-gm:+.3f}" for a in alphas if a in ams]
        trend = " → ".join(trend_parts)
        row = f"{cat:<10} {gm:<8.3f} " + "".join(f"{ams.get(a, 0):<14.3f}" for a in alphas)
        print(row + f"Δvs greedy: " + trend)


def per_category_trajectory():
    """EM1 per category across steps (greedy). Shows training dynamics."""
    print("\n=== PER-CATEGORY TRAJECTORY (greedy, EM1 over steps) ===")
    print(f"{'category':<10} " + " ".join(f"{s//1000}k':<7" for s in STEPS))
    print(f"{'category':<10} " + " ".join(f"{str(s//1000)+'k':<8}" for s in STEPS) + "range")
    for cat in CATS:
        vals = []
        for step in STEPS:
            r = load(step, "greedy")
            v = cat_em1(r, cat) if r else None
            vals.append(v)
        cells = " ".join(f"{v:<8.3f}" if v is not None else f"{'-':<8}" for v in vals)
        numeric = [v for v in vals if v is not None]
        rng = f"[{min(numeric):.2f}-{max(numeric):.2f}] spread={max(numeric)-min(numeric):+.3f}" if numeric else ""
        print(f"{cat:<10} " + cells + rng)


def main():
    lines = []
    flat_table(lines)
    print("\n".join(lines))
    best_per_step()
    nolm_vs_lm()
    per_category_lm_effect()
    per_category_alpha_sensitivity()
    per_category_trajectory()


if __name__ == "__main__":
    main()
