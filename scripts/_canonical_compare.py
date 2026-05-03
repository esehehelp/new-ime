"""Compare WSL canonical 3-variant bench output to past
benchmark_comparison.md anchor numbers."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]

# (model, bench) -> (EM1, EM5, CharAcc, p50_ms) from past doc 2026-04-22
ANCH = {
    ("suiko-v1-small-greedy",    "probe_v3"):    (0.601, 0.601, 0.944,  9),
    ("suiko-v1-small-greedy",    "ajimee_jwtd"): (0.580, 0.580, 0.951, 10),
    ("suiko-v1-small-kenlm",     "probe_v3"):    (0.664, 0.776, 0.947, 17),
    ("suiko-v1-small-kenlm",     "ajimee_jwtd"): (0.670, 0.830, 0.959, 21),
    ("suiko-v1-small-kenlm-moe", "probe_v3"):    (0.672, 0.784, 0.949, 22),
    ("suiko-v1-small-kenlm-moe", "ajimee_jwtd"): (0.670, 0.820, 0.959, 28),
}

H = ("config", "bench", "EM1", "(a)", "ΔEM1", "EM5", "(a)", "ΔEM5",
     "CAcc", "(a)", "ΔCAcc", "p50", "(a)", "Δp50")
print(f"{H[0]:<28s} {H[1]:<14s} {H[2]:>6s} {H[3]:>6s} {H[4]:>6s} "
      f"{H[5]:>6s} {H[6]:>6s} {H[7]:>6s} {H[8]:>6s} {H[9]:>6s} {H[10]:>6s} "
      f"{H[11]:>5s} {H[12]:>4s} {H[13]:>5s}")
print("-" * 130)

for name in ("suiko-v1-small-greedy", "suiko-v1-small-kenlm",
             "suiko-v1-small-kenlm-moe"):
    p = REPO / f"results/bench/{name}/summary.json"
    if not p.is_file():
        continue
    rows = json.loads(p.read_text(encoding="utf-8"))
    for row in rows:
        b = row["bench"]
        a = ANCH.get((name, b))
        if a is None:
            continue
        em1, em5, ca, p50 = row["em1"], row["em5"], row["char_acc"], row["p50_ms"]
        a_em1, a_em5, a_ca, a_p50 = a
        d_em1 = em1 - a_em1
        d_em5 = em5 - a_em5
        d_ca = ca - a_ca
        d_p50 = p50 - a_p50
        print(f"{name:<28s} {b:<14s} "
              f"{em1:>6.4f} {a_em1:>6.3f} {d_em1:>+6.4f} "
              f"{em5:>6.4f} {a_em5:>6.3f} {d_em5:>+6.4f} "
              f"{ca:>6.4f} {a_ca:>6.3f} {d_ca:>+6.4f} "
              f"{p50:>5.1f} {a_p50:>4d} {d_p50:>+5.1f}")
