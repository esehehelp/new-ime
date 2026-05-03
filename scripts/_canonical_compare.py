"""WSL canonical 9-model bench output と過去 benchmark_comparison.md
(2026-04-22) アンカーの差分テーブル。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]

# (model name, bench) -> (EM1, EM5, CharAcc, p50_ms) 過去 doc 値
ANCH = {
    ("suiko-v1-small-greedy",    "probe_v3"):    (0.601, 0.601, 0.944,    9),
    ("suiko-v1-small-greedy",    "ajimee_jwtd"): (0.580, 0.580, 0.951,   10),
    ("suiko-v1-small-kenlm",     "probe_v3"):    (0.664, 0.776, 0.947,   17),
    ("suiko-v1-small-kenlm",     "ajimee_jwtd"): (0.670, 0.830, 0.959,   21),
    ("suiko-v1-small-kenlm-moe", "probe_v3"):    (0.672, 0.784, 0.949,   22),
    ("suiko-v1-small-kenlm-moe", "ajimee_jwtd"): (0.670, 0.820, 0.959,   28),
    ("zenz-v2.5-xsmall",         "probe_v3"):    (0.695, 0.813, 0.953,  118),
    ("zenz-v2.5-xsmall",         "ajimee_jwtd"): (0.695, 0.845, 0.953,  139),
    ("zenz-v2.5-small",          "probe_v3"):    (0.713, 0.848, 0.959,  376),
    ("zenz-v2.5-small",          "ajimee_jwtd"): (0.840, 0.955, 0.977,  418),
    ("zenz-v2.5-medium",         "probe_v3"):    (0.747, 0.876, 0.966, 1173),
    ("zenz-v2.5-medium",         "ajimee_jwtd"): (0.875, 0.970, 0.982, 1361),
    ("zenz-v3.1-small",          "probe_v3"):    (0.718, 0.856, 0.959,  417),
    ("zenz-v3.1-small",          "ajimee_jwtd"): (0.860, 0.930, 0.983,  470),
    ("jinen-v1-xsmall",          "probe_v3"):    (0.609, 0.747, 0.929,  115),
    ("jinen-v1-xsmall",          "ajimee_jwtd"): (0.395, 0.525, 0.917,  124),
    ("jinen-v1-small",           "probe_v3"):    (0.672, 0.776, 0.944,  278),
    ("jinen-v1-small",           "ajimee_jwtd"): (0.655, 0.835, 0.952,  309),
}

ORDER = [
    "suiko-v1-small-greedy",
    "suiko-v1-small-kenlm",
    "suiko-v1-small-kenlm-moe",
    "zenz-v2.5-xsmall",
    "zenz-v2.5-small",
    "zenz-v2.5-medium",
    "zenz-v3.1-small",
    "jinen-v1-xsmall",
    "jinen-v1-small",
]

H = ("config", "bench", "EM1", "(a)", "ΔEM1", "EM5", "(a)", "ΔEM5",
     "CAcc", "(a)", "ΔCAcc", "p50", "(a)", "Δp50")
print(f"{H[0]:<26s} {H[1]:<13s} {H[2]:>6s} {H[3]:>6s} {H[4]:>7s} "
      f"{H[5]:>6s} {H[6]:>6s} {H[7]:>7s} {H[8]:>6s} {H[9]:>6s} {H[10]:>7s} "
      f"{H[11]:>6s} {H[12]:>5s} {H[13]:>7s}")
print("-" * 130)

worst = {"em1": (None, 0.0), "em5": (None, 0.0), "char_acc": (None, 0.0), "p50": (None, 0.0)}

for name in ORDER:
    p = REPO / f"results/bench/{name}/summary.json"
    if not p.is_file():
        print(f"{name:<26s}  (not found: {p})")
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
        for k, d in (("em1", d_em1), ("em5", d_em5), ("char_acc", d_ca), ("p50", d_p50)):
            if abs(d) > abs(worst[k][1]):
                worst[k] = ((name, b), d)
        print(f"{name:<26s} {b:<13s} "
              f"{em1:>6.4f} {a_em1:>6.3f} {d_em1:>+7.4f} "
              f"{em5:>6.4f} {a_em5:>6.3f} {d_em5:>+7.4f} "
              f"{ca:>6.4f} {a_ca:>6.3f} {d_ca:>+7.4f} "
              f"{p50:>6.1f} {a_p50:>5d} {d_p50:>+7.1f}")

print()
print("=== worst delta ===")
for k, (loc, d) in worst.items():
    if loc is None:
        continue
    sign = "+" if d >= 0 else "-"
    print(f"  {k:<8s} {sign}{abs(d):.4f}  at {loc[0]}/{loc[1]}")
