"""CVAE 検証 probe 評価 runner。

現段階では CVAE 未実装のため、まずベースライン測定: 同じ reading に対し
モデルは domain 非依存で 1 つの surface を出す。per-domain EM で「デフォ
ルト出力が各 domain の期待にどれだけ合うか」を測定する。

将来 CVAE が実装されたら、backend の convert(reading, context, z) を
domain z 付きで呼ぶ版に差し替える。

評価指標:

  per-domain EM:
      各 domain ごとの EM1。baseline は各 domain に対し default surface が
      合う割合。CVAE 有効時に上がるべき。

  disagreement-rate:
      各 reading について domain ごとに期待 surface が異なっていた割合。
      これは probe 自体のメタ指標で、CVAE に学ばせる価値がある reading が
      どのくらいあるかを示す。

  domain-ceiling EM:
      もし z が perfect ならどこまで EM が上がるかの理論上限。全 probe
      item で仮に「その domain の expected」を出したと仮定した EM = 1.00
      (全 item 正解) になる。baseline との差 = CVAE が狙う gain。

Usage:
    uv run python -m scripts.run_cvae_probe \
        --probe datasets/eval/cvae_probe.tsv \
        --backend ctc_nat_90m \
        --out results/cvae_probe/baseline_90m.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from models.src.eval.ctc_nat_backend import CTCNATBackend


def load_probe(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        out.append({"domain": parts[0], "reading": parts[1],
                    "expected": parts[2]})
    return out


def meta_stats(probe: list[dict]) -> dict:
    """Compute probe-level meta stats (disagreement rate, etc)."""
    per_reading: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in probe:
        per_reading[item["reading"]].append((item["domain"], item["expected"]))

    disagree = 0
    agree_but_multi_domain = 0
    single_domain = 0
    for r, pairs in per_reading.items():
        surfaces = set(p[1] for p in pairs)
        if len(pairs) == 1:
            single_domain += 1
        elif len(surfaces) == 1:
            agree_but_multi_domain += 1
        else:
            disagree += 1

    return {
        "n_items": len(probe),
        "n_readings": len(per_reading),
        "readings_with_domain_disagreement": disagree,
        "readings_with_domain_agreement": agree_but_multi_domain,
        "readings_single_domain": single_domain,
        "disagreement_rate": round(disagree / len(per_reading), 3),
        "domain_distribution": Counter(i["domain"] for i in probe),
    }


def evaluate_baseline(backend: CTCNATBackend, probe: list[dict]) -> dict:
    """Baseline: domain 非依存で 1 つの出力、per-domain EM を計算。

    同じ reading は backend も同じ出力を返すため、一度だけ呼んで cache。
    """
    reading_to_cands: dict[str, list[str]] = {}
    for item in probe:
        r = item["reading"]
        if r not in reading_to_cands:
            # 文脈無しで呼ぶ (CVAE probe は context を入れない設計)
            try:
                reading_to_cands[r] = backend.convert(r, "")
            except Exception as e:
                reading_to_cands[r] = [f"<err:{e}>"]

    per_domain: dict[str, dict] = defaultdict(lambda: {"n": 0, "em1": 0, "em5": 0})
    results = []
    for item in probe:
        cands = reading_to_cands[item["reading"]]
        top1 = cands[0] if cands else ""
        em1 = 1 if top1 == item["expected"] else 0
        em5 = 1 if item["expected"] in cands[:5] else 0
        per_domain[item["domain"]]["n"] += 1
        per_domain[item["domain"]]["em1"] += em1
        per_domain[item["domain"]]["em5"] += em5
        results.append({
            "domain": item["domain"],
            "reading": item["reading"],
            "expected": item["expected"],
            "top1": top1,
            "top5": cands[:5],
            "em1": em1,
            "em5": em5,
        })

    for d, v in per_domain.items():
        v["em1_rate"] = round(v["em1"] / v["n"], 3)
        v["em5_rate"] = round(v["em5"] / v["n"], 3)

    n_total = len(results)
    overall_em1 = sum(r["em1"] for r in results) / n_total
    overall_em5 = sum(r["em5"] for r in results) / n_total

    return {
        "backend": backend.name,
        "overall_em1": round(overall_em1, 3),
        "overall_em5": round(overall_em5, 3),
        "per_domain": dict(per_domain),
        "n_readings_cached": len(reading_to_cands),
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="datasets/eval/cvae_probe.tsv")
    ap.add_argument("--backend", default="ctc_nat_90m",
                    choices=["ctc_nat_30m", "ctc_nat_90m"])
    ap.add_argument("--ckpt", default="",
                    help="override default checkpoint path")
    ap.add_argument("--lm-path", default="")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--out", default="results/cvae_probe/baseline.json")
    args = ap.parse_args()

    probe = load_probe(Path(args.probe))
    meta = meta_stats(probe)

    print("=== Probe meta ===")
    print(f"  items:                  {meta['n_items']}")
    print(f"  unique readings:        {meta['n_readings']}")
    print(f"  domain-disagreement:    {meta['readings_with_domain_disagreement']} "
          f"({meta['disagreement_rate']*100:.1f}%)")
    print(f"  domain-agreement:       {meta['readings_with_domain_agreement']}")
    print(f"  single-domain:          {meta['readings_single_domain']}")
    print(f"  domain dist:            {dict(meta['domain_distribution'])}")
    print()

    default_ckpts = {
        "ctc_nat_30m": "models/checkpoints/ctc_nat_30m/checkpoint_step_50000.pt",
        "ctc_nat_90m": "models/checkpoints/ctc_nat_90m/checkpoint_step_27500.pt",
    }
    ckpt = args.ckpt or default_ckpts[args.backend]

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Loading {args.backend} from {ckpt} ===")
    backend = CTCNATBackend(
        ckpt, device=device,
        beam_width=args.beam,
        lm_path=args.lm_path or None,
        lm_alpha=args.alpha,
        lm_beta=args.beta,
    )
    print(f"  {backend.name}")
    print()

    result = evaluate_baseline(backend, probe)

    print(f"=== Baseline results ===")
    print(f"  overall EM1: {result['overall_em1']:.3f}")
    print(f"  overall EM5: {result['overall_em5']:.3f}")
    print(f"  per-domain:")
    for d, v in sorted(result["per_domain"].items()):
        print(f"    {d:<12} EM1={v['em1_rate']:.2f} EM5={v['em5_rate']:.2f} "
              f"(n={v['n']})")

    # Disagreement-aware metric: looking only at readings where domains
    # disagree, what's the best-case vs current?
    result["meta"] = meta
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
