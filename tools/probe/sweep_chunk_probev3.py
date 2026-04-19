"""Chunk-decoding sweep on probe_v3.

Tests whether 30m_v2's probe_v3 regression is due to bunsetsu-span training:
if the model learned "reading chunk → 1-2 bunsetsu surface", it might fail
on whole-sentence conversion. Splitting probe_v3 inputs into bunsetsu-sized
chunks should restore performance if that hypothesis is correct.

For comparison, also runs 30m (v1, no bunsetsu training) with chunk decoding
— v1 shouldn't benefit (may even hurt) if the hypothesis holds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from models.src.eval.bench_loaders import load_probe
from models.src.eval.ctc_nat_backend import CTCNATBackend
from tools.probe.run_probe_v3 import evaluate


def run(ckpt, label, items, chunk_threshold, chunk_size, lm_path=None, alpha=0, beta=0, beam=1):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = dict(
        device=device,
        chunk_threshold=chunk_threshold,
        chunk_size=chunk_size,
        beam_width=beam,
    )
    if lm_path:
        kwargs.update(lm_path=lm_path, lm_alpha=alpha, lm_beta=beta)
    backend = CTCNATBackend(ckpt, **kwargs)
    t0 = time.perf_counter()
    res = evaluate(backend, items)
    elapsed = time.perf_counter() - t0
    em = res.get("exact_match_top1", 0)
    ca = res.get("char_acc_top1", 0)
    p50 = res["latency_ms"]["p50"]
    print(
        f"{label:<60} EM1={em:.4f} CharAcc={ca:.4f} p50={p50}ms ({elapsed:.0f}s)",
        flush=True,
    )
    per_cat = {c: d.get("exact_match_top1", 0) for c, d in res["per_category"].items()}
    return {"em1": em, "char_acc": ca, "p50": p50, "per_category": per_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="datasets/eval/probe/probe.json")
    ap.add_argument("--out", default="results/probe_v3_chunk_sweep/summary.json")
    ap.add_argument("--lm", default="models/kenlm/kenlm_general_train_4gram_probing.bin")
    args = ap.parse_args()

    items = load_probe(args.probe)
    print(f"probe_v3: {len(items)} items", flush=True)

    ckpt_v2 = "models/checkpoints/ctc_nat_30m_v2_dryrun/best.pt"
    ckpt_v1 = "models/checkpoints/ctc_nat_30m/best.pt"

    results: dict[str, dict] = {}

    # v2 baselines
    results["v2_greedy_nochunk"] = run(ckpt_v2, "v2 greedy no-chunk (baseline)", items, 0, 0)
    results["v2_beam5_nolm_nochunk"] = run(ckpt_v2, "v2 beam=5 no LM no-chunk", items, 0, 0, beam=5)

    # v2 chunk sweep (greedy)
    for thr, sz in [(12, 6), (12, 8), (16, 8), (16, 12), (20, 8), (20, 12), (20, 14)]:
        tag = f"v2_greedy_chunk{thr}x{sz}"
        results[tag] = run(ckpt_v2, f"v2 greedy chunk thr={thr} size={sz}", items, thr, sz)

    # v2 chunk + KenLM (best config from prior: α=0.4, β=0.6)
    results["v2_lm_chunk16x8"] = run(
        ckpt_v2, "v2 beam=5 LM(α=0.4,β=0.6) chunk16x8", items, 16, 8,
        lm_path=args.lm, alpha=0.4, beta=0.6, beam=5,
    )

    # v1 control: if bunsetsu hypothesis correct, chunking shouldn't help v1
    results["v1_greedy_nochunk"] = run(ckpt_v1, "v1 greedy no-chunk (control)", items, 0, 0)
    results["v1_greedy_chunk16x8"] = run(ckpt_v1, "v1 greedy chunk16x8 (control)", items, 16, 8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out}")

    # Compact table
    print("\n=== Summary (per-category EM1) ===")
    cats = ["edge", "general", "homophone", "names", "numeric", "particle", "tech"]
    header = ["tag", "EM1", "CA", "p50"] + cats
    print(" | ".join(f"{h:<10}" for h in header))
    for tag, r in results.items():
        row = [tag, f"{r['em1']:.3f}", f"{r['char_acc']:.3f}", f"{r['p50']}"]
        for c in cats:
            v = r["per_category"].get(c, 0)
            row.append(f"{v:.2f}")
        print(" | ".join(f"{c:<10}" for c in row))


if __name__ == "__main__":
    main()
