"""AJIMEE JWTD_v2 (full 200) × α/β sweep for v3 bunsetsu checkpoints.

Mirrors the probe_v3 sweep shell but uses AJIMEE items (no category field).
Writes per-(ckpt, config) summary.json under results/ajimee_v3_bunsetsu_sweep/.

CPU only. Run via WSL (torch 2.11.0+cpu + KenLM).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from models.src.eval.bench_loaders import load_ajimee_jwtd
from models.src.eval.ctc_nat_backend import CTCNATBackend
from models.src.eval.metrics import EvalResult

AJIMEE = "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json"
LM = "models/kenlm/kenlm_general_train_4gram_probing.bin"
OUT_ROOT = Path("results/ajimee_v3_bunsetsu_sweep")

CKPTS = [
    ("ctc-nat-30m-bunsetsu-v3-best",       "models/checkpoints/ctc-nat-30m-bunsetsu-v3/best.pt"),
    ("ctc-nat-30m-bunsetsu-v3-step60000",  "models/checkpoints/ctc-nat-30m-bunsetsu-v3/checkpoint_step_60000.pt"),
    ("ctc-nat-30m-bunsetsu-v3-step73000",  "models/checkpoints/ctc-nat-30m-bunsetsu-v3/checkpoint_step_73000.pt"),
]
CONFIGS = [
    ("greedy",      dict(beam_width=1)),
    ("beam5_nolm",  dict(beam_width=5)),
    ("a0.2_b0.3",   dict(beam_width=5, lm_path=LM, lm_alpha=0.2, lm_beta=0.3)),
    ("a0.2_b0.6",   dict(beam_width=5, lm_path=LM, lm_alpha=0.2, lm_beta=0.6)),
    ("a0.4_b0.3",   dict(beam_width=5, lm_path=LM, lm_alpha=0.4, lm_beta=0.3)),
    ("a0.4_b0.6",   dict(beam_width=5, lm_path=LM, lm_alpha=0.4, lm_beta=0.6)),
    ("a0.6_b0.3",   dict(beam_width=5, lm_path=LM, lm_alpha=0.6, lm_beta=0.3)),
    ("a0.6_b0.6",   dict(beam_width=5, lm_path=LM, lm_alpha=0.6, lm_beta=0.6)),
]


def evaluate(backend, items, top_k=5):
    overall = EvalResult()
    hits5 = []
    lat = []
    for it in items:
        t0 = time.perf_counter()
        try:
            cands = backend.convert(it["reading"], it["context"])
        except Exception as e:
            cands = [f"<err:{e}>"]
        lat.append((time.perf_counter() - t0) * 1000)
        refs = it["references"]
        cands_k = cands[:top_k] if cands else []
        overall.add_multi(refs, cands_k)
        hits5.append(int(any(c in refs for c in cands_k)))
    s = overall.summary()
    s["em5"] = round(sum(hits5) / max(len(hits5), 1), 4)
    lat.sort()
    n = len(lat)
    s["p50"] = round(lat[n // 2], 1)
    s["p95"] = round(lat[int(n * 0.95)], 1)
    s["n"] = len(items)
    return s


def main():
    items = load_ajimee_jwtd(AJIMEE)
    print(f"ajimee: {len(items)} items", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for name, ckpt_path in CKPTS:
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for cfg_name, kwargs in CONFIGS:
            print(f"\n=== {name} / {cfg_name} ===", flush=True)
            t0 = time.perf_counter()
            backend = CTCNATBackend(ckpt_path, device="cpu", **kwargs)
            print(f"  load: {time.perf_counter()-t0:.1f}s", flush=True)
            t0 = time.perf_counter()
            r = evaluate(backend, items)
            dt = time.perf_counter() - t0
            em1 = r.get("exact_match_top1", 0)
            em5 = r["em5"]
            ca = r.get("char_acc_top1", 0)
            print(f"  EM1={em1:.3f} EM5={em5:.3f} CharAcc={ca:.3f} p50={r['p50']}ms ({dt:.0f}s)",
                  flush=True)
            (out_dir / cfg_name).mkdir(exist_ok=True)
            (out_dir / cfg_name / "summary.json").write_text(
                json.dumps({name: r}, indent=2, ensure_ascii=False), encoding="utf-8")
            del backend

    print("\nDONE")


if __name__ == "__main__":
    main()
