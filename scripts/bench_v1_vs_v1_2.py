"""One-off comparison bench: Suiko-v1-small (ctc-nat-41m-maskctc-student-wp
step100000) vs Suiko-v1.2-medium (best.pt @ ~step 15000).

Runs the same canonical setup as legacy/tools/tools/misc/bench_all.py:
  - probe_v3 (348 items)
  - AJIMEE JWTD_v2 (200 items)
  - device CPU, greedy (beam_width=1)
  - top_k=5

Output: results/bench_v1_vs_v1_2/<model>__<bench>.json + summary table.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")

from models.src.eval.bench_loaders import load_ajimee_jwtd, load_probe
from models.src.eval.ctc_nat_backend import CTCNATBackend
from models.src.eval.metrics import EvalResult

DEVICE = "cpu"
OUT = Path("results/bench_v1_vs_v1_2")
OUT.mkdir(parents=True, exist_ok=True)


def evaluate(backend, items, top_k=5):
    overall = EvalResult()
    em5 = []
    lat = []
    fails = []
    for item in items:
        t0 = time.perf_counter()
        try:
            cands = backend.convert(item["reading"], item["context"])
        except Exception as e:
            cands = [f"<error:{e}>"]
        lat.append((time.perf_counter() - t0) * 1000)
        refs = item["references"]
        cands_k = cands[:top_k] if cands else []
        overall.add_multi(refs, cands_k)
        em5.append(int(any(c in refs for c in cands_k)))
        if cands_k and cands_k[0] not in refs and len(fails) < 5:
            fails.append({"reading": item["reading"][:30], "ref": refs[0][:30],
                          "pred": cands_k[0][:30]})
    s = overall.summary()
    s["n"] = len(items)
    s["em5"] = round(sum(em5) / len(em5), 4)
    lat.sort()
    n = len(lat)
    s["latency_ms"] = {
        "p50": round(lat[n // 2], 1),
        "p95": round(lat[int(n * 0.95)], 1),
        "mean": round(sum(lat) / n, 1),
    }
    s["sample_failures"] = fails
    return s


def normalize_compile_prefix(ckpt_path: str) -> str:
    """If state_dict keys were saved through torch.compile (prefix
    `_orig_mod.`), produce a temp ckpt with the prefix stripped and return
    its path. Otherwise return the original path unchanged."""
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("model_state_dict") if isinstance(obj, dict) else None
    if not sd:
        return ckpt_path
    if not any(k.startswith("_orig_mod.") for k in sd):
        return ckpt_path
    new_sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
              for k, v in sd.items()}
    obj["model_state_dict"] = new_sd
    out = Path(tempfile.gettempdir()) / (Path(ckpt_path).stem + "__stripped.pt")
    torch.save(obj, out)
    # CTCNATBackend expects a `<stem>_tokenizer.json` sidecar next to the ckpt.
    src_tok = Path(ckpt_path).parent / (Path(ckpt_path).stem + "_tokenizer.json")
    dst_tok = out.parent / (out.stem + "_tokenizer.json")
    if src_tok.exists() and not dst_tok.exists():
        import shutil
        shutil.copy(src_tok, dst_tok)
    print(f"  [strip] {ckpt_path} -> {out} (compiled prefix removed)")
    return str(out)


MODELS = [
    ("Suiko-v1-small__greedy",
        "models/checkpoints/ctc-nat-41m-maskctc-student-wp/checkpoint_step_100000.pt"),
    ("Suiko-v1.2-small-step100000__greedy",
        "models/checkpoints/Suiko-v1.2-small/checkpoint_step_100000.pt"),
    ("Suiko-v1.5-small-step30000__greedy",
        "models/checkpoints/Suiko-v1.5-small/checkpoint_step_30000.pt"),
]


def main() -> None:
    probe_items = load_probe("datasets/eval/probe/probe.json")
    ajimee_items = load_ajimee_jwtd(
        "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json"
    )
    print(f"probe: {len(probe_items)} items, ajimee: {len(ajimee_items)} items")

    rows = []
    for label, ckpt in MODELS:
        print(f"\n=== {label} ===")
        if not Path(ckpt).exists():
            print(f"  SKIP: {ckpt} not found")
            continue
        ckpt_use = normalize_compile_prefix(ckpt)
        be = CTCNATBackend(ckpt_use, device=DEVICE, beam_width=1)
        for bench, items in [("probe_v3", probe_items), ("ajimee_jwtd", ajimee_items)]:
            t = time.time()
            s = evaluate(be, items)
            dt = time.time() - t
            out = OUT / f"{label}__{bench}.json"
            out.write_text(json.dumps(s, ensure_ascii=False, indent=2))
            row = {
                "model": label, "bench": bench,
                "n": s["n"],
                "em1": s.get("exact_match_top1", 0),
                "em5": s["em5"],
                "char_acc": s.get("char_acc_top1", 0),
                "p50_ms": s["latency_ms"]["p50"],
                "p95_ms": s["latency_ms"]["p95"],
                "wall_s": round(dt, 1),
            }
            rows.append(row)
            print(f"  {bench}: EM1={row['em1']:.4f} EM5={row['em5']:.4f} "
                  f"CharAcc={row['char_acc']:.4f} p50={row['p50_ms']}ms ({dt:.0f}s)")

    summary = OUT / "summary.json"
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print("\n=== summary ===")
    print(f"{'model':<35} {'bench':<14} {'EM1':>7} {'EM5':>7} {'CharAcc':>8} {'p50ms':>7}")
    for r in rows:
        print(f"{r['model']:<35} {r['bench']:<14} {r['em1']:>7.4f} {r['em5']:>7.4f} "
              f"{r['char_acc']:>8.4f} {r['p50_ms']:>7}")


if __name__ == "__main__":
    main()
