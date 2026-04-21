"""ONNX × (KenLM single / KenLM-MoE) benchmark on probe_v3 + AJIMEE.

For the v2 step160000 ckpt, measure 4 combos:
  fp32 ONNX + KenLM single       — parity with PT KenLM
  int8 ONNX + KenLM single       — production candidate
  fp32 ONNX + KenLM-MoE          — MoE upper bound
  int8 ONNX + KenLM-MoE          — production candidate with MoE

All use α=0.2 β=0.6 beam=5. Reuses models.src.eval.ctc_beam.prefix_beam_search
with ONNX-produced logits + existing KenLMCharScorer / KenLMMixtureScorer.
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnxruntime as ort
import torch

from models.src.data.tokenizer import BLANK_ID, SharedCharTokenizer
from models.src.eval.bench_loaders import load_ajimee_jwtd, load_probe
from models.src.eval.ctc_beam import prefix_beam_search
from models.src.eval.kenlm_mixture import CategoryEstimator, KenLMMixtureScorer
from models.src.eval.kenlm_scorer import KenLMCharScorer
from models.src.eval.metrics import EvalResult

V2_ONNX_FP32 = "models/onnx/ctc-nat-30m-student-step160000.fp32.onnx"
V2_ONNX_INT8 = "models/onnx/ctc-nat-30m-student-step160000.int8.onnx"
V2_TOKENIZER = "models/onnx/ctc-nat-30m-student-step160000.fp32.tokenizer.json"

LM_GENERAL = "models/kenlm/kenlm_general_train_4gram_probing.bin"
LM_TECH = "models/kenlm/kenlm_tech_4gram.bin"
LM_ENTITY = "models/kenlm/kenlm_entity_4gram.bin"

PROBE = "datasets/eval/probe/probe.json"
AJIMEE = "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json"

OUT_ROOT = Path("results/onnx_kenlm_moe_bench")
SEQ_LEN = 128
MAX_CTX = 40
ALPHA = 0.2
BETA = 0.6
BEAM_WIDTH = 5
TOP_K = 5


def run_bench(session, tokenizer, items, scorer, estimator, per_cat: bool) -> dict:
    overall = EvalResult()
    cats: dict[str, EvalResult] = defaultdict(EvalResult)
    em5: list[int] = []
    lat: list[float] = []

    for it in items:
        ctx = it["context"][-MAX_CTX:] if it["context"] else ""
        ids = tokenizer.encode_with_special(ctx, it["reading"])[:SEQ_LEN]
        ilen = len(ids)
        x = np.zeros((1, SEQ_LEN), dtype=np.int64)
        m = np.zeros((1, SEQ_LEN), dtype=np.int64)
        x[0, :ilen] = ids
        m[0, :ilen] = 1

        t0 = time.perf_counter()
        out = session.run(["logits"], {"input_ids": x, "attention_mask": m})[0]
        logits_valid = torch.from_numpy(out[0, :ilen])
        log_probs = torch.log_softmax(logits_valid, dim=-1)

        if estimator is not None and isinstance(scorer, KenLMMixtureScorer):
            scorer.set_weights(estimator.estimate(it["reading"], it["context"]))

        beam = prefix_beam_search(
            log_probs,
            blank_id=BLANK_ID,
            beam_width=BEAM_WIDTH,
            top_k_per_step=16,
            lm_scorer=scorer,
            lm_alpha=ALPHA,
            lm_beta=BETA,
        )
        lat.append((time.perf_counter() - t0) * 1000)

        seen = set()
        cands: list[str] = []
        for toks, _ in beam[:TOP_K]:
            s = tokenizer.decode(toks)
            if s and s not in seen:
                seen.add(s)
                cands.append(s)
            if len(cands) >= TOP_K:
                break

        refs = it["references"]
        overall.add_multi(refs, cands)
        em5.append(int(any(c in refs for c in cands)))
        if per_cat:
            cat = it.get("category")
            if cat:
                cats[cat].add_multi(refs, cands)

    s = overall.summary()
    s["em5"] = round(sum(em5) / max(len(em5), 1), 4)
    lat.sort()
    n = len(lat)
    s["p50"] = round(lat[n // 2], 1)
    s["p95"] = round(lat[int(n * 0.95)], 1)
    s["n"] = len(items)
    if per_cat and cats:
        s["per_category"] = {
            c: {
                "em1": round(r.summary().get("exact_match_top1", 0), 3),
                "n": r.summary().get("n", 0),
            }
            for c, r in sorted(cats.items())
        }
    return s


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tokenizer = SharedCharTokenizer.load(V2_TOKENIZER)
    probe_items = load_probe(PROBE)
    ajimee_items = load_ajimee_jwtd(AJIMEE)
    print(f"probe={len(probe_items)} ajimee={len(ajimee_items)}", flush=True)

    lm_paths = {"general": LM_GENERAL, "tech": LM_TECH, "entity": LM_ENTITY}

    configs = [
        ("fp32_kenlm_single", V2_ONNX_FP32, False),
        ("int8_kenlm_single", V2_ONNX_INT8, False),
        ("fp32_kenlm_moe",    V2_ONNX_FP32, True),
        ("int8_kenlm_moe",    V2_ONNX_INT8, True),
    ]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4

    summary: dict[str, dict] = {}
    for name, onnx_path, use_moe in configs:
        print(f"\n=== {name} ({Path(onnx_path).name}) ===", flush=True)
        t0 = time.perf_counter()
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        if use_moe:
            scorer = KenLMMixtureScorer(lm_paths, tokenizer)
            estimator = CategoryEstimator(available_domains=set(lm_paths))
        else:
            scorer = KenLMCharScorer(LM_GENERAL, tokenizer)
            estimator = None
        print(f"  load: {time.perf_counter()-t0:.1f}s", flush=True)

        entry: dict[str, dict] = {}
        for bench_name, items, per_cat in (
            ("probe", probe_items, True),
            ("ajimee", ajimee_items, False),
        ):
            t0 = time.perf_counter()
            r = run_bench(sess, tokenizer, items, scorer, estimator, per_cat)
            dt = time.perf_counter() - t0
            print(
                f"  {bench_name:<7} EM1={r.get('exact_match_top1', 0):.3f} "
                f"EM5={r['em5']:.3f} CharAcc={r.get('char_acc_top1', 0):.3f} "
                f"p50={r['p50']}ms ({dt:.0f}s)",
                flush=True,
            )
            entry[bench_name] = r
        summary[name] = entry

    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== TABLE ===")
    print(f"{'config':<20} {'probe EM1':<10} {'probe EM5':<10} {'ajimee EM1':<11} {'ajimee EM5':<11}")
    for n, r in summary.items():
        p = r["probe"]; a = r["ajimee"]
        print(f"{n:<20} {p.get('exact_match_top1', 0):<10.3f} {p.get('em5',0):<10.3f} "
              f"{a.get('exact_match_top1',0):<11.3f} {a.get('em5',0):<11.3f}")


if __name__ == "__main__":
    main()
