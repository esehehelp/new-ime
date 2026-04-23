"""ONNX × KenLM α/β sweep benchmark on probe_v3.

Reuses the existing `prefix_beam_search` + `KenLMCharScorer` (Python) and
replaces the PyTorch forward with an onnxruntime session. Handles both
fp32 and int8 ONNX models produced via export_onnx_ctc_nat.py +
onnxruntime.quantization.quantize_dynamic.

Configs per ONNX model:
  greedy             — argmax + CTC collapse, no LM
  beam5_nolm         — prefix beam search, no LM
  a{α}_b{β}          — prefix beam + KenLM shallow fusion (α × β grid)

CPU only. Run via WSL (torch cpu + kenlm + onnxruntime installed).
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

from models.src.data.tokenizer import BLANK_ID, CLS_ID, PAD_ID, SEP_ID, SharedCharTokenizer
from models.src.eval.bench_loaders import load_probe
from models.src.eval.ctc_beam import prefix_beam_search
from models.src.eval.kenlm_scorer import KenLMCharScorer
from models.src.eval.metrics import EvalResult


def forward_onnx(session: ort.InferenceSession, input_ids: np.ndarray, attn_mask: np.ndarray) -> np.ndarray:
    out = session.run(
        ["logits"],
        {"input_ids": input_ids, "attention_mask": attn_mask},
    )
    return out[0]


def greedy_decode_from_logits(
    logits: np.ndarray, blank_id: int, input_len: int
) -> list[int]:
    argmax = logits[:input_len].argmax(axis=-1).tolist()
    out: list[int] = []
    prev = -1
    for tok in argmax:
        if tok != blank_id and tok != prev:
            out.append(tok)
        prev = tok
    return out


def run_one_config(
    session: ort.InferenceSession,
    items: list[dict],
    tokenizer: SharedCharTokenizer,
    seq_len: int,
    max_ctx: int,
    beam_width: int,
    lm_scorer: KenLMCharScorer | None,
    lm_alpha: float,
    lm_beta: float,
    top_k: int = 5,
) -> dict:
    overall = EvalResult()
    per_cat: dict[str, EvalResult] = defaultdict(EvalResult)
    em5: list[int] = []
    lat: list[float] = []

    for it in items:
        ctx = it["context"][-max_ctx:] if it["context"] else ""
        ids = tokenizer.encode_with_special(ctx, it["reading"])[:seq_len]
        input_len = len(ids)
        input_ids = np.zeros((1, seq_len), dtype=np.int64)
        attn_mask = np.zeros((1, seq_len), dtype=np.int64)
        input_ids[0, :input_len] = ids
        attn_mask[0, :input_len] = 1

        t0 = time.perf_counter()
        logits_np = forward_onnx(session, input_ids, attn_mask)[0]  # (T, V)
        if beam_width <= 1:
            tokens = greedy_decode_from_logits(logits_np, BLANK_ID, input_len)
            cands_ids = [tokens]
        else:
            logits_valid = torch.from_numpy(logits_np[:input_len])
            log_probs = torch.log_softmax(logits_valid, dim=-1)
            beam = prefix_beam_search(
                log_probs,
                blank_id=BLANK_ID,
                beam_width=beam_width,
                top_k_per_step=16,
                lm_scorer=lm_scorer,
                lm_alpha=lm_alpha,
                lm_beta=lm_beta,
            )
            cands_ids = [toks for toks, _ in beam[:top_k]] if beam else [[]]
        dt = (time.perf_counter() - t0) * 1000
        lat.append(dt)

        seen = set()
        cands_str: list[str] = []
        for toks in cands_ids:
            s = tokenizer.decode(toks)
            if s and s not in seen:
                seen.add(s)
                cands_str.append(s)
            if len(cands_str) >= top_k:
                break
        refs = it["references"]
        overall.add_multi(refs, cands_str)
        per_cat[it["category"]].add_multi(refs, cands_str)
        em5.append(int(any(c in refs for c in cands_str)))

    s = overall.summary()
    s["em5"] = round(sum(em5) / max(len(em5), 1), 4)
    lat.sort()
    n = len(lat)
    s["p50"] = round(lat[n // 2], 1)
    s["p95"] = round(lat[int(n * 0.95)], 1)
    s["n"] = len(items)
    s["per_category"] = {
        c: {
            "em1": round(r.summary().get("exact_match_top1", 0), 3),
            "n": r.summary().get("n", 0),
        }
        for c, r in sorted(per_cat.items())
    }
    return s


CONFIGS = [
    ("greedy",      dict(beam_width=1, alpha=0.0, beta=0.0)),
    ("beam5_nolm",  dict(beam_width=5, alpha=0.0, beta=0.0)),
    ("a0.2_b0.3",   dict(beam_width=5, alpha=0.2, beta=0.3)),
    ("a0.2_b0.6",   dict(beam_width=5, alpha=0.2, beta=0.6)),
    ("a0.4_b0.3",   dict(beam_width=5, alpha=0.4, beta=0.3)),
    ("a0.4_b0.6",   dict(beam_width=5, alpha=0.4, beta=0.6)),
    ("a0.6_b0.3",   dict(beam_width=5, alpha=0.6, beta=0.3)),
    ("a0.6_b0.6",   dict(beam_width=5, alpha=0.6, beta=0.6)),
]


def run_model(onnx_path: str, tokenizer_path: str, probe_path: str,
              lm_path: str, out_root: Path, seq_len: int = 128, max_ctx: int = 40):
    print(f"=== {onnx_path} ===", flush=True)
    tokenizer = SharedCharTokenizer.load(tokenizer_path)
    items = load_probe(probe_path)
    print(f"  tokenizer vocab={tokenizer.vocab_size}  items={len(items)}", flush=True)

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    lm_scorer = KenLMCharScorer(lm_path, tokenizer)

    name = Path(onnx_path).stem
    model_out = out_root / name
    model_out.mkdir(parents=True, exist_ok=True)

    for cfg_name, kwargs in CONFIGS:
        print(f"  -- {cfg_name}", flush=True)
        scorer = lm_scorer if (kwargs["alpha"] != 0.0 or kwargs["beta"] != 0.0) else None
        t0 = time.perf_counter()
        res = run_one_config(
            session, items, tokenizer, seq_len, max_ctx,
            beam_width=kwargs["beam_width"],
            lm_scorer=scorer,
            lm_alpha=kwargs["alpha"],
            lm_beta=kwargs["beta"],
        )
        dt = time.perf_counter() - t0
        em1 = res.get("exact_match_top1", 0)
        em5 = res["em5"]
        p50 = res["p50"]
        print(f"     EM1={em1:.3f} EM5={em5:.3f} p50={p50}ms ({dt:.0f}s)", flush=True)
        (model_out / cfg_name).mkdir(exist_ok=True)
        (model_out / cfg_name / "summary.json").write_text(
            json.dumps({name: res}, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def main():
    out_root = Path("results/onnx_kenlm_sweep")
    tokenizer = "models/onnx/ctc-nat-30m-student-step160000.fp32.tokenizer.json"
    probe = "datasets/eval/probe/probe.json"
    lm = "models/kenlm/kenlm_general_train_4gram_probing.bin"

    for onnx_name in [
        "models/onnx/ctc-nat-30m-student-step160000.fp32.onnx",
        "models/onnx/ctc-nat-30m-student-step160000.int8.onnx",
    ]:
        run_model(onnx_name, tokenizer, probe, lm, out_root)


if __name__ == "__main__":
    main()
