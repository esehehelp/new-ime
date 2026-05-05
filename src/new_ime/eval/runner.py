"""Bench runner orchestration.

Format-preserving port of the pre-v2 evaluate() in
scripts/bench_v1_vs_v1_2.py: same JSON shape, same latency keys, same
sample_failures cap. Adds probe categorical EM as a top-level field.

Verbose mode (`-v` on the CLI) writes a per-item JSONL with the FULL
candidate list returned by the backend (not just the top_k slice used
for metrics) so any post-hoc audit can recompute metrics or inspect
specific failures without re-running the model.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import IO, List, Protocol

from new_ime.eval.loaders import BenchItem, load_bench
from new_ime.eval.metrics import (
    EvalResult,
    character_accuracy,
    latency_summary,
    nfkc_normalize,
    top_k_character_accuracy,
)


class ConversionBackend(Protocol):
    """Minimal contract: take (reading, context), return ranked candidates.

    `runtime` / `artifact_format` are surfaced in the bench JSON so a
    summary row carries enough metadata for cross-backend comparison
    (Suiko ONNX-via-Rust vs HF PyTorch).
    """

    name: str
    runtime: str
    artifact_format: str

    def convert(self, reading: str, context: str) -> List[str]: ...


_FAILURE_CAP = 5


def evaluate(
    backend: ConversionBackend,
    items: List[BenchItem],
    top_k: int = 5,
    verbose_log: IO[str] | None = None,
    bench_name: str = "",
) -> dict:
    """Run a backend over a bench. Returns the per-bench JSON dict.

    If `verbose_log` is given, writes one NDJSON record per item with the
    full backend output, so the run is verifiable after the fact.
    """
    overall = EvalResult()
    overall_nfkc = EvalResult()
    em5_flags: List[int] = []
    em5_nfkc_flags: List[int] = []
    latencies_ms: List[float] = []
    failures: list[dict] = []
    cat_total: dict[str, int] = defaultdict(int)
    cat_em1: dict[str, int] = defaultdict(int)
    cat_em1_nfkc: dict[str, int] = defaultdict(int)

    for i, item in enumerate(items):
        t0 = time.perf_counter()
        error: str | None = None
        try:
            cands = backend.convert(item.reading, item.context)
        except Exception as e:  # noqa: BLE001 — record-and-continue intentional for bench
            error = f"{type(e).__name__}:{e}"
            cands = [f"<error:{error}>"]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(latency_ms)

        refs = item.references
        cands_k = list(cands[:top_k]) if cands else []
        overall.add_multi(refs, cands_k)

        em1 = int(bool(cands_k) and cands_k[0] in refs)
        em5 = int(any(c in refs for c in cands_k))
        em5_flags.append(em5)
        char_acc_top1 = (
            max((character_accuracy(r, cands_k[0]) for r in refs), default=0.0)
            if cands_k
            else 0.0
        )
        char_acc_topk = (
            max(
                (top_k_character_accuracy(r, cands_k, top_k) for r in refs),
                default=0.0,
            )
            if cands_k
            else 0.0
        )

        # NFKC-normalised parallel computation. AJIMEE-Bench publishes
        # `Acc@1 (NFKC)` as the canonical practical-use number; we
        # surface both raw and NFKC so callers can pick the right one
        # per benchmark.
        refs_nfkc = [nfkc_normalize(r) for r in refs]
        cands_k_nfkc = [nfkc_normalize(c) for c in cands_k]
        overall_nfkc.add_multi(refs_nfkc, cands_k_nfkc)
        em1_nfkc = int(bool(cands_k_nfkc) and cands_k_nfkc[0] in refs_nfkc)
        em5_nfkc = int(any(c in refs_nfkc for c in cands_k_nfkc))
        em5_nfkc_flags.append(em5_nfkc)
        char_acc_top1_nfkc = (
            max(
                (character_accuracy(r, cands_k_nfkc[0]) for r in refs_nfkc),
                default=0.0,
            )
            if cands_k_nfkc
            else 0.0
        )

        if item.category is not None:
            cat_total[item.category] += 1
            cat_em1[item.category] += em1
            cat_em1_nfkc[item.category] += em1_nfkc

        if cands_k and cands_k[0] not in refs and len(failures) < _FAILURE_CAP:
            failures.append(
                {
                    "reading": item.reading[:30],
                    "ref": refs[0][:30] if refs else "",
                    "pred": cands_k[0][:30],
                }
            )

        if verbose_log is not None:
            record = {
                "i": i,
                "bench": bench_name,
                "index": item.index,
                "category": item.category,
                "reading": item.reading,
                "context": item.context,
                "references": refs,
                "candidates": list(cands),  # FULL list, not capped at top_k
                "em1": em1,
                "em5": em5,
                "em1_nfkc": em1_nfkc,
                "em5_nfkc": em5_nfkc,
                "char_acc_top1": round(char_acc_top1, 4),
                "char_acc_topk": round(char_acc_topk, 4),
                "char_acc_top1_nfkc": round(char_acc_top1_nfkc, 4),
                "latency_ms": round(latency_ms, 3),
            }
            engine_ms = getattr(backend, "last_engine_ms", None)
            if engine_ms is not None:
                # pure engine time as reported by the backend, e.g. the Rust
                # daemon — excludes Python orchestration / IPC overhead.
                record["engine_ms"] = round(float(engine_ms), 3)
            if error is not None:
                record["error"] = error
            verbose_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            verbose_log.flush()

    s = overall.summary()
    n = len(items)
    s["n"] = n
    s["em5"] = round(sum(em5_flags) / n, 4) if n else 0.0
    s["latency_ms"] = latency_summary(latencies_ms)
    s["sample_failures"] = failures
    # Backend provenance — same 3 fields as summary.json so per-bench
    # files are self-contained.
    runtime = getattr(backend, "runtime", "unknown")
    s["runtime"] = runtime
    s["artifact_format"] = getattr(backend, "artifact_format", "unknown")
    # Rust daemon is the only backend whose `latency_ms` excludes large
    # framework overheads (HF backends include tokenizer + model + Python
    # generate loop). Cross-runtime latency comparison is therefore not
    # apples-to-apples; flag it explicitly.
    s["latency_comparable"] = runtime == "rust-engine-daemon"

    # NFKC-normalised metrics — AJIMEE-Bench's "Acc@1 (NFKC)" canonical
    # number. Side-by-side with the raw fields so consumers can pick.
    nfkc = overall_nfkc.summary()
    s["exact_match_top1_nfkc"] = nfkc.get("exact_match_top1", 0.0)
    s["exact_match_top5_nfkc"] = nfkc.get("exact_match_top5", 0.0)
    s["char_acc_top1_nfkc"] = nfkc.get("char_acc_top1", 0.0)
    s["char_acc_top5_nfkc"] = nfkc.get("char_acc_top5", 0.0)
    s["em5_nfkc"] = round(sum(em5_nfkc_flags) / n, 4) if n else 0.0

    if cat_total:
        s["probe_categories"] = {
            cat: {
                "n": cat_total[cat],
                "em1": round(cat_em1[cat] / cat_total[cat], 4),
                "em1_nfkc": round(cat_em1_nfkc[cat] / cat_total[cat], 4),
            }
            for cat in sorted(cat_total)
        }
    return s


def run_bench_suite(
    backend: ConversionBackend,
    benches: dict[str, str | Path],
    out_dir: Path,
    decode_mode: str,
    top_k: int = 5,
    verbose: bool = False,
) -> list[dict]:
    """Run one backend over all benches in `benches` (name -> dataset path).

    Writes per-bench `<bench>__<mode>.json` and a combined `summary.json`
    in `out_dir`. With `verbose=True`, also writes
    `<bench>__<mode>.full.jsonl` containing every backend output for
    audit/verification.

    Returns the rows that landed in summary.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for bench_name, ds_path in benches.items():
        items = load_bench(bench_name, ds_path)
        print(
            f"[bench] {bench_name}: {len(items)} items "
            f"(backend={backend.name} decode={decode_mode}"
            f"{' verbose' if verbose else ''})",
            file=sys.stderr,
        )

        wall_t0 = time.time()
        log_path = out_dir / f"{bench_name}__{decode_mode}.full.jsonl"
        log_ctx = (
            open(log_path, "w", encoding="utf-8")
            if verbose
            else nullcontext(None)
        )
        with log_ctx as log_fh:
            s = evaluate(
                backend,
                items,
                top_k=top_k,
                verbose_log=log_fh,
                bench_name=bench_name,
            )
        wall = time.time() - wall_t0

        per_bench_path = out_dir / f"{bench_name}__{decode_mode}.json"
        per_bench_path.write_text(
            json.dumps(s, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        row = {
            "model": backend.name,
            "bench": bench_name,
            "n": s["n"],
            "em1": s.get("exact_match_top1", 0.0),
            "em5": s["em5"],
            "char_acc": s.get("char_acc_top1", 0.0),
            "em1_nfkc": s["exact_match_top1_nfkc"],
            "em5_nfkc": s["em5_nfkc"],
            "char_acc_nfkc": s["char_acc_top1_nfkc"],
            "p50_ms": s["latency_ms"]["p50"],
            "p95_ms": s["latency_ms"]["p95"],
            "wall_s": round(wall, 1),
            "runtime": s["runtime"],
            "artifact_format": s["artifact_format"],
            "latency_comparable": s["latency_comparable"],
        }
        rows.append(row)
        print(
            f"[bench] {bench_name}: EM1={row['em1']:.4f} EM5={row['em5']:.4f} "
            f"CharAcc={row['char_acc']:.4f} "
            f"p50={row['p50_ms']}ms ({wall:.0f}s)"
            + (f" -> {log_path.name}" if verbose else ""),
            file=sys.stderr,
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows
