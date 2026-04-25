from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from models.src.eval.bench_loaders import load_ajimee_jwtd, load_general, load_probe
from models.src.eval.ctc_nat_backend import CTCNATBackend
from models.src.eval.jinen_backend import JinenV1Backend
from models.src.eval.metrics import EvalResult, character_accuracy
from models.src.eval.teacher_backend import TeacherBackend
from models.src.eval.zenz_backend import ZenzV2Backend


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_wsl_path(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    if not drive:
        return path.as_posix()
    tail = path.as_posix()[2:]
    return f"/mnt/{drive}{tail}"


def _is_missing_kenlm_error(exc: BaseException) -> bool:
    return "kenlm is not installed in this environment" in str(exc)


def _run_python_ctc_backend_via_wsl(
    root: Path,
    config_path: Path,
    model: dict,
    bench_names: list[str],
    output_root: Path,
) -> list[dict]:
    model_dir = output_root / _safe(model["name"])
    model_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = _to_wsl_path(root)
    cmd = [
        "wsl",
        "--cd",
        _to_wsl_path(root),
        "python3",
        "-m",
        "models.tools.eval.run_bench_all_from_config",
        "--config",
        _to_wsl_path(config_path),
        "--models",
        model["name"],
        "--benches",
        ",".join(bench_names),
        "--output-root",
        _to_wsl_path(output_root),
    ]
    print(
        f"[run_bench_all] fallback-wsl model={model['name']} exec: {' '.join(cmd)}",
        flush=True,
    )
    subprocess.run(cmd, check=True, env=env)
    summary = _load_json(model_dir / "summary.json")
    return summary["results"]


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_models(cfg: dict) -> list[dict]:
    expanded: list[dict] = []
    for model in cfg["models"]:
        if "variants" not in model:
            expanded.append(model)
            continue
        base = {k: v for k, v in model.items() if k != "variants"}
        base_name = base.pop("base_name", None) or base.get("name")
        if not base_name:
            raise ValueError("benchmark model missing `base_name`/`name`")
        for variant in model["variants"]:
            merged = _deep_merge_dict(base, variant)
            suffix = merged.pop("name_suffix", "").strip()
            merged["name"] = base_name if not suffix else f"{base_name} {suffix}"
            expanded.append(merged)
    return expanded


def _load_benches(root: Path, cfg: dict, selected: list[str] | None) -> tuple[list[str], dict[str, list[dict]], dict[str, Path]]:
    bench_cfg = cfg["defaults"]["benches"]
    bench_names = selected or [
        name for name, meta in bench_cfg.items() if meta.get("enabled_by_default", True)
    ]
    benches: dict[str, list[dict]] = {}
    paths: dict[str, Path] = {}
    for name in bench_names:
        meta = bench_cfg[name]
        path = _resolve(root, meta.get("path"))
        if path is not None:
            paths[name] = path
        if name == "probe_v3":
            benches[name] = load_probe(str(path))
        elif name == "ajimee_jwtd_v2":
            benches[name] = load_ajimee_jwtd(str(path))
        elif name == "general_dev":
            items = load_general(str(path))
            sample_count = int(meta.get("sample_count", 0) or 0)
            benches[name] = items[:sample_count] if sample_count > 0 else items
        else:
            raise ValueError(f"unsupported bench `{name}`")
    return bench_names, benches, paths


def _summarize_latencies(latencies: list[float]) -> dict:
    if not latencies:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    latencies = sorted(latencies)
    n = len(latencies)
    return {
        "p50_ms": round(latencies[n // 2], 3),
        "p95_ms": round(latencies[min(n - 1, int(n * 0.95))], 3),
        "mean_ms": round(sum(latencies) / n, 3),
    }


def _pattern_counts(counts: dict[str, int]) -> list[dict]:
    items = [{"key": k, "count": v} for k, v in counts.items()]
    items.sort(key=lambda x: (-x["count"], x["key"]))
    return items


def _round_to_step(value: float, step: float) -> float:
    return round(round(value / step) * step, 10)


def _grid_values(lo: float, hi: float, step: float) -> list[float]:
    start = _round_to_step(lo, step)
    end = _round_to_step(hi, step)
    if end < start:
        start, end = end, start
    values = []
    cur = start
    while cur <= end + 1e-9:
        values.append(round(cur, 10))
        cur = round(cur + step, 10)
    return values or [start]


def _summary_score(summary: dict) -> tuple[float, float, float, float]:
    return (
        float(summary.get("exact_match_top1", 0.0)),
        float(summary.get("char_acc_top1", 0.0)),
        float(summary.get("exact_match_top5", 0.0)),
        -float((summary.get("latency") or {}).get("p50_ms", 0.0)),
    )


def _evaluate_items_summary(backend, items: list[dict]) -> dict:
    result = EvalResult()
    latencies: list[float] = []
    for item in items:
        t0 = time.perf_counter()
        cands = backend.convert(item["reading"], item.get("context", "")) or []
        latencies.append((time.perf_counter() - t0) * 1000.0)
        result.add_multi(list(item["references"]), cands)
    summary = result.summary()
    summary["latency"] = _summarize_latencies(latencies)
    return summary


def _set_backend_lm_params(backend, alpha: float, beta: float) -> None:
    backend.lm_alpha = float(alpha)
    backend.lm_beta = float(beta)


def _axis_tune(
    backend,
    items: list[dict],
    axis: str,
    alpha: float,
    beta: float,
    lo: float,
    hi: float,
    step: float,
) -> tuple[float, dict]:
    lo = _round_to_step(lo, step)
    hi = _round_to_step(hi, step)
    best_value = alpha if axis == "alpha" else beta
    best_summary: dict | None = None

    while hi - lo > step + 1e-9:
        mid = _round_to_step((lo + hi) / 2.0, step)
        left = _round_to_step(max(lo, mid - step), step)
        right = _round_to_step(min(hi, mid + step), step)
        candidates = sorted({lo, left, mid, right, hi})
        scored: list[tuple[float, dict]] = []
        for value in candidates:
            cand_alpha = value if axis == "alpha" else alpha
            cand_beta = value if axis == "beta" else beta
            _set_backend_lm_params(backend, cand_alpha, cand_beta)
            summary = _evaluate_items_summary(backend, items)
            scored.append((value, summary))
        value, summary = max(scored, key=lambda item: _summary_score(item[1]))
        best_value, best_summary = value, summary
        idx = candidates.index(value)
        new_lo = candidates[max(0, idx - 1)]
        new_hi = candidates[min(len(candidates) - 1, idx + 1)]
        if new_lo == lo and new_hi == hi:
            break
        lo, hi = new_lo, new_hi

    final_scored: list[tuple[float, dict]] = []
    for value in _grid_values(lo, hi, step):
        cand_alpha = value if axis == "alpha" else alpha
        cand_beta = value if axis == "beta" else beta
        _set_backend_lm_params(backend, cand_alpha, cand_beta)
        summary = _evaluate_items_summary(backend, items)
        final_scored.append((value, summary))
    value, summary = max(final_scored, key=lambda item: _summary_score(item[1]))
    return value, summary


def _maybe_tune_lm(root: Path, defaults: dict, model: dict, backend, benches: dict[str, list[dict]]) -> dict | None:
    kenlm = model.get("kenlm", {})
    kenlm_moe = model.get("kenlm_moe", {})
    if not kenlm.get("enabled") and not kenlm_moe.get("enabled"):
        return None

    tuning = _deep_merge_dict(defaults.get("tuning", {}), model.get("tuning", {}))
    if not tuning.get("enabled", False):
        return None

    bench_name = tuning.get("bench", "probe_v3")
    items = benches.get(bench_name)
    if not items:
        raise ValueError(
            f"tuning bench `{bench_name}` is not loaded; include it in selected benches or defaults"
        )

    alpha_cfg = tuning.get("alpha", {})
    beta_cfg = tuning.get("beta", {})
    alpha = float((kenlm if kenlm.get("enabled") else kenlm_moe).get("alpha", 0.0))
    beta = float((kenlm if kenlm.get("enabled") else kenlm_moe).get("beta", 0.0))
    passes = int(tuning.get("passes", 2))
    step = float(alpha_cfg.get("step", beta_cfg.get("step", 0.05)))
    best_summary = None

    print(
        f"[run_bench_all] tuning model={model['name']} bench={bench_name} "
        f"alpha=[{alpha_cfg.get('min', 0.0)},{alpha_cfg.get('max', 0.8)}] "
        f"beta=[{beta_cfg.get('min', 0.0)},{beta_cfg.get('max', 0.8)}] step={step}",
        flush=True,
    )

    for pass_idx in range(passes):
        alpha, best_summary = _axis_tune(
            backend,
            items,
            "alpha",
            alpha,
            beta,
            float(alpha_cfg.get("min", 0.0)),
            float(alpha_cfg.get("max", 0.8)),
            step,
        )
        print(
            f"[run_bench_all] tuning pass={pass_idx + 1} axis=alpha best={alpha:.2f} "
            f"EM1={best_summary.get('exact_match_top1', 0.0):.4f}",
            flush=True,
        )
        beta, best_summary = _axis_tune(
            backend,
            items,
            "beta",
            alpha,
            beta,
            float(beta_cfg.get("min", 0.0)),
            float(beta_cfg.get("max", 0.8)),
            step,
        )
        print(
            f"[run_bench_all] tuning pass={pass_idx + 1} axis=beta best={beta:.2f} "
            f"EM1={best_summary.get('exact_match_top1', 0.0):.4f}",
            flush=True,
        )

    _set_backend_lm_params(backend, alpha, beta)
    if kenlm.get("enabled"):
        kenlm["alpha"] = alpha
        kenlm["beta"] = beta
    if kenlm_moe.get("enabled"):
        kenlm_moe["alpha"] = alpha
        kenlm_moe["beta"] = beta

    print(
        f"[run_bench_all] tuned model={model['name']} alpha={alpha:.2f} beta={beta:.2f}",
        flush=True,
    )
    return {
        "bench": bench_name,
        "alpha": round(alpha, 2),
        "beta": round(beta, 2),
        "step": step,
        "passes": passes,
        "summary": best_summary or {},
    }


def _evaluate_backend(name: str, params_label: str, backend, benches: dict[str, list[dict]], tuning_info: dict | None = None) -> list[dict]:
    results: list[dict] = []
    for bench_name, items in benches.items():
        print(f"[run_bench_all] bench={bench_name} samples={len(items)} model={name}", flush=True)
        overall = EvalResult()
        by_source: dict[str, EvalResult] = {}
        by_category: dict[str, EvalResult] = {}
        latencies: list[float] = []
        failures: list[dict] = []
        ref_top1_counts: dict[str, int] = {}
        reading_top1_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        t_start = time.perf_counter()
        for i, item in enumerate(items):
            t0 = time.perf_counter()
            cands = backend.convert(item["reading"], item.get("context", "")) or []
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)
            refs = list(item["references"])
            overall.add_multi(refs, cands)
            by_source.setdefault(item["source"], EvalResult()).add_multi(refs, cands)
            category = item.get("category")
            if category:
                by_category.setdefault(category, EvalResult()).add_multi(refs, cands)
            top1 = cands[0] if cands else ""
            if top1 not in refs:
                char1 = max((character_accuracy(ref, top1) for ref in refs), default=0.0)
                failures.append(
                    {
                        "index": item.get("_index", i),
                        "source": item["source"],
                        "category": category,
                        "context": item.get("context", ""),
                        "reading": item["reading"],
                        "references": refs,
                        "candidates": cands,
                        "top1": top1,
                        "char_acc_top1": char1,
                        "exact_match_top1": False,
                        "latency_ms": round(lat_ms, 3),
                    }
                )
                first_ref = refs[0] if refs else ""
                ref_top1_counts[f"{first_ref} => {top1}"] = ref_top1_counts.get(
                    f"{first_ref} => {top1}", 0
                ) + 1
                reading_top1_counts[f"{item['reading']} => {top1}"] = reading_top1_counts.get(
                    f"{item['reading']} => {top1}", 0
                ) + 1
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1
                source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1
        summary = overall.summary()
        result = {
            "backend": backend.name,
            "model": name,
            "params": params_label,
            "bench": bench_name,
            "device": "CPU only",
            "canonical": True,
            "decoding": {
                "num_beams": getattr(backend, "beam_width", 1),
                "num_return": max(1, getattr(backend, "num_return", getattr(backend, "beam_width", 1))),
            },
            "total": summary.get("total", 0),
            "exact_match_top1": summary.get("exact_match_top1", 0.0),
            "exact_match_top5": summary.get("exact_match_top5", 0.0),
            "char_acc_top1": summary.get("char_acc_top1", 0.0),
            "char_acc_top5": summary.get("char_acc_top5", 0.0),
            "latency": _summarize_latencies(latencies),
            "per_source": {k: v.summary() for k, v in by_source.items()},
            "per_category": {k: v.summary() for k, v in by_category.items()},
            "failures": failures,
            "failure_patterns": {
                "reference_to_top1": _pattern_counts(ref_top1_counts),
                "reading_to_top1": _pattern_counts(reading_top1_counts),
                "by_category": _pattern_counts(category_counts),
                "by_source": _pattern_counts(source_counts),
            },
            "total_time_s": round(time.perf_counter() - t_start, 3),
        }
        if tuning_info is not None:
            result["tuning"] = tuning_info
        results.append(result)
    return results


def _run_rust_native(
    root: Path,
    bench_bin: Path,
    defaults: dict,
    model: dict,
    bench_names: list[str],
    bench_paths: dict[str, Path],
    output_root: Path,
) -> list[dict]:
    model_dir = output_root / _safe(model["name"])
    model_dir.mkdir(parents=True, exist_ok=True)
    decoding = model.get("decoding", {})
    if model.get("kenlm", {}).get("enabled") or model.get("kenlm_moe", {}).get("enabled"):
        raise ValueError(f"rust_native model `{model['name']}` does not support KenLM options yet")
    cmd = [
        str(bench_bin),
        "--config",
        str(_resolve(root, model.get("config") or defaults["train_config"])),
        "--checkpoint",
        str(_resolve(root, model["model_path"])),
        "--out-dir",
        str(model_dir),
        "--markdown",
        str(model_dir / "benchmark_tables.md"),
        "--model-name",
        model["name"],
        "--benches",
        ",".join(bench_names),
        "--num-beams",
        str(decoding.get("num_beams", 1)),
        "--num-return",
        str(decoding.get("num_return", 1)),
    ]
    path_flags = {
        "probe_v3": "--probe-path",
        "ajimee_jwtd_v2": "--ajimee-path",
        "general_dev": "--general-path",
    }
    for bench_name in bench_names:
        bench_path = bench_paths.get(bench_name)
        flag = path_flags.get(bench_name)
        if flag and bench_path is not None:
            cmd.extend([flag, str(bench_path)])
    print(f"[run_bench_all] exec: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    summary = _load_json(model_dir / "summary.json")
    return summary["results"]


def _build_python_ctc_backend(root: Path, model: dict):
    decoding = model.get("decoding", {})
    kenlm = model.get("kenlm", {})
    kenlm_moe = model.get("kenlm_moe", {})
    if kenlm.get("enabled") and kenlm_moe.get("enabled"):
        raise ValueError(
            f"model `{model['name']}` enables both kenlm and kenlm_moe; choose one"
        )
    kwargs = {
        "checkpoint_path": str(_resolve(root, model["model_path"])),
        "device": model.get("device", "cpu"),
        "beam_width": max(int(decoding.get("num_beams", 1)), int(decoding.get("num_return", 1))),
        "name": model["name"],
    }
    if kenlm_moe.get("enabled"):
        paths = {
            k: str(_resolve(root, v))
            for k, v in (kenlm_moe.get("paths_by_domain") or {}).items()
        }
        kwargs["lm_paths_by_domain"] = paths
        kwargs["lm_alpha"] = float(kenlm_moe.get("alpha", 0.0))
        kwargs["lm_beta"] = float(kenlm_moe.get("beta", 0.0))
    elif kenlm.get("enabled"):
        kwargs["lm_path"] = str(_resolve(root, kenlm.get("path")))
        kwargs["lm_alpha"] = float(kenlm.get("alpha", 0.0))
        kwargs["lm_beta"] = float(kenlm.get("beta", 0.0))
    return CTCNATBackend(**kwargs)


def _build_python_zenz_backend(model: dict):
    decoding = model.get("decoding", {})
    return ZenzV2Backend(
        model["model_path"],
        device=model.get("device", "cpu"),
        num_beams=int(decoding.get("num_beams", 1)),
        num_return=int(decoding.get("num_return", 1)),
        name_suffix="",
    )


def _build_python_jinen_backend(model: dict):
    decoding = model.get("decoding", {})
    return JinenV1Backend(
        model["model_path"],
        device=model.get("device", "cpu"),
        num_beams=int(decoding.get("num_beams", 1)),
        num_return=int(decoding.get("num_return", 1)),
    )


def _build_python_teacher_backend(root: Path, model: dict):
    return TeacherBackend(
        str(_resolve(root, model["model_path"])),
        device=model.get("device", "cpu"),
        name=model["name"],
    )


def _build_python_onnx_ctc_greedy(root: Path, model: dict):
    import numpy as np
    import onnxruntime as ort
    from models.src.data.tokenizer import BLANK_ID, SharedCharTokenizer

    model_path = str(_resolve(root, model["model_path"]))
    tokenizer_path = str(_resolve(root, model["tokenizer_path"]))
    tokenizer = SharedCharTokenizer.load(tokenizer_path)
    seq = int(model.get("max_seq_len", 128))
    max_ctx = int(model.get("max_context", 40))
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(model.get("intra_op_threads", 4))
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

    class OnnxGreedyBackend:
        def __init__(self, name: str) -> None:
            self._name = name

        @property
        def name(self) -> str:
            return self._name

        def convert(self, reading: str, context: str) -> list[str]:
            ctx = context[-max_ctx:] if context else ""
            ids = tokenizer.encode_with_special(ctx, reading)[:seq]
            ilen = len(ids)
            x = np.zeros((1, seq), dtype=np.int64)
            m = np.zeros((1, seq), dtype=np.int64)
            x[0, :ilen] = ids
            m[0, :ilen] = 1
            out = sess.run(["logits"], {"input_ids": x, "attention_mask": m})[0]
            argmax = out[0, :ilen].argmax(axis=-1).tolist()
            toks, prev = [], -1
            for t in argmax:
                if t != BLANK_ID and t != prev:
                    toks.append(int(t))
                prev = int(t)
            return [tokenizer.decode(toks)]

    return OnnxGreedyBackend(model["name"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark_models.json")
    ap.add_argument("--models", default="", help="comma separated model names to run")
    ap.add_argument("--benches", default="", help="comma separated benches to run")
    ap.add_argument("--output-root", default="", help="override output root")
    args = ap.parse_args()

    root = _repo_root()
    config_path = _resolve(root, args.config)
    cfg = _load_json(config_path)
    selected_models = {m.strip() for m in args.models.split(",") if m.strip()}
    selected_benches = [b.strip() for b in args.benches.split(",") if b.strip()] or None
    bench_names, benches, bench_paths = _load_benches(root, cfg, selected_benches)
    models = _expand_models(cfg)

    defaults = cfg["defaults"]
    output_root = _resolve(root, args.output_root or defaults["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    bench_bin = root / "build" / "release" / "kkc-bench.exe"

    all_results: list[dict] = []
    for model in models:
        if not model.get("enabled", True):
            continue
        if selected_models and model["name"] not in selected_models:
            continue
        runner = model["runner"]
        if runner == "rust_native":
            results = _run_rust_native(
                root, bench_bin, defaults, model, bench_names, bench_paths, output_root
            )
        elif runner == "python_ctc_nat":
            try:
                backend = _build_python_ctc_backend(root, model)
                model_dir = output_root / _safe(model["name"])
                model_dir.mkdir(parents=True, exist_ok=True)
                tuning_info = _maybe_tune_lm(root, defaults, model, backend, benches)
                results = _evaluate_backend(
                    model["name"],
                    model.get("params", "python"),
                    backend,
                    benches,
                    tuning_info,
                )
                for result in results:
                    out = model_dir / f"{_safe(model['name'])}__{result['bench']}.json"
                    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                (model_dir / "summary.json").write_text(
                    json.dumps({"canonical": False, "device": "CPU only", "results": results}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except RuntimeError as exc:
                if sys.platform != "win32" or not _is_missing_kenlm_error(exc):
                    raise
                results = _run_python_ctc_backend_via_wsl(
                    root=root,
                    config_path=config_path,
                    model=model,
                    bench_names=bench_names,
                    output_root=output_root,
                )
        elif runner == "python_zenz":
            backend = _build_python_zenz_backend(model)
            model_dir = output_root / _safe(model["name"])
            model_dir.mkdir(parents=True, exist_ok=True)
            results = _evaluate_backend(
                model["name"],
                model.get("params", "python"),
                backend,
                benches,
            )
            for result in results:
                out = model_dir / f"{_safe(model['name'])}__{result['bench']}.json"
                out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            (model_dir / "summary.json").write_text(
                json.dumps({"canonical": True, "device": "CPU only", "results": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif runner == "python_jinen":
            backend = _build_python_jinen_backend(model)
            model_dir = output_root / _safe(model["name"])
            model_dir.mkdir(parents=True, exist_ok=True)
            results = _evaluate_backend(
                model["name"],
                model.get("params", "python"),
                backend,
                benches,
            )
            for result in results:
                out = model_dir / f"{_safe(model['name'])}__{result['bench']}.json"
                out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            (model_dir / "summary.json").write_text(
                json.dumps({"canonical": True, "device": "CPU only", "results": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif runner == "python_teacher":
            backend = _build_python_teacher_backend(root, model)
            model_dir = output_root / _safe(model["name"])
            model_dir.mkdir(parents=True, exist_ok=True)
            results = _evaluate_backend(
                model["name"],
                model.get("params", "python"),
                backend,
                benches,
            )
            for result in results:
                out = model_dir / f"{_safe(model['name'])}__{result['bench']}.json"
                out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            (model_dir / "summary.json").write_text(
                json.dumps({"canonical": True, "device": "CPU only", "results": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif runner == "python_onnx_ctc_greedy":
            backend = _build_python_onnx_ctc_greedy(root, model)
            model_dir = output_root / _safe(model["name"])
            model_dir.mkdir(parents=True, exist_ok=True)
            results = _evaluate_backend(
                model["name"],
                model.get("params", "python"),
                backend,
                benches,
            )
            for result in results:
                out = model_dir / f"{_safe(model['name'])}__{result['bench']}.json"
                out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            (model_dir / "summary.json").write_text(
                json.dumps({"canonical": True, "device": "CPU only", "results": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            raise ValueError(f"unsupported runner `{runner}` for model `{model['name']}`")
        all_results.extend(results)

    (output_root / "summary.json").write_text(
        json.dumps({"results": all_results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[run_bench_all] summary: {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
