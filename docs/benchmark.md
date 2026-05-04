# Benchmark

## 条件

- device: CPU only (WSL)
- bench 1: `datasets/eval/probe/probe.json` (`probe_v3`, 348 items)
- bench 2: `references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json` (200 items)
- decode: `num_beams=5`, `num_return=5` (canonical) / `num_beams=1`, `num_return=1` (greedy)
- metrics: EM1, EM5, CharAcc, latency p50/p95/mean (ms)
- input: 片仮名は `jaconv.kata2hira` で hiragana に正規化

## 比較対象

| name | type | source |
|---|---|---|
| `suiko-v1-small-greedy` | ctc-nat | `checkpoints/suiko-v1-small/checkpoint_step_100000.pt` (greedy) |
| `suiko-v1-small-kenlm` | ctc-nat + KenLM single | 同 + `assets/kenlm/kenlm_general_train_4gram_probing.bin` (α=0.2 β=0.6) |
| `suiko-v1-small-kenlm-moe` | ctc-nat + KenLM MoE | 同 + `general / tech / entity` 4-gram (α=0.2 β=0.6) |
| `zenz-v2.5-xsmall` | zenz-v2.5 | `references/zenz-v2.5-xsmall/` |
| `zenz-v2.5-small` | zenz-v2.5 | `references/zenz-v2.5-small/` |
| `zenz-v2.5-medium` | zenz-v2.5 | `references/zenz-v2.5-medium/` |
| `zenz-v3.1-small` | zenz-v3.1 | `references/zenz-v3.1-small/` |
| `jinen-v1-xsmall` | jinen-v1 | HF `togatogah/jinen-v1-xsmall` |
| `jinen-v1-small` | jinen-v1 | HF `togatogah/jinen-v1-small` |

## 実行

```bash
# WSL から
cd /mnt/d/Dev/new-ime
PYTHONPATH=src ~/.venvs/new-ime/bin/python -m new_ime.cli.bench [flags]

# または scripts/_uv_env.sh を source 経由で
source scripts/_uv_env.sh
uv run python -m new_ime.cli.bench [flags]
```

flags:

| flag | |
|---|---|
| `-c DIR` | config dir 上書き (default `configs/bench/`) |
| `-m NAME...` | `[run] name` whitelist |
| `-t NAME...` | `[benches]` キー whitelist |
| `-v` | per-item NDJSON log (`<bench>__<mode>.full.jsonl`) を有効化 |

差分計算:

```bash
~/.venvs/new-ime/bin/python scripts/_canonical_compare.py
```

## TOML

`configs/bench/<name>.toml` を 1 ファイル = 1 実験。`[run] name` が出力 dir 名。

```toml
[run]
name = "suiko-v1-small-greedy"
out_dir = "results/bench/suiko-v1-small-greedy"

[model]
type = "ctc-nat"  # | "zenz-v2.5" | "zenz-v3.1" | "jinen-v1"
checkpoint = "checkpoints/suiko-v1-small/checkpoint_step_100000.pt"
tokenizer  = "checkpoints/suiko-v1-small/checkpoint_step_100000_tokenizer.json"
preset = "phase3_30m"

[decode]
mode = "greedy"   # | "beam"
num_beams = 1
num_return = 1
top_k = 5

[benches]
probe_v3    = "datasets/eval/probe/probe.json"
ajimee_jwtd = "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json"

[device]
backend = "cpu"

# CTC-NAT 専用、optional
[lm]
mode = "single"   # | "moe"
path = "assets/kenlm/kenlm_general_train_4gram_probing.bin"  # single
alpha = 0.2
beta = 0.6

# moe の場合は path の代わりに:
# [lm.paths_by_domain]
# general = "..."
# tech    = "..."
# entity  = "..."
```

HF 系 model:

```toml
[model]
type = "zenz-v2.5"   # または "zenz-v3.1" / "jinen-v1"
path = "references/zenz-v2.5-small"   # local dir or HF Hub ID
max_new_tokens = 80
max_context_chars = 40
```

## 出力

`<run.out_dir>/<bench>__<mode>.json`:

```json
{
  "total": 348, "n": 348,
  "char_acc_top1": 0.944, "exact_match_top1": 0.6006,
  "char_acc_top5": 0.944, "exact_match_top5": 0.6006,
  "char_acc_top10": 0.944, "exact_match_top10": 0.6006,
  "em5": 0.6006,
  "latency_ms": { "p50": 9.3, "p95": 18.0, "mean": 10.7 },
  "sample_failures": [{ "reading": "...", "ref": "...", "pred": "..." }],
  "probe_categories": { "edge": { "n": 50, "em1": 0.45 } }
}
```

`<run.out_dir>/<bench>__<mode>.full.jsonl` (`-v` 時、1 行 1 item):

```json
{"i": 0, "bench": "probe_v3", "index": "0001", "category": "general",
 "reading": "...", "context": "", "references": ["..."],
 "candidates": ["...", "..."],
 "em1": 1, "em5": 1, "char_acc_top1": 1.0, "char_acc_topk": 1.0,
 "latency_ms": 16.32}
```

`<run.out_dir>/summary.json`:

```json
[{"model": "...", "bench": "probe_v3", "n": 348,
  "em1": 0.6006, "em5": 0.6006, "char_acc": 0.944,
  "p50_ms": 9.3, "p95_ms": 18.0, "wall_s": 4.0}]
```

## アンカー (`legacy/docs/benchmark_comparison.md` 2026-04-22)

probe_v3:

| model | params | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|---:|
| zenz-v2.5-medium | 310M | 0.747 | 0.876 | 0.966 | 1173 |
| zenz-v3.1-small | 91M | 0.718 | 0.856 | 0.959 | 417 |
| zenz-v2.5-small | 91M | 0.713 | 0.848 | 0.959 | 376 |
| zenz-v2.5-xsmall | 30M | 0.695 | 0.813 | 0.953 | 118 |
| suiko-v1-small-kenlm-moe | 41M | 0.672 | 0.784 | 0.949 | 22 |
| jinen-v1-small | 110M | 0.672 | 0.776 | 0.944 | 278 |
| suiko-v1-small-kenlm | 41M | 0.664 | 0.776 | 0.947 | 17 |
| jinen-v1-xsmall | 35.8M | 0.609 | 0.747 | 0.929 | 115 |
| suiko-v1-small-greedy | 41M | 0.601 | 0.601 | 0.944 | 9 |

ajimee_jwtd:

| model | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|
| zenz-v2.5-medium | 0.875 | 0.970 | 0.982 | 1361 |
| zenz-v3.1-small | 0.860 | 0.930 | 0.983 | 470 |
| zenz-v2.5-small | 0.840 | 0.955 | 0.977 | 418 |
| zenz-v2.5-xsmall | 0.695 | 0.845 | 0.953 | 139 |
| suiko-v1-small-kenlm | 0.670 | 0.830 | 0.959 | 21 |
| suiko-v1-small-kenlm-moe | 0.670 | 0.820 | 0.959 | 28 |
| jinen-v1-small | 0.655 | 0.835 | 0.952 | 309 |
| suiko-v1-small-greedy | 0.580 | 0.580 | 0.951 | 10 |
| jinen-v1-xsmall | 0.395 | 0.525 | 0.917 | 124 |

## 現環境結果 (WSL CPU, torch 2.11.0+cpu, 2026-05-03)

| config | bench | EM1 | ΔEM1 | EM5 | ΔEM5 | CharAcc | ΔCharAcc | p50 | Δp50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| suiko-v1-small-greedy    | probe_v3    | 0.6006 | -0.0004 | 0.6006 | -0.0004 | 0.9440 | +0.0000 |   9.3 |    +0.3 |
| suiko-v1-small-greedy    | ajimee_jwtd | 0.5800 | +0.0000 | 0.5800 | +0.0000 | 0.9509 | -0.0001 |  11.2 |    +1.2 |
| suiko-v1-small-kenlm     | probe_v3    | 0.6580 | -0.0060 | 0.7759 | -0.0001 | 0.9468 | -0.0002 |  12.2 |    -4.8 |
| suiko-v1-small-kenlm     | ajimee_jwtd | 0.6700 | +0.0000 | 0.8300 | +0.0000 | 0.9591 | +0.0001 |  14.3 |    -6.7 |
| suiko-v1-small-kenlm-moe | probe_v3    | 0.6695 | -0.0025 | 0.7845 | +0.0005 | 0.9484 | -0.0006 |  13.9 |    -8.1 |
| suiko-v1-small-kenlm-moe | ajimee_jwtd | 0.6700 | +0.0000 | 0.8200 | +0.0000 | 0.9592 | +0.0002 |  15.5 |   -12.5 |
| zenz-v2.5-xsmall         | probe_v3    | 0.6954 | +0.0004 | 0.8132 | +0.0002 | 0.9527 | -0.0003 | 124.9 |    +6.9 |
| zenz-v2.5-xsmall         | ajimee_jwtd | 0.6950 | +0.0000 | 0.8450 | +0.0000 | 0.9530 | +0.0000 | 133.1 |    -5.9 |
| zenz-v2.5-small          | probe_v3    | 0.7126 | -0.0004 | 0.8477 | -0.0003 | 0.9585 | -0.0005 | 361.3 |   -14.7 |
| zenz-v2.5-small          | ajimee_jwtd | 0.8400 | +0.0000 | 0.9550 | +0.0000 | 0.9767 | -0.0003 | 424.8 |    +6.8 |
| zenz-v2.5-medium         | probe_v3    | 0.7471 | +0.0001 | 0.8764 | +0.0004 | 0.9655 | -0.0005 | 1123.9 |  -49.1 |
| zenz-v2.5-medium         | ajimee_jwtd | 0.8750 | +0.0000 | 0.9700 | +0.0000 | 0.9819 | -0.0001 | 1205.1 | -155.9 |
| zenz-v3.1-small          | probe_v3    | 0.7184 | +0.0004 | 0.8563 | +0.0003 | 0.9594 | +0.0004 | 378.5 |   -38.5 |
| zenz-v3.1-small          | ajimee_jwtd | 0.8600 | +0.0000 | 0.9300 | +0.0000 | 0.9833 | +0.0003 | 433.5 |   -36.5 |
| jinen-v1-xsmall          | probe_v3    | 0.6092 | +0.0002 | 0.7471 | +0.0001 | 0.9286 | -0.0004 |  81.8 |   -33.2 |
| jinen-v1-xsmall          | ajimee_jwtd | 0.3950 | +0.0000 | 0.5250 | +0.0000 | 0.9172 | +0.0002 |  85.7 |   -38.3 |
| jinen-v1-small           | probe_v3    | 0.6724 | +0.0004 | 0.7759 | -0.0001 | 0.9437 | -0.0003 | 216.2 |   -61.8 |
| jinen-v1-small           | ajimee_jwtd | 0.6550 | +0.0000 | 0.8350 | +0.0000 | 0.9524 | +0.0004 | 246.9 |   -62.1 |
