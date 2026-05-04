# Benchmark

## 条件

- device: CPU only (WSL)
- bench 1: `datasets/eval/probe/probe.json` (`probe_v3`, 348 items)
- bench 2: `references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json` (200 items)
- decode: `num_beams=5`, `num_return=5` (canonical) / `num_beams=1`, `num_return=1` (greedy)
- metrics: EM1, EM5, CharAcc, latency p50/p95/mean (ms)
- input: 片仮名は `jaconv.kata2hira` で hiragana に正規化

主判断は `probe_v3`。`ajimee_jwtd` は n=200 と小さいため補助指標として見る。

## 比較対象

| name | type | source |
|---|---|---|
| `suiko-v1-small-greedy` | ctc-nat | `checkpoints/suiko-v1-small/checkpoint_step_100000.pt` (greedy) |
| `suiko-v1-small-kenlm` | ctc-nat + KenLM single | 同 + `assets/kenlm/kenlm_general_train_4gram_probing.bin` (α=0.2 β=0.6) |
| `suiko-v1-small-kenlm-moe` | ctc-nat + KenLM MoE | 同 + `general / tech / entity` 4-gram (α=0.2 β=0.6) |
| `suiko-v1-small-kenlm-6gram-q8` | ctc-nat + KenLM single | 同 + `assets/kenlm/kenlm_general_6gram_q8.bin` (α=0.2 β=0.6) |
| `suiko-v1-small-kenlm-6gram-q8-moe` | ctc-nat + KenLM MoE | 同 + general/tech 6-gram q8 + entity 4-gram (α=0.2 β=0.6) |
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

## 結果

probe_v3:

| config | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|
| suiko-v1-small-kenlm-6gram-q8-moe | 0.6782 | 0.7874 | 0.9495 | 14.0 |
| suiko-v1-small-kenlm-moe | 0.6695 | 0.7845 | 0.9484 | 15.1 |
| suiko-v1-small-kenlm-6gram-q8 | 0.6667 | 0.7874 | 0.9487 | 11.9 |
| suiko-v1-small-kenlm | 0.6580 | 0.7759 | 0.9468 | 13.5 |

ajimee_jwtd (補助, n=200):

| config | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|
| suiko-v1-small-kenlm-6gram-q8-moe | 0.6800 | 0.8150 | 0.9563 | 17.7 |
| suiko-v1-small-kenlm-moe | 0.6700 | 0.8200 | 0.9592 | 19.0 |
| suiko-v1-small-kenlm-6gram-q8 | 0.6700 | 0.8100 | 0.9558 | 14.5 |
| suiko-v1-small-kenlm | 0.6700 | 0.8300 | 0.9591 | 19.4 |
