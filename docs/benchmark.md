# Benchmark Protocol

`new-ime` の **canonical benchmark 条件** と canonical 比較対象モデル、
出力フォーマットを定義する。文書は protocol だけを保持。実測値は
`results/` 配下の JSON、比較は TOML diff で追う。

## Canonical 条件

過去の `legacy/docs/benchmark_comparison.md` (archive/pre-v2 で参照可)
を継承。

- **device**: CPU only
- **bench 1**: `datasets/eval/probe/probe.json` (`probe_v3`, 348 items)
- **bench 2**: `references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json` (200 items)
- **canonical decode**: `num_beams=5`, `num_return=5` (top-5 候補返却)
- **metrics** (全モデル必須):
  - `EM1` — top-1 が references のいずれかに一致 (= `exact_match_top1`)
  - `EM5` — top-5 のいずれかが references に一致
  - `CharAcc` — top-1 char-level accuracy (= `char_acc_top1`)
  - `latency`: p50 / p95 / mean (ms)
- **probe categorical EM**: `probe.json` の `category` ごとに EM1 を集計
  (edge / general / homophone / names / numeric / particle / tech)

`greedy` (beam=1, num_return=1) は baseline 計測としてのみ使用。EM5 は
EM1 と縮退するが過去 doc と整合性を保つために維持する。

入力 `reading` は片仮名であれば hiragana に正規化してから backend に渡す
(過去 bench と同じ挙動、`jaconv.kata2hira`)。

## Canonical 比較対象 (過去 doc 由来)

新規エントリは追加せず、過去 doc に記載のあるものに限定する。削除済の
ckpt (ctc-nat-30m-student / ar-31m / ctc-nat-30m-scratch / teacher-150m)
は再現不能のため放棄。

### new-ime 系 (Suiko-v1-small × 3 modes)

| name | mode | 備考 |
|---|---|---|
| suiko-v1-small-greedy | greedy | EM5 == EM1 (縮退、baseline) |
| suiko-v1-small-kenlm | beam5 + KenLM (general) | canonical 条件 |
| suiko-v1-small-kenlm-moe | beam5 + KenLM MoE (general / entity / tech) | canonical 条件 |

### reference 系 (HF transformers backend 経由)

| name | params | source |
|---|---|---|
| zenz-v2.5-xsmall | 30M | references/zenz-v2.5-xsmall/ |
| zenz-v2.5-small | 91M | references/zenz-v2.5-small/ |
| zenz-v2.5-medium | 310M | references/zenz-v2.5-medium/ |
| zenz-v3.1-small | 91M | references/zenz-v3.1-small/ |
| jinen-v1-xsmall | 35.8M | references/jinen-* (HF) |
| jinen-v1-small | 110M | 同上 |

## TOML 契約

bench は config 1 ファイル + 1 引数で起動 (`<tool> <config.toml>`)。
schema 定義: `src/new_ime/config/bench.py`。

例 (`configs/bench/suiko-v1-small-greedy.toml`):

```toml
[run]
name = "suiko-v1-small-greedy"
out_dir = "results/bench/suiko-v1-small-greedy"

[model]
checkpoint = "checkpoints/suiko-v1-small/checkpoint_step_100000.pt"
tokenizer  = "checkpoints/suiko-v1-small/checkpoint_step_100000_tokenizer.json"
preset = "phase3_30m"

[decode]
mode = "greedy"
num_beams = 1
num_return = 1
top_k = 5

[benches]
probe_v3    = "datasets/eval/probe/probe.json"
ajimee_jwtd = "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json"

[device]
backend = "cpu"
```

## 実行

```bash
ime-bench                                # configs/bench/*.toml 全部
ime-bench -m suiko-v1-small-greedy       # 特定モデル ([run] name)
ime-bench -t probe_v3                    # 特定テスト ([benches] key)
ime-bench -m a b -t probe_v3             # 組合せ
ime-bench -v -m a                        # 全候補 NDJSON log
ime-bench -c path/to/configs/            # config dir 上書き
```

CLI 引数 (実験定義は TOML が真、flag は runtime knob のみ):

| flag | 役割 |
|---|---|
| `-c / --config-dir DIR` | bench TOML の置き場 (default `configs/bench/`) |
| `-m / --models NAME ...` | `[run] name` whitelist |
| `-t / --tests NAME ...` | `[benches]` キー whitelist |
| `-v / --verbose` | 全候補 NDJSON log を有効化 |

## 出力フォーマット

### 各 bench JSON

`<run.out_dir>/<bench_name>__<decode.mode>.json`:

```json
{
  "total": 348,
  "char_acc_top1": 0.944,
  "exact_match_top1": 0.6006,
  "char_acc_top5": 0.944,
  "exact_match_top5": 0.6006,
  "char_acc_top10": 0.944,
  "exact_match_top10": 0.6006,
  "n": 348,
  "em5": 0.6006,
  "latency_ms": { "p50": 16.4, "p95": 29.9, "mean": 17.8 },
  "sample_failures": [
    { "reading": "<input>", "ref": "<expected>", "pred": "<got>" }
  ],
  "probe_categories": {
    "edge": { "n": 50, "em1": 0.45 }
  }
}
```

`probe_categories` は probe bench のみ含まれる。

### 詳細 log (`-v` 時のみ)

`<run.out_dir>/<bench>__<mode>.full.jsonl`, 1 行 1 item の NDJSON:

```json
{
  "i": 0, "bench": "probe_v3", "index": "0001", "category": "general",
  "reading": "わたしはがくせいです", "context": "",
  "references": ["私は学生です"],
  "candidates": ["私は学生です", "わたしは学生です", "..."],
  "em1": 1, "em5": 1, "char_acc_top1": 1.0, "char_acc_topk": 1.0,
  "latency_ms": 16.32
}
```

`candidates` は backend が返した **全候補** (top_k で truncate しない)。
metrics 再計算は各行の `references` と `candidates` から可能。

### summary JSON

`<run.out_dir>/summary.json`: `[{model, bench, n, em1, em5, char_acc,
p50_ms, p95_ms, wall_s}]` の配列。

## アンカー (過去 benchmark_comparison.md, 2026-04-22 確定値)

### probe_v3

| model | EM1 | EM5 | CharAcc | p50 |
|---|---:|---:|---:|---:|
| zenz-v2.5-medium | 0.747 | 0.876 | 0.966 | 1173 |
| zenz-v3.1-small | 0.718 | 0.856 | 0.959 | 417 |
| zenz-v2.5-small | 0.713 | 0.848 | 0.959 | 376 |
| zenz-v2.5-xsmall | 0.695 | 0.813 | 0.953 | 118 |
| **suiko-v1-small-kenlm-moe** | 0.672 | 0.784 | 0.949 | 22 |
| jinen-v1-small | 0.672 | 0.776 | 0.944 | 278 |
| **suiko-v1-small-kenlm** | 0.664 | 0.776 | 0.947 | 17 |
| jinen-v1-xsmall | 0.609 | 0.747 | 0.929 | 115 |
| **suiko-v1-small-greedy** | 0.601 | 0.601 | 0.944 | 9 |

### AJIMEE JWTD_v2

| model | EM1 | EM5 | CharAcc | p50 |
|---|---:|---:|---:|---:|
| zenz-v2.5-medium | 0.875 | 0.970 | 0.982 | 1361 |
| zenz-v3.1-small | 0.860 | 0.930 | 0.983 | 470 |
| zenz-v2.5-small | 0.840 | 0.955 | 0.977 | 418 |
| zenz-v2.5-xsmall | 0.695 | 0.845 | 0.953 | 139 |
| **suiko-v1-small-kenlm** | 0.670 | 0.830 | 0.959 | 21 |
| **suiko-v1-small-kenlm-moe** | 0.670 | 0.820 | 0.959 | 28 |
| jinen-v1-small | 0.655 | 0.835 | 0.952 | 309 |
| **suiko-v1-small-greedy** | 0.580 | 0.580 | 0.951 | 10 |
| jinen-v1-xsmall | 0.395 | 0.525 | 0.917 | 124 |
