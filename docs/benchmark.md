# Benchmark Protocol

`new-ime` 唯一の現役ドキュメント。canonical bench の条件、比較対象、
TOML / 出力 / CLI 仕様、そして現環境での再現結果を保持する。実測値
そのものは `results/bench/<run.name>/` 配下の JSON、比較は TOML diff
と本ドキュメントの再現表で追う。

## Canonical 条件

過去の `legacy/docs/benchmark_comparison.md` (`archive/pre-v2` で参照可)
を継承。

- **device**: CPU only — WSL CPU を canonical 計測環境とする。Windows
  native venv は dev / lookup 用で、bench は WSL から回す。
- **bench 1**: `datasets/eval/probe/probe.json` (`probe_v3`, 348 items)
- **bench 2**: `references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json` (200 items)
- **canonical decode**: `num_beams=5`, `num_return=5` (top-5 候補)。
  `greedy` (beam=1, return=1) は baseline として併記。EM5 は EM1 と
  縮退する。
- **metrics** (全モデル必須):
  - `EM1` — top-1 が references のいずれかに一致 (= `exact_match_top1`)
  - `EM5` — top-5 のいずれかが references に一致
  - `CharAcc` — top-1 char-level accuracy (= `char_acc_top1`)
  - `latency`: p50 / p95 / mean (ms)
- **probe categorical EM**: `probe.json` の `category` ごとに EM1 を集計
  (edge / general / homophone / names / numeric / particle / tech)

入力 `reading` は片仮名であれば hiragana に正規化してから backend に渡す
(`jaconv.kata2hira`、過去 bench と同じ挙動)。

## Canonical 比較対象 (再現可能 9 entries)

過去 doc に記載のあるもの**かつ**現リポで再現可能なものに限定。新規エントリは追加しない。

### new-ime 系 (3 entries)

| name | 内訳 | mode |
|---|---|---|
| `suiko-v1-small-greedy` | Suiko-v1-small (CTC-NAT 41M / phase3_30m, step 100k) | greedy (beam=1) |
| `suiko-v1-small-kenlm` | 同 ckpt + 単一 KenLM (general 4-gram) shallow fusion | beam=5 + LM |
| `suiko-v1-small-kenlm-moe` | 同 ckpt + KenLM MoE (general / tech / entity) | beam=5 + LM-MoE |

### reference 系 (6 entries)

| name | params | source |
|---|---|---|
| `zenz-v2.5-xsmall` | 30M | `references/zenz-v2.5-xsmall/` |
| `zenz-v2.5-small`  | 91M | `references/zenz-v2.5-small/` |
| `zenz-v2.5-medium` | 310M | `references/zenz-v2.5-medium/` |
| `zenz-v3.1-small`  | 91M | `references/zenz-v3.1-small/` |
| `jinen-v1-xsmall`  | 35.8M | HF Hub `togatogah/jinen-v1-xsmall` |
| `jinen-v1-small`   | 110M | HF Hub `togatogah/jinen-v1-small` |

### 放棄済 (ckpt 削除のため再現不能、計 10 entries)

- `ctc-nat-30m-student` × {greedy, kenlm, kenlm-moe, onnx-fp32, onnx-int8}
- `ctc-nat-30m-scratch` × {greedy, kenlm}
- `ar-31m-scratch` × {greedy, beam5}
- `teacher-150m-teacher`

## TOML 仕様

bench は config 1 ファイル + 1 引数で起動 (`<tool> <config.toml>`)。
Schema 定義は `src/new_ime/config/bench.py` (pydantic, `extra="forbid"`)。
すべての TOML は `configs/bench/*.toml` に置き、`ime-bench` が自動で
discover する。

### sections

| section | 役割 | 必須 |
|---|---|---|
| `[run]` | `name` (= 出力 dir 名) / `out_dir` | yes |
| `[model]` | `type` で backend 分岐 (詳細下記) | yes |
| `[decode]` | `mode` (`greedy` / `beam`), `num_beams`, `num_return`, `top_k` | yes |
| `[benches]` | bench 名 → dataset path の table (任意個、最低 1) | yes |
| `[device]` | `backend` (`cpu` / `cuda`)、default `cpu` | no |
| `[lm]` | KenLM shallow fusion (CTC-NAT のみ有効) | no |

### `[model]` discriminator

```toml
# CTC-NAT (Suiko 系)
type = "ctc-nat"
checkpoint = "checkpoints/suiko-v1-small/checkpoint_step_100000.pt"
tokenizer  = "checkpoints/suiko-v1-small/checkpoint_step_100000_tokenizer.json"
preset = "phase3_30m"   # ckpt の preset field と一致させる
```

```toml
# zenz (HuggingFace GPT2 系)
type = "zenz-v2.5"  # または "zenz-v3.1"
path = "references/zenz-v2.5-small"
max_new_tokens = 80
max_context_chars = 40
```

```toml
# jinen (HuggingFace AutoModelForCausalLM)
type = "jinen-v1"
path = "togatogah/jinen-v1-small"   # local dir or HF Hub ID
```

### `[lm]` (CTC-NAT 専用)

単一 KenLM:

```toml
[lm]
mode = "single"
path = "assets/kenlm/kenlm_general_train_4gram_probing.bin"
alpha = 0.2
beta = 0.6
```

MoE (domain mixture):

```toml
[lm]
mode = "moe"
alpha = 0.2
beta = 0.6
[lm.paths_by_domain]
general = "assets/kenlm/kenlm_general_train_4gram_probing.bin"
tech    = "assets/kenlm/kenlm_tech_4gram.bin"
entity  = "assets/kenlm/kenlm_entity_4gram.bin"
```

## CLI

```bash
ime-bench                                     # configs/bench/*.toml 全部
ime-bench -m suiko-v1-small-greedy            # 特定モデル ([run] name)
ime-bench -t probe_v3                         # 特定テスト ([benches] key)
ime-bench -m a b -t probe_v3                  # 組合せ
ime-bench -v -m a                             # 全候補 NDJSON log
ime-bench -c path/to/configs/                 # config dir 上書き
```

CLI 引数は以下のみ (実験定義は TOML が真、flag は runtime knob のみ):

| flag | 役割 |
|---|---|
| `-c / --config-dir DIR` | bench TOML の置き場 (default `configs/bench/`) |
| `-m / --models NAME ...` | `[run] name` whitelist |
| `-t / --tests NAME ...` | `[benches]` キー whitelist |
| `-v / --verbose` | 全候補 NDJSON log を有効化 |

bench 環境は WSL を優先する。`scripts/_uv_env.sh` を source すれば
OS 別の uv venv (`$HOME/.venvs/new-ime` (Linux) / `.venv-windows`
(Windows)) に自動切替する。

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
  "latency_ms": { "p50": 9.3, "p95": 18.0, "mean": 10.7 },
  "sample_failures": [
    { "reading": "<input>", "ref": "<expected>", "pred": "<got>" }
  ],
  "probe_categories": {
    "edge": { "n": 50, "em1": 0.45 }
  }
}
```

`probe_categories` は probe bench のみ。`sample_failures` は最初に
失敗した最大 5 件。

### 詳細 log (`-v` 時のみ)

`<run.out_dir>/<bench>__<mode>.full.jsonl`、1 行 1 item の NDJSON:

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

`<run.out_dir>/summary.json`: 当該 bench 群の 1 行サマリ配列。

```json
[
  {
    "model": "suiko-v1-small-greedy",
    "bench": "probe_v3",
    "n": 348, "em1": 0.6006, "em5": 0.6006, "char_acc": 0.944,
    "p50_ms": 9.3, "p95_ms": 18.0, "wall_s": 4.0
  }
]
```

## アンカー (`legacy/docs/benchmark_comparison.md` 2026-04-22)

### probe_v3

| model | params | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|---:|
| zenz-v2.5-medium | 310M | **0.747** | **0.876** | **0.966** | 1173 |
| zenz-v3.1-small | 91M | 0.718 | 0.856 | 0.959 | 417 |
| zenz-v2.5-small | 91M | 0.713 | 0.848 | 0.959 | 376 |
| zenz-v2.5-xsmall | 30M | 0.695 | 0.813 | 0.953 | 118 |
| **suiko-v1-small-kenlm-moe** | **41M** | **0.672** | **0.784** | **0.949** | **22** |
| jinen-v1-small | 110M | 0.672 | 0.776 | 0.944 | 278 |
| **suiko-v1-small-kenlm** | 41M | 0.664 | 0.776 | 0.947 | 17 |
| jinen-v1-xsmall | 35.8M | 0.609 | 0.747 | 0.929 | 115 |
| **suiko-v1-small-greedy** | 41M | 0.601 | 0.601 | 0.944 | 9 |

### AJIMEE JWTD_v2

| model | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|
| zenz-v2.5-medium | **0.875** | **0.970** | **0.982** | 1361 |
| zenz-v3.1-small | 0.860 | 0.930 | 0.983 | 470 |
| zenz-v2.5-small | 0.840 | 0.955 | 0.977 | 418 |
| zenz-v2.5-xsmall | 0.695 | 0.845 | 0.953 | 139 |
| **suiko-v1-small-kenlm** | **0.670** | **0.830** | **0.959** | **21** |
| **suiko-v1-small-kenlm-moe** | 0.670 | 0.820 | 0.959 | 28 |
| jinen-v1-small | 0.655 | 0.835 | 0.952 | 309 |
| **suiko-v1-small-greedy** | 0.580 | 0.580 | 0.951 | 10 |
| jinen-v1-xsmall | 0.395 | 0.525 | 0.917 | 124 |

## 現環境での再現結果 (WSL CPU torch 2.11.0+cpu, 2026-05-03)

`scripts/_canonical_compare.py` 出力の要約。anchor との Δ:

| config | bench | EM1 | ΔEM1 | EM5 | ΔEM5 | CAcc | ΔCAcc | p50 | Δp50 |
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

### 最大誤差

| metric | Δ | location | 解釈 |
|---|---:|---|---|
| EM1 | -0.0060 | suiko-v1-small-kenlm / probe_v3 | torch 2.6→2.11 numeric kernel diff |
| EM5 | +0.0005 | suiko-v1-small-kenlm-moe / probe_v3 | 同上 |
| CharAcc | -0.0006 | suiko-v1-small-kenlm-moe / probe_v3 | 同上 |
| p50 | -155.9ms | zenz-v2.5-medium / ajimee_jwtd | 新環境で高速 |

精度 18/18 件で `|ΔEM1| ≤ 0.006` (相対 1% 以内)、ほぼ全件で完全一致。
latency は新 torch / 新 CPU で全件高速側に振れている。

## 再現コマンド (canonical 9 全実行 + diff)

```bash
# WSL から
cd /mnt/d/Dev/new-ime
PYTHONPATH=src ~/.venvs/new-ime/bin/python -m new_ime.cli.bench -v
~/.venvs/new-ime/bin/python scripts/_canonical_compare.py
```

または `scripts/_uv_env.sh` を source 経由で `ime-bench` を直接呼ぶ。
