# Benchmark Protocol

`new-ime` の **canonical benchmark 条件** を定義する。文書は protocol の
仕様だけを保つ。実測値は `results/` 配下の JSON、比較は TOML diff で
追う。

## Canonical 条件

- **device**: CPU only (Windows native venv 経由でも可)
- **bench 1**: `datasets/eval/probe/probe.json` (`probe_v3`, 348 items)
- **bench 2**: `references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json` (200 items)
- **decoding**:
  - canonical: `greedy` (`num_beams=1`, `num_return=1`)
  - 比較用: `beam5` (`num_beams=5`, `num_return=5`)
- **metrics**:
  - `EM1` — top-1 が references のいずれかに一致
  - `EM5` — top-5 のいずれかが references に一致 (beam モードのみ意味あり)
  - `CharAcc` — 文字単位 accuracy
  - `p50`, `p95` latency (ms)
- **probe categorical EM**: `probe.json` のカテゴリタグごとに EM1 を集計
  (edge / general / homophone / names / numeric / particle / tech)

## TOML 契約

bench は config 1 ファイル + 1 引数で起動する:

```toml
# configs/bench/canonical-greedy.toml
[run]
name = "suiko-v1-small-greedy"
out_dir = "results/bench/suiko-v1-small-greedy"

[model]
checkpoint = "checkpoints/suiko-v1-small/checkpoint_step_100000.pt"
tokenizer  = "checkpoints/suiko-v1-small/checkpoint_step_100000_tokenizer.json"
preset = "ctc-nat-41m"

[decode]
mode = "greedy"        # "greedy" | "beam"
num_beams = 1
num_return = 1
top_k = 5

[benches]
probe_v3 = "datasets/eval/probe/probe.json"
ajimee_jwtd = "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json"

[device]
backend = "cpu"
```

## 出力フォーマット

各 bench につき 1 JSON:

```
results/bench/<run.name>/<bench>__<decode.mode>.json
results/bench/<run.name>/summary.json
```

`summary.json` schema:

```json
[
  {
    "model": "suiko-v1-small-greedy",
    "bench": "probe_v3",
    "n": 348,
    "em1": 0.6006,
    "em5": 0.6006,
    "char_acc": 0.944,
    "p50_ms": 16.4,
    "p95_ms": 29.9,
    "wall_s": 6.6,
    "probe_categories": {
      "edge": 0.45,
      "general": 0.47,
      "homophone": 0.24,
      "names": 0.42,
      "numeric": 0.40,
      "particle": 0.97,
      "tech": 0.43
    }
  }
]
```

## 既知のアンカー (Suiko-v1-small, greedy)

| bench | n | EM1 | CharAcc | p50 ms |
|---|---|---|---|---|
| probe_v3 | 348 | 0.6006 | 0.944 | 16.4 |
| ajimee_jwtd | 200 | 0.58 | 0.9509 | 18.7 |

(2026-04-26 計測, `archive/pre-v2` の `results/bench_v1_vs_v1_2/summary.json`)

v2 の最初の smoke は **これを再現できること**。再現できなければ
restructure 過程で何かを壊している。
