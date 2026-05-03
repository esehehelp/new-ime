# Benchmark Protocol

`new-ime` の **canonical benchmark 条件** と出力フォーマットを定義する。
文書は protocol だけを保持。実測値は `results/` 配下の JSON、比較は
TOML diff で追う。

## Canonical 条件

- **device**: CPU only
- **bench 1**: `datasets/eval/probe/probe.json` (`probe_v3`, 348 items)
- **bench 2**: `references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json` (200 items)
- **decoding** (両方を計測する):
  - `greedy`: `num_beams=1`, `num_return=1` (top_k=5 まで返す model も top_k=5 で評価)
  - `beam5`: `num_beams=5`, `num_return=5`
- **metrics**:
  - `EM1` — top-1 が references のいずれかに一致 (= `exact_match_top1`)
  - `EM5` — top-5 のいずれかが references に一致
  - `CharAcc` — top-1 char-level accuracy (= `char_acc_top1`)
  - `latency`: p50 / p95 / mean (ms)
- **probe categorical EM**: `probe.json` の `category` ごとに EM1 を集計
  (edge / general / homophone / names / numeric / particle / tech)

入力 `reading` は片仮名であれば hiragana に正規化してから backend に渡す
(過去 bench と同じ挙動、`jaconv.kata2hira`)。

## TOML 契約

bench は config 1 ファイル + 1 引数で起動:

```toml
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
probe_v3    = "datasets/eval/probe/probe.json"
ajimee_jwtd = "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json"

[device]
backend = "cpu"
```

実行:

```bash
ime-bench configs/bench/canonical-greedy.toml
# または
python -m new_ime.cli.bench configs/bench/canonical-greedy.toml

# 詳細 log 付き (全 LLM 出力を残し検証可能にする)
ime-bench -v configs/bench/canonical-greedy.toml
```

`-v` を付けると、各 bench につき `<out_dir>/<bench>__<mode>.full.jsonl`
が追加で書かれる。1 行 1 item の NDJSON で、backend が返した全候補
(top_k で truncate せず) を保持する。後から metrics の再計算や個別
失敗例の調査に使う。

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
    "edge": { "n": 50, "em1": 0.45 },
    "general": { "n": 50, "em1": 0.47 },
    "homophone": { "n": 50, "em1": 0.24 },
    "names": { "n": 50, "em1": 0.42 },
    "numeric": { "n": 50, "em1": 0.40 },
    "particle": { "n": 50, "em1": 0.97 },
    "tech": { "n": 48, "em1": 0.43 }
  }
}
```

`probe_categories` は probe bench のみ含まれる。

### 詳細 log (`-v` 時のみ)

`<run.out_dir>/<bench>__<mode>.full.jsonl`, 1 行 1 item の NDJSON:

```json
{
  "i": 0,
  "bench": "probe_v3",
  "index": "0001",
  "category": "general",
  "reading": "わたしはがくせいです",
  "context": "",
  "references": ["私は学生です"],
  "candidates": ["私は学生です", "わたしは学生です", "..."],
  "em1": 1,
  "em5": 1,
  "char_acc_top1": 1.0,
  "char_acc_topk": 1.0,
  "latency_ms": 16.32
}
```

`candidates` は backend が返した **全候補** (top_k で truncate しない)。
metrics 再計算は各行の `references` と `candidates` から可能。

### summary JSON

`<run.out_dir>/summary.json`:

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
    "wall_s": 6.6
  }
]
```

## アンカー (Suiko-v1-small, 2026-04-26 計測)

| bench | mode | EM1 | EM5 | CharAcc | p50 ms |
|---|---|---:|---:|---:|---:|
| probe_v3 | greedy | 0.601 | 0.601 | 0.944 | 9 |
| ajimee_jwtd | greedy | 0.580 | 0.580 | 0.951 | 10 |
| probe_v3 | kenlm | 0.664 | 0.776 | 0.947 | 17 |
| ajimee_jwtd | kenlm | 0.670 | 0.830 | 0.959 | 21 |

(kenlm は LM fusion 有効時。v2 では別 config で扱う予定)

v2 restructure の **smoke 目標**: greedy で probe_v3 EM1 ≥ 0.59 を再現
できること。下回ったら restructure 過程で何かを壊している。
