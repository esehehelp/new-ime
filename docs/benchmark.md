# Benchmark

## 実行

WSL CPU を前提:

```bash
cd /mnt/d/Dev/new-ime
UV_PROJECT_ENVIRONMENT=.venv-wsl PYTHONPATH=src uv run python -m new_ime.cli.bench
```

`configs/bench/*.toml` がすべて実行される。絞る場合のみ `-m` / `-t` を使う。

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl PYTHONPATH=src \
  uv run python -m new_ime.cli.bench -m suiko-v1-small-kenlm-6gram-q8-moe -t probe_v3
```

## 評価指標

| 列 | 定義 |
|---|---|
| EM1 | top-1 候補が references に完全一致する割合 (素値) |
| EM1 (NFKC) | EM1 を NFKC 正規化後に比較 |
| EM5 (NFKC) | top-5 のうち少なくとも 1 件が references に一致する割合 (NFKC) |
| CharAcc (NFKC) | 1 − normalized Levenshtein distance、top-1 vs best reference (NFKC) |
| p50 ms | per-item latency 中央値 |
| runtime | `rust` = Rust daemon (ONNX Runtime CPU) / `hf` = HF Transformers (PyTorch CPU) |

p50 は同一 runtime 内のみ比較可。`rust` と `hf` は framework overhead が異なるため別軸。

## 比較対象

| config | type | source / 設定 |
|---|---|---|
| `suiko-v1-small-current` | ctc-nat + KenLM MoE 6gram-q8 (Rust) | beam=10, α=0.3 β=0.6 — 現行 TSF と同設定 |
| `zenz-v2.5-xsmall` | zenz-v2.5 (HF) | `references/zenz-v2.5-xsmall/` |
| `zenz-v2.5-small` | zenz-v2.5 (HF) | `references/zenz-v2.5-small/` |
| `zenz-v2.5-medium` | zenz-v2.5 (HF) | `references/zenz-v2.5-medium/` |
| `zenz-v3.1-small` | zenz-v3.1 (HF) | `references/zenz-v3.1-small/` |
| `jinen-v1-xsmall` | jinen-v1 (HF) | HF `togatogah/jinen-v1-xsmall` |
| `jinen-v1-small` | jinen-v1 (HF) | HF `togatogah/jinen-v1-small` |

## 結果

`probe_v3` (n=348)、EM1 (NFKC) 降順:

| config | EM1 | EM1 (NFKC) | EM5 (NFKC) | CharAcc (NFKC) | p50 ms | runtime |
|---|---:|---:|---:|---:|---:|:---:|
| `zenz-v2.5-medium` | 0.7471 | 0.7557 | 0.8822 | 0.9669 | 1215.7 | hf |
| `zenz-v3.1-small` | 0.7184 | 0.7213 | 0.8592 | 0.9596 | 398.7 | hf |
| `zenz-v2.5-small` | 0.7126 | 0.7184 | 0.8534 | 0.9596 | 403.0 | hf |
| `zenz-v2.5-xsmall` | 0.6954 | 0.7011 | 0.8161 | 0.9533 | 124.3 | hf |
| `jinen-v1-small` | 0.6724 | 0.6983 | 0.8075 | 0.9479 | 230.5 | hf |
| `jinen-v1-xsmall` | 0.6092 | 0.6264 | 0.7730 | 0.9341 | 90.4 | hf |
| `suiko-v1-small-current` | 0.5776 | 0.5948 | 0.6925 | 0.8254 | 33.6 | rust |

`ajimee_jwtd` (n=200)、EM1 (NFKC) 降順:

| config | EM1 | EM1 (NFKC) | EM5 (NFKC) | CharAcc (NFKC) | p50 ms | runtime |
|---|---:|---:|---:|---:|---:|:---:|
| `zenz-v2.5-medium` | 0.8750 | 0.8800 | 0.9700 | 0.9821 | 1344.2 | hf |
| `zenz-v3.1-small` | 0.8600 | 0.8650 | 0.9350 | 0.9835 | 437.6 | hf |
| `zenz-v2.5-small` | 0.8400 | 0.8450 | 0.9600 | 0.9769 | 438.6 | hf |
| `zenz-v2.5-xsmall` | 0.6950 | 0.6950 | 0.8450 | 0.9532 | 133.1 | hf |
| `jinen-v1-small` | 0.6550 | 0.6950 | 0.8900 | 0.9589 | 262.5 | hf |
| `suiko-v1-small-current` | 0.6300 | 0.6700 | 0.7950 | 0.9272 | 39.2 | rust |
| `jinen-v1-xsmall` | 0.3950 | 0.4050 | 0.5400 | 0.9221 | 95.8 | hf |
