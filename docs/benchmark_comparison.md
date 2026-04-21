---
status: current
last_updated: 2026-04-21
---

# Benchmark Comparison

この文書は `new-ime` の **canonical benchmark 条件** と、その条件で比較した主要モデルの結果だけを残します。細かな sweep や一時的な測定メモはここに増やしません。

## Canonical 条件

比較条件は固定です。

- device: `CPU only`
- bench 1: `datasets/eval/probe/probe.json` (`probe_v3`, 348 items)
- bench 2: `references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json` (200 items)
- decoding:
  - `num_beams=5`
  - `num_return=5`
- metrics:
  - `EM1`
  - `EM5`
  - `CharAcc`
  - `p50 latency`

`jinen` 系だけは Hugging Face 実装都合で native Windows CPU `.venv` から回しています。精度比較は可能ですが、latency は WSL CPU 計測と完全同条件ではありません。

## いま比較するモデル

### new-ime 系

- `ctc-nat-30m-student step160000 greedy`
- `ctc-nat-30m-student step160000 kenlm`
- `ctc-nat-30m-student step160000 kenlm-moe`
- `ctc-nat-30m-student step160000 onnx-fp32 greedy`
- `ctc-nat-30m-student step160000 onnx-int8 greedy`

### reference

- `zenz-v2.5-xsmall`
- `zenz-v2.5-small`
- `zenz-v2.5-medium`
- `zenz-v3.1-small`
- `jinen-v1-xsmall`
- `jinen-v1-small`
- `teacher-150m-teacher`

## probe_v3

| model | params | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|---:|
| zenz-v2.5-medium | 310M | **0.747** | **0.876** | **0.966** | 1173 |
| zenz-v3.1-small | 91M | 0.718 | 0.856 | 0.959 | 417 |
| zenz-v2.5-small | 91M | 0.713 | 0.848 | 0.959 | 376 |
| teacher-150m-teacher | 150M | 0.698 | 0.698 | 0.950 | 288 |
| zenz-v2.5-xsmall | 30M | 0.695 | 0.813 | 0.953 | 118 |
| jinen-v1-small | 110M | 0.672 | 0.776 | 0.944 | 278 |
| **ctc-nat-30m-student kenlm-moe** | **30M** | **0.669** | **0.770** | **0.948** | **22** |
| ctc-nat-30m-student kenlm | 30M | 0.658 | 0.759 | 0.943 | 17 |
| ctc-nat-30m-student greedy | 30M | 0.609 | 0.609 | 0.942 | **10** |
| jinen-v1-xsmall | 35.8M | 0.609 | 0.747 | 0.929 | 115 |
| ctc-nat-30m-student onnx-fp32 greedy | 30M | 0.609 | 0.609 | 0.942 | 23 |
| ctc-nat-30m-student onnx-int8 greedy | 30M | 0.603 | 0.603 | 0.940 | 13 |
| ar-31m-scratch beam5 | 31M | 0.575 | 0.756 | 0.899 | 281 |
| ar-31m-scratch greedy | 31M | 0.563 | 0.563 | 0.897 | 76 |
| ctc-nat-30m-scratch kenlm | 30M | 0.540 | 0.655 | 0.900 | 20 |
| ctc-nat-30m-scratch greedy | 30M | 0.417 | 0.417 | 0.884 | 11 |

## AJIMEE JWTD_v2

| model | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|
| zenz-v2.5-medium | **0.875** | **0.970** | **0.982** | 1361 |
| zenz-v3.1-small | 0.860 | 0.930 | 0.983 | 470 |
| zenz-v2.5-small | 0.840 | 0.955 | 0.977 | 418 |
| teacher-150m-teacher | 0.715 | 0.715 | 0.964 | 322 |
| zenz-v2.5-xsmall | 0.695 | 0.845 | 0.953 | 139 |
| **jinen-v1-small** | **0.655** | **0.835** | **0.952** | 309 |
| **ctc-nat-30m-student kenlm-moe** | **0.655** | 0.810 | 0.959 | **26** |
| ctc-nat-30m-student kenlm | 0.650 | 0.815 | 0.959 | 20 |
| ctc-nat-30m-student greedy | 0.550 | 0.550 | 0.949 | **11** |
| jinen-v1-xsmall | 0.395 | 0.525 | 0.917 | 124 |
| ctc-nat-30m-student onnx-fp32 greedy | 0.550 | 0.550 | 0.949 | 22 |
| ctc-nat-30m-student onnx-int8 greedy | 0.515 | 0.515 | 0.945 | 13 |
| ar-31m-scratch beam5 | 0.480 | 0.725 | 0.882 | 321 |
| ctc-nat-30m-scratch kenlm | 0.480 | 0.640 | 0.909 | 23 |
| ar-31m-scratch greedy | 0.470 | 0.470 | 0.883 | 81 |
| ctc-nat-30m-scratch greedy | 0.280 | 0.280 | 0.882 | 14 |

## probe_v3 category 別 EM1

| model | edge | general | homo | names | numeric | particle | tech |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctc-nat-30m-student greedy | 0.575 | 0.587 | 0.405 | 0.618 | 0.585 | 0.875 | 0.682 |
| ctc-nat-30m-student kenlm | 0.650 | 0.667 | 0.460 | 0.673 | 0.615 | 0.906 | 0.682 |
| **ctc-nat-30m-student kenlm-moe** | **0.675** | 0.667 | 0.460 | 0.673 | **0.631** | 0.906 | **0.727** |
| ctc-nat-30m-student onnx-fp32 greedy | 0.575 | 0.587 | 0.405 | 0.618 | 0.585 | 0.875 | 0.682 |
| ctc-nat-30m-student onnx-int8 greedy | 0.575 | 0.600 | 0.405 | 0.618 | 0.554 | 0.875 | 0.659 |
| zenz-v2.5-xsmall | 0.825 | 0.667 | 0.460 | 0.709 | 0.661 | 0.875 | 0.727 |
| jinen-v1-xsmall | 0.725 | 0.613 | 0.432 | 0.618 | 0.523 | 0.812 | 0.614 |
| zenz-v2.5-small | 0.800 | 0.680 | 0.486 | 0.727 | 0.661 | 0.906 | 0.795 |
| jinen-v1-small | 0.825 | 0.653 | 0.486 | 0.673 | 0.615 | 0.781 | 0.727 |
| zenz-v2.5-medium | 0.825 | 0.707 | 0.540 | 0.818 | 0.677 | 0.875 | 0.841 |
| zenz-v3.1-small | 0.775 | 0.667 | 0.486 | 0.818 | 0.661 | 0.875 | 0.795 |
| teacher-150m-teacher | 0.775 | 0.667 | 0.486 | 0.727 | 0.661 | 0.906 | 0.727 |

## 要点

### ctc-nat-30m-student の立ち位置

- `kenlm-moe` が現状の最良構成です
- 同サイズ帯の `zenz-v2.5-xsmall` にはまだ少し届きません
- ただし CPU レイテンシは大きく勝っています

### KenLM の効き方

- greedy から `kenlm` で大きく伸びます
- `kenlm-moe` は probe と tech / edge 側の改善に効きます
- int8 の劣化は LM 併用でかなり吸収できます

### jinen の位置づけ

- `jinen-v1-small` は精度だけ見ればかなり近いです
- ただし CPU latency は `ctc-nat-30m-student` よりかなり重いです
- `jinen-v1-xsmall` は速度は近い帯ですが、精度差が大きいです

## production 候補

いま production 寄りに見るなら次の 2 つです。

- 最高精度寄り: `ctc-nat-30m-student kenlm-moe`
- 実装・配布寄り: `ctc-nat-30m-student onnx-int8 + KenLM`

後者は size と latency が良く、実運用への落とし込みがしやすいです。

## 実行コマンド

canonical 再生成はこれです。

```bash
cd /mnt/d/Dev/new-ime
PYTHONPATH=. python3 tools/misc/bench_all.py
```

`jinen` を個別に回すときは専用 runner も使えます。

```bash
PYTHONPATH=. python models/tools/eval/run_jinen_bench.py
```
