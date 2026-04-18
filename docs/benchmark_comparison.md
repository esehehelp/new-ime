---
status: current
last_updated: 2026-04-19
---

# ベンチマーク比較 (living doc)

現状の最良結果を集約。probe_v2 (phrase-level)・cvae_probe (domain)・
旧 sentence-level bench (参考) の 3 層。新しい測定が出たら更新。

## phrase-level (probe_v2, 467 項目, 7 category)

| モデル | 設定 | EM1 | EM5 | p50 ms (WSL CPU) |
|---|---|---:|---:|---:|
| **zenz-v3.1-small** | beam=5 | **0.715** | 0.925 | 274 |
| **zenz-v2.5-small** | beam=5 | **0.700** | 0.916 | 266 |
| CTC-NAT 90M step27500 | α=0.2, β=0.6, beam=5 + KenLM | **0.612** | 0.612 | 21 |
| CTC-NAT 30M step50000 | α=0.4, β=0.3, beam=5 + KenLM | 0.499 | 0.499 | 26 |
| CTC-NAT 90M step27500 | greedy (no LM) | 0.580 | 0.580 | 12 |
| ar_v3_vast | beam=5 | 0.360 | 0.540 | 104 |

- **zenz-small との差**: 90M best で -9pt、speed 13x 有利 (CPU)
- **KenLM gain**: 90M で +3.2pt、30M で +12.4pt
- **CTC-NAT EM5==EM1**: beam が候補多様性を出せていない (Mask-CTC / 温度
  サンプリング未実装)

詳細: `docs/probe_v2_4way_results.md`

## domain-conditional (cvae_probe, 188 項目, 10 domain)

CVAE **未実装** の baseline。47% の reading で domain 間正解分岐あり。

| モデル | 設定 | EM1 |
|---|---|---:|
| CTC-NAT 90M step27500 | greedy | 0.585 |
| CTC-NAT 90M step27500 | α=0.2, β=0.6, beam=5 + KenLM | 0.574 |

domain 別 (greedy):

| domain | n | EM1 |
|---|---:|---:|
| formal | 22 | 0.77 |
| medical | 11 | 0.73 |
| general | 40 | 0.70 |
| casual | 15 | 0.67 |
| news | 6 | 0.67 |
| literary | 19 | 0.53 |
| tech | 21 | 0.52 |
| business | 14 | 0.50 |
| academic | 27 | 0.48 |
| legal | 13 | 0.15 |

理論上限 (z perfect) = 1.00。baseline → 上限で +42pt が**理論最大幅**。CVAE
実装で実際に取れる gain の期待値は未知 (10-30pt が妥当レンジ、
下は posterior collapse の 0pt)。

詳細: `docs/cvae_probe_baseline.md`

## 旧 sentence-level (参考値)

2026-04-18 取得。CTC-NAT 90M step15000 + KenLM α=0.80 β=1.0 beam=8 (WSL CPU)。

| bench | CTC baseline EM | best EM | Δ |
|---|---:|---:|---:|
| manual_test (100) | 0.700 | **0.870** | +17pt |
| ajimee_jwtd (80) | 0.225 | **0.413** | +18.75pt |
| eval_v3_dev (200) | 0.115 | **0.255** | +14pt |

KenLM は eval_v3/train で学習。eval_v3/dev は hold-out 同ドメイン、
manual/ajimee は LM 未露出ゆえ genuine gain。

### zenz-v2.5-medium (参考)

| model | manual_test EM | eval_v3_dev EM |
|---|---:|---:|
| zenz-v2.5-medium (310M) | 0.86 | 0.575 |
| zenz-v2.5-small (91M) | 0.80 | 0.50 |
| CTC-NAT 90M step15000 + KenLM | 0.87 | 0.26 |

**注**: manual_test で zenz-medium を僅差で超えているが、probe_v2 (phrase-level)
では逆転している。main 比較指標は probe_v2 に一本化する方針 (vision.md の
位置付け参照)。

## 更新履歴

- 2026-04-19: probe_v2 467 項目 4-way + CVAE probe baseline 追加、docs 整理
- 2026-04-18: CTC-NAT 90M + KenLM sweep、zenz-v2.5 3 sizes 対比 初回取得
