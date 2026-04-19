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
| **CTC-NAT 30m_v2 step49000** | α=0.2, β=0.6, beam=5 + KenLM | **0.779** | 0.779 | 12 |
| **CTC-NAT 30m_v2 step49000** | greedy (no LM) | **0.739** | 0.739 | 8 |
| **teacher-150m-teacher step100000** | greedy (AR) | **0.739** | 0.739 | 40 |
| **zenz-v3.1-small** | beam=5 | 0.715 | 0.925 | 274 |
| **zenz-v2.5-small** | beam=5 | 0.700 | 0.916 | 266 |
| CTC-NAT 90M step27500 | α=0.2, β=0.6, beam=5 + KenLM | 0.612 | 0.612 | 21 |
| CTC-NAT 30M step50000 | α=0.4, β=0.3, beam=5 + KenLM | 0.499 | 0.499 | 26 |
| CTC-NAT 90M step27500 | greedy (no LM) | 0.580 | 0.580 | 12 |
| ar_v3_vast | beam=5 | 0.360 | 0.540 | 104 |

- **30m_v2 (新 mix, synth_numeric 含む) が greedy で zenz-v3.1 を +2.4pt 超え**。
  KenLM 付きで +6.4pt、速度 23x 有利 (CPU)
- **KenLM gain は 30m_v2 で +4.0pt** (旧 30M の +12.4pt から縮小) — numeric
  を学習済みで LM 依存度が低下。α=0.6 で numeric 崩壊 (0.88→0.01) するため
  強めの α は禁忌
- **teacher 150m は numeric 0.96**、30m_v2 は 0.88 — 両者とも旧 90M の 0.02
  から大幅改善
- **edge カテゴリ退行**: 30m_v2 edge 0.39、teacher 0.18、旧 90M step27500
  の 0.68 から退行。corpus_v2 の mix / tokenizer 変更を疑う
  (`project_phase3_edge_regression` memo)
- **CTC-NAT EM5==EM1**: beam が候補多様性を出せていない (Mask-CTC / 温度
  サンプリング未実装)

詳細: `docs/probe_v2_4way_results.md`

### 30m_v2 KenLM α×β sweep (probe_v2)

| 設定 | EM1 | numeric | edge | homo | particle |
|---|---:|---:|---:|---:|---:|
| greedy | 0.739 | 0.880 | 0.394 | 0.562 | 0.500 |
| **α=0.2 β=0.6** | **0.779** | 0.863 | 0.521 | 0.630 | 0.650 |
| α=0.4 β=0.6 | 0.711 | 0.538 | 0.535 | 0.630 | 0.700 |
| α=0.6 β=0.6 | 0.582 | 0.009 | 0.535 | 0.616 | 0.750 |

phrase 単位では **α=0.2 が最適**。α を上げると particle / edge は伸びるが
numeric が破壊されて総合では負ける。

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

## sentence-level (manual / ajimee / general)

### 2026-04-19: 30m_v2 + KenLM α×β sweep (beam=5)

`results/eval_runs_30mv2_kenlm_sweep/`。manual=100, ajimee=80, general_dev=80。

| 設定 | manual EM | ajimee EM | general eval EM |
|---|---:|---:|---:|
| greedy (no LM) | 0.830 | 0.338 | 0.263 |
| α=0.2 β=0.6 | 0.860 | 0.500 | 0.300 |
| **α=0.4 β=0.6** | **0.870** | **0.575** | **0.300** |
| α=0.4 β=0.3 | 0.870 | 0.575 | 0.300 |
| α=0.6 β=0.6 | 0.870 | 0.525 | 0.250 |

- **文単位の最適 α は 0.4** (phrase の 0.2 より大きい)。**α=0.6 は general eval で
  baseline 以下に劣化**
- **ajimee で +23.75pt** (0.338→0.575)。文単位 KenLM の効果が最大
- **general の gain は +3.75pt** と小さい。長文 wiki は LM では埋めにくく、
  chunk decoding との併用が必要
- 単一設定で使うなら **α=0.4, β=0.6**。probe_v2 では 0.711 (最適 0.779 から
  -6.8pt だが zenz 0.715 と同等)

### 3-bench 4-way 比較 (2026-04-19)

`results/eval_runs_2026_04_19_30mv2_teacher/`。greedy。

| model | manual EM / p50 | ajimee EM / p50 | general eval EM / p50 |
|---|---:|---:|---:|
| zenz-v2.5-small | 0.890 / 197ms | 0.750 / 370ms | 0.375 / 593ms |
| teacher-150m-teacher step100000 | 0.850 / 80ms | 0.588 / 153ms | 0.363 / 265ms |
| ctc-nat-30m-student step49000 | 0.830 / 21ms | 0.338 / 21ms | 0.263 / 20ms |
| ctc-nat-30m-student + α=0.4 β=0.6 | 0.870 / ~15ms | 0.575 / ~22ms | 0.300 / 37ms |

**CharAcc** は teacher 150m が general eval で 0.893 と zenz-small の 0.857 を
**超えている**。bidirectional encoder による長文の局所整合性の優位。

### 旧 sentence-level (参考値)

2026-04-18 取得。CTC-NAT 90M step15000 + KenLM α=0.80 β=1.0 beam=8 (WSL CPU)。

| bench | CTC baseline EM | best EM | Δ |
|---|---:|---:|---:|
| manual_test (100) | 0.700 | **0.870** | +17pt |
| ajimee_jwtd (80) | 0.225 | **0.413** | +18.75pt |
| general_dev (200) | 0.115 | **0.255** | +14pt |

KenLM は general/train で学習。general/dev は hold-out 同ドメイン、
manual/ajimee は LM 未露出ゆえ genuine gain。

### zenz-v2.5-medium (参考)

| model | manual_test EM | general_dev EM |
|---|---:|---:|
| zenz-v2.5-medium (310M) | 0.86 | 0.575 |
| zenz-v2.5-small (91M) | 0.80 | 0.50 |
| CTC-NAT 90M step15000 + KenLM | 0.87 | 0.26 |

**注**: manual_test で zenz-medium を僅差で超えているが、probe_v2 (phrase-level)
では逆転している。main 比較指標は probe_v2 に一本化する方針 (vision.md の
位置付け参照)。

## 更新履歴

- 2026-04-19 (追記): ctc-nat-30m-student step49000 + teacher-150m-teacher step100000 を
  probe_v2 / 3-bench で評価、30m_v2 KenLM α×β sweep を probe_v2 と 3-bench
  両方で実施。30m_v2 が phrase / sentence 両領域で zenz-small 比肩に到達、
  edge カテゴリ退行を検出
- 2026-04-19: probe_v2 467 項目 4-way + CVAE probe baseline 追加、docs 整理
- 2026-04-18: CTC-NAT 90M + KenLM sweep、zenz-v2.5 3 sizes 対比 初回取得
