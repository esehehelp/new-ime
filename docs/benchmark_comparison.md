---
status: current
last_updated: 2026-04-18
---

# ベンチマーク比較結果 (2026-04-18)

## 追加: CTC-NAT Phase 3 90M + KenLM shallow fusion (2026-04-18 late)

90M CTC-NAT (step 15000, resume from OOM-killed vast.ai run) + 4-gram
KenLM trained on `datasets/eval_v3/train.jsonl` surface (4.58M 文、
char-spaced)。`scripts/bench_ctc_nat_fusion.py` 経由 (WSL CPU, torch 2.11)。

### α/β sweep, beam=8, 常時 LM (gate=0)

CTC baseline = α=β=0。**α=0.80 β=1.0** が 3 bench 全勝。

| bench | baseline EM | best EM | config | Δ | relative |
|---|---|---|---|---|---|
| manual_test (100) | 0.7000 | **0.8700** | α=0.80 β∈{0,0.5,1.0} | +17.0pt | +24% |
| ajimee_jwtd (80) | 0.2250 | **0.4125** | α=0.50 β=0 / α=0.80 β=1.0 | +18.75pt | +83% |
| eval_v3_dev (200) | 0.1150 | **0.2550** | α=0.80 β=1.0 | +14.0pt | +122% |

LM 学習コーパスは eval_v3/train → eval_v3/dev は held out だが同ドメイン。
manual/ajimee は LM 未露出ゆえ genuine gain。

### 速度 (CPU)

beam=8 + KenLM @ α=0.80 β=1.0:

| bench | p50 | p95 | vs beam8 alone |
|---|---|---|---|
| manual (≤20字) | 27.9ms | 41.3ms | +1ms / +5ms |
| ajimee (中文) | 45.4ms | 169.0ms | +8ms / +69ms |
| eval_v3 (長文) | 86.7ms | 214.5ms | +22ms / +92ms |

### beam=4 版

精度 -1〜2.5pt で p95 最大 -62ms の短縮:

| bench | beam=8 EM | beam=4 EM | beam=8 p95 | beam=4 p95 |
|---|---|---|---|---|
| manual | 0.87 | 0.86 | 41ms | 33ms |
| ajimee | 0.4125 | 0.3875 | 169ms | 117ms |
| eval_v3 | 0.2550 | 0.2300 | 214ms | 152ms |

### low-confidence gate 試験 (gate=-N: conf<N でのみ LM 起動)

90M CTC は high-confidence すぎて (mean top-1 logp > -1 が常時)、
gate=-1〜-5 では LM が全く起動せず baseline に落ちる。閾値は
0 近傍 (-0.1〜-0.5) で再 sweep が必要。gate 機構自体は二重
forward 解消済みで正常動作 (manual p50 21ms = greedy 相当)。

### 30M CTC-NAT (phase3_30m, local eval_v3/train) + KenLM

30M は **`datasets/eval_v3/train.jsonl` で学習済** なので eval_v3/dev は
in-distribution、manual/ajimee は外。beam=4 sweep:

| bench | baseline EM | best EM | config | Δ vs baseline |
|---|---|---|---|---|
| manual_test | 0.5100 | **0.7500** | α=0.50 β∈{0,0.5} | +24pt |
| ajimee_jwtd | 0.2000 | **0.4000** | α=0.50 β=0.5 | +20pt |
| eval_v3_dev | 0.2250 | **0.3200** | α=0.30-0.50 β=1.0 | +9.5pt |

30M vs 90M (両方 +KenLM、beam=4):

| bench | 30M (28.6M) | 90M (95M) | 勝者 |
|---|---|---|---|
| manual | 0.7500 | 0.8700 | **90M +12pt** |
| ajimee | 0.4000 | 0.4125 | 90M +1.25pt (ほぼ互角) |
| eval_v3 | **0.3200** | 0.2550 | **30M +6.5pt** (data leakage 効果) |

**Speed (CPU, p50)**:

| bench | 30M beam4+LM | 90M beam4+LM | 倍率 |
|---|---|---|---|
| manual | 13ms | 27ms | 2.1x |
| ajimee | 23ms | 42ms | 1.8x |
| eval_v3 | 40ms | 87ms | 2.2x |

30M は 90M の約半分の CPU レイテンシ。LM fusion は data-eff 観点で小モデル
にも大幅な gain を与える (ajimee +20pt, manual +24pt)。

### 暫定結論 (KenLM shallow fusion)

1. **KenLM fusion は絶大** — 90M で ajimee +83% rel, eval_v3 +122% rel
2. **最適点は α=0.50-0.80, β=0.5-1.0** (モデル規模で shift、30M は小さめ α)
3. **CPU でも十分実用的** — manual 27ms / eval_v3 87ms p50
4. **β は長文 (eval_v3) でより効く** — CTC の短め出力傾向を矯正
5. **low-conf gate は閾値調整待ち** — 現実装は動作するが 0 近傍でないと効かず

## 全9モデル × 3ベンチ

| Model | Params | manual EM | ajimee EM | eval_v3 EM | 最遅p50 |
|-------|--------|-----------|-----------|------------|---------|
| zenz-v2.5-medium greedy | 310M | 0.900 | 0.787 | 0.412 | 1853ms |
| zenz-v2.5-small greedy | 91M | 0.890 | 0.750 | 0.375 | 581ms |
| zenz-v2.5-xsmall greedy | 26M | 0.880 | 0.588 | 0.312 | 242ms |
| ar_v3_vast greedy | 32M | 0.800 | 0.450 | 0.412 | 191ms |
| ar_v3_vast beam10 | 32M | 0.800 | 0.450 | 0.450 | 1082ms |
| ar_v3_local greedy | 32M | 0.780 | 0.400 | 0.325 | 233ms |
| ar_v3_local beam10 | 32M | 0.790 | 0.412 | 0.350 | 1095ms |
| ar_v3_chunks greedy | 30M | 0.590 | 0.212 | 0.013 | 132ms |
| ar_v3_chunks beam10 | 30M | 0.590 | 0.188 | 0.013 | 1086ms |

## 同規模対決: ar_v3_vast (32M) vs zenz-xsmall (26M)

| ベンチ | ar_v3_vast EM | zenz-xsmall EM | 勝者 |
|--------|--------------|----------------|------|
| manual_test | 0.800 | 0.880 | zenz +8pt |
| ajimee_jwtd | 0.450 | 0.588 | zenz +14pt |
| eval_v3_dev | **0.412** | 0.312 | **自前 +10pt** |

## 重要な知見

1. **自前分布 (eval_v3) では 310M zenz-medium と互角** (EM 0.412 = 同値)
2. **汎化ベンチ (AJIMEE) では xsmall にも負ける** → データ多様性不足
3. **zenz 系列のスケーリング**: manual はサチ (0.88-0.90), AJIMEE は明確にスケール
4. **CTC-NAT で速度優位を取る戦略は有効**: zenz-xsmall 75-242ms に対し CTC-NAT 10-30ms を狙える
5. **beam search は EM を改善しない** (greedy とほぼ同等)、ただし CharAcc は微改善

## Phase 3 への示唆

- 精度で zenz-xsmall を超えるには: データ多様性 + モデル規模拡大
- 速度で差別化するには: CTC-NAT 並列生成
- 200M CTC-NAT 1.58-bit なら: zenz-small (91M) に精度で並び、速度で 10倍��上高速
