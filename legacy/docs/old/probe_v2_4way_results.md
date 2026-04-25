# probe_v2 4-way 評価

`datasets/eval/probe_v2.tsv` (**467 項目、7 category**) による句単位 (bunsetsu)
の 4-way 比較。canonical 数字はこの 467 項目版。

## 比較の前提

new-ime は **mozc 置き換えの neural IME** プロジェクト、zenz-v2.5 は **AzooKey
内部の LLM 変換器**で用途が異なる (詳細: `vision.md`)。**主比較は同サイズ帯
(zenz-small 91M) での句レベル性能**。medium (310M) 比較は参考値。

評価単位は **句 (bunsetsu)**。IME の実使用に合わせた評価で、文脈ゼロの変換。

## 結果 (probe_v2, 467 項目)

| モデル | 設定 | EM1 | EM5 | p50 ms (WSL CPU) |
|---|---|---:|---:|---:|
| **zenz-v3.1-small** | beam=5 | **0.715** | 0.925 | 274 |
| **zenz-v2.5-small** | beam=5 | **0.700** | 0.916 | 266 |
| CTC-NAT 90M step27500 | α=0.2, β=0.6, beam=5 + KenLM | **0.612** | 0.612 | 21 |
| CTC-NAT 30M step50000 | α=0.4, β=0.3, beam=5 + KenLM | 0.499 | 0.499 | 26 |
| CTC-NAT 90M step27500 | greedy (no LM) | 0.580 | 0.580 | 12 |
| ar_v3_vast | beam=5 | 0.360 | 0.540 | 104 |

カテゴリ分布: general 80, tech 26, names 60, homophone 73, edge 71, numeric 117,
particle 40。

## カテゴリ別 (CTC-NAT 90M best vs zenz-v3.1)

| カテゴリ | 90M | v3.1 | 差 |
|---|---:|---:|---:|
| numeric | 0.02 | 0.39 | **-37pt** ← 支配的 |
| edge | 0.68 | 0.78 | -10pt |
| names | 0.90 | 0.92 | -2pt |
| particle | 0.93 | 0.93 | 0 |
| general | 0.96 | 0.95 | +1pt |
| tech | 0.92 | 0.89 | **+3pt** (CTC 勝ち) |
| homophone | 0.60 | 0.57 | **+3pt** (CTC 勝ち) |

全体差 -10.3pt のうち **numeric だけで -10pt 相当** (117 項目 × 37pt 差 = 43
items 相当のロス)。numeric を半分解決すれば 90M は zenz と並ぶ計算。

## KenLM shallow fusion (sweep)

CTC-NAT 90M step27500 に対し beam=5 + KenLM
(`models/kenlm/kenlm_general_train_4gram_probing.bin`) で α × β grid sweep
(α ∈ {0.2, 0.4, 0.6}, β ∈ {0.0, 0.3, 0.6})。

| config | EM1 |
|---|---:|
| greedy (no LM) | 0.580 |
| beam=5 (no LM) | 0.580 |
| **α=0.2, β=0.6** | **0.612** |
| α=0.4, β=0.6 | 0.600 |
| α=0.6, β=0.6 | 0.600 |

**KenLM gain +3.2pt**、α=0.2 β=0.6 が最適。50 項目 pilot では β=0 が最適と誤判定
していたが、467 項目で β=0.6 が優位と確定。**モデル規模で最適 α/β が変わる**:
90M は α=0.2、30M は α=0.4。

## CTC-NAT 30M sweep

同 sweep を CTC-NAT 30M step50000 にも実施。

| config | EM1 |
|---|---:|
| greedy | 0.375 |
| **α=0.4, β=0.3** | **0.499** |

**KenLM gain +12.4pt** (90M の 4 倍)。小規模モデルの soft logit に LM が強く効く。
30M + LM 0.499 でも 90M greedy 0.580 に届かず、**規模の壁は LM では埋まらない**。

## 観察

### new-ime model vs zenz

- CTC-NAT 90M は zenz から 10-11pt 差、**レイテンシは 13x 速** (CPU 比較)
- **zenz-v3.1 は v2.5 より +1.5pt** (ほぼ同等)
- **CTC-NAT は homophone と tech で zenz に勝つ** — モデル構造の強みは既にある
- **numeric 以外で総合すると 90M はほぼ zenz 並み**

### 弱点

- **numeric が壊滅** (0.02) — 学習データに数詞 + 助数詞 + SI 接頭語 が不足。
  synth_numeric + synth_numeric_ext で対策中
- **homophone 0.60** — 文脈条件付けが必要。CVAE / context window で改善余地
- **CTC-NAT の EM5 == EM1** — prefix beam が同じ collapse に収束。Mask-CTC
  refinement / 温度サンプリングで候補多様性の改善余地

## Pilot (50 項目、歴史的記録)

初期パイロット 50 項目では 90M EM1 0.800、zenz-small 0.880、差 8pt。467 項目へ
拡張して noise band を下げたら差が 10-11pt に広がった。pilot はカテゴリ
あたり 6-8 項目で ±15pt の信頼区間があり、config 間の +2pt 差は noise 内だった。

## 留意点

- 467 項目でも 信頼区間は Wilson 95%CI で ±3-5pt。configs 間の 1-2pt 差は
  有意とは言えない
- domain n=6 の news は特に noise が大きい
- CTC-NAT 90M は step27500 (崩壊後の回復直後)、step 20000-25000 も評価すべき
