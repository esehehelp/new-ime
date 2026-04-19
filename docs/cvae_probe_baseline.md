# CVAE 検証 probe — ベースライン測定

`datasets/eval/cvae-probe/probe.tsv` (188 項目、70 unique reading、10 domain) で
現行モデル (CVAE **未実装**) のベースラインを測定。

**この probe の位置付け**: **CVAE 仮説の試金石**。実装後に probe EM を測って
効果を判定するためのベースライン。「CVAE 必要性の証明」ではなく「CVAE を
試す価値の定量化」。

## Probe のメタ特性

| 指標 | 値 |
|---|---:|
| 総項目数 | 188 |
| unique reading | 70 |
| domain-disagreement reading | 33 (47.1%) |
| domain-agreement | 20 |
| single-domain | 17 |

47% の reading で domain 別に正解が分岐 = **「条件付けなしでは原理的に全 domain 正解
できない」item が 33 reading × 複数 domain 分存在する**という構造的事実。

domain 分布: general 40, academic 27, formal 22, tech 21, literary 19,
casual 15, business 14, legal 13, medical 11, news 6。

**信頼区間の注意**: n=188 全体、domain 別で n=6-40。domain 別 EM は Wilson 95%CI
で概ね ±15pt の幅がある。特に news (n=6) や legal (n=13) の数字は幅が大きく、
1-2 項目の正誤で domain EM が ±10-15pt 動く。pilot として扱い、probe 拡張後に
再測定する前提。

## ベースライン結果 (CTC-NAT 90M step27500)

| domain | greedy | +KenLM (α=0.2, β=0.6, beam=5) | 差 |
|---|---:|---:|---:|
| overall | 0.585 | 0.574 | -1.1 |
| formal (n=22) | 0.77 | 0.68 | -9 |
| medical (n=11) | 0.73 | 0.73 | 0 |
| general (n=40) | 0.70 | 0.72 | +2 |
| casual (n=15) | 0.67 | 0.67 | 0 |
| news (n=6) | 0.67 | 0.33 | -34 |
| literary (n=19) | 0.53 | 0.53 | 0 |
| tech (n=21) | 0.52 | 0.52 | 0 |
| business (n=14) | 0.50 | 0.50 | 0 |
| academic (n=27) | 0.48 | 0.48 | 0 |
| legal (n=13) | 0.15 | 0.23 | +8 |

## 観察 (過度な一般化を避ける)

### 1. デフォルト出力は formal/general 寄り

- 高 EM: formal 0.77, medical 0.73, general 0.70
- 低 EM: legal 0.15, academic 0.48, business 0.50, tech 0.52

これは **学習データの domain 偏りの反映** と推測される。Wiki (academic 寄り) +
aozora (literary 寄り) + chunks (general/formal 寄り) の混合で、legal/tech 専門
表記は訓練露出が少ない。**CVAE なしでも corpus rebalance (legal/tech 専門文書の
追加) で formal 以外を底上げできる可能性がある** — この仮説は corpus_v2 bunsetsu
化完了後に再測定して検証する。

### 2. KenLM (domain-agnostic) の効果は方向性混在

- 全体 -1.1pt、legal +8pt、news -34pt、formal -9pt
- これは domain-agnostic な LM (general/train) を使った場合の結果。LM コーパスの
  domain 偏り (wiki/news 寄り) が probe 側の domain 別正解と一致しない所で崩れる

**注意**: これは「KenLM では domain 問題を解けない」を示す結果ではない。**domain
ごとに別の KenLM を用意して domain-specific に切り替え**れば、かなり解ける可能性
がある。ここは未測定。CVAE の代替候補として domain-specific KenLM / domain
embedding / domain-specific fine-tuning が残っている。

### 3. CVAE 試行の動機付けとしては十分、ただし効果量は未知

- **47% の reading で正解が domain 分岐** = 条件付けなしでは原理的に天井あり
- この天井の数値化は baseline 0.585 vs 上限 1.00 = **+41.5pt が理論最大幅**
- ただし **これは z_domain が完璧に機能かつモデルが完璧に出し分ける場合** の
  上限。実際の CVAE 実装で得られる gain の期待値ではない
- posterior collapse で 0pt、中途半端な disentangle で 5-15pt、うまく噛み合えば
  20-30pt、という幅を想定しておくのが妥当

## CVAE 実装時の評価枠組み

現状 0.585 に対し、実装後に以下で判定 (baseline からの lift で閾値を切る。
pilot n=188 の Wilson 95%CI は ±7pt、domain 別は ±15pt なので、fail 判定には
baseline +5pt の有意帯を確保):

| 到達点 | overall EM1 | baseline からの差 | 判定 |
|---|---:|---:|---|
| fail | < 0.635 | < +5pt (noise 帯) | 仮説棄却、代替手法 (domain-specific KenLM 等) へ |
| marginal | 0.635-0.70 | +5 〜 +11pt | 部分的に働いている、原因分析が必要 |
| working | 0.70-0.80 | +11 〜 +21pt | CVAE に学習シグナルが入っている |
| strong | > 0.80 | > +21pt | 仮説支持、z_domain が意味のある disentangle |
| ceiling | 1.00 | +41pt | 理論上限 (z perfect + 完璧な出し分け) |

domain 別では legal / tech / academic の改善幅を重視 (baseline 最低位で余地大)。

## 代替手法との比較枠組み (CVAE を他と並べて評価)

CVAE 実装前に「CVAE が最良」と決めつけず、以下の代替を比較測定すべき:

| 手法 | 期待 gain | 実装コスト | 特徴 |
|---|---|---|---|
| CVAE (z_writer + z_domain) | 不明、理論上限 +42pt | 大 | 柔らかい条件付け、ユーザ適応可能 |
| domain-specific KenLM (N 個切替) | 不明、+10-20pt 想像 | 小-中 | 実装軽い、推論時 domain 選択必要 |
| domain embedding (hard concat) | 不明、+10-30pt 想像 | 中 | CVAE より簡素、柔軟性は低い |
| corpus rebalance のみ | +5-15pt 想像 | 中 | 専門 corpus 追加で底上げ |

CVAE を本線で検証する方針自体は妥当だが、**fail 判定時の撤退先として代替手法を
並列で準備**しておくのが安全。

## 留意点

- probe 188 項目、domain あたり n=6-40 は **pilot 規模**。domain 別 EM は ±15pt
  ノイズ帯。実装比較時は probe を 500+ に拡張してから。
- writer-id accuracy と KL divergence 指標は CVAE 実装後に追加予定 (domain 条件
  付き出力が出せないと測れない)。現在は per-domain EM のみ。
- 青空文庫 author 情報が v2 corpus に残っていないため writer 粒度は source/domain
  レベル。作者別 CVAE 検証には raw 青空文庫 XML からの再抽出が必要。
- formal 0.77 が高い原因は「CVAE なしで解ける = 学習データが formal 偏り」である
  可能性があり、**corpus rebalance だけで formal 以外を底上げできるか**を bunsetsu
  化完了後に再測定する価値がある。この結果次第で CVAE の期待 gain も再計算。
