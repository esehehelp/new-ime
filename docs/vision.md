---
status: current
last_updated: 2026-04-19
---

# new-ime: 最終構成ビジョン

v1.0 最終到達点の設計文書。実行中の実装計画は `phase3_v2_dryrun_runbook.md`、
現状ベンチは `benchmark_comparison.md`。

## プロジェクトの目的と位置付け

**new-ime は mozc を置き換える次世代の neural IME を作るプロジェクト**。設計の核は
「**書き手適応 CVAE + 非自己回帰 CTC-NAT**」の組み合わせで、これが IME の使用パターン
(短文・高頻度・個人差大) に構造的に適合する、という仮説の検証が目的。

- **短文**: 変換単位は句 (bunsetsu) が中心。→ NAT の並列デコードが効く。
- **高頻度**: 1 文字タイプごとに 10ms 以下で候補更新したい。→ AR の O(seq_len)
  生成は構造的に厳しい。CTC-NAT は O(1) ステップ。
- **個人差大**: 書き手・ドメイン・セッションで語彙と表記揺れが変わる。→ CVAE
  で z をユーザ側に持たせ、モデル本体は共有したまま書き手別に適応。

### zenz-v2.5 との比較方針

zenz-v2.5 (Miwa-Keita 氏) は同じ kana-kanji 変換タスクを扱うが**用途が異なる**:

| | zenz-v2.5 | new-ime |
|---|---|---|
| 用途 | AzooKey 内部の LLM 変換器 (macOS/iOS) | mozc 置き換えの IME (Windows 中心) |
| アーキテクチャ | GPT-2 系 auto-regressive | CTC-NAT + CVAE |
| 評価単位 | 文 (context 込み) | **句 (bunsetsu) が主** |
| 書き手適応 | 想定外 | **コア機能** |

**主比較は同サイズ帯 (zenz-small 91M) での phrase レベル性能**。medium (310M) との
比較は参考値。zenz-small を句レベル EM で追い抜き、**レイテンシで明確に優位**であれば
仮説成立と見なす。

## アーキテクチャ: CTC-NAT Encoder-Decoder

```
[左文脈 + ひらがな入力]  + CVAE z (writer/domain/session)
    │
    ▼ Encoder (scratch, h=640, L=8, FiLM 条件付け)
    │
    ▼ Decoder (NAT, h=640, L=8, self+cross+ffn, FiLM 条件付け)
    │
    ▼ CTC Head → collapse / beam search + KenLM shallow fusion
    │
    ▼ Mask-CTC refinement (2-3 回, オプション)
    │
    漢字かな混じり出力 (top-K)
```

### CTC-NAT を選ぶ理由

- 日本語読み→表記は単調アラインメント → CTC の前提と合致
- 並列生成でレイテンシが入力長に依存しない
- AR の構造崩壊 (「人口のうの」) が原理的に発生しない
- Beam search の exposure bias がない

### 規模 (`new-ime-model-90M`, 本命)

| コンポーネント | 構成 | params |
|---|---|---|
| shared char embedding (tied) | 6500 × 640 | ~4.2M |
| encoder (scratch) | 8 層 | ~39M |
| decoder (scratch) | 8 層 | ~52M |
| CTC head | Linear(640, vocab) | ~4.2M |
| CVAE (posterior biGRU + prior + FiLM) | — | ~7M |
| 合計 | | **~104M** |

テスト用に `phase3_30m` (h=384, L=6+6) も並置。

## ユーザ適応: 階層的 CVAE (**現状未実装、後工程**)

### 潜在変数設計

```
z = (z_writer[32], z_domain[16], z_session[16]) = 64 次元
```

- **z_writer**: 書き手の文体・語彙選好 (端末ローカル、オンライン更新)
- **z_domain**: ビジネス/カジュアル/技術 (アプリ別)
- **z_session**: 直近入力の傾向 (セッション内)

### 検証方法

`cvae_probe_baseline.md` の probe (188 項目 / 10 domain) で baseline EM 0.585 から
どこまで動くかを試金石とする。**+42pt は理論最大幅であって期待値ではない**。
domain-specific KenLM 等の代替手法も並列比較する枠組みで評価する。

## 量子化: 1.58-bit QAT (研究線、クリティカルパス外)

- fp16 学習完走 → Continual QAT (fp16 → 1.58-bit) → activation 8-bit
- 本命モデル ~104M × 1.58-bit ≈ **15-25 MB**
- Nielsen 2024 "BitNet b1.58 Reloaded" の前提で設計。v1.0 では int8 ORT 経路が
  主配布、1.58-bit は bitnet.cpp 経由の研究成果物。

## データ戦略 (現状 2026-04-19)

- `datasets/mixes/train_v1_200m.jsonl` 200M rows (既存 v1 mix: chunks + zenz_llmjp
  + wiki + aozora + fineweb2 + hplt3、コンポーネント: `datasets/corpus/v1/`)
- `datasets/corpus/v2/bunsetsu/` 約 10M 句 (wikinews/aozora_dialogue/tatoeba/wikibooks/
  wiktionary から Ginza で bunsetsu 分割)
- 合成データ: synth_numeric 37K (数詞×助数詞) + synth_numeric_ext 150K (時刻/
  日付/通貨/分数/小数/連番)

学習 mix の最新比率は `phase3_v2_dryrun_runbook.md` を参照。

### 読み付与

MeCab unidic-lite `features[17]` (仮名形出現形) を使用。features[6/7/8/9] は
Phase 2 v1/v2 の失敗で不正と確定 (詳細: `old/phase2_results.md`)。

## 撤退経路

| リスク | 撤退先 |
|---|---|
| CTC-NAT 精度不足 (AR との差 3pt 以上) | AR + 投機的デコード |
| CTC-NAT 精度不足 (AR との差 2-3pt) | DAT (DAG 構造) |
| 1.58-bit 品質劣化 | int8 ORT (30-40 MB) |
| CVAE posterior collapse | domain-specific KenLM / domain embedding / LoRA ベースのアプリ別分離 |
| KenLM 効果薄 | ユーザ辞書強化のみ |

## 採用しない選択

- **llama.cpp**: decoder-only 前提、encoder-decoder 非対応
- **1.58-bit を初期から**: QAT 再学習のコストは fp16 完走後にのみ発生する方が安全
- **AR beam search**: 40-60x の速度ペナルティ、length normalization でも改善限定

## 関連 docs

- 現状ベンチ集約: `benchmark_comparison.md` (living doc)
- 実行中の実装: `phase3_v2_dryrun_runbook.md`
- probe_v2 詳細: `probe_v2_4way_results.md`
- CVAE probe ベースライン: `cvae_probe_baseline.md`
- 過去 plan / 完了 phase: `old/` 配下
