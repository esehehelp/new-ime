---
status: current
last_updated: 2026-04-20
---

# new-ime: 最終構成ビジョン

v1.0 / v2.0 の到達点と、そこへ至るロードマップの設計文書。
実装の現行計画は `phase3_v2_dryrun_runbook.md`、ベンチ実値は `benchmark_comparison.md`。

## プロジェクトの目的と位置付け

**CPU で動く neural IME。mozc の置き換え候補**として構築する。書き手適応 CVAE と
非自己回帰 CTC-NAT の組み合わせが、IME の使用パターン (短文・高頻度・個人差大) に
構造的に適合するという仮説の検証が核心。

- **短文**: 変換単位は句 (bunsetsu) 中心 → NAT の並列デコードが効く
- **高頻度**: 変換レイテンシで UX が決まる → AR の O(seq_len) は構造的に厳しい
- **個人差大**: 書き手・ドメインで語彙と表記揺れが変わる → CVAE で z をユーザ側に

### 市場座標 (CPU/GPU 軸)

| | 前提 | 対応 OS | UX |
|---|---|---|---|
| **new-ime** | **CPU** | Windows/Linux | v1.0: 変換キー / v2.0: ライブ |
| karukan (togatoga) | GPU (llama.cpp + zenz) | Linux (fcitx5) | 変換キー |
| AzooKey | Apple Silicon NPU | macOS/iOS | ライブ変換 |
| mozc | CPU (統計ベース) | Windows/Linux | 変換キー |
| zenz-v2.5/v3.1 | GPU 40ms / CPU 240-1000ms | (ライブラリ) | — |

**「CPU で動く neural IME」は空白市場**。GPU 前提の karukan/AzooKey と直接競合せず、
速度で mozc に近づけつつ精度で neural の優位を示すのが狙い。

## v1.0 / v2.0 スコープ分離

| | v1.0 | v2.0 |
|---|---|---|
| UX | 変換キー型 (commit on space) | ライブ変換 (per-keystroke) |
| レイテンシ予算 | 単発 200-300ms 許容 | 単発 40ms 以下必須 |
| 累積考慮 | 不要 | 必須 |
| 主力モデル | 30m_v2 step160k + KenLM | 同モデル (17ms) を継続 |
| 量子化 | int8 (production) | int8 → 1.58-bit 段階移行 |
| ユーザ適応 | 頻度学習 (minimal) | CVAE 本格投入 |
| データ | 現 20M | 200M → 5B |

**v1.0 の 17ms レイテンシが既に v2.0 の 40ms 要件を満たす** ため、アーキの連続性が高い。
v1.0 はモデル選定と IME 統合が主、v2.0 はデータと CVAE / MoE の研究フェーズ。

## 主力モデル (v1.0 確定)

**`ctc-nat-30m-student` step160k + KenLM α=0.2, β=0.6, beam=5**

- probe_v3: **EM1=0.655, p50=17ms** (CPU, 2026-04-20 測定)
- 同サイズ帯 zenz-small (91M) との phrase レベル比較が主軸
- teacher-v1-150m (probe_v3 greedy 0.681) は参照値に後退。production から外す
  - ただしkdの教師モデルとして価値あり，データセット含めた大型化も考慮

### アーキテクチャ: CTC-NAT Encoder-Decoder

```
[左文脈 + ひらがな入力]  + CVAE z (v2.0 以降)
    │
    ▼ Encoder (scratch, FiLM 条件付け)
    │
    ▼ Decoder (NAT, self+cross+ffn, FiLM)
    │
    ▼ CTC Head → collapse / beam + KenLM shallow fusion
    │
    ▼ Mask-CTC refinement (v1.0 終盤、オプション)
    │
    漢字かな混じり出力 (top-K)
```

### CTC-NAT を選ぶ理由

- 日本語読み→表記は単調アラインメント → CTC の前提と合致
- 並列生成でレイテンシが入力長に依存しない
- AR の構造崩壊 (「人口のうの」) が原理的に発生しない
- Beam search の exposure bias がない

## アーキテクチャ候補

### 採用 / 採用予定

- **CTC-NAT (現行)**: 30M でも十分実用。データ + step でスケール可能
- **Mask-CTC refinement**: EM5 = EM1 問題 (候補多様性ゼロ) への対応。低信頼位置を
  mask して再予測。実装コスト小、**v1.0 向け優先**

### 将来検証

- **DAT (Directed Acyclic Transformer)**: 実行時 K-path 可変で速度-精度切り替え。
  30M CTC-NAT で十分のため不要。ledger に残すのみ
- **MoE (Mixture of Experts)**: category 別 α 感度が明確に逆方向 (後述) → 構造的解。
  候補構成: explicit category / data-driven sparse / adapter / CVAE + MoE hybrid。
  v1.0 は category-wise α で代替、**v1.x 以降の候補**
- **CVAE (domain/writer/source 条件付け)**: モジュールは実装済み (192 行)、推論時 z 管理
  が未実装。**5B データでラベル付与してから本格投入**

## 推論側の改善

### 採用: KenLM shallow fusion

- 現設定: α=0.2, β=0.6, beam=5
- probe_v3 で +5.2pt (EM1 0.603 → 0.655 @ step160k)

### カテゴリ別 α 感度 (2026-04-20 計測、step 120k-160k 平均)

```
              α=0.2    α=0.4    α=0.6   (greedy 比、β=0.6 固定)
names      +0.073 → +0.094 → +0.109   LM 強化で単調向上
homophone  +0.054 → +0.068 → +0.068   α 強めで頭打ち
particle   +0.021 → +0.021 → +0.026   飽和 (0.86 base)
general    +0.049 → +0.027 → +0.022   α 強いとわずかに劣化
tech       +0.004 → +0.008 → +0.000   ほぼ無反応
numeric    +0.054 → +0.000 → -0.077   α 強化で退行
edge       +0.079 → -0.037 → -0.183   α 強化で破滅
```

**α=0.2 は全カテゴリで安全**。α ≥ 0.4 は names/homophone には良いが edge/numeric を
壊す → 現 KenLM の train corpus が edge/numeric 系の表記を十分カバーしていない。

### v1.x 検討項目

- **category-wise adaptive α**: 推論時に category 推定 + α 切り替え。
  names/homophone α=0.4-0.6、edge/numeric α=0.2、tech α=0.0 (LM 外) が目安
- **複数 KenLM mixture**: general + tech domain の別 LM。tech 特化 LM で tech gap 補償
- **温度 / Nucleus sampling**: EM5 対応の軽量手段 (Mask-CTC と並列の選択肢)
- **MBR decoding**: 複数候補から minimum Bayes risk 選択。候補多様性確保の中コスト

## 量子化

| 形式 | 位置付け |
|---|---|
| **int8 (ORT)** | **v1.0 production 形式**。fp16 → int8 劣化が 1-2pt 以内なら採用 |
| 1.58-bit QAT | v2.0 以降の大型モデル対応。CPU メモリ帯域ボトルネック対策 |
| bitnet.cpp | 1.58bit 本格実装、research 段階。v1.0 では int8 fallback で十分 |

## データ戦略 (段階的スケール)

**20M → 200M (中間) → 5B (最終)**。著作権法 30条の4 + 非公開データの組合せで
データ可動性を確保。5B の 25x は zenz 190M を明確に超える規模。

### 5B 段階の LLM 整形パイプライン

3 層構造:
- **Layer 1 (2-3B)**: Web crawl + MeCab 読み付与
- **Layer 2 (1-1.5B)**: Qwen 等で読み推定 / ラベル付与 / augmentation
- **Layer 3 (500M-1B)**: 完全合成

CVAE ラベル (writer / domain / source) の土台を Layer 2 で用意。

### カテゴリ別弱点対応 (vs zenz 帯)

| cat | gap | 対策 |
|---|---|---|
| names | -15pt | Wikipedia entity 厚く、固有名詞辞書 |
| tech | -12pt | Qiita/Zenn、tech domain KenLM |
| edge | -13pt | IT 用語保持、英字/カタカナ判断 |
| homophone | -3pt | context 長拡張、CVAE domain 条件付け |

### mix 比率 / 量の調整

- `scripts/build_phase3_train.py` の ratio-chunks **0.50 → 0.10** (plan.md 準拠)
- 200M pool sampling 解除で、**新規クロール不要で 20M → 200M に到達**

### contamination 管理

- probe_v3 / AJIMEE 元テキストを学習データから除外
- 非公開データセット使用でも評価純潔性は保持

### 読み付与

MeCab unidic-lite `features[17]` (仮名形出現形)。features[6/7/8/9] は Phase 2 v1/v2
の失敗で不正と確定 (詳細: `old/phase2_results.md`)。

## IME 統合層

### fcitx5 プラグイン (Linux)

karukan-im (MIT/Apache-2.0) を参考。変換キー押下型の基本フローは流用可能。

### TSF (Windows)

Microsoft docs 参照で独自実装。既存 OSS IME を参考。
**karukan にない価値**で、v1.0 の差別化要素。

### 辞書層

- ユーザ辞書 + システム辞書 (Sudachi or mozc 辞書)
- 頻度学習 (v1.x 候補)

### ユーザ適応

- v1.0: 変換履歴からの最低限の頻度学習 (副機能)
- v1.x 以降: CVAE 本格適応

## 学習パイプライン

### teacher-v1-150m 200k 完走

AR 150M、probe_v3 greedy 0.681。参照値として保持。v1.0 production からは外れた。

### KD 再設計

現 kd_alpha=0.001 で実質 KD なし。必要なら kd_alpha schedule 見直し。
ユーザ方針が minimal change の場合は保留。

### step 追加学習

- probe_v3 で step 160k は完全収束していない可能性 (120→160k で EM1 +0.025、greedy 単調増加)
- lr 再起動で step 200k+ の可能性
- **200M 移行との ROI 比較**で判断

## 評価体系

### canonical: probe_v3

- step 160k + KenLM α=0.2 b=0.6: EM1 0.655 を**公式記録**
- eval_v3/dev は学習監視用、probe_v3 を production 指標に昇格

### 使い分け

| bench | 用途 |
|---|---|
| eval_v3/dev | 長文、dev-set、学習監視 |
| probe_v3 (348 items, 7 cats) | canonical、phrase 中文、AJIMEE 寄せ |
| AJIMEE 本家 200 件 | 公平比較 |
| cvae_probe | CVAE 効果測定 |

### zenz 公平比較

- 両者 int8 / beam / KenLM 条件を統一
- AJIMEE 本家で直接比較
- **v1.0 出荷前に実施**

### best.pt 信頼性

dev loss が non-monotonic で best.pt が true best にならないことを確認済み。
**canonical には final.pt または step_{N}.pt を採用**。

## ロードマップ

### v1.0 構成 (確定寄り)

- 30m_v2 step160k + KenLM + int8 + IME 統合 (fcitx5/TSF)
- Mask-CTC refinement で EM5 対応
- category-wise α で精度微調整
- **現 20M data のまま出荷可能**

### v0.9 preview (選択肢)

- 現 20M 版を early release、β feedback で v1.0 に反映
- 5B / 200M 移行の裏で時間稼ぎ

### v1.x 中期

- 200M data 再学習 (30m_v2 継続)
- Mask-CTC refinement 成熟
- fcitx5 / TSF 統合完成
- 条件揃え zenz 比較で精度確立

### v2.0 ライブ変換

- ライブ変換 UX (40ms 要件、現 30m_v2 で対応可)
- CVAE 本格投入 (domain/writer/source FiLM)
- MoE PoC (200M 以上のデータ前提)
- モバイル対応 (将来)

### 5B 路線

- Layer 1 (2-3B): Web crawl + MeCab
- Layer 2 (1-1.5B): LLM 整形 + ラベル付与
- Layer 3 (500M-1B): 完全合成
- 段階的: 20M → 200M → 5B

## ライブ変換 (v2.0 への構造的課題)

CTC-NAT の encoder は双方向 attention のため、入力末尾に 1 token 足すと既存位置の
表現も変わる。AR の KV-cache 再利用 (past の KV は不変) と構造的に噛み合わない。
結果として **毎 keystroke で encoder 全再計算**。ただし現 30m_v2 は p50 17ms なので
40ms 要件は満たす。30 字タイプで累積が問題になる場合の緩和策:

1. **chunk-cached encoding**: 5-8 字 chunk で区切り、chunk 境界で attention を切る近似
2. **live mode は refinement 省略**: Mask-CTC は commit 時のみ
3. **Hybrid (軽量 AR + CTC-NAT)**: ライブは蒸留 AR (10-20M、KV-cache)、commit 時に
   CTC-NAT + KenLM で精密変換。2 モデル保持コスト増
4. **Mask-CTC 多様性活用**: 低信頼位置を複数補完で動的候補提示

## プロジェクト運営

### オープン性の段階

- **v1.0**: code 公開、model 公開、データセット公開
- **v1.x**: データセット非公開に移行 (5B で)
- zenz / AzooKey / karukan と同じ標準運用

### 評価再現性

code と model は公開、data が非公開でも benchmark で証明。
probe_v3 / AJIMEE 本家 / 公開 dev set で評価純潔性を担保。

## 撤退経路

| リスク | 撤退先 |
|---|---|
| CTC-NAT 精度不足 (AR との差 3pt 以上) | AR + 投機的デコード |
| CTC-NAT 精度不足 (AR との差 2-3pt) | DAT (DAG 構造) |
| 1.58-bit 品質劣化 | int8 ORT (30-40 MB) |
| CVAE posterior collapse | domain-specific KenLM / domain embedding / LoRA 分離 |
| KenLM 効果薄 | ユーザ辞書強化のみ |

## 関連 docs

- ベンチ living doc: `benchmark_comparison.md`
- 実装 runbook: `phase3_v2_dryrun_runbook.md`
- probe_v3 raw results: `results/probe_v3_30m_student_120_140k/`
- 過去 plan / 完了 phase: `old/` 配下
