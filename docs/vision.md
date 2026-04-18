---
status: current
last_updated: 2026-04-18
supersedes: docs/architecture_decisions.md
---

# new-ime: 最終構成ビジョン

v1.0 最終到達点の設計文書。Phase 2 AR ベースラインを教師として Phase 3 以降で実現する。

## プロジェクトの目的と位置付け

**new-ime は mozc を置き換える次世代の neural IME を作るプロジェクト**。設計の核は
「**書き手適応 CVAE + 非自己回帰 CTC-NAT**」の組み合わせで、これが IME の使用パターン
(短文・高頻度・個人差大) に構造的に適合する、という仮説の検証が目的。

- **短文**: 変換単位は句 (bunsetsu) が中心。長文を丸ごと変換する場面は少ない。
  → NAT (並列デコード) の速度優位が効く。
- **高頻度**: 1 文字タイプごとに 10ms 以下で候補を更新したい。
  → AR の O(seq_len) 生成は構造的に厳しい。CTC-NAT は O(1) ステップ。
- **個人差大**: 書き手・ドメイン・セッションで語彙と表記揺れが変わる。
  → CVAE で z をユーザ側に持たせ、モデル本体は共有したまま書き手別に
    適応する。辞書 LoRA のような局所パッチではなく、潜在空間での条件付け。

### zenz-v2.5 との比較方針

zenz-v2.5 (Miwa-Keita 氏) は同じ kana-kanji 変換タスクを扱うが**用途が異なる**:

| | zenz-v2.5 | new-ime |
|---|---|---|
| 用途 | AzooKey 内部の LLM 変換器 (macOS/iOS) | mozc 置き換えの IME (Windows 中心) |
| アーキテクチャ | GPT-2 系 auto-regressive | CTC-NAT + CVAE |
| 評価単位 | 文 (context 込み) | **句 (bunsetsu) が主**、文も副次 |
| 書き手適応 | 想定外 | **コア機能** |

したがって **主比較は同サイズ帯 (zenz-small 91M) での phrase レベル性能に絞る**。
medium (310M) との比較は参考値に留め、zenz-medium を超えることは目標にしない
(用途が違うため)。zenz-v2.5-small を 句レベル EM で追い抜き、かつ**レイテンシで
明確に優位**であれば、IME としての設計仮説は成立したと見なす。

## 設計原則

- タスクの本質に合致した構造を選ぶ
- 精度・速度・サイズ・適応性の全てで最高点
- 将来拡張性を持たせる

## アーキテクチャ: CTC-NAT Encoder-Decoder

```
[左文脈 + ひらがな入力]  + CVAE z (writer/domain/session)
    │
    ▼ Encoder (scratch, h=640, L=8, FiLM 条件付け)
    │
    ▼ Decoder (NAT, h=640, L=8, self+cross+ffn, FiLM 条件付け)
    │
    ▼ CTC Head → CTC collapse / beam search + KenLM shallow fusion
    │
    ▼ Mask-CTC refinement (2-3回, オプション)
    │
    ▼ ユーザ辞書ハード制約 + ユーザ学習スコア加算
    │
    漢字かな混じり出力 (top-K, K=10)
```

### モデル名

- **`new-ime-model`** (シリーズ名)
  - **`new-ime-model-90M`**: 本命研究対象。`configs/phase3_90m.yaml`、CVAE 有効
  - **`new-ime-model-20M`**: テスト/速度検証用の縮小プリセット。`configs/phase3_20m.yaml`、CVAE 無効

成果物 (checkpoint, ONNX, bitnet.cpp パック) のファイル名・HF リポジトリ名もこの規約に従う。
配布ライセンスは `MODEL_LICENSE` (CC BY-SA 4.0)、attribution は `ATTRIBUTION.md`。

### 規模 (`new-ime-model-90M`)

| コンポーネント | 構成 | パラメータ |
|---------------|------|-----------|
| Shared char embedding (tied) | 6500 × 640 | ~4.2M |
| Encoder (scratch) | 8層, h=640 | ~39M |
| Decoder (scratch) | 8層, h=640, self+cross+ffn | ~52M |
| CTC Head | Linear(640, 6500+1) | ~4.2M |
| CVAE (posterior biGRU + prior + FiLM) | - | ~7M |
| 合計 | | **~104M** (zenz-v2.5-small ~91M と同等帯) |

テスト/速度検証用に `phase3_20m` (h=384, L=6+6) プリセットを並置。BERT
(cl-tohoku/bert-base-japanese-char-v3, 110M) は単独で予算超過のため本体初期化からは
外し、Step B 任意オプションの MLM warm-up のみに使う。

### CTC-NAT を選ぶ理由

- 日本語読み→表記は単調アラインメント → CTC の前提と合致
- 並列生成でレイテンシが入力長に依存しない
- AR の構造崩壊 (「人口のうの」) が原理的に発生しない
- Beam search の exposure bias 問題がない
- Phase 2 の実験で beam 内に 95% 正解があることを確認済み

## 量子化: 1.58-bit QAT

### 段階的移行

1. **fp16 完全学習** → 安定精度確保、KD 教師として使用
2. **Continual QAT (fp16→1.58-bit)** → 中央値スケーリング (Nielsen 2024)
3. **Activation 8-bit 量子化** → weight 1.58-bit + activation 8-bit 混合

### 最終サイズ

- ~104M × 1.58-bit ≈ **15-25MB** (モデル本体)
- Activation + KV cache 込みで **25-40MB** (実メモリ)

### 根拠

- Nielsen 2024 "BitNet b1.58 Reloaded": 100K-48M で検証済み
- encoder-decoder (BERT/T5系) でも検証済み
- 暗黙的正則化効果で fp16 を超える場合もあり
- h=640 で fp16 同等性能確保の見込み (h=512-1024 の検証範囲内)

## ユーザ適応: 階層的 CVAE

### 潜在変数設計

```
z = (z_writer[32], z_domain[16], z_session[16]) = 64次元
```

- **z_writer**: 書き手の文体・語彙選好 (ユーザ固有、オンライン更新)
- **z_domain**: ビジネス/カジュアル/技術 (アプリ別に推定、fcitx5 からアプリ名取得可)
- **z_session**: 直近入力の傾向 (セッション内で動的更新)

### 推論時の適応

- ユーザが数十〜数百文入力 → z_writer をローカルでオンライン更新
- モデル本体は固定、z のみ更新 → プライバシー保護
- データは端末外に出ない

## データ戦略

### 目標: 100-200億トークン (長期)

**現状 (2026-04-18):**

| ソース | フィルタ後ペア | 品質 (監査) |
|--------|---------------|------------|
| Wikipedia JA | 18.4M | 87.8% clean |
| 青空文庫 | 2.4M | 92.0% clean |
| Livedoor News | 84K | - |
| Tatoeba JP | 228K | - |
| 合計 | **~20.8M** | |

さらに `tools/chunk-generator` (Rust) で文節チャンク **100M 件**を生成。詳細は `data_pipeline.md`。

**将来追加候補 (MIT 互換のみ、詳細は `dataset_candidates.md`):**

| ソース | 規模 | 用途 |
|--------|------|------|
| zenz-v2.5-dataset (llm-jp-corpus v3 サブセット, ODC-BY) | 32.4GB | 最小工数で即投入可 |
| CulturaX ja / FineWeb-2 jpn_Jpan | 100B+ tokens | 大規模 Web |
| HPLT v2/v3 ja (CC0) | 901B chars | クリーンな Web |
| おーぷん2ちゃんねる (Apache-2.0) | 8.14M 対話 | 口語 |
| 政府白書 (政府標準利用規約 2.0) | 数百MB | 硬文 |
| 国会会議録 (著作権法 40 条 1 項) | 数十GB | 政治演説 |
| 青空文庫 | 17,000 作品 | 文学 |

### 読み付与の精緻化

- MeCab unidic-lite `features[17]` (仮名形出現形) を使用
  - features[6/7/8/9] ではなく 17。Phase 2 v1/v2 の失敗を経て確定 (詳細は phase2_results.md)
- 青空文庫のルビ情報を教師信号に
- 数万文の人手検証サブセット

## 推論パイプライン

```
ひらがな入力 + 左文脈 + z (writer/domain/session)
    │
    ▼ Encoder (並列, ~5ms)
    │
    ▼ Decoder (NAT並列, ~5ms)
    │
    ▼ CTC beam search (beam=20) + KenLM shallow fusion
    │
    ▼ Mask-CTC refinement (2-3回, ~5ms each)
    │
    ▼ ユーザ辞書ハード制約 + ユーザ学習スコア加算
    │
    最終 top-K 候補 (K=10)
```

### 候補ランキングパイプライン (Phase 5)

モデル内部の N-best 生成とは独立に、以下を**別機構**で扱う:

```
CTC beam search (beam=20, KenLM fusion)
    ↓ top-20 候補
ユーザ辞書ルックアップ (ハード制約)
    ↓
ユーザ学習スコア加算 (SQLite, 指数減衰)
    ↓
最終 top-K (K=10)
```

最終スコア:
```
final_score(y) = model_score(y) + α·user_freq(y) + β·recency(y)
```

**SQLite スキーマ案:**
```sql
CREATE TABLE user_history (
    reading TEXT, context_hash INTEGER, candidate TEXT,
    count INTEGER DEFAULT 1, last_used TIMESTAMP,
    PRIMARY KEY (reading, context_hash, candidate)
);
CREATE TABLE user_dict (
    reading TEXT, candidate TEXT, priority INTEGER DEFAULT 0,
    PRIMARY KEY (reading, candidate)
);
```

- ユーザ学習: (読み, 文脈hash, 候補) → (count, last_used)
- アプリ別学習データ分離 (fcitx5 で取得可能)
- CVAE は v2 送り、まずはアプリ別分離のみ

### 目標速度

| 入力長 | 目標レイテンシ |
|--------|-------------|
| 短文 (5文字) | < 20ms |
| 中文 (15文字) | < 30ms |
| 長文 (30文字) | < 50ms |

## 推論基盤

- ONNX Runtime (C++) — encoder/decoder 別 ONNX ファイル
- CTC beam search: C++ 実装 (`tools/chunk-generator` の CTCDecoder を流用)
- KenLM shallow fusion: beam search のスコアに n-gram LM を加算
- fcitx5 統合: クライアント・サーバー方式 (Hazkey パターン)
- Windows TSF 統合: mozc フォーク + DLL 差し替え

## 配布プラットフォーム

| プラットフォーム | 方式 | 状態 |
|----------------|------|------|
| Windows (TSF) | mozc フォーク + DLL 差し替え | **DLL 動作確認済み** (engine/win32/) |
| Linux (fcitx5) | エンジンプラグイン | 設計済み、未ビルド |
| macOS (IMKit) | 将来拡張 | |
| モバイル (iOS/Android) | 将来拡張 | |

## 学習ロードマップ

詳細は `phase3_plan.md` (Step A-H)。概要:

### 同時検証 (CTC-NAT + CVAE + 1.58-bit を単一研究線で)

Phase 3 は 3 要素を段階分離せず、1 本の学習スケジュール (S0-S5) で順次オンにする。
両立不能が判明したときだけ撤退経路 (下記) に落とす。

- S0-S3: scratch CTC-NAT + GLAT + オンライン KD + Mask-CTC (fp16)
- S4: CVAE 有効化 (FiLM 条件付け、KL annealing)
- S5: 1.58-bit QAT 有効化 (BitLinear 置換、bitnet.cpp 経路)
- 並行: KenLM shallow fusion (Step E)、int8 ORT fallback (Step G)、bitnet.cpp カーネル (Step H)

## 目標精度

| ベンチ | Phase 2 (32M AR) | Phase 3 目標 |
|--------|------------------|-------------|
| manual_test EM | 0.800 | 0.900+ |
| AJIMEE EM | 0.450 | 0.800+ |
| gold_1k EM | 0.660 | 0.900+ |
| eval_v3 EM | 0.412 | 0.500+ |

比較ベースラインは `benchmark_comparison.md` 参照。

## 撤退経路

| リスク | 撤退先 |
|--------|--------|
| CTC-NAT 精度不足 (AR との差 3pt 以上) | AR + 投機的デコード |
| CTC-NAT 精度不足 (AR との差 2-3pt) | DAT (DAG 構造、`thu-coai/DA-Transformer` 流用) |
| 1.58-bit 品質劣化 | int8 量子化 (30-40MB) |
| CVAE posterior collapse | 条件付きモデル (非変分版) |
| KenLM 効果薄 | ユーザ辞書強化のみ |

## 採用しない選択と理由

### llama.cpp
- Decoder-only 前提、encoder-decoder 非対応
- zenz-v1 では使えるが CTC-NAT では使えない

### 1.58-bit を初期から
- Phase 3 fp16 → CQAT で 1.58-bit の段階的移行が安全
- QAT 再学習のコストは fp16 学習後にのみ発生

### Beam search (AR)
- 40-60x の速度ペナルティ (実装依存)
- Length normalization + repetition penalty でも改善限定的
- CTC beam search が構造的に優れている

## この構成の独自価値

既存 IME (mozc, zenz, Google, ATOK) のどれも持っていない組み合わせ:

- **小型だが高性能** (zenz-v2.5-small 水準)
- **極小サイズ** (1.58-bit で 15-40MB)
- **ユーザ適応可能** (CVAE)
- **並列生成で高速** (CTC-NAT)
- **多ドメイン対応** (100-200億トークン)
- **プライバシー保護** (ローカル適応)
