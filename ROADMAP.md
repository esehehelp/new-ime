# 実装ロードマップ: CTC-NAT ベースかな漢字変換エンジン

調査結果 (2026-04-17) を踏まえた具体的実装計画。PLAN.md の Phase 0〜6 を実装レベルに落とし込む。

---

## アーキテクチャ確定

### 変更点 (PLAN.md からの差分)

| 項目 | PLAN.md 時点 | 調査後 |
|------|-------------|--------|
| 主損失 | Cross-entropy on parallel output | **CTC loss** (単調アラインメント前提) |
| 長さ予測 | 分類ヘッドで予測 | **不要** (CTC が内部処理) |
| Refinement | CMLM mask-predict 1回 | **Mask-CTC** (CTC + mask-predict 複合) |
| ベース実装 | fairseq CMLM | **スクラッチ (PyTorch)** (fairseq は 2026-03 アーカイブ済み) |
| エンコーダ初期化 | cl-tohoku/bert-japanese | **cl-tohoku/bert-base-japanese-char-v3** (文字単位、HF直接) |
| 候補生成 | 未定 | **CTC beam search** (top-k パス抽出) |
| 出力語彙 | SentencePiece BPE 32k | **文字単位 ~6000** (CTC は細粒度トークンが有利) |

### 確定アーキテクチャ

```
入力: [左文脈(40文字)] [SEP] [ひらがな入力]
                │
       ┌────────▼─────────┐
       │     Encoder       │  cl-tohoku/bert-base-japanese-char-v3
       │  12層, hidden 768 │  事前学習済みを初期化に使用
       │  入力語彙: ~100   │  (ひらがな + カタカナ + 制御トークン)
       └────────┬─────────┘
                │
       ┌────────▼─────────┐
       │     Decoder       │  6層, hidden 768, 8ヘッド
       │  (Non-autoregressive) │  cross-attention でエンコーダ参照
       │  出力長 = 入力長   │  CTC が圧縮を処理
       │  出力語彙: ~6000  │  (漢字 + かな + 記号、文字単位)
       └────────┬─────────┘
                │
       ┌────────▼─────────┐
       │    CTC Collapse   │  blank トークン除去 + 重複統合
       │    or             │
       │    Mask-Predict   │  低信頼トークンをマスク → 1回再予測
       └────────┬─────────┘
                │
       漢字かな混じり出力 (top-k 候補)
```

### パラメータ見積もり

| コンポーネント | 層数 | hidden | パラメータ数 |
|---------------|------|--------|-------------|
| Encoder (BERT初期化) | 12 | 768 | ~85M |
| Decoder (スクラッチ) | 6 | 768 | ~45M |
| CTC出力ヘッド | - | 768→6000 | ~4.6M |
| 合計 | | | **~135M** |

エンコーダは凍結→段階的解凍で学習コスト削減可能。

---

## Step 0: プロジェクト基盤 (1週間)

### 0.1 開発環境セットアップ

```
new-ime/
├── src/
│   ├── model/          # モデル定義
│   │   ├── encoder.py  # BERT エンコーダラッパー
│   │   ├── decoder.py  # NAT デコーダ
│   │   ├── ctc_nat.py  # CTC-NAT 統合モデル
│   │   └── mask_predict.py  # Mask-CTC refinement
│   ├── data/           # データパイプライン
│   │   ├── tokenizer.py     # 入出力トークナイザ
│   │   ├── dataset.py       # PyTorch Dataset
│   │   └── preprocessing.py # MeCab読み付与、フィルタ
│   ├── training/       # 学習ループ
│   │   ├── trainer.py
│   │   ├── glat.py     # GLAT サンプリング
│   │   └── distillation.py  # KD (Phase 2 の AR から)
│   ├── inference/      # 推論
│   │   ├── ctc_decode.py    # CTC greedy / beam search
│   │   └── export_onnx.py   # ONNX エクスポート
│   └── eval/           # 評価
│       ├── metrics.py       # 文字精度、文節境界、レイテンシ
│       └── benchmark.py     # zenz-v1 / mozc 比較
├── configs/            # 学習設定 YAML
├── scripts/            # データ準備スクリプト
├── tests/              # ユニットテスト
├── references/         # 参照リポジトリ (gitignore)
├── PLAN.md
├── ROADMAP.md
└── .gitignore
```

### 0.2 依存関係

```
# core
torch >= 2.2
transformers >= 4.40  # cl-tohoku/bert-base-japanese-char-v3
sentencepiece
tokenizers

# data
mecab-python3
unidic-lite  # or ipadic-neologd
jaconv

# training
wandb
accelerate  # multi-GPU, mixed precision

# eval
onnxruntime

# dev
pytest
ruff
```

### 0.3 成果物

- [ ] プロジェクト構造作成
- [ ] pyproject.toml / requirements.txt
- [ ] CI: ruff lint + pytest
- [ ] cl-tohoku/bert-base-japanese-char-v3 のダウンロードと動作確認
- [ ] 出力語彙の設計: Unicode 漢字+かな+記号から ~6000 文字を選定する基準の確定

---

## Step 1: トークナイザとデータパイプライン (3〜4週間)

### 1.1 入力トークナイザ (文字単位)

- ひらがな 83文字 + カタカナ 86文字 + 長音符・濁点等 + 制御トークン ([PAD], [SEP], [UNK], [CLS])
- 合計 ~180 トークン
- BERT の既存トークナイザの部分集合として構成可能か確認 → 可能なら BERT トークナイザをそのまま使う

### 1.2 出力トークナイザ (文字単位)

- zenz-v1 と同じ方針: 文字単位 + バイトフォールバック
- JIS X 0208 の漢字 (~6355字) + ひらがな + カタカナ + ASCII + 記号 + [BLANK] (CTC用) + [MASK] (refinement用)
- 語彙サイズ: ~6500
- 未知漢字はバイトフォールバック (UTF-8 バイト列として出力)

### 1.3 読み付与パイプライン

```
原文 (漢字かな混じり)
    │
    ▼ MeCab + ipadic-NEologd
形態素列: [(表層, 読み, 品詞), ...]
    │
    ▼ jaconv でカタカナ→ひらがな統一
    │
    ▼ フィルタ:
    │   - 読み不明形態素が含まれる文を除外
    │   - 外国語比率 > 30% の文を除外
    │   - 文長 3〜80文字
    │
    ▼ 三つ組生成
(ひらがな読み, 漢字表記, 前文脈)
```

### 1.4 データソースと量

| ソース | 推定文数 | 用途 |
|--------|---------|------|
| Wikipedia JA | ~15M 文 | 主力 |
| 青空文庫 | ~2M 文 | 文体多様性 |
| CC-100 JA (フィルタ後) | ~5M 文 | 口語補強 |
| **合計** | **~10M 文ペア (初期)** | |

### 1.5 評価セット

- Wikipedia から時期で分割 (2024年以降の記事をテスト用)
- 分野別層化: 技術、文学、日常、固有名詞
- 各 5000 文、計 20000 文

### 1.6 成果物

- [ ] 入力トークナイザ実装 + テスト
- [ ] 出力トークナイザ実装 + テスト (バイトフォールバック含む)
- [ ] MeCab 読み付与スクリプト
- [ ] フィルタリングパイプライン
- [ ] Wikipedia ダンプ処理スクリプト
- [ ] 青空文庫処理スクリプト
- [ ] データ統計レポート生成
- [ ] 評価セット作成
- [ ] mozc ベースライン精度測定

---

## Step 2: AR ベースライン (2〜3週間)

Phase 2 の AR モデルは **KD 教師として必須**。NAT 学習前に用意する。

### 2.1 モデル

- GPT-2 スタイル decoder-only, ~90M パラメータ (zenz-v1 と同規模)
- `ku-nlp/gpt2-small-japanese-char` から初期化してファインチューン
- 入力形式: `[左文脈][SEP][ひらがな読み][OUT][漢字表記][EOS]`

### 2.2 学習

- Causal LM, teacher forcing
- bf16 混合精度, AdamW, warmup + cosine decay
- A100 1枚, ~10-20時間
- 予算: ~$30-60

### 2.3 KD データ生成

学習データの各ひらがな入力に対して AR モデルで greedy decode し、出力を保存。
これが Step 3 の CTC-NAT 学習データ (教師出力) になる。

### 2.4 成果物

- [ ] AR ベースラインモデル
- [ ] 評価: 文字精度、mozc/zenz-v1 比較
- [ ] KD 用デコード出力 (10M文分)

### 2.5 Go/No-Go

mozc を文字精度で超えること。超えなければデータ品質を見直す。

---

## Step 3: CTC-NAT 本体実装 (6〜8週間)

### 3.1 モデル実装

#### Encoder (encoder.py, ~100行)

```python
# cl-tohoku/bert-base-japanese-char-v3 をラップ
# 入力: [左文脈][SEP][ひらがな] → token_ids
# 出力: encoder_hidden (batch, seq_len, 768)
```

- HuggingFace の `BertModel.from_pretrained()` で初期化
- 最初はエンコーダを凍結、Step 3.4 で段階的に解凍

#### Decoder (decoder.py, ~200行)

```python
# 6層 Transformer デコーダ
# Self-attention: 双方向 (causal mask なし — NAT なので)
# Cross-attention: エンコーダ出力を参照
# 入力: エンコーダと同じ長さの query token 列
# 出力: (batch, seq_len, 6000) — 各位置の文字確率
```

- Query tokens の初期化: エンコーダ出力をそのまま使う (upsampling なし、入力長 = 出力長)
- CTC の blank トークンが長さの差を吸収する

#### CTC ヘッド (ctc_nat.py, ~150行)

```python
# Linear(768, vocab_size + 1)  # +1 for CTC blank
# CTC loss: torch.nn.CTCLoss()
# 入力系列長 >= 出力系列長 が CTC の制約 → かな長 >= 漢字長で通常成立
```

#### Mask-CTC Refinement (mask_predict.py, ~150行)

```python
# CTC の出力で信頼度が低い位置を [MASK] に置換
# デコーダを再度実行して [MASK] 位置を再予測
# 1回のみ (追加コスト: デコーダ 1パス分)
```

### 3.2 GLAT 学習 (glat.py, ~100行)

```python
# 1. Forward pass → 予測を得る
# 2. 予測と正解の Hamming distance を計算
# 3. distance / length * lambda (lambda=0.5) の割合で
#    正解トークンをデコーダ入力に「見せる」
# 4. 残りの位置について CTC loss を計算
```

### 3.3 Knowledge Distillation

- Step 2 の AR モデル出力を正解として使う
- 実データの正解も混ぜる (比率は要実験: KD 100% vs KD 50% + real 50%)

### 3.4 学習スケジュール

| フェーズ | エンコーダ | デコーダ | 損失 | ステップ数 |
|---------|-----------|---------|------|-----------|
| 3.4a: デコーダ学習 | 凍結 | 学習 | CTC + GLAT (KDデータ) | 100k |
| 3.4b: 全体ファインチューン | 低lr (1e-5) | 通常lr (3e-4) | CTC + GLAT (KD + real混合) | 200k |
| 3.4c: Mask-CTC 追加 | 低lr | 低lr | CTC + Mask-predict | 50k |

推定学習時間: H100 で 20〜40 時間
予算: $100〜300

### 3.5 推論パイプライン

```
ひらがな入力 + 左文脈
    │
    ▼ Encoder (1パス, ~5ms CPU)
    │
    ▼ Decoder (1パス並列, ~5ms CPU)
    │
    ▼ CTC greedy collapse  → 第1候補 (~0.1ms)
    │   or
    ▼ CTC beam search (beam=5) → top-5候補 (~1ms)
    │
    ▼ (オプション) Mask-CTC refinement 1回 (~5ms)
    │
    最終出力: 候補リスト
```

目標レイテンシ: refinement 込みで **< 20ms on modern CPU**

### 3.6 成果物

- [ ] encoder.py + テスト
- [ ] decoder.py + テスト
- [ ] ctc_nat.py (統合モデル) + テスト
- [ ] glat.py + テスト
- [ ] mask_predict.py + テスト
- [ ] 学習スクリプト (configs/ に YAML)
- [ ] CTC greedy / beam search デコード
- [ ] ONNX エクスポートスクリプト
- [ ] ベンチマーク: 精度 + レイテンシ vs AR ベースライン vs zenz-v1

### 3.7 Go/No-Go

| 条件 | 基準 |
|------|------|
| 精度 | AR ベースライン (Step 2) と同等以上 |
| レイテンシ | AR の 1/3 以下 |
| 両立 | 両方満たすこと |

**失敗時の撤退**:
- CTC-NAT の精度が不十分 → DAT (DAG構造) にエスカレート (`thu-coai/DA-Transformer` のアルゴリズムをスクラッチ移植)
- DAT も不十分 → AR + 投機的デコードに切り替え (データパイプラインと評価基盤は再利用)

---

## Step 4: 評価と改善 (4〜6週間)

### 4.1 評価項目

| 指標 | 測定方法 |
|------|---------|
| 文字精度 (CharAcc) | 正解文字数 / 全文字数 |
| 文節境界精度 | MeCab で文節分割し境界一致率 |
| 完全一致率 | 文単位で出力が正解と完全一致 |
| 推論レイテンシ | CPU (i7/Ryzen) での p50/p95/p99 |
| モデルサイズ | fp16, int8, ONNX それぞれ |

### 4.2 誤り分析

- 同音異義語の誤変換パターン分類
- 固有名詞の扱い
- 送りがなの誤り
- 文脈依存の変換精度

### 4.3 改善サイクル

誤り分析 → データ拡張 or 損失調整 → 再学習 → 再評価 (最大3サイクル)

---

## Step 5: 推論エンジンと fcitx5 統合 (4〜6週間)

### 5.0 アーキテクチャ決定: クライアント・サーバー方式

Hazkey (MIT ライセンス) と fcitx5-mozc の両方がクライアント・サーバー方式を採用。我々も同じパターンに従う。

```
┌─────────────────────────────────────────────────┐
│                 fcitx5 プロセス                    │
│                                                   │
│  ┌──────────────────────────────┐                │
│  │  new-ime エンジンプラグイン     │  C++, ~1500行 │
│  │  (InputMethodEngineV2)        │                │
│  │                               │                │
│  │  - keyEvent 処理              │                │
│  │  - preedit 表示               │                │
│  │  - 候補リスト管理              │                │
│  │  - ローマ字→ひらがな変換       │                │
│  │  - サーバー接続管理            │                │
│  └──────────┬───────────────────┘                │
│             │ Unix domain socket                  │
│             │ protobuf (length-prefixed)          │
└─────────────┼─────────────────────────────────────┘
              │
┌─────────────▼─────────────────────────────────────┐
│           new-ime-server プロセス                    │
│           C++, ~2000行                               │
│                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │ Socket Server │ │ ONNX Runtime │ │ CTC Decoder │ │
│  │ (Unix domain) │ │ (推論)       │ │ (beam search)│ │
│  └──────┬───────┘ └──────┬──────┘ └──────┬──────┘ │
│         │                │               │         │
│  ┌──────▼────────────────▼───────────────▼───────┐ │
│  │             ConversionEngine                    │ │
│  │  - encode(context + kana)                       │ │
│  │  - decode(encoder_out) → CTC logits             │ │
│  │  - beam_search(logits, beam=5) → candidates     │ │
│  │  - mask_refine(low_conf) → refined (optional)   │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  ┌────────────────┐  ┌──────────────────┐           │
│  │ User Dictionary │  │ Encoder KV Cache  │          │
│  │ (制約デコード用) │  │ (prefix 再利用)   │          │
│  └────────────────┘  └──────────────────┘           │
└──────────────────────────────────────────────────────┘
```

**クライアント・サーバーを選ぶ理由**:

| 理由 | 詳細 |
|------|------|
| fcitx5 はシングルスレッド | keyEvent で 15-20ms ブロックすると全入力コンテキストが止まる |
| クラッシュ隔離 | ONNX Runtime の障害が IME フレームワークを巻き込まない |
| メモリ管理 | ~135M param モデル (int8 で ~135MB) をサーバープロセスに閉じ込める |
| 実績 | Hazkey, fcitx5-mozc が同パターンで安定稼働 |

### 5.1 ONNX エクスポートとモデル最適化

#### エクスポート

Encoder と Decoder は**別 ONNX ファイル**にする:
- `new-ime-encoder.onnx` — BERT encoder (入力 → hidden states)
- `new-ime-decoder.onnx` — NAT decoder (hidden states → CTC logits)
- `new-ime-refiner.onnx` — Mask-CTC refinement (オプション)

```python
torch.onnx.export(
    encoder, (input_ids, attention_mask),
    "new-ime-encoder.onnx",
    opset_version=17,
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}}
)
```

分割理由: Encoder KV キャッシュの再利用が容易、個別に量子化レベルを変えられる。

#### 量子化

| 形式 | サイズ見積もり | 手法 |
|------|-------------|------|
| fp32 | ~540MB | - |
| fp16 | ~270MB | ONNX cast |
| int8 dynamic | ~135MB | `onnxruntime.quantization` (推奨) |
| int8 static | ~135MB | calibration データ必要 |

### 5.2 推論サーバー (new-ime-server, C++)

#### コンポーネント構成

| コンポーネント | 行数目安 | 役割 |
|---------------|---------|------|
| `socket_server.cpp` | ~300行 | Unix domain socket, length-prefixed protobuf, select() ループ, シグナルハンドリング |
| `onnx_inference_engine.cpp` | ~400行 | Encoder/Decoder ONNX セッション管理, KV キャッシュ |
| `ctc_decoder.cpp` | ~300行 | CTC greedy collapse, prefix beam search (Hannun 2014) |
| `conversion_engine.cpp` | ~500行 | 変換 API 統合, ユーザ辞書, インクリメンタル変換 |
| `protocol_handler.cpp` | ~200行 | protobuf dispatch |
| **合計** | **~1700行** | |

#### ONNX Runtime 設定

```cpp
Ort::SessionOptions opts;
opts.SetIntraOpNumThreads(4);
opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
opts.EnableMemPattern(true);
// int8 モデルの場合、追加設定不要 (量子化済み ONNX をそのままロード)
```

#### ソケットパス

```
${XDG_RUNTIME_DIR}/new-ime-server.${UID}.sock
```

自動起動: エンジンプラグインがサーバー未起動を検出 → `startProcess()` で起動、最大8回リトライ (250ms間隔)。Hazkey と同じパターン。

### 5.3 IPC プロトコル (protobuf)

Hazkey のプロトコル (12種コマンド) を大幅に簡略化。ローマ字→ひらがな変換・カーソル管理はエンジン側で行い、サーバーには変換リクエストのみ送る。

```protobuf
syntax = "proto3";
package newime;
option optimize_for = LITE_RUNTIME;

message RequestEnvelope {
    oneof payload {
        ConvertRequest convert = 1;
        ConvertIncrementalRequest convert_incremental = 2;
        SetContextRequest set_context = 3;
        ReloadModelRequest reload_model = 4;
        ShutdownRequest shutdown = 5;
    }
}

message ConvertRequest {
    string kana_input = 1;       // ひらがな入力
    string left_context = 2;     // 左文脈 (最大40文字)
    int32 num_candidates = 3;    // 候補数 (default 5)
    bool use_refinement = 4;     // Mask-CTC refinement
}

message ConvertIncrementalRequest {
    string kana_input = 1;
    string left_context = 2;
    string cached_prefix = 3;    // encoder キャッシュ用
    int32 num_candidates = 4;
}

message SetContextRequest {
    string surrounding_text = 1;
    int32 cursor_position = 2;
}

message ReloadModelRequest { string model_path = 1; }
message ShutdownRequest {}

message ResponseEnvelope {
    enum Status { SUCCESS = 0; FAILED = 1; }
    Status status = 1;
    string error_message = 2;
    oneof payload {
        CandidatesResult candidates = 3;
    }
}

message CandidatesResult {
    repeated Candidate candidates = 1;
}

message Candidate {
    string text = 1;             // 変換結果 (漢字かな混じり)
    string reading = 2;          // 読み (ひらがな)
    float score = 3;             // 信頼度スコア
}
```

### 5.4 fcitx5 エンジンプラグイン

#### クラス構成

| クラス | 継承元 (fcitx5) | 行数目安 |
|--------|-----------------|---------|
| `NewImeEngine` | `InputMethodEngineV2` | ~200行 |
| `NewImeState` | `InputContextProperty` | ~500行 |
| `NewImeCandidateList` | `CommonCandidateList` | ~100行 |
| `NewImeCandidateWord` | `CandidateWord` | ~30行 |
| `NewImePreedit` | (独自) | ~150行 |
| `NewImeServerConnector` | (独自) | ~250行 |
| `NewImeComposingText` | (独自) | ~300行 |
| **合計** | | **~1530行** |

#### キーイベント ステートマシン

```
[直接入力] ──かな/ローマ字キー──▶ [Composing]
                                    │
                 preedit にひらがな表示
                 (逐次: ConvertIncrementalRequest で予測変換)
                                    │
                       スペース/変換キー
                                    │
                              ▼
                          [候補選択] ── 候補リスト表示
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         数字キー          スペース        Escape
         候補確定          次候補          キャンセル
              │               │               │
              ▼               ▼               ▼
         commitString    次候補表示     [Composing]
              │
              ▼
         [直接入力]
```

#### ローマ字→ひらがな変換 (NewImeComposingText)

```cpp
// エンジン側で処理 (サーバーに送らない)
// ローマ字テーブル: "ka"→"か", "shi"→"し", "xtu"→"っ"
// n + 子音 → "ん" + 子音, nn → "ん"
class NewImeComposingText {
    std::string romaji_buffer_;  // 未確定ローマ字
    std::string hiragana_;       // 確定ひらがな
    int cursor_;

    void input_char(char c);
    void delete_left();
    void move_cursor(int offset);
    std::string get_hiragana() const;
    std::string get_display() const;  // ひらがな + 未確定ローマ字
};
```

#### メタデータ

```ini
# new-ime-addon.conf
[Addon]
Name=new-ime
Category=InputMethod
Type=SharedLibrary
Library=fcitx5-new-ime
OnDemand=True

# new-ime-im.conf
[InputMethod]
Name=new-ime
LangCode=ja
Addon=new-ime
```

### 5.5 ビルドシステム (CMake)

```cmake
cmake_minimum_required(VERSION 3.21)
project(new-ime VERSION 0.1.0)

find_package(Fcitx5Core REQUIRED)
find_package(Fcitx5Utils REQUIRED)
find_package(Fcitx5Config REQUIRED)
find_package(Protobuf 3.12 REQUIRED)
find_package(onnxruntime REQUIRED)

# Protobuf code generation
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS protocol/new_ime.proto)

# --- fcitx5 engine plugin ---
add_library(fcitx5-new-ime SHARED
    engine/new_ime_engine.cpp
    engine/new_ime_state.cpp
    engine/new_ime_candidate.cpp
    engine/new_ime_preedit.cpp
    engine/new_ime_server_connector.cpp
    engine/new_ime_composing_text.cpp
    ${PROTO_SRCS}
)
set_target_properties(fcitx5-new-ime PROPERTIES PREFIX "")
target_link_libraries(fcitx5-new-ime
    Fcitx5::Core Fcitx5::Config
    protobuf::libprotobuf-lite
)

# --- inference server ---
add_executable(new-ime-server
    server/main.cpp
    server/socket_server.cpp
    server/onnx_inference_engine.cpp
    server/ctc_decoder.cpp
    server/conversion_engine.cpp
    server/protocol_handler.cpp
    ${PROTO_SRCS}
)
target_link_libraries(new-ime-server
    onnxruntime
    protobuf::libprotobuf-lite
)
```

### 5.6 ファイル配置

```
/usr/lib/fcitx5/fcitx5-new-ime.so             # エンジンプラグイン
/usr/lib/new-ime/new-ime-server                # 推論サーバー
/usr/share/fcitx5/addon/new-ime-addon.conf     # アドオン登録
/usr/share/fcitx5/inputmethod/new-ime-im.conf  # 入力メソッド登録
/usr/share/new-ime/models/encoder.onnx         # Encoder モデル
/usr/share/new-ime/models/decoder.onnx         # Decoder モデル
/usr/share/new-ime/tokenizer/vocab.json        # トークナイザ語彙
```

### 5.7 パッケージング

| 形式 | 優先度 | 備考 |
|------|--------|------|
| Arch AUR | 最優先 | 自分で使う環境 |
| .deb | 次点 | Ubuntu/Debian 向け |
| Flatpak | 後回し | sandbox と Unix socket の相性検証要 |

### 5.8 成果物チェックリスト

- [ ] ONNX エクスポート (encoder.onnx, decoder.onnx, refiner.onnx)
- [ ] ONNX int8 量子化 + 精度劣化検証
- [ ] protobuf スキーマ定義 (new_ime.proto)
- [ ] new-ime-server
  - [ ] socket_server (Unix domain socket)
  - [ ] onnx_inference_engine (encoder + decoder)
  - [ ] ctc_decoder (greedy + beam search)
  - [ ] conversion_engine (統合 API)
  - [ ] protocol_handler
- [ ] fcitx5 エンジンプラグイン
  - [ ] NewImeEngine
  - [ ] NewImeState (キーイベント ステートマシン)
  - [ ] NewImeCandidateList / CandidateWord
  - [ ] NewImePreedit
  - [ ] NewImeServerConnector (IPC クライアント)
  - [ ] NewImeComposingText (ローマ字→ひらがな)
- [ ] CMake ビルドシステム
- [ ] メタデータファイル (.conf)
- [ ] Arch AUR PKGBUILD
- [ ] 結合テスト (エンジン ↔ サーバー ↔ モデル)
- [ ] end-to-end レイテンシ計測 (キー入力→候補表示)

---

## Step 6: 量子化とユーザ適応 (v1.0 後)

v1.0 リリース後の拡張。PLAN.md Phase 6 と同じ。

- 1.58-bit QAT (BitNet 方式)
- CVAE ユーザ適応
- LoRA による軽量パーソナライズ (CVAE の代替として先に試す価値あり)

---

## タイムライン

```
Week 1       : Step 0 — プロジェクト基盤
Week 2-5     : Step 1 — データパイプライン
Week 6-8     : Step 2 — AR ベースライン + KD データ生成
Week 9-16    : Step 3 — CTC-NAT 本体 (ここが山場)
Week 17-22   : Step 4 — 評価・改善サイクル
Week 23-28   : Step 5 — ONNX + fcitx5 統合
```

**合計: ~28 週間 (7ヶ月)** at 週 10-20 時間

---

## 技術的リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| CTC-NAT の精度が AR に届かない | Step 3 停滞 | DAT にエスカレート、最終的に AR+投機的デコード |
| エンコーダ凍結では不十分 | 精度不足 | 段階的解凍 (3.4b) で対応済み |
| CTC の入力長 >= 出力長制約に違反 | 学習不能 | かな入力にパディング or upsampling 層追加 |
| ONNX エクスポートで CTC beam search が対応不可 | 推論エンジン問題 | beam search は ONNX 外で Python/C++ 実装 |
| Hazkey のライセンスが非互換 | fcitx5統合の設計変更 | **確認済み: MIT ライセンス。問題なし** |
| fcitx5 シングルスレッドで推論ブロック | UI フリーズ | クライアント・サーバー方式で解決済み (5.0) |
| ONNX エクスポートで動的形状が問題 | 推論不可 | dynamic_axes 指定 + opset 17 で対応 |

---

## 参考文献 (実装時に参照)

### 必読

1. Ghazvininejad et al. 2019 — Mask-Predict (CMLM): アルゴリズムの基本
2. Qian et al. 2021 — GLAT: 学習安定化の核心テクニック
3. Higuchi et al. 2021 — Mask-CTC: CTC + mask-predict の複合 (ASR 文脈)
4. Saharia et al. 2020 — CTC-based NAT for translation

### 参考

5. Gu et al. 2018 — 元祖 NAT (fertility、KD の背景理解)
6. Huang et al. 2022 — DAT (撤退先の技術理解)
7. Shao & Feng 2022 — NMLA-NAT (CTC + 非単調対応、NeurIPS Spotlight)

### 実装参考

8. fairseq `cmlm_transformer.py` — CMLM のリファレンス実装 (~450行)
9. thu-coai/DA-Transformer — DAT 実装 (DAG DP のロジック)
10. ictnlp/nmla-nat — CTC-NAT 実装
