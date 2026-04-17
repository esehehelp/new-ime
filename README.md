# new-ime

CTC-NAT (Connectionist Temporal Classification + Non-Autoregressive Transformer) ベースの日本語かな漢字変換エンジン。fcitx5 向け。

## 概要

既存の自己回帰モデル (zenz-v1 等) に対し、**並列生成アーキテクチャ**でレイテンシの大幅削減を狙う。

- **Encoder**: cl-tohoku/bert-base-japanese-char-v3 (事前学習済み、12層, 768dim)
- **Decoder**: Non-autoregressive Transformer (双方向 self-attention, 6層)
- **出力**: CTC loss による並列デコード + Mask-CTC refinement
- **学習安定化**: GLAT (Glancing Language Model Training) + Knowledge Distillation
- **推論**: ONNX Runtime (CPU, int8 量子化)
- **IME統合**: fcitx5 プラグイン (クライアント・サーバー方式)

## アーキテクチャ

```
[左文脈 + ひらがな入力]
        │
   ┌────▼─────────┐
   │   Encoder     │  BERT (事前学習済み)
   └────┬─────────┘
        │
   ┌────▼─────────┐
   │   Decoder     │  NAT (並列生成)
   │   (NAT)       │  cross-attention でエンコーダ参照
   └────┬─────────┘
        │
   ┌────▼─────────┐
   │   CTC Head    │  CTC collapse / beam search
   └────┬─────────┘
        │
   漢字かな混じり出力 (top-k 候補)
```

## プロジェクト構成

```
new-ime/
├── src/                    # Python (モデル・学習・データ)
│   ├── model/              #   CTC-NAT モデル定義
│   │   ├── encoder.py      #     BERT エンコーダ
│   │   ├── decoder.py      #     NAT デコーダ
│   │   └── ctc_nat.py      #     統合モデル + GLAT + Mask-CTC
│   ├── data/               #   トークナイザ・データセット
│   │   └── tokenizer.py    #     入力 (かな) / 出力 (漢字) トークナイザ
│   ├── training/           #   学習ループ (TODO)
│   ├── inference/          #   推論・ONNX エクスポート (TODO)
│   └── eval/               #   評価 (TODO)
├── engine/                 # C++ (fcitx5 エンジンプラグイン)
│   └── src/
│       ├── engine.cpp/h    #   fcitx5 InputMethodEngineV2
│       ├── composing_text  #   ローマ字→ひらがな変換
│       ├── preedit         #   preedit 表示管理
│       └── server_connector#   IPC クライアント
├── server/                 # C++ (推論サーバー)
│   └── src/
│       ├── main.cpp        #   サーバーエントリポイント
│       ├── socket_server   #   Unix domain socket
│       ├── ctc_decoder     #   CTC greedy / beam search
│       └── conversion_engine#  変換ロジック (ONNX Runtime)
├── protocol/               # protobuf IPC スキーマ
│   └── new_ime.proto
├── scripts/                # データパイプライン
│   ├── process_wiki.py     #   Wikipedia dump → JSONL (マルチプロセス)
│   ├── process_aozora.py   #   青空文庫 → JSONL
│   ├── postprocess.py      #   品質フィルタ (旧仮名・ト書き除去等)
│   ├── build_vocab.py      #   文字頻度 → 出力語彙構築
│   ├── audit_data.py       #   データ品質監査
│   └── dataset_stats.py    #   統計レポート
├── tests/                  # テスト
│   ├── test_tokenizer.py   #   Python トークナイザ (19 tests)
│   ├── test_model.py       #   CTC-NAT モデル (17 tests)
│   └── cpp/
│       ├── test_composing_text.cpp  # ローマ字変換 (12 tests)
│       └── test_ctc_decoder.cpp     # CTC デコーダ (5 tests)
├── PLAN.md                 # 全体計画 (6フェーズ)
├── ROADMAP.md              # 実装ロードマップ (詳細)
├── CMakeLists.txt          # C++ ビルド
└── pyproject.toml          # Python (uv)
```

## セットアップ

### Python (モデル・データパイプライン)

```bash
uv sync --group dev
uv run pytest
```

### C++ テスト (gcc)

```bash
g++ -std=c++20 -I engine/src -o test_composing tests/cpp/test_composing_text.cpp engine/src/composing_text.cpp && ./test_composing
g++ -std=c++20 -I server/src -o test_ctc tests/cpp/test_ctc_decoder.cpp server/src/ctc_decoder.cpp && ./test_ctc
```

### C++ ビルド (CMake, Linux)

```bash
cmake -B build -DENABLE_FCITX5=ON  # fcitx5 プラグイン
cmake --build build
```

## データパイプライン

```bash
# Wikipedia
uv run python scripts/process_wiki.py \
    --input datasets/src/jawiki-latest-pages-articles.xml.bz2 \
    --output datasets/wiki_sentences.jsonl \
    --workers 16

# 青空文庫
uv run python scripts/process_aozora.py \
    --input datasets/src/utf8_all.csv.gz \
    --output datasets/aozora_sentences.jsonl

# 後処理 (品質フィルタ)
uv run python scripts/postprocess.py \
    --input datasets/wiki_sentences.jsonl datasets/aozora_sentences.jsonl \
    --output datasets/all_clean.jsonl \
    --stats

# 語彙構築
uv run python scripts/build_vocab.py \
    --input datasets/all_clean.jsonl \
    --output datasets/vocab.json
```

## 参考文献

- [Mask-Predict (CMLM)](https://arxiv.org/abs/1904.09324) — Ghazvininejad et al. 2019
- [GLAT](https://arxiv.org/abs/2008.07905) — Qian et al. 2021
- [Mask-CTC](https://arxiv.org/abs/2005.08700) — Higuchi et al. 2021
- [DA-Transformer](https://arxiv.org/abs/2205.07459) — Huang et al. 2022
- [zenz-v1](https://huggingface.co/Miwa-Keita/zenz-v1) — Miwa 2024
- [Hazkey](https://github.com/7ka-Hiira/hazkey) — fcitx5 日本語 IME

## ライセンス

MIT
