# new-ime

CTC-NAT (Connectionist Temporal Classification + Non-Autoregressive Transformer) ベースの日本語かな漢字変換エンジン。Windows TSF + fcitx5 対応。

## 概要

既存の自己回帰モデル (zenz-v1 等) に対し、**並列生成アーキテクチャ** でレイテンシ削減と
**1.58-bit 量子化** で極小サイズを狙う。

- **Encoder**: cl-tohoku/bert-base-japanese-char-v3 (事前学習済み、12層, 768dim)
- **Decoder**: Non-autoregressive Transformer (双方向 self-attention, 6層)
- **出力**: CTC loss による並列デコード + Mask-CTC refinement
- **学習安定化**: GLAT (Glancing Language Model Training) + Knowledge Distillation
- **推論**: ONNX Runtime (CPU, int8 量子化 → v1 以降 1.58-bit QAT)
- **IME 統合**: Windows TSF (DLL 動作確認済) / fcitx5 プラグイン (クライアント・サーバー方式)

設計詳細は `docs/vision.md`、実装計画は `docs/roadmap.md` を参照。

## アーキテクチャ

```
[左文脈 + ひらがな入力]
        │
   ┌────▼─────────┐
   │   Encoder     │  BERT (事前学習済み, 12層)
   └────┬─────────┘
        │
   ┌────▼─────────┐
   │   Decoder     │  NAT (並列生成, 6層)
   │   (NAT)       │  cross-attention でエンコーダ参照
   └────┬─────────┘
        │
   ┌────▼─────────┐
   │   CTC Head    │  CTC collapse / beam search + KenLM
   └────┬─────────┘
        │
   漢字かな混じり出力 (top-K 候補)
```

## 現状 (2026-04-18)

| Phase | 状態 |
|-------|------|
| Phase 0 (設計) | 完了 |
| Phase 1 (データパイプライン) | 完了 — 20.8M ペア + 100M チャンク |
| Phase 2 (AR ベースライン) | 完了 — 31.9M, manual 80/100, eval_v3 EM 0.412 |
| Phase 3 (CTC-NAT) | 未着手 |
| Phase 5 (Windows TSF DLL) | 動作確認済 |
| Phase 5 (fcitx5 プラグイン) | ソース実装済、未ビルド |

Phase 2 の詳細は `docs/phase2_results.md`、ベンチ比較は `docs/benchmark_comparison.md`。

## プロジェクト構成

```
new-ime/
├── src/                       # Python (モデル・学習・データ・評価)
│   ├── model/                 #   CTC-NAT モデル定義
│   │   ├── encoder.py         #     BERT エンコーダ
│   │   ├── decoder.py         #     NAT デコーダ
│   │   └── ctc_nat.py         #     統合モデル + GLAT + Mask-CTC
│   ├── data/                  #   トークナイザ・データセット
│   │   └── tokenizer.py
│   ├── training/              #   学習ループ
│   ├── inference/             #   推論
│   └── eval/                  #   評価
│       ├── metrics.py         #     edit distance, CharAcc, EM
│       ├── run_eval.py        #     バックエンド抽象 + レイテンシ測定
│       ├── bench_loaders.py   #     ベンチデータローダ
│       ├── ar_backend.py      #     AR ベースライン
│       ├── zenz_backend.py    #     zenz-v2.5 比較
│       └── fast_gen.py
├── engine/                    # IME エンジンプラグイン
│   ├── src/                   #   fcitx5 InputMethodEngineV2 (C++)
│   │   ├── engine.{cpp,h}
│   │   ├── composing_text.*   #   ローマ字→ひらがな
│   │   ├── preedit.*
│   │   └── server_connector.* #   IPC クライアント
│   ├── win32/                 #   Windows TSF 統合 (DLL 動作確認済)
│   │   ├── ffi_impl.cpp       #     ONNX Runtime C++ FFI
│   │   ├── interactive.cpp    #     対話型コンソールデモ
│   │   ├── test_ffi.cpp
│   │   └── build.bat
│   ├── new-ime-addon.conf     # fcitx5 メタデータ
│   └── new-ime-im.conf
├── server/                    # 推論サーバー (C++)
│   └── src/
│       ├── main.cpp
│       ├── socket_server.*    #   Unix domain socket
│       ├── ctc_decoder.*      #   CTC greedy / beam search
│       └── conversion_engine.*#   変換ロジック (ONNX Runtime)
├── protocol/                  # protobuf IPC スキーマ
│   └── new_ime.proto
├── scripts/                   # データ・学習・評価スクリプト
│   ├── process_wiki.py        #   Wikipedia dump → JSONL (16 workers)
│   ├── process_aozora.py
│   ├── process_livedoor.py
│   ├── process_tatoeba.py
│   ├── postprocess.py         #   品質フィルタ (旧仮名・ト書き除去等)
│   ├── build_vocab.py
│   ├── build_eval_set.py      #   train/dev/test 分離
│   ├── audit_data.py
│   ├── dataset_stats.py
│   ├── generate_chunks.py     #   文節チャンク生成呼び出し
│   ├── mecab_to_tsv.py
│   ├── run_all_evals.py       #   全ベンチ一括実行
│   ├── eval_ar_checkpoint.py
│   ├── eval_gold.py
│   ├── manual_test.py / manual_test_beam.py
│   └── vast_train.sh          #   Vast.ai 学習スクリプト
├── tools/                     # Rust ツール (WSL ビルド)
│   └── chunk-generator/       #   文節分割 + 1-3文節ウィンドウ
├── tests/                     # テスト
│   ├── test_tokenizer.py      #   Python トークナイザ
│   ├── test_model.py          #   CTC-NAT モデル
│   ├── test_metrics.py        #   評価指標
│   └── cpp/
│       ├── test_composing_text.cpp
│       └── test_ctc_decoder.cpp
├── configs/                   # 学習設定 YAML
├── checkpoints/               # 学習チェックポイント (gitignore)
├── datasets/                  # データ (gitignore)
├── docker/                    # Docker 関連
├── docs/                      # ドキュメント
│   ├── vision.md              #   最終構成ビジョン
│   ├── roadmap.md             #   実装ロードマップ (Phase 0〜6)
│   ├── data_pipeline.md       #   データパイプライン詳細
│   ├── dataset_candidates.md  #   学習データ追加候補 (ライセンス整理)
│   ├── phase2_results.md      #   Phase 2 AR 実験結果
│   ├── benchmark_comparison.md#   zenz-v2.5 との比較
│   └── external_review_notes.md
├── CHANGELOG.md
├── CMakeLists.txt             # C++ ビルド
└── pyproject.toml             # Python (uv)
```

## セットアップ

### Python (モデル・データパイプライン・評価)

```bash
uv sync --group dev
uv run pytest
```

### C++ テスト (gcc)

```bash
g++ -std=c++20 -I engine/src -o test_composing tests/cpp/test_composing_text.cpp engine/src/composing_text.cpp && ./test_composing
g++ -std=c++20 -I server/src -o test_ctc tests/cpp/test_ctc_decoder.cpp server/src/ctc_decoder.cpp && ./test_ctc
```

### Windows エンジン (MSVC, 動作確認済)

```cmd
cd engine\win32
build.bat
test_ffi.exe
interactive.exe
```

### Linux fcitx5 プラグイン (未ビルド検証)

```bash
cmake -B build -DENABLE_FCITX5=ON
cmake --build build
```

## データパイプライン

詳細は `docs/data_pipeline.md`。

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

# Livedoor / Tatoeba
uv run python scripts/process_livedoor.py ...
uv run python scripts/process_tatoeba.py ...

# 後処理 (品質フィルタ)
uv run python scripts/postprocess.py \
    --input datasets/wiki_sentences.jsonl datasets/aozora_sentences.jsonl \
    --output datasets/all_clean.jsonl --stats

# 評価セット分離
uv run python scripts/build_eval_set.py \
    --input datasets/all_clean.jsonl \
    --out-dir datasets/splits/

# 語彙構築
uv run python scripts/build_vocab.py \
    --input datasets/all_clean.jsonl \
    --output datasets/vocab.json
```

MeCab feature は **`features[17]` (仮名形出現形)** を使用。
`features[6/7/8/9]` は不正 (詳細は `docs/phase2_results.md`)。

## 評価

```bash
# 全ベンチ一括 (zenz-v2.5, 自前 AR, チャンクモデル対比)
uv run python scripts/run_all_evals.py

# ゴールド (1000件、人手検証)
uv run python scripts/eval_gold.py --backend ar --checkpoint ...

# 手動テスト (100問)
uv run python scripts/manual_test.py
```

## 参考文献

- [Mask-Predict (CMLM)](https://arxiv.org/abs/1904.09324) — Ghazvininejad et al. 2019
- [GLAT](https://arxiv.org/abs/2008.07905) — Qian et al. 2021
- [Mask-CTC](https://arxiv.org/abs/2005.08700) — Higuchi et al. 2021
- [DA-Transformer](https://arxiv.org/abs/2205.07459) — Huang et al. 2022
- [BitNet b1.58 Reloaded](https://arxiv.org/abs/2407.09527) — Nielsen 2024
- [zenz-v1](https://huggingface.co/Miwa-Keita/zenz-v1) — Miwa 2024
- [Hazkey](https://github.com/7ka-Hiira/hazkey) — fcitx5 日本語 IME (MIT)

## ライセンス

MIT
