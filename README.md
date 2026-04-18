# new-ime

日本語かな漢字変換の研究・実験プロトタイプ。CTC-NAT (Connectionist Temporal Classification + Non-Autoregressive Transformer) を主軸に、自己回帰モデル、蒸留、量子化、推論統合を比較検証する。

## 位置づけ

このリポジトリは、使える IME の出荷よりも研究検証を優先する。

- 学習・評価・推論の再現実験を主目的とする
- コードと、モデル/データ成果物のライセンスを分離して管理する
- 学習済みモデルや混合学習データは実験的成果物として扱う

## 概要

既存の自己回帰モデル (zenz-v1 等) に対し、**並列生成アーキテクチャ** でレイテンシ削減と
**1.58-bit 量子化** で極小サイズを狙う。

- **モデル名**: `new-ime-model`
  - `new-ime-model-90M`: 本命 (scratch h=640, L_enc=8, L_dec=8, CTC-NAT base ~97M + CVAE ~7M = ~104M、config: `configs/phase3_90m.yaml`)
  - `new-ime-model-20M`: テスト/速度検証用 (h=384, L=6+6, CVAE 無効、config: `configs/phase3_20m.yaml`)
- **Encoder**: scratch Transformer (事前学習は Step B 任意オプションで cl-tohoku/bert-base-japanese-char-v3 を MLM warm-up)
- **Decoder**: Non-autoregressive Transformer (双方向 self-attention + cross-attention + FiLM 条件付け)
- **出力**: CTC loss による並列デコード + Mask-CTC refinement
- **学習安定化**: GLAT (Glancing Language Model Training) + Knowledge Distillation
- **推論**: ONNX Runtime / bitnet.cpp / 比較用ベンチハーネス
- **IME 統合**: 研究段階では `interactive.cpp` を中心とした CLI / デモ優先

設計詳細は `docs/vision.md`、実装計画は `docs/roadmap.md` を参照。

## アーキテクチャ

```
[左文脈 + ひらがな入力] + CVAE z (writer/domain/session)
        │
   ┌────▼─────────┐
   │   Encoder     │  scratch Transformer (h=640, L=8)
   │   (FiLM 条件) │  本命は事前学習初期化なし
   └────┬─────────┘
        │
   ┌────▼─────────┐
   │   Decoder     │  NAT (並列生成, h=640, L=8)
   │   (NAT)       │  self-attn + cross-attn + FiLM
   └────┬─────────┘
        │
   ┌────▼─────────┐
   │   CTC Head    │  CTC collapse / beam search + KenLM
   │   + Mask-CTC  │  低信頼位置のみ refinement
   └────┬─────────┘
        │
   漢字かな混じり出力 (top-K 候補)
```

## 現状 (2026-04-18)

| Phase | 状態 |
|-------|------|
| Phase 0 (設計) | 完了 |
| Phase 1 (データパイプライン) | 完了 — ~21M ペア + 100M チャンク、加えて HPLT v3 / FineWeb-2 / zenz-llmjp サブセット追加、`datasets/phase3/train.jsonl` 200M rows 生成済 |
| Phase 2 (AR ベースライン) | 完了 — 31.9M, manual 80/100, eval_v3 EM 0.412 |
| Phase 3 (CTC-NAT + CVAE + 1.58-bit) | 進行中 — 受け入れテスト / `SharedCharTokenizer` / `CTCNAT` / `CVAE` / `BitLinear` / `curriculum_sampler` / オンライン KD 実装済。20M/90M の学習と速度評価を進行中 |
| Phase 5 (Windows TSF DLL, AR 版) | 動作確認済 (`phase3_plan.md` で v1.0 範囲外に再定義、CLI `interactive.cpp` が出口) |
| Phase 5 (fcitx5 プラグイン) | ソース実装済、未ビルド |

Phase 2 の詳細は `docs/phase2_results.md`、Phase 3 計画は `docs/phase3_plan.md`、
ベンチ比較は `docs/benchmark_comparison.md`。

## プロジェクト構成

```
new-ime/
├── src/                       # Python (モデル・学習・データ・評価)
│   ├── model/
│   │   ├── encoder.py         #   BERT / scratch エンコーダ
│   │   ├── decoder.py         #   NAT デコーダ (FiLM 条件付け)
│   │   ├── ctc_nat.py         #   統合モデル + GLAT + Mask-CTC
│   │   ├── cvae.py            #   writer/domain/session 潜在変数
│   │   └── bit_linear.py      #   1.58-bit BitLinear (median scaling + STE)
│   ├── data/
│   │   ├── tokenizer.py       #   SharedCharTokenizer + 旧 Input/Output 互換
│   │   ├── dataset.py         #   reservoir sampling 対応 JSONL dataset
│   │   ├── curriculum_sampler.py  # プール混合 S0-S5 サンプラー
│   │   └── mecab_pipeline.py  #   features[17] 共通ワーカー
│   ├── training/
│   │   ├── train_ar.py        #   Phase 2 AR
│   │   ├── train_ctc_nat.py   #   Phase 3 本線 (KD/GLAT/Mask-CTC/CVAE/QAT)
│   │   └── kd.py              #   オンライン KD (hard-example 中心)
│   ├── inference/
│   └── eval/
│       ├── metrics.py         #   edit distance, CharAcc, EM
│       ├── run_eval.py        #   バックエンド抽象 + レイテンシ測定
│       ├── bench_loaders.py
│       ├── ar_backend.py
│       ├── zenz_backend.py
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
│   ├── process_{wiki,aozora,livedoor,tatoeba}.py    # Phase 1 ソース処理
│   ├── process_{hplt,fineweb2,zenz_subset}.py       # Phase 3 Step C 追加ソース
│   ├── download_{hplt3_ja,fineweb2_ja,zenz_subset}.py  # HF ダウンローダ
│   ├── postprocess.py / audit_data.py / audit_pools.py / audit_tokenizer.py
│   ├── build_vocab.py / build_eval_set.py / dataset_stats.py
│   ├── generate_chunks.py / mecab_to_tsv.py
│   ├── build_phase3_train.py  # プール混合 + reservoir で train.jsonl 生成
│   ├── run_all_evals.py / eval_ar_checkpoint.py / eval_gold.py
│   ├── manual/                #   manual_test*.py (旧パス wrapper 互換あり)
│   ├── bench/                 #   bench_*_speed.py (旧パス wrapper 互換あり)
│   ├── gold/                  #   gen_gold_*.py (旧パス wrapper 互換あり)
│   ├── print_comparison.py
│   └── vast_train.sh          #   Vast.ai 学習スクリプト
├── tools/                     # Rust ツール (WSL ビルド、rustc 1.85)
│   ├── chunk-generator/       #   文節分割 + 1-3文節ウィンドウ
│   ├── postprocess/           #   品質フィルタ (Rust 版)
│   ├── build-vocab/           #   語彙構築
│   ├── build-train-mix/       #   phase3 train.jsonl 混合 (Python scripts/build_phase3_train.py の Rust port)
│   ├── process-zenz/          #   zenz サブセット処理 (Rust port)
│   ├── audit-pools/           #   プール別監査 (Rust port)
│   ├── audit-tokenizer/       #   tokenizer 検証
│   ├── datacore/              #   共通データ型・I/O
│   └── mecab-test/            #   mecab-rs 動作確認
├── tests/                     # テスト (Python 128 pass + C++ 17)
│   ├── test_tokenizer.py / test_model.py / test_metrics.py
│   ├── test_mecab_pipeline.py / test_curriculum_sampler.py
│   ├── test_bit_linear.py / test_kd.py / test_train_ctc_nat.py
│   ├── test_dataset_reservoir.py / test_audit_pools.py
│   └── cpp/
│       ├── test_composing_text.cpp
│       └── test_ctc_decoder.cpp
├── configs/                   # 学習設定 YAML
├── checkpoints/               # 学習チェックポイント (gitignore)
├── datasets/                  # データ (gitignore)
├── docker/                    # Docker 関連
├── docs/                      # ドキュメント
│   ├── vision.md              #   最終構成ビジョン
│   ├── roadmap.md             #   実装ロードマップ (Phase 0〜5)
│   ├── phase3_plan.md         #   Phase 3 (CTC-NAT + CVAE + 1.58-bit) 詳細計画
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

このリポジトリでは、コードとモデル/データ成果物でライセンスが異なる。

- コード (`src/`, `server/`, `engine/`, `tools/`, `scripts/` など): [MIT](LICENSE)
- モデル重み・学習チェックポイント・混合学習 JSONL・蒸留成果物: [CC BY-SA 4.0](MODEL_LICENSE)
- データソースごとの注意と帰属: [DATA_LICENSES.md](DATA_LICENSES.md), [ATTRIBUTION.md](ATTRIBUTION.md)

重要:

- `checkpoints/`, `datasets/`, `results/` 以下の成果物は、内容に応じて MIT ではなく追加条件が付く
- 特に Wikipedia 由来データや、それを含む派生成果物は ShareAlike 条件の影響を受けうる
- 配布や再学習に使う前に、必ず `MODEL_LICENSE`, `DATA_LICENSES.md`, `ATTRIBUTION.md` を確認すること
