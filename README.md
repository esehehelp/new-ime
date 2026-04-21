# new-ime

日本語かな漢字変換のプロトタイプ。CTC-NAT (Connectionist Temporal Classification + Non-Autoregressive Transformer) を主軸に、自己回帰モデル、蒸留、量子化、推論統合を比較検証する。

## 位置づけ

このリポジトリは、使える IME の完成よりも検証を優先する。

- 学習・評価・推論の再現実験を主目的とする
- コードと、モデル/データ成果物のライセンスを分離して管理する
- 学習済みモデルや混合学習データは実験的成果物として扱う

## 概要

既存の自己回帰モデル (zenz-v2.5 等) に対し、**並列生成アーキテクチャ** でレイテンシ削減と
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

## アーキテクチャ

```
[左文脈 + ひらがな入力] + CVAE z (writer/domain/session)
        │
   ┌────▼─────────┐
   │   Encoder    │  scratch Transformer (h=640, L=8)
   │   (FiLM 条件)│  本命は事前学習初期化なし
   └────┬─────────┘
        │
   ┌────▼─────────┐
   │   Decoder    │  NAT (並列生成, h=640, L=8)
   │   (NAT)      │  self-attn + cross-attn + FiLM
   └────┬─────────┘
        │
   ┌────▼─────────┐
   │   CTC Head   │  CTC collapse / beam search + KenLM
   │   + Mask-CTC │  低信頼位置のみ refinement
   └────┬─────────┘
        │
   漢字かな混じり出力 (top-K 候補)
```

## プロジェクト構成

```
new-ime/
├── models/src/                # Python メイン実装 (モデル・学習・データ・評価)
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
│   │   └── kd.py              #   オンライン KD (AR/CTC teacher 両対応)
│   └── eval/
│       ├── metrics.py         #   edit distance, CharAcc, EM
│       ├── run_eval.py        #   バックエンド抽象 + レイテンシ測定
│       ├── bench_loaders.py
│       ├── ar_backend.py
│       ├── zenz_backend.py
│       └── fast_gen.py
├── engine/                    # IME エンジン一式 (C++)
│   ├── src/                   #   fcitx5 InputMethodEngineV2 プラグイン
│   │   ├── engine.{cpp,h}
│   │   ├── composing_text.*   #     ローマ字→ひらがな
│   │   ├── preedit.*
│   │   └── server_connector.* #     IPC クライアント
│   ├── win32/                 #   Windows TSF 統合 (DLL 動作確認済)
│   │   ├── ffi_impl.cpp       #     ONNX Runtime C++ FFI
│   │   ├── interactive.cpp    #     AR 版対話デモ
│   │   ├── interactive_ctc.cpp #    CTC-NAT 版対話デモ (dry-run 出口)
│   │   ├── build.bat / build_ctc.bat
│   │   └── test_ffi.cpp
│   ├── server/                #   推論サーバー + CTC decoder + KenLM
│   │   └── src/
│   │       ├── main.cpp
│   │       ├── socket_server.*    #  Unix domain socket IPC
│   │       ├── ctc_decoder.*      #  CTC greedy / prefix beam
│   │       ├── lm_scorer_kenlm.*  #  KenLM shallow fusion
│   │       └── conversion_engine.*
│   ├── protocol/              #   protobuf IPC (engine ↔ server、現状 mock)
│   │   └── new_ime.proto
│   ├── new-ime-addon.conf     #   fcitx5 addon メタデータ
│   └── new-ime-im.conf        #   fcitx5 IM メタデータ
├── datasets/                  # データ (gitignore)
│   ├── raw/                   #   一次ソース (XML/CSV)
│   ├── corpus/
│   │   ├── sentence/          #     文レベル (yomi 付)
│   │   ├── bunsetsu/          #     句レベル (Ginza)
│   │   ├── synth/             #     合成 (numeric / numeric_ext)
│   │   └── legacy/            #     旧 sentence-level コンポーネント pool
│   ├── mixes/
│   │   ├── scratch-200m.jsonl #    基礎 mix (chunks + zenz + wiki + aozora + ...)
│   │   ├── student-20m.jsonl  #    CTC student 用 mix
│   │   └── teacher-20m.jsonl  #    teacher 用 mix
│   ├── eval/
│   │   ├── general/           #    dev/test/train (旧 eval_v3)
│   │   ├── probe/probe.json   #    467 項目 phrase-level
│   │   ├── cvae-probe/probe.tsv #  188 項目 domain-labeled
│   │   └── legacy/            #    旧 probe_v1/v2 他
│   ├── tokenizers/            #   char-5k.json (shared char tokenizer)
│   ├── audits/                #   プール監査ログ
│   └── tools/                 #   データ処理スクリプト
│       ├── corpus/            #     Python: bunsetsu_split, clean, synth_*
│       ├── probe/             #     Python: probe 生成・評価ランナー
│       ├── mix/               #     Rust: build-train-mix (sentence + bunsetsu + synth)
│       ├── chunk-generator/   #     Rust: 文節チャンク生成
│       ├── audit/             #     Rust: プール別監査 (旧 audit-pools)
│       ├── process-zenz/      #     Rust: zenz サブセット処理
│       └── datacore/          #     Rust: 共通データ型・I/O
├── models/
│   ├── src/                   # Python メイン実装
│   │   ├── model/             #   encoder / decoder / ctc_nat / cvae / bit_linear
│   │   ├── data/              #   tokenizer / dataset / curriculum_sampler / mecab_pipeline
│   │   ├── training/          #   train_ar / train_ctc_nat / train_teacher / kd
│   │   └── eval/              #   metrics / run_eval / bench_loaders / *_backend
│   ├── tools/                 # モデル系スクリプト
│   │   ├── eval/              #   Python: run_all_evals / eval_gold / print_comparison
│   │   ├── bench/             #   Python: speed bench (ar/ctc/zenz/fusion)
│   │   ├── manual/            #   Python: 手動テスト 100問
│   │   ├── export/            #   Python: ONNX export
│   │   ├── dict/              #   Python: mozc dict import
│   │   ├── build-vocab/       #   Rust: 語彙構築
│   │   ├── audit-tokenizer/   #   Rust: tokenizer 検証
│   │   └── postprocess/       #   Rust: 品質フィルタ
│   ├── checkpoints/           # 学習チェックポイント (gitignore)
│   │   ├── ar-31m-scratch/    #   Phase 2 AR 31.9M (KD teacher)
│   │   ├── ctc-nat-30m-scratch/    # 30M scratch
│   │   ├── ctc-nat-30m-student/    # 30M (student-20m mix)
│   │   ├── ctc-nat-90m-scratch/    # 90M scratch
│   │   ├── teacher-150m-teacher/   # teacher seq2seq 150M
│   │   └── archive/           #   旧 smoke / 実験用 (zstd 圧縮済)
│   ├── onnx/                  # 配布 ONNX + sidecar
│   ├── dicts/                 # 辞書層 (fixed_dict_*, user_dict)
│   ├── kenlm/                 # KenLM 言語モデル (.bin)
│   └── tests/                 # Python + C++ テスト
├── engine/                    # IME エンジン一式 (C++)
│   ├── src/                   #   fcitx5 InputMethodEngineV2 プラグイン
│   ├── win32/                 #   Windows TSF 統合 (ffi_impl / interactive*)
│   ├── server/                #   推論サーバー + CTC decoder + KenLM
│   ├── protocol/              #   protobuf IPC
│   ├── new-ime-addon.conf
│   └── new-ime-im.conf
├── tools/                     # プロジェクト全体の雑多ツール
│   ├── misc/                  #   shell: deploy/mirror/compress_archive
│   ├── old/                   #   legacy one-shot scripts
│   └── onnxruntime-win-x64-1.22.0/  # Windows 向け ONNX Runtime
├── configs/                   # 学習設定 YAML + env
├── docs/                      # ドキュメント
│   ├── vision.md              #   最終構成ビジョン
│   ├── benchmark_comparison.md      # 現状ベンチ集約 (living doc)
│   ├── phase3_v2_dryrun_runbook.md  # dry-run 実行手順
│   ├── probe_v2_4way_results.md     # probe_v2 測定詳細
│   ├── cvae_probe_baseline.md       # CVAE probe ベースライン
│   └── old/                   #   過去 plan / superseded 資料
├── Cargo.toml                 # Rust workspace ルート
└── CHANGELOG.md
│       ├── roadmap.md               # 旧 Phase 0-5 ロードマップ
│       ├── phase3_plan.md           # 旧 Phase 3 計画
│       ├── phase2_results.md        # Phase 2 AR 実験結果
│       ├── data_pipeline.md         # 旧データパイプライン詳細
│       ├── dataset_candidates.md    # 学習データ候補整理
│       ├── corpus_candidates_v2.md  # corpus_v2 候補
│       ├── deployment.md            # vast.ai deploy (local 移行で superseded)
│       ├── sweep_probe_v1_results.md # probe_v1 sweep (v2 で置換)
│       ├── external_review_notes.md # 外部 review ノート
│       └── 30m-50k-sample.txt       # 前回 30M training log
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
g++ -std=c++20 -I engine/src -o test_composing models/tests/cpp/test_composing_text.cpp engine/src/composing_text.cpp && ./test_composing
g++ -std=c++20 -I engine/server/src -o test_ctc models/tests/cpp/test_ctc_decoder.cpp engine/server/src/ctc_decoder.cpp && ./test_ctc
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

本リポジトリ (GitHub) は **コード + docs のみ**。モデル重み・学習データは
`.gitignore` 済、将来 HuggingFace / Release 経由で配布する想定。

- **コード** (`models/src/`, `engine/`, `tools/` 等): [MIT](LICENSE)
- **モデル重み・学習 JSONL・蒸留成果物** (HF / Release 配布物):
  [CC BY-SA 4.0](MODEL_LICENSE)
- **上流データソースごとの注意と帰属**:
  [DATA_LICENSES.md](DATA_LICENSES.md), [ATTRIBUTION.md](ATTRIBUTION.md)
- **3rd-party ライブラリ依存** (KenLM LGPL 等):
  [LICENSE_NOTICES.md](LICENSE_NOTICES.md)

重要:

- Wikipedia 由来データや、それを含む派生成果物は ShareAlike 条件の影響を受ける
- KenLM は LGPL 2.1 — 本リポジトリには source 不在だが、binary 再配布時は
  LGPL 義務発生。詳細 `LICENSE_NOTICES.md`
- 配布や再学習に使う前に、 `MODEL_LICENSE`, `DATA_LICENSES.md`,
  `ATTRIBUTION.md`, `LICENSE_NOTICES.md` を確認すること
