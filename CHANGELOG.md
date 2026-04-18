# Changelog

## 2026-04-18 (later-3): Phase 3 student train.jsonl 生成 (200M rows)

### 実データ build 完了

- `datasets/phase3/train.jsonl`: 200,000,000 rows, **37.5 GB**
- 混合比 (short-heavy spec、CTC-NAT 生徒向け)
  | pool | % | rows | oversample |
  |---|---:|---:|---:|
  | chunks (surface ≥ 8 chars) | 50% | 100M | 1.3x |
  | super-short (surface < 8 chars) | 10% | 20M | 0.8x |
  | zenz_llmjp | 15% | 30M | 0.27x |
  | wiki + aozora | 10% | 20M | 0.96x |
  | fineweb2_ja | 10% | 20M | 0.17x |
  | hplt3_ja | 5% | 10M | 0.48x |
- `--super-cutoff 8` は **四字熟語が super-short に含まれる** ことを意図 (surface ≤ 7 chars)
- 汚染フィルタ: chunks + 旧 wiki/aozora に対し `eval_v3/test.jsonl + dev.jsonl` 6-gram で実行
  (新 web pools は上流で処理済み)
- weighted-least-served によるストリーム交互書き込み → DataLoader shuffle で混合完成

### Dataset の大規模化対応

- `KanaKanjiDataset` が 37.5GB train.jsonl を全量 RAM 化しようとする問題を回避: `max_samples`
  指定時に **Algorithm R (Knuth reservoir sampling)** で 1 回のストリームで抽出、RAM は
  `max_samples` 行分のみ。既存 eval JSONL 小サイズはそのまま全読み (後方互換)。
- `tests/test_dataset_reservoir.py` 6 件追加、全 suite 128 pass

### 教師モデル向け (予定)

- 200M パラメータ AR 教師用に 400M-row train.jsonl を別途生成予定
- `--total 400000000` + 別 `--ratio-*` で同 script 流用可能

## 2026-04-18 (later-2): データソース拡充 (Step C) scaffolding

### 追加ソース (downloader + processor)

- **zenz-v2.5 llm-jp-corpus-v3 subset** (ODC-BY, 30.2 GB): 既に kana-kanji pair 形式。
  `scripts/download_zenz_subset.py` / `scripts/process_zenz_subset.py`。
  入力カタカナ → ひらがなへ `jaconv.kata2hira` 変換、長さフィルタ、オプション 6-gram 汚染監査。
  CC-BY-SA な `train_wikipedia.jsonl` は明示拒否。
- **HPLT v3 ja (jpn_Jpan)** (CC0-1.0): `.jsonl.zst` の 38 shard (tier 10 = 3.1 GB ~ tier 5 = 37.8 GB/shard)。
  `scripts/download_hplt3_ja.py` (shard map fetch, tier filter, resume 対応) + `scripts/process_hplt.py`。
- **FineWeb-2 jpn_Jpan** (ODC-By, 474 parquet shards × ~4.6 GB): CulturaX は gated のため代替採用。
  `scripts/download_fineweb2_ja.py` / `scripts/process_fineweb2.py`。

### 共通化

- `src/data/mecab_pipeline.py` 新規: `worker_init`, `text_to_pairs`, `attach_context`, sentence
  filters, `reading_from_mecab` (features[17] 固定)。`scripts/process_wiki.py` を
  thin wrapper に refactor、HPLT / FineWeb-2 から同一 worker を再利用。
- `scripts/audit_pools.py` 拡張: source タグ集計 + `SOURCE_LICENSE` テーブル引き、
  `ATTRIBUTION.md` 同梱検査、デフォルトの eval 汚染比較対象に `eval_v3/test.jsonl` 追加。
- `pyproject.toml`: `zstandard` を依存に追加 (HPLT shard 展開)。

### 依存

- MIT 互換のみ (ODC-BY / CC0 / ODC-By の 3 種)。
- 各ソース DL 先に `ATTRIBUTION.md` 自動生成、attribution 義務を監査で検出可能に。
- `datasets/src/<source>/` 構造を plan 通り踏襲。

### テスト

- `tests/test_mecab_pipeline.py` 8 件追加 (sentence split, filters, context attach, worker_init guard)。
- 全 suite 120 pass。

## 2026-04-18 (later): オンライン KD 実装

### 追加

- `src/training/kd.py`: AR 教師 (`SimpleGPT2`) の online 蒸留モジュール
  - `ARTeacher`: eval/no_grad/fp16、バッチ greedy 生成、frozen-vocab エンコード (UNK fallback で
    推論時の vocab 拡張を禁止)
  - `KDConfig`: α 線形 ramp、hard-example threshold、`kd_every` による optimizer-step ゲート
  - `compute_kd_ctc_loss`: hard sample 限定の CTC 蒸留損失 (input<target の例は除外)
- `src/training/train_ctc_nat.py`: `--kd-*` フラグで KD を有効化。KD 関連メタを checkpoint 保存
  し、resume 時に不一致で fail-fast
- `tests/test_kd.py` (17) / `tests/test_train_ctc_nat.py` (KD validation, 2 追加)

### 方針

- **教師 vocab ≠ 生徒 vocab** のため、ID 空間を合わせず**テキスト経由で再エンコード**する
  hard-target 蒸留。ソフト KL は採用せず。
- **Hard-example 基準**: 教師の mean top-1 confidence < threshold のサンプルだけを蒸留対象に。
  容易例は gold のみ。
- **設定漂流防止**: KD teacher path / alpha / threshold / schedule は checkpoint に固着。
  resume 時の差分は `ValueError`。

## 2026-04-18: 長期ビジョン・ベンチ比較・ドキュメント整理

### 長期ビジョン確定

- `docs/vision.md`: 200M CTC-NAT + 1.58-bit QAT + 階層 CVAE の最終到達点を明文化
- 撤退経路 (CTC-NAT→DAT→AR+投機的デコード) を全階層で整備

### ベンチマーク比較

- `docs/benchmark_comparison.md`: 9モデル × 3ベンチの全量比較
  - zenz-v2.5 medium/small/xsmall (greedy)
  - 自前 ar_v3 (vast / local / chunks, greedy + beam10)
- **自前 ar_v3_vast (32M) は eval_v3 で zenz-v2.5-medium (310M) と EM 0.412 同値**
- AJIMEE では xsmall にも負け (0.450 vs 0.588) → 汎化データ多様性不足
- Phase 3 の戦略指針: データ多様性 + 規模拡大 + CTC-NAT 並列生成

### ドキュメント整理

- `PLAN.md` + `ROADMAP.md` → `docs/roadmap.md` に統合 (旧 2 ファイル削除)
- `docs/architecture_decisions.md` → `docs/vision.md` に吸収
- `README.md` 刷新 (現状反映、Windows TSF を追記)
- データ追加候補を `docs/dataset_candidates.md` に整理 (MIT 互換のみ)

---

## 2026-04-17 (continued): データ v3 + 追加コーパス + Windows エンジン

### データパイプライン v3

- **MeCab feature 確定**: features[17] (仮名形出現形) を採用
  - v1 features[7] (書字形基本形): 漢字が reading に混入
  - v2 features[9] (発音形出現形): 長音化 (よほー)、助詞変化 (は→わ) で IME 入力と不一致
  - v3 **features[17]**: 活用後の正しいカタカナ、正解
- **追加コーパス**:
  - Livedoor News: 7,367 記事 → 84,182 クリーンペア (現代ニュース文体)
  - Tatoeba Japanese: 248K 文 → 228,440 クリーンペア (短文・会話調、p50=17文字)
- **Wikipedia v3**: features[17] で全再処理 → 1842万ペア
- **統合データセット v3**: 19.8M train + 2K dev + 10K test
- **チャンクデータ**: tools/chunk-generator (Rust) で 100M 文節チャンク生成
- **HuggingFace にアップロード**: esehe/new-ime-dataset (private)

### AR ベースライン完成 (Phase 2 完了)

- SimpleGPT2 31.9M params (hidden 512, 8 layers, 8 heads, vocab ~6500)
- vast.ai で 2012万全量学習 (5090)、ベスト step 70000
  - Dev CharAcc 91.4%, 手動テスト 80/100
  - eval_v3 EM 0.412, manual_test EM 0.800, AJIMEE EM 0.450
- RTX 3060 ローカル 200万サンプル版: 天井 87-88% (容量飽和)
- チャンクのみモデル: 文レベル崩壊、混合学習が必須

### 重要な知見

- **beam search 内正解率 95%** (greedy top1 は 60%) → KenLM/ユーザ辞書でリランキング可
- teacher-forced 87% vs AR 81% の乖離 → CTC-NAT の優位性を示す材料
- 31.9M 天井は ~91-92%。これ以上は 200M が必要

### Windows IME エンジン (新規)

- `engine/win32/ffi_impl.cpp`: ONNX Runtime C++ FFI 実装
- `engine/win32/new-ime-engine.dll`: DLL ビルド・動作確認済
- `engine/win32/interactive.cpp`: 対話型コンソールデモ
- `engine/win32/test_ffi.cpp`: FFI 単体テスト
- ONNX エクスポート経路構築 (encoder/decoder 別ファイル方針)

### 失敗モードの分類

- **カテゴリA (コピー問題)**: 学習量不足 → step 増加で解消確認
- **カテゴリB (同音異義語)**: データ頻度通り (感じ>漢字) → 文脈モデリング強化が必要
- **カテゴリC (構造崩壊)**: AR の連鎖失敗 (人工知能→人口のう) → CTC-NAT で解消見込み
- **カテゴリD (局所最適)**: 文字単位 AR で「にほん→二本」 → CTC-NAT の並列生成で解消見込み

---

## 2026-04-17: プロジェクト立ち上げ + データパイプライン + 評価基盤

### 設計・計画

- PLAN.md: NAT ベース並列生成日本語 IME の 6 フェーズ開発計画を策定 (2026-04-18 に roadmap.md へ統合)
- ROADMAP.md: CTC-NAT アーキテクチャの実装レベルロードマップ作成 (同上)
- 設計判断の確定:
  - CMLM → **CTC-NAT** (Mask-CTC) に変更。かな漢字変換の単調アラインメントに CTC が最適
  - fairseq (2026-03 アーカイブ済み) → **スクラッチ実装 (PyTorch)**
  - エンコーダ: cl-tohoku/bert-base-japanese-char-v3 から初期化
  - 推論: ONNX Runtime (C++, int8 量子化)
  - IME 統合: クライアント・サーバー方式 (Hazkey/mozc と同パターン)
- zenz-v1 のコードリーディング完了:
  - GPT-2 90M, vocab 6000, GGUF Q8_0, greedy decode
  - ラティス→ニューラル検証のドラフト・検証ループ
  - プロンプト形式、トークナイザ構成を把握
- 外部レビュー議論の知見整理 (docs/external_review_notes.md)

### データパイプライン (初期構築、後に v3 へ進化)

- Wikipedia 処理パイプライン (scripts/process_wiki.py):
  - mwxml + mwparserfromhell でテキスト抽出 (wikiextractor は Python 3.13 非互換で却下)
  - MeCab (unidic-lite) で読み付与
  - 16 ワーカーのマルチプロセスで並列化
  - 結果: 243万記事 → **2683万ペア**
- 青空文庫処理パイプライン (scripts/process_aozora.py):
  - 形態素解析済み CSV (utf8_all.csv.gz) から直接変換
  - 結果: **375万ペア**
- 後処理品質フィルタ (scripts/postprocess.py):
  - 旧仮名遣い除去 (ゐ/ゑ、歴史的促音つた/つて、思ふ系、せう/ませう)
  - 戯曲ト書き、章番号、裸引用タイトル除去
  - 読みの純度検証 (ひらがなのみ)、POS タグ漏れ修正
  - 重複排除、全角空白正規化
- 品質監査 (scripts/audit_data.py):
  - 1% 抽出 x 2回で分布安定を確認
  - Wikipedia: 87.8% clean, 青空文庫: 92.0% clean

### モデルスケルトン (Python)

- encoder.py: BertEncoder (cl-tohoku ラッパー, freeze/unfreeze) + MockEncoder
- decoder.py: NATDecoder (双方向 self-attention, cross-attention, causal mask なし)
- ctc_nat.py: CTCNAT 統合モデル + GLATSampler (線形アニーリング) + MaskCTCRefiner
- リファレンス実装クローン: fairseq, NMLA-NAT, DA-Transformer

### トークナイザ (Python)

- InputTokenizer: ひらがな/カタカナ文字単位、vocab ~180
- OutputTokenizer: 漢字かな混じり文字単位、バイトフォールバック、vocab ~6500+
- 周波数ベース語彙構築スクリプト (scripts/build_vocab.py)

### fcitx5 エンジンプラグイン (C++)

- composing_text: ローマ字→ひらがな変換 (促音、撥音、shi/chi/tsu 対応)
- preedit: preedit 表示管理 (simple/highlighted/multi-segment)
- server_connector: Unix domain socket IPC クライアント
- engine: fcitx5 InputMethodEngineV2 統合 (Direct/Composing/CandidateSelection ステートマシン)
- メタデータ: new-ime-addon.conf, new-ime-im.conf

### 推論サーバー (C++)

- socket_server: Unix domain socket, length-prefixed メッセージ, select() ループ
- ctc_decoder: CTC greedy collapse + prefix beam search (Hannun 2014)
- conversion_engine: 推論バックエンド抽象化 + MockInferenceEngine
- protobuf スキーマ: protocol/new_ime.proto (5種コマンド)

### 評価基盤

- metrics.py: edit distance, CharAcc Top-K, ExactMatch, EvalResult 集計
- run_eval.py: バックエンド抽象化、レイテンシ分布測定
- build_eval_set.py: 記事/作品単位の train/dev/test 分離、層化サンプリング
  - Dev: 2,000 / Test: 10,000 / Train: 19,802,641
- Identity ベースライン: CharAcc=0.247 (下限)
- CTC T_in >= T_out 検証: 違反率 0.38% (中黒由来、無視可能)

### ビルドシステム

- pyproject.toml (uv + ruff)
- CMakeLists.txt (gcc/cmake, -DENABLE_FCITX5, -DENABLE_ONNX オプション)

### テスト

| テスト | 件数 |
|--------|------|
| Python トークナイザ | 19 |
| Python CTC-NAT モデル | 17 |
| Python 評価指標 | 20 |
| C++ composing_text | 12 |
| C++ ctc_decoder | 5 |
| **合計** | **73** |
