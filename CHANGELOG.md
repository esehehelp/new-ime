# Changelog

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
