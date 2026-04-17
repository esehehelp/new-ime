# Changelog

## 2026-04-17 (continued): データ v2 + 追加コーパス + AR ベースライン学習

### データパイプライン v2

- **MeCab feature 修正**: features[9] (発音形: よほー, わ, お) → features[6] (書字形: よほう, は, を)
  - 発音形は音声学的変換を含み IME 入力と不一致だった
  - 書字形は正書法に準拠し IME の読みと一致する
- **追加コーパス**:
  - Livedoor News: 7,367記事 → 84,182 クリーンペア (現代ニュース文体)
  - Tatoeba Japanese: 248K文 → 228,440 クリーンペア (短文・会話調、p50=17文字)
- **Wikipedia v2**: features[6] で全再処理 → 1842万ペア
- **統合データセット v2**: 2012万 train + 2K dev + 10K test
- **HuggingFace にアップロード**: esehe/new-ime-dataset (private)

### AR ベースライン学習 (参考モデル、旧データ)

- SimpleGPT2 28.6M params (512 hidden, 8 layers)
- RTX 3060 12GB でローカル学習
- step 18000 時点の手動テスト結果 (20問中 ○15, △2, ×3):
  - 東京都渋谷区、学校に行く、新聞を読む、経済学を専攻する etc. → 正解
  - 交渉が難航している、散歩に行きましょう、明日は雨が降るでしょう → step 8000 で失敗 → step 18000 で正解に改善
  - 感じ変換の制度を評価する (漢字→感じ) → 同音異義語、文脈不足
  - にほんご → ニホンゴ (文字単位 AR の局所最適問題)
- 発見: teacher-forced eval (87%) vs autoregressive eval (81%) の乖離 → CTC-NAT の優位性を示す材料

### 分析: 失敗モードの分類

- **カテゴリA (コピー問題)**: 学習量不足 → step 増加で解消確認
- **カテゴリB (同音異義語)**: データ頻度通り (感じ>漢字) → 文脈モデリング強化が必要
- **カテゴリC (構造崩壊)**: AR の連鎖失敗 (人工知能→人口のう) → CTC-NAT で解消見込み
- **カテゴリD (局所最適)**: 文字単位 AR で「にほん→二本」 → CTC-NAT の並列生成で解消見込み

---

## 2026-04-17: プロジェクト立ち上げ + データパイプライン + 評価基盤

### 設計・計画

- PLAN.md: NAT ベース並列生成日本語 IME の6フェーズ開発計画を策定
- ROADMAP.md: CTC-NAT アーキテクチャの実装レベルロードマップ作成
- 設計判断の確定:
  - CMLM → **CTC-NAT** (Mask-CTC) に変更。かな漢字変換の単調アラインメントに CTC が最適
  - fairseq (2026-03 アーカイブ済み) → **スクラッチ実装 (PyTorch)**
  - エンコーダ: cl-tohoku/bert-base-japanese-char-v3 から初期化
  - 推論: ONNX Runtime (C++, int8 量子化)
  - fcitx5 統合: クライアント・サーバー方式 (Hazkey/mozc と同パターン)
- zenz-v1 のコードリーディング完了:
  - GPT-2 90M, vocab 6000, GGUF Q8_0, greedy decode
  - ラティス→ニューラル検証のドラフト・検証ループ
  - プロンプト形式、トークナイザ構成を把握
- 外部レビュー議論の知見整理 (docs/external_review_notes.md)

### データパイプライン

- Wikipedia 処理パイプライン (scripts/process_wiki.py):
  - mwxml + mwparserfromhell でテキスト抽出 (wikiextractor は Python 3.13 非互換で却下)
  - MeCab (unidic-lite) で読み付与。**features[9] (発音形出現形)** を使用 (features[7] は書字形で漢字が混入する致命的バグを発見・修正)
  - 16ワーカーのマルチプロセスで並列化
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

**最終データセット:**

| ソース | 生ペア | フィルタ後 | 保持率 |
|--------|--------|-----------|--------|
| Wikipedia | 26,838,817 | 18,426,765 | 68.7% |
| 青空文庫 | 3,758,957 | 2,420,414 | 64.4% |
| **合計** | **30,597,774** | **20,847,179** | |

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
