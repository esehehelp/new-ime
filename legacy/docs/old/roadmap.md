---
status: current
last_updated: 2026-04-18
supersedes: PLAN.md, ROADMAP.md
---

# new-ime 実装ロードマップ

Phase 0〜6 の開発計画。最終ビジョンは `vision.md` 参照。本ドキュメントは実装タスクと
Go/No-Go 基準に焦点を絞る。

---

## 現状 (2026-04-18)

| Phase | 状態 |
|-------|------|
| Phase 0 (設計) | **完了** |
| Phase 1 (データパイプライン) | **完了** (Wiki+青空+Livedoor+Tatoeba ~21M ペア + 100M チャンク、+ Step C で HPLT v3 / FineWeb-2 / zenz-llmjp 追加) |
| Phase 2 (AR ベースライン) | **完了** (31.9M, manual 80%, eval_v3 EM 0.412) |
| Phase 3 (CTC-NAT + CVAE + 1.58-bit) | **進行中** — Step A 受け入れテスト / Step B コード基盤 / Step C データ拡充・混合 / KD 実装 完了、Step D 本線学習 未着手 (詳細 `phase3_plan.md`) |
| Phase 4 (評価・改善) | 評価基盤構築済、改善サイクル未着手 |
| Phase 5 (fcitx5/Windows 統合) | `phase3_plan.md` で v1.0 範囲外に再定義。Windows TSF/DLL は AR 版で動作確認済だが CTC-NAT 統合は次計画 |
| Phase 6 (旧: 量子化・CVAE) | **Phase 3 に吸収** (CTC-NAT と同時検証、`phase3_plan.md` 参照) |

---

## アーキテクチャ確定事項

PLAN 初版からの主要変更点:

| 項目 | 初版 | 確定 |
|------|------|------|
| 主損失 | Cross-entropy on parallel output | **CTC loss** |
| 長さ予測 | 分類ヘッド | **不要** (CTC が内部処理) |
| Refinement | CMLM mask-predict 1回 | **Mask-CTC** (CTC + mask-predict 複合) |
| ベース実装 | fairseq CMLM | **スクラッチ (PyTorch)** (fairseq 2026-03 アーカイブ済) |
| エンコーダ初期化 | cl-tohoku/bert-japanese | **scratch (h=640, L_enc=8)** を本命。BERT 初期化は MLM warm-up の任意オプションに格下げ |
| 候補生成 | 未定 | **CTC beam search + KenLM shallow fusion** |
| 出力語彙 | SentencePiece BPE 32k | **shared char vocab ~6500** (入出力統合、`SharedCharTokenizer`) |
| MeCab feature | features[7/9] | **features[17]** (仮名形出現形) |
| パラメータ規模 | 100M | **~104M (CTC-NAT base 97M + CVAE 7M)**。縮小は h=384 L=6+6 ≈ 20-30M を速度/テスト用 |

### 確定アーキテクチャ

```
入力: [CLS] 左文脈(最大40字) [SEP] かな入力     ← 最大 128 tokens
        │
   ┌────▼─────────┐
   │   Encoder     │  scratch, h=640, L=8
   │   (FiLM 条件) │  CVAE z を FiLM で混ぜる
   └────┬─────────┘
        │
   ┌────▼─────────┐
   │   Decoder     │  scratch, h=640, L=8, self+cross+ffn
   │  (NAT)        │  cross-attention でエンコーダ参照
   │  出力長=入力長 │  CTC blank で圧縮処理
   └────┬─────────┘
        │
   ┌────▼─────────┐
   │   CTC Head    │  blank + 重複統合 + prefix beam + KenLM
   │   + Mask-CTC  │  低信頼位置を再予測
   └────┬─────────┘
        │
   漢字かな混じり出力 (top-k 候補)
```

### パラメータ見積もり (phase3_90m プリセット)

| コンポーネント | 層数 | hidden | パラメータ数 |
|---------------|------|--------|-------------|
| Shared char embedding (tied) | - | 6500×640 | ~4.2M |
| Encoder (scratch) | 8 | 640 | ~39M |
| Decoder (scratch, self+cross+ffn) | 8 | 640 | ~52M |
| CTC 出力ヘッド | - | 640→6500 | ~4.2M |
| CVAE (posterior biGRU + prior + FiLM) | - | - | ~7M |
| 合計 | | | **~104M** |

テスト用縮小プリセット (`phase3_20m`) は h=384, L=6+6 でデバッグ・速度ベンチ専用。
scratch encoder が前提だが、MLM プリトレ (BERT-japanese-char-v3) を Step B の任意
オプションとして残す。

### NAT を選ぶ理由 (2026-04-17 確定)

zenz-v1 と同じ GPT-2 アーキテクチャでデータ品質のみで勝負する路線は採らない。

- かな漢字変換は**単調アラインメント** → NAT が最も得意
- GLAT + KD で AR との精度差は BLEU ~0.5 まで縮まる
- CTC は**長さ予測を不要にする** (blank トークンで長さ差を吸収)
- 主な精度リスクは同音異義語の曖昧性解消 → Mask-CTC + KenLM で緩和

**撤退経路**: Phase 3 Go/No-Go で AR+投機的デコードに切り替え可能。
この場合でもデータパイプラインと評価基盤は再利用できる。

### 参考実装

| 実装 | URL | 備考 |
|------|-----|------|
| fairseq CMLM (参考のみ) | `facebookresearch/fairseq` | アーカイブ済み、コード参照のみ |
| DA-Transformer | `thu-coai/DA-Transformer` | ICML 2022、撤退先 (DAT) の参考 |
| NMLA-NAT | `ictnlp/nmla-nat` | CTC-NAT 実装、最も関連が近い |

---

## Phase 0: 設計確定 ✅

**完了 (2026-04-17)**

- zenz-v1 コードリーディング完了
  - GPT-2 90M, vocab 6000, GGUF Q8_0, greedy decode
  - ラティス→ニューラル検証のドラフト・検証ループ
- NAT 系論文速習 (Gu 2018, Ghazvininejad 2019, Qian 2020, Huang 2022)
- アーキテクチャ決定 (CTC-NAT + Mask-CTC)
- 評価指標確定 (CharAcc, EM, 文節境界, p50/p95/p99)

---

## Phase 1: データパイプライン ✅

**完了 (2026-04-17〜18)**

### 成果

| ソース | 生ペア | フィルタ後 | 保持率 |
|--------|--------|-----------|--------|
| Wikipedia JA | 26.8M | **18.4M** | 68.7% |
| 青空文庫 | 3.76M | **2.42M** | 64.4% |
| Livedoor News | 7.4K 記事 | 84K | - |
| Tatoeba JP | 248K | 228K | - |
| **合計** | | **~20.8M** | |

+ `tools/chunk-generator` (Rust) で文節チャンク **100M 件** を生成。

### MeCab feature

**確定: `features[17]` (仮名形出現形)** を使用。

過去の失敗:
- v1: features[7] (書字形基本形) → 漢字が reading に混入
- v2: features[9] (発音形出現形) → 長音化・助詞変化で IME 入力と不一致
- v3: **features[17]** → 活用形も正しい、正解

詳細は `phase2_results.md`。データフロー詳細は `data_pipeline.md`。

### 評価セット

- Dev: 2,000 / Test: 10,000 / Train: 19.8M
- 記事/作品単位で train/dev/test 分離 (テスト汚染防止)
- Identity ベースライン: CharAcc=0.247 (下限)
- CTC `T_in >= T_out` 検証: 違反率 0.38% (中黒由来、無視可能)

---

## Phase 2: AR ベースライン ✅

**完了 (2026-04-18)** — KD 教師として Phase 3 で使用。

### モデル

- SimpleGPT2: 31.9M params (hidden 512, 8 layers, 8 heads, vocab ~6500)
- 文字単位トークナイザ (入力 ~180, 出力 ~6500)
- Causal LM, fp16 混合精度, AdamW + cosine decay

### 結果 (ベストモデル: step 70000)

| 指標 | 結果 |
|------|------|
| Dev CharAcc (teacher-forced) | 91.4% |
| 手動テスト 100問 | 80/100 |
| eval_v3 EM | 0.412 |
| manual_test EM | 0.800 |
| AJIMEE EM | 0.450 |

詳細ベンチ比較は `benchmark_comparison.md`。

### 重要な知見

- teacher-forced eval (87%) vs autoregressive eval (81%) の乖離 → **CTC-NAT の優位性を示す材料**
- Beam search (beam=10): top1 精度は greedy と同等だが、**in-beam 正解率 95%** → KenLM/ユーザ辞書でリランキングすれば 95% 到達可能
- 31.9M モデル容量天井は ~91-92%。これ以上は 200M が必要

### 失敗モード分類

- **カテゴリA (コピー問題)**: 容量限界 → 200M で解消見込み
- **カテゴリB (同音異義語)**: データ頻度通り → 文脈モデリング強化必要
- **カテゴリC (AR 構造崩壊)**: 先頭誤りが後続に波及 → CTC-NAT で解消
- **カテゴリD (文字 AR 局所最適)**: 「にほん→二本」等 → CTC-NAT で解消

---

## Phase 3: CTC-NAT + CVAE + 1.58-bit の同時検証 (6〜8週間)

**現在進行中 (Step A-C + KD 実装済)**。詳細実装計画は `phase3_plan.md` に集約。
本節はハイレベル概要のみ残す (個別 Step や受け入れ基準は phase3_plan.md 正)。

### 3.1 モデル実装

#### Encoder (encoder.py, ~100行)

- cl-tohoku/bert-base-japanese-char-v3 をラップ (`BertModel.from_pretrained`)
- 入力: `[左文脈][SEP][ひらがな]` → token_ids
- 出力: `encoder_hidden` (batch, seq_len, 768)
- 最初はエンコーダを凍結、3.4b で段階的に解凍

#### Decoder (decoder.py, ~200行)

- 6層 Transformer デコーダ
- Self-attention: 双方向 (causal mask なし)
- Cross-attention: エンコーダ出力を参照
- Query tokens の初期化: エンコーダ出力そのまま (upsampling なし)
- 出力: (batch, seq_len, 6000) — 各位置の文字確率

#### CTC ヘッド (ctc_nat.py, ~150行)

- `Linear(768, vocab_size + 1)` (+1 for CTC blank)
- `torch.nn.CTCLoss()`
- 入力系列長 >= 出力系列長 制約 → Phase 1 で違反率 0.38% 確認済

#### Mask-CTC Refinement (mask_predict.py, ~150行)

- CTC 出力で信頼度が低い位置を `[MASK]` に置換
- デコーダを再度実行して `[MASK]` 位置を再予測
- 1回のみ (追加コスト: デコーダ 1パス分)

### 3.2 GLAT 学習 (glat.py, ~100行)

```python
# 1. Forward pass → 予測を得る
# 2. 予測と正解の Hamming distance を計算
# 3. distance / length * lambda (lambda=0.5) の割合で
#    正解トークンをデコーダ入力に「見せる」
# 4. 残りの位置について CTC loss を計算
```

### 3.3 Knowledge Distillation

- Phase 2 AR モデル (step 70000) の greedy decode を正解として使う
- 実データの正解も混ぜる (比率は要実験: KD 100% vs KD 50% + real 50%)

### 3.4 学習スケジュール

| フェーズ | エンコーダ | デコーダ | 損失 | ステップ |
|---------|-----------|---------|------|---------|
| 3.4a: デコーダ学習 | 凍結 | 学習 | CTC + GLAT (KDデータ) | 100k |
| 3.4b: 全体ファインチューン | 低lr (1e-5) | 通常lr (3e-4) | CTC + GLAT (KD + real混合) | 200k |
| 3.4c: Mask-CTC 追加 | 低lr | 低lr | CTC + Mask-predict | 50k |

推定学習時間: H100 で 20〜40 時間 / 予算: $100〜300

### 3.5 KenLM shallow fusion (並行実装)

Phase 3 と同時に導入:

```
score(y|x) = log P_CTC(y|x) + λ · log P_LM(y)
```

- Wikipedia コーパスで 4-gram KenLM 学習 (数時間)
- C++ サーバー側: KenLM C++ API を直接統合
- CTC prefix beam search に LM スコア加算 (~100行の修正)
- `alpha` (LM重み) と `beta` (word insertion penalty) を dev セットで調整

### 3.6 推論パイプライン

```
ひらがな入力 + 左文脈
    │
    ▼ Encoder (1パス, ~5ms CPU)
    ▼ Decoder (1パス並列, ~5ms CPU)
    ▼ CTC greedy collapse → 第1候補 (~0.1ms)
       or
    ▼ CTC beam search (beam=5, KenLM 融合) → top-5候補 (~1ms)
    ▼ (オプション) Mask-CTC refinement 1回 (~5ms)
    │
    最終候補リスト
```

目標レイテンシ: refinement 込みで **< 20ms on modern CPU**

### 3.7 Go/No-Go

| 条件 | 基準 |
|------|------|
| 精度 | Phase 2 AR (eval_v3 EM 0.412) と同等以上 |
| レイテンシ | AR の 1/3 以下 |
| 両立 | 両方満たすこと |

**失敗時の撤退**:
- 精度差 1-2pt: 速度優先で CTC-NAT 採用
- 精度差 2-3pt: Imputer 的反復 CTC
- 精度差 3pt 以上: AR + 投機的デコードに撤退
- ハイブリッド案 (高信頼は CTC、低信頼時のみ AR で再ランキング): 最終手段

---

## Phase 4: 評価と改善 (4〜6週間)

### 4.1 評価項目 (基盤構築済)

| 指標 | 測定方法 | 実装 |
|------|---------|------|
| 文字精度 (CharAcc) | 正解文字数 / 全文字数 | `src/eval/metrics.py` |
| 完全一致率 (EM) | 文単位で出力が正解と完全一致 | `src/eval/metrics.py` |
| 文節境界精度 | MeCab で文節分割し境界一致率 | 未実装 |
| 推論レイテンシ | CPU (i7/Ryzen) での p50/p95/p99 | `src/eval/run_eval.py` |
| モデルサイズ | fp16, int8, ONNX それぞれ | - |

### 4.2 誤り分析

- 同音異義語の誤変換パターン分類
- 固有名詞の扱い
- 送りがなの誤り
- 文脈依存の変換精度

### 4.3 改善サイクル

誤り分析 → データ拡張 or 損失調整 → 再学習 → 再評価 (最大3サイクル)

---

## Phase 5: 推論エンジンと IME 統合 (4〜6週間)

### 5.0 クライアント・サーバー方式 (確定)

Hazkey (MIT) と fcitx5-mozc の両方がクライアント・サーバー方式。同じパターンを採用。

```
┌──────────────────────────────┐
│  IME フレームワーク            │
│  (fcitx5 / Windows TSF)       │
│  ↕ protobuf over IPC          │
└──────────┬───────────────────┘
           │
┌──────────▼───────────────────┐
│  new-ime-server (C++)         │
│  - Socket Server              │
│  - ONNX Runtime (encoder+dec) │
│  - CTC Decoder (beam search)  │
│  - ConversionEngine           │
│  - User Dict / KV Cache       │
└───────────────────────────────┘
```

**クライアント・サーバーを選ぶ理由**:

| 理由 | 詳細 |
|------|------|
| IME フレームワークのブロック回避 | keyEvent で 15-20ms ブロックすると全入力コンテキストが止まる |
| クラッシュ隔離 | ONNX Runtime の障害が IME フレームワークを巻き込まない |
| メモリ管理 | ~135M param モデル (int8 で ~135MB) をサーバープロセスに閉じ込める |

### 5.1 Windows TSF 統合 ✅ (DLL 動作確認済)

**engine/win32/ に実装済:**

- `ffi_impl.cpp` — ONNX Runtime C++ 経由の FFI
- `new-ime-engine.dll` — DLL ビルド済
- `interactive.exe` — 対話型コンソールデモ
- `test_ffi.exe` — FFI 単体テスト

### 5.2 fcitx5 プラグイン (未ビルド)

**engine/src/ にソース実装済:**

| コンポーネント | 役割 |
|---------------|------|
| `NewImeEngine` | fcitx5 `InputMethodEngineV2` |
| `NewImeState` | キーイベント ステートマシン (Direct/Composing/CandidateSelection) |
| `NewImeCandidateList/Word` | 候補リスト管理 |
| `NewImePreedit` | preedit 表示 |
| `NewImeServerConnector` | Unix domain socket IPC クライアント |
| `NewImeComposingText` | ローマ字→ひらがな変換 (促音・撥音対応) |

メタデータ: `engine/new-ime-addon.conf`, `engine/new-ime-im.conf` 配置済。

### 5.3 ONNX エクスポート

Encoder と Decoder は**別 ONNX ファイル**:
- `new-ime-encoder.onnx` — BERT encoder
- `new-ime-decoder.onnx` — NAT decoder
- `new-ime-refiner.onnx` — Mask-CTC refinement (オプション)

分割理由: Encoder KV キャッシュの再利用が容易、個別に量子化レベルを変えられる。

| 形式 | サイズ見積もり | 手法 |
|------|-------------|------|
| fp32 | ~540MB | - |
| fp16 | ~270MB | ONNX cast |
| int8 dynamic | ~135MB | `onnxruntime.quantization` (推奨) |
| int8 static | ~135MB | calibration データ必要 |

### 5.4 IPC プロトコル

`protocol/new_ime.proto` に定義済 (5種コマンド: Convert, ConvertIncremental, SetContext, ReloadModel, Shutdown)。
ローマ字→ひらがな変換・カーソル管理はエンジン側で行い、サーバーには変換リクエストのみ送る。

### 5.5 パッケージング

| 形式 | 優先度 | 備考 |
|------|--------|------|
| Arch AUR | Linux 最優先 | 自分で使う環境 |
| Windows インストーラ | Windows 最優先 | |
| .deb | Linux 次点 | Ubuntu/Debian 向け |
| Flatpak | 後回し | sandbox と Unix socket の相性検証要 |

---

## Phase 6: (旧) 量子化とユーザ適応 — **Phase 3 に吸収**

旧版では v1.0 後の別トラックとしていたが、2026-04-18 の方針改訂で CTC-NAT と同時検証に
統合した。1.58-bit QAT / CVAE / LoRA フォールバックの扱いは `phase3_plan.md` の
Step D3・Step H・CVAE fallback 節を参照。

---

## タイムライン

```
Week 1       : Step 0 — プロジェクト基盤                  ✅ 完了
Week 2-5     : Step 1 — データパイプライン                ✅ 完了
Week 6-8     : Step 2 — AR ベースライン + KD              ✅ 完了
Week 9-10    : Phase 3 Step A/B/C (受け入れ + 基盤 + 混合) ✅ 完了
Week 11-16   : Phase 3 Step D (本線学習 fp16+CVAE+1.58-bit) ← 現在ここ
Week 17-22   : Phase 3 Step E/F/G/H (KenLM / bench / int8 / bitnet)
Week 23-28   : Phase 4 評価・改善 → interactive.cpp CLI 出口
```

**合計: ~28 週間 (7ヶ月)** at 週 10-20 時間

---

## 技術的リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| CTC-NAT の精度が AR に届かない | Phase 3 停滞 | DAT にエスカレート、最終的に AR+投機的デコード |
| エンコーダ凍結では不十分 | 精度不足 | 段階的解凍 (3.4b) で対応 |
| CTC の入力長 >= 出力長制約違反 | 学習不能 | 検証済み 0.38%、パディング or upsampling 層で対応可 |
| ONNX export で CTC beam search が対応不可 | 推論エンジン問題 | beam search は ONNX 外で C++ 実装 |
| Hazkey ライセンス非互換 | fcitx5 統合の設計変更 | **確認済: MIT。問題なし** |
| fcitx5 シングルスレッドで推論ブロック | UI フリーズ | クライアント・サーバー方式で解決済 |
| 同音異義語誤変換 | 実用精度不足 | Mask-CTC + KenLM + ユーザ辞書の3層防衛 |

---

## 参考文献

### 必読

1. Ghazvininejad et al. 2019 — Mask-Predict (CMLM): アルゴリズムの基本
2. Qian et al. 2021 — GLAT: 学習安定化の核心テクニック
3. Higuchi et al. 2021 — Mask-CTC: CTC + mask-predict の複合 (ASR 文脈)
4. Saharia et al. 2020 — CTC-based NAT for translation

### 参考

5. Gu et al. 2018 — 元祖 NAT (fertility、KD の背景理解)
6. Huang et al. 2022 — DAT (撤退先の技術理解)
7. Shao & Feng 2022 — NMLA-NAT (CTC + 非単調対応、NeurIPS Spotlight)
8. Nielsen 2024 — BitNet b1.58 Reloaded (1.58-bit 量子化)

### 実装参考

9. fairseq `cmlm_transformer.py` — CMLM のリファレンス実装 (~450行)
10. thu-coai/DA-Transformer — DAT 実装 (DAG DP のロジック)
11. ictnlp/nmla-nat — CTC-NAT 実装
12. Hazkey — fcitx5 日本語 IME (MIT)
