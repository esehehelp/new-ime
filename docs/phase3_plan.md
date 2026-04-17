---
status: current
last_updated: 2026-04-18
depends_on: docs/vision.md, docs/roadmap.md, docs/phase2_results.md, docs/dataset_candidates.md
---

# Phase 3 実装計画: CTC-NAT + CVAE + 1.58-bit (研究プロトタイプ検証)

## Context

Phase 2 AR ベースライン (31.9M, eval_v3 EM 0.412) が完了し、Phase 3 に進む段階。
ユーザ方針:

- **v1.0 の目的は研究プロトタイプ検証**。「使える IME の出荷」ではない。
- CTC-NAT + CVAE + 1.58-bit の**両立可能性を本線で同時検証**する。分離フェーズではない。
  両立不能が確認されたときのみ段階的妥協 (撤退経路) に落とす。
- 規模基準は **~90M (zenz-v2.5-small 級)**。20-40M は速度実証用の縮小実験に格下げ。
- **1.58-bit は bitnet.cpp 専用**。ORT には載せない。ORT 経路は int8 fallback に限定。
- **Windows IME (TSF/DLL/fcitx5) 統合はスコープ外**。`engine/win32/interactive.cpp` を更新した
  CLI デモが出口。
- データ混合比の設計が本計画の中心論点。データセット追加より重要。
- Go/No-Go は **速度優先**。精度は AR (eval_v3 EM 0.412) と致命的に離れない下限でよい。

一次レビューと二次レビューの指摘 (Windows 工数過小評価・1.58-bit→ORT 未定義・29M 根拠薄弱・
KD 事前展開が重い・推論スタック二重化・CVAE 推論時状態未仕様・zenz サブセットのライセンス
慎重論・受け入れテスト先行) をすべて反映する。

## スコープ

### In-scope (v1.0 研究プロトタイプ)

- ~90M CTC-NAT 本体 (fp16 → 1.58-bit QAT 両方の checkpoint を残す)
- CVAE (writer/domain/session、FiLM 条件付け、推論時状態管理仕様込み)
- データ混合カリキュラム (P1〜P5 プール分離、段階的サンプリング)
- KenLM shallow fusion (beam search 統合)
- オンライン KD (Phase 2 AR 31.9M を教師、事前展開しない)
- C++ 推論ベンチ harness (受け入れテストの一部)
- `engine/win32/interactive.cpp` を CTC-NAT 対応に書き換え (CLI デモ)
- ONNX int8 fallback (bitnet.cpp が間に合わない場合の代替)
- bitnet.cpp カスタム推論カーネル (1.58-bit 研究線)

### Out-of-scope (この計画では触らない)

- Windows TSF/DLL の IME 本体統合 (`ffi_impl.cpp` の API 互換維持は**しない**、必要なら次計画)
- fcitx5 プラグインのビルド・配布
- ランキング/ユーザ辞書 (v2 計画)
- モバイル・macOS
- Swallow Corpus v3 (ライセンス未確認)
- CC-BY-SA データ全般

## アーキテクチャ

### 規模と構成

**本命: h=640, L_enc=8, L_dec=8 scratch** ≈ 97M base + 7M CVAE ≈ **~104M**

| コンポーネント | 計算 | params |
|---|---|---|
| Shared char embedding (tied) | 6500 × 640 | 4.16M |
| Positional (enc+dec) | 2 × 512 × 640 | 0.66M |
| Encoder × 8: 12h² per layer | 8 × 12 × 640² | 39.3M |
| Decoder × 8: 16h² per layer (self+cross+ffn) | 8 × 16 × 640² | 52.4M |
| LayerNorms etc | - | ~0.5M |
| **CTC-NAT base** | | **~97M** |
| FiLM(z→γ,β) × 8 decoder layers | 8 × 2 × 64 × 640 | 0.66M |
| Posterior biGRU (h=256) over surface | | ~3M |
| Writer prior encoder | | ~1M |
| Domain/session embed + small MLP | | ~2M |
| **CVAE 上乗せ** | | **~7M** |
| **合計** | | **~104M** |

**縮小実験用: h=384, L=6+6** ≈ 29M + CVAE 1M = 30M (速度検証用、A/B 可能な config 化)

**エンコーダ初期化**: scratch を第一候補。cl-tohoku/bert-base-japanese-char-v3 (110M) は
~90M budget を単独で超えるため、本体初期化としては使わない。MLM プリトレで数 epoch
加熱する余地は残す (Step B の任意オプション)。

### 入出力

- Tokenizer: **shared char vocab ~6500** (入出力統合)。現行 `InputTokenizer` (180 vocab, kana only) は
  破棄。左文脈に漢字が入るため encoder 側も ~6500 必要。
- 入力: `[CLS] 左文脈(最大40字) [SEP] かな入力` — 最大長 **128 tokens** (計画内で一貫)
- 出力: CTC blank 付きで 128 tokens、collapse 後に漢字かな混じり

### CVAE 仕様 (推論時状態管理含む、先に確定)

**潜在変数**:
```
z = concat(z_writer[32], z_domain[16], z_session[16]) ∈ ℝ^64
```

**学習時 posterior q(z | x, y)**:
- biGRU(h=256) on target surface embedding → μ_q, logσ²_q ∈ ℝ^64

**学習時 prior p(z | 粗ラベル)**:
- 書き手ラベルを青空文庫作者 (約 1500) + Wiki 分野 (20 クラス) + ソース (5 クラス) で付与
- 不明時は N(0, I)

**推論時状態管理 (ここを先に仕様化する)**:

| 変数 | 更新規則 | 保持期間 | ソース |
|---|---|---|---|
| z_writer | 入力確定文から q(z_writer\|x,y) 事後を EMA (decay 0.995) で更新 | 端末 SQLite に永続化 | ローカルのみ、ネット非送信 |
| z_domain | 起動時に設定 or アプリ名ハッシュでルックアップ。設定 UI は本計画では作らない | プロセス存続中固定 | ユーザ設定 or デフォルト |
| z_session | 直近 N=16 の確定文から posterior を再計算 | セッション (プロセス) 内 | 揮発メモリ |

**FiLM 条件付け**: 各 decoder 層で `h ← γ(z) ⊙ h + β(z)`、Linear(64, 640) × 2 per layer。

**損失**:
```
L = L_CTC + β(t) · max(KL(q(z|x,y) || p(z|x)), free_bits)
```
- β: 0 → 0.5 を 50k ステップで線形 ramp
- free_bits: 1.0 nat per latent dim (posterior collapse 防止)

**fallback**: posterior collapse (KL がほぼ 0) が発生した場合、**LoRA ベースのアプリ別分離**に
撤退 (vision.md / roadmap.md と整合)。

## データ混合カリキュラム (中心論点)

### プール分離

| プール | ソース | サイズ | 特性 |
|---|---|---|---|
| P1 高品質文ペア | Wikipedia (features[17]) + 青空文庫 | 20.8M | quality-verified、主軸 |
| P2 外部整形済 | zenz-v2.5-dataset の llm-jp-corpus-v3 サブセット (ODC-BY 解釈) | 32.4GB | 即投入可、ただしライセンス・汚染監査必須 |
| P3 読み付与 Web | HPLT v3 ja (CC0) サブセット 5-10GB | 数 M ペア | features[17] パイプラインで加工 |
| P4 チャンク (短文) | `tools/chunk-generator` 出力 | 100M | **総量で押し込まない**。低比率混入のみ |
| P5 KD overlay | Phase 2 AR greedy をオンライン生成 | - | **事前展開しない**、hard-example 中心 |

### 段階的サンプリング (固定比率ではない)

| ステージ | ステップ | P1 | P2 | P3 | P4 | P5 overlay |
|---|---|---|---|---|---|---|
| S0 warmup | 0-10k | 100% | 0 | 0 | 0 | 0 |
| S1 base | 10k-80k | 60% | 30% | 5% | 5% | 20% of batch |
| S2 diverse | 80k-180k | 40% | 30% | 20% | 10% | 30% of batch (hard) |
| S3 clean finetune | 180k-230k | 80% | 20% | 0 | 0 | 10% of batch |
| S4 +CVAE | 230k-300k | 60% (labeled) | 20% (labeled) | 10% | 10% | 20% |
| S5 +QAT 1.58 | 300k-360k | 70% | 30% | 0 | 0 | 10% |

- **混合はバッチ単位でランダム**、プール比は期待値。
- **P4 (chunk) は常に低比率**。chunk-only で文レベル崩壊する既存結果 (`phase2_results.md`) を
  踏まえ、最大 10% に制限。
- **P5 KD overlay は hard-example 指定**。AR が自信を持って出す容易例は蒸留しない
  (AR の上限を縛らないため)。実装: オンライン生成時に AR の top-1 confidence < 閾値の
  サンプルだけ KD ターゲットに使う。

### 受け入れ条件 (Step A 完了判定)

1. 各プールの件数・分布・平均文長・漢字含有率レポート
2. zenz-v2.5-dataset のライセンス遵守チェックリスト (ODC-BY attribution 明記、CC-BY-SA 非混入確認)
3. test 汚染監査: eval_v3/manual_test/AJIMEE 各テストセットと lexical / 6-gram 完全一致チェック
4. dev セット 2000 文が S1 データと分離されていることを確認

## 推論スタックの統一

現状:
- `engine/win32/ffi_impl.cpp` は AR 専用 (`ar_v3_vast_fixed.onnx`)、[context][SEP][hiragana][OUT] プロンプト組立
- `server/src/ctc_decoder.cpp` は C++ CTC greedy + prefix beam search 実装あり (fcitx5 前提)
- `engine/win32/interactive.cpp` は `ffi_impl.cpp` を呼ぶ CLI

**方針**: **`server/src/` を CTC-NAT 本体の推論スタックとし、Windows 側は server コードを
ライブラリとしてリンクする**。二重実装を回避。

- `server/src/ctc_decoder.cpp` に KenLM shallow fusion を追加 (Step E)
- `server/src/conversion_engine.cpp` を ONNX Runtime 経路 (encoder/decoder 別 ONNX) に対応
- `engine/win32/interactive.cpp` を `conversion_engine` を呼ぶ形に書き換え
- `ffi_impl.cpp` は**触らない** (AR 用としてそのまま保管)
- 1.58-bit 経路は**別ディレクトリ** (`server/src/bitnet/`) に分離、int8 ORT とコードパスを共有しない

## 実装ステップ

### Step A: 受け入れテスト定義 (先行、3 日)

実装前に acceptance test を pytest/ctest で定義。失敗すれば Step B 以降ブロック。

- [ ] shared tokenizer invariants: encode→decode 可逆、BLANK/MASK/CLS/SEP の ID 固定
- [ ] PyTorch/ONNX 出力一致 test: dummy input で ±1e-4 以内
- [ ] C++ CTC beam + KenLM golden test: 既知 (input, expected top-k) の一致
- [ ] レイテンシベンチ harness: interactive.cpp から呼べる wall-clock 計測
- [ ] データプール監査スクリプト: 各プールの件数・ライセンス・汚染チェック出力
- [ ] CVAE posterior 動作 test: KL > free_bits になることを 1k ステップで確認

### Step B: コード基盤 (1 週)

- [ ] `src/data/tokenizer.py`: `SharedCharTokenizer` 追加 (InputTokenizer を deprecate)
- [ ] `src/model/encoder.py`: `SmallEncoder` (scratch, h=640/L=8) 追加
- [ ] `src/model/decoder.py`: FiLM 条件付けを decoder layer に追加 (`film_z` 引数)
- [ ] `src/model/cvae.py` 新規: PosteriorEncoder, Prior, FiLMProjector
- [ ] `src/model/bit_linear.py` 新規: BitLinear (median scaling + STE) + 1.58-bit pack/unpack
- [ ] `src/model/ctc_nat.py`: 上記を統合。`config=` で規模を切り替え (90M / 30M の 2 プリセット)
- [ ] `configs/phase3_90m.yaml` / `configs/phase3_30m.yaml`

### Step C: データ拡充と混合 (1 週)

- [ ] `scripts/download_zenz_subset.py`: llm-jp-corpus-v3 サブセットのみ DL、ライセンスチェック
- [ ] `scripts/download_hplt3_ja.py`: HPLT v3 ja サブセット 5-10GB
- [ ] `scripts/process_hplt.py`: features[17] パイプライン適用
- [ ] `scripts/label_for_cvae.py`: 青空文庫作者・Wiki 分野・ソース ラベル付与
- [ ] `scripts/audit_pools.py`: プール別監査 + 汚染チェック (受け入れテスト)
- [ ] `src/data/curriculum_sampler.py`: 段階別のプール混合サンプラー

### Step D: 本線学習 (4-6 週)

同一研究線上で S0〜S5 を順次実行。チェックポイントを S3/S4/S5 時点で保存
(rollback 用)。

- [ ] D0: `src/training/train_ctc_nat.py` 実装 (fp16, GLAT, Mask-CTC 対応)
- [ ] D1: S0-S3 完走 (fp16 base, 230k steps)
- [ ] D2: S4 完走 (CVAE 有効化、KL annealing、70k steps)
- [ ] D3: S5 完走 (1.58-bit QAT 有効化、BitLinear 置換、60k steps)

**KD 方針**: オンライン生成。AR 教師 (`checkpoints/ar_baseline/best.pt`) を別 GPU/別プロセス
で走らせ、hard-example のみ KD ターゲットとして差し替える。**全件事前展開の JSONL は作らない**。

**計算**: 90M × 360k = H100 で ~60-80 時間、$300-500。ローカル 3060 は S0-S1 の
smoke test のみ。

### Step E: KenLM shallow fusion (1 週、D1 完了後に並行可)

- [ ] Wiki surface で 4-gram KenLM 学習 (`kenlm/bin/lmplz -o 4`)
- [ ] `server/src/ctc_decoder.cpp` に KenLM スコア加算追加 (`-DENABLE_KENLM`)
- [ ] golden test: 既知 (input, KenLM スコア込み top-k) の一致
- [ ] α, β を dev で tuning、`configs/phase3_kenlm.yaml` に保存

### Step F: 推論ベンチ harness (4 日、Step E と並行)

- [ ] `server/bench/latency_bench.cpp`: encoder + decoder + beam + KenLM の p50/p95/p99
- [ ] 入力長別 (5/15/30/60 文字) のレイテンシ分布
- [ ] `scripts/run_phase3_bench.py`: PyTorch 側と合わせた total レポート生成
- [ ] **受け入れテスト連動**: Step D 各 checkpoint で自動実行

### Step G: interactive.cpp 統合 (int8 ORT 経路、1 週)

- [ ] `scripts/export_onnx.py` 再作成: encoder/decoder 別 ONNX
- [ ] `onnxruntime.quantization.quantize_dynamic` で int8 化
- [ ] `engine/win32/interactive.cpp` を `server/src/conversion_engine` 呼び出しに書き換え
- [ ] `engine/win32/build.bat` 更新、CMake 経由に切替
- [ ] end-to-end latency bench と精度 (eval_v3 EM) 比較レポート

### Step H: bitnet.cpp 経路 (研究線、2 週)

**Step G と独立。int8 ORT が動いていれば Step H の遅延は本線を止めない。**

- [ ] bitnet.cpp を vendored 依存として追加 (`third_party/bitnet.cpp/`)
- [ ] `server/src/bitnet/ctc_nat_bitnet.cpp` 新規: 1.58-bit 推論カーネル呼び出し
- [ ] D3 の weight を bitnet.cpp 形式に pack (`tools/pack_bitnet.py`)
- [ ] bitnet 経路専用の interactive デモ (`engine/win32/interactive_bitnet.exe`)
- [ ] fp16 / int8 ORT / 1.58-bit bitnet.cpp の 3-way 比較レポート

## Go/No-Go

**速度主目標、精度は下限条件**。

| ステージ | Go 条件 |
|---|---|
| Step D1 (fp16 base) | **p95 レイテンシ < 30ms @ 中文** AND eval_v3 EM ≥ 0.380 (AR - 3pt 以内) |
| Step D2 (+CVAE) | 書き手別 eval で +2pt 以上の改善、または汎用 eval で劣化なし |
| Step D3 (+1.58-bit) | EM 低下 < 3pt (fp16 比)、サイズ ≤ 20MB |
| Step H (bitnet.cpp) | p95 レイテンシが int8 ORT より速い、または同等 |

**失敗時の撤退**:

| 失敗 | 撤退先 |
|---|---|
| D1 EM < 0.380 かつ速度も遅い | DAT (DA-Transformer) にエスカレート。不可なら AR + 投機的デコード |
| D2 posterior collapse (KL < free_bits) | LoRA ベースのアプリ別分離に差し替え (CVAE 廃止) |
| D3 EM 低下 > 5pt | 1.58-bit を諦め、int8 ORT を最終配布形式に |
| H bitnet.cpp 統合失敗 | 研究線として保留、本線は int8 ORT のみ |

## 量子化経路の分離 (明文化)

| 経路 | 形式 | ランタイム | 用途 | 本計画での位置付け |
|---|---|---|---|---|
| A: fp16 | PyTorch | GPU | 学習と基礎評価 | D1/D2 完了後の保存チェックポイント |
| B: int8 ORT dynamic | ONNX | onnxruntime | 配布用 fallback | **Step G の出口**、interactive.cpp CLI |
| C: 1.58-bit bitnet | 独自 pack | bitnet.cpp | 研究線、最小サイズ | **Step H の出口**、別 CLI |

**A→B→C は片道のみ**。B と C は互いに変換しない。ORT に 1.58-bit を載せる作業は
計画から除外。

## 重要ファイル (編集対象)

- 新規: `src/model/cvae.py`, `src/model/bit_linear.py`, `src/training/train_ctc_nat.py`,
  `src/data/curriculum_sampler.py`, `server/src/bitnet/*`, `server/bench/latency_bench.cpp`,
  `configs/phase3_*.yaml`, `tools/pack_bitnet.py`
- 大改: `src/data/tokenizer.py` (Shared 化), `src/model/encoder.py` (SmallEncoder),
  `src/model/decoder.py` (FiLM), `src/model/ctc_nat.py` (config 化),
  `server/src/ctc_decoder.cpp` (KenLM), `server/src/conversion_engine.cpp` (ONNX 経路),
  `engine/win32/interactive.cpp` (conversion_engine 呼び出し化)
- 不変: `engine/win32/ffi_impl.cpp` (AR 用として維持), `server/src/socket_server.*`

## 検証 (acceptance tests, end-to-end)

- `uv run pytest` で全 Python 単体テスト green
- `ctest` で C++ 単体テスト + golden test green
- `scripts/audit_pools.py` でライセンス・汚染違反ゼロ
- `scripts/run_phase3_bench.py` で D1/D2/D3 各 checkpoint のレイテンシ + 精度レポート
- `engine/win32/interactive.exe` (int8 ORT) で手動変換 20 問を通して動作確認
- `engine/win32/interactive_bitnet.exe` (1.58-bit) で同 20 問、結果差分レポート
- `docs/phase3_results.md` に最終結果まとめ、`docs/vision.md` と `docs/roadmap.md` を更新
