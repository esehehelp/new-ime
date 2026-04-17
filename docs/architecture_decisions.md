# アーキテクチャ設計判断の記録

## Phase 3 本命: 200M CTC-NAT 1.58-bit

### 構成

```
[左文脈 + ひらがな入力]
    │
    ▼ Encoder (BERT 85M, cl-tohoku/bert-base-japanese-char-v3 初期化)
    │
    ▼ Decoder (NAT 6層, 双方向 self-attention + cross-attention)
    │
    ▼ CTC Head → CTC collapse / beam search + KenLM shallow fusion
    │
    漢字かな混じり出力 (top-k 候補)
```

### 量子化戦略

- Nielsen 2024 "BitNet b1.58 Reloaded" で小モデル (100K-48M) の実証あり
- Encoder-decoder も対応済み (BERT/T5系で検証)
- CQAT: fp16 で学習 → 1.58-bit に遷移
- Hidden 512 → 1024 で fp16 同等性能
- 200M 1.58-bit → ~40MB メモリ

### 推論基盤

- ONNX Runtime (C++) — encoder/decoder 別 ONNX ファイル
- CTC beam search: C++ 実装 (tools/chunk-generator の CTCDecoder を流用)
- KenLM shallow fusion: beam search のスコアに n-gram LM を加算
- fcitx5 統合: クライアント・サーバー方式 (Hazkey パターン)

## 採用しない選択

### llama.cpp
- Decoder-only 前提、encoder-decoder 非対応
- zenz-v1 は使えるが、CTC-NAT では使えない

### 1.58-bit を初期から
- Phase 2 → Phase 3 fp16 → CQAT で 1.58-bit の段階的移行が安全
- QAT 再学習のコストは fp16 学習後にのみ発生

### Beam search (AR)
- 40-60x の速度ペナルティ (実装依存)
- Length normalization + repetition penalty でも改善限定的
- CTC beam search が構造的に優れている

## 候補ランキングパイプライン (Phase 5)

```
CTC beam search (beam=20, KenLM fusion)
    ↓ top-20 候補
ユーザ辞書ルックアップ (ハード制約)
    ↓
ユーザ学習スコア加算 (SQLite, 指数減衰)
    ↓
最終 top-K (K=10)
```

- ユーザ学習: (読み, 文脈hash, 候補) → (count, last_used)
- アプリ別学習データ分離 (fcitx5 で取得可能)
- CVAE はv2送り
