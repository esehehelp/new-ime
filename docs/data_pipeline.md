---
status: current
last_updated: 2026-04-18
---

# データパイプライン設計メモ

現状の処理フローとデータ特性。MeCab feature は `features[17]` (仮名形出現形) を使用 (確定)。

## 処理フロー

```
Wikipedia dump (4.4GB bz2)
    │ scripts/process_wiki.py (Python, 16 workers)
    │ mwxml + mwparserfromhell + MeCab features[17]
    ▼
wiki_sentences_v3.jsonl (26.8M pairs)
    │
    │ tools/postprocess (Rust) or scripts/postprocess.py
    │ 旧仮名除去, POS leak, dedup, 長さフィルタ
    ▼
wiki_clean_v3.jsonl (18.4M pairs, 68.7%)

青空文庫 CSV (形態素解析済み)
    │ scripts/process_aozora.py (row[11] = 読みカタカナ)
    ▼
aozora_clean.jsonl (2.4M pairs)

Livedoor / Tatoeba
    │ scripts/process_livedoor.py / process_tatoeba.py
    │ MeCab features[17]
    ▼
livedoor_clean_v3.jsonl (84K) / tatoeba_clean_v3.jsonl (228K)

全クリーンデータ → scripts/build_eval_set.py
    ▼
train.jsonl (20.1M) / dev.jsonl (2K) / test.jsonl (10K)

train.jsonl → scripts/mecab_to_tsv.py (Python, MeCab → TSV)
    ▼
all_morphemes.tsv (9GB, 4.4億行)
    │ tools/chunk-generator (Rust, WSL)
    │ 文節分割 → 1-3文節ウィンドウ → JSONL
    ▼
chunks_v3_100m.jsonl (100M chunks, 13GB)

[Phase 3 Step C 追加ソース]
HPLT v3 ja (CC0 jsonl.zst)   → scripts/download_hplt3_ja.py → scripts/process_hplt.py
FineWeb-2 jpn_Jpan (ODC-By)  → scripts/download_fineweb2_ja.py → scripts/process_fineweb2.py
zenz-v2.5-dataset (ODC-BY サブセット) → scripts/download_zenz_subset.py → scripts/process_zenz_subset.py
    │ いずれも src/data/mecab_pipeline.py (features[17]) で共通処理
    ▼
datasets/src/{hplt3_ja,fineweb2_ja,zenz_llmjp}/*.jsonl

[Phase 3 train.jsonl 生成]
全プール → scripts/build_phase3_train.py (weighted least-served 交互書き込み)
          or tools/build-train-mix (Rust port)
          + scripts/audit_pools.py (ライセンス・汚染監査)
    ▼
datasets/phase3/train.jsonl (200M rows, ~37.5 GB)
    │ src/data/dataset.py (Algorithm R reservoir sampling) で学習時ロード
```

## 品質監査プロセス

1. 0.1% 無作為抽出 x 2回
2. scripts/audit_data.py で自動パターン検出
3. 分布安定を確認 (両回の結果が一致)
4. v3 最終結果: wiki 100.0% clean (2回とも)

## データの特性

| ソース | 件数 | p50文字数 | Surface/Reading比 | 文脈あり率 |
|--------|------|---------|-------------------|-----------|
| Wikipedia | 18.4M | 28 | 0.752 | 94% |
| 青空文庫 | 2.4M | 26 | 0.816 | 92% |
| Livedoor | 84K | 32 | 0.812 | 92% |
| Tatoeba | 228K | 17 | 0.807 | 0% |
| チャンク | 100M | 9 | - | 0% |
| ゴールド | 1K | ~10 | - | ~30% |

## 既知の問題

- MeCab unidic-lite は features[17] が最適だが、WSL の NAIST jdic とはフォーマットが異なる
- 「にほん」の読みが features[17] でもカタカナ「ニホン」として出る場面がある (ローカルモデルで「ニホン語」エラー)
- チャンクのみで学習すると文レベルの変換能力が崩壊 (混合が必須)
- 短文 (2-5文字) は 31.9M モデルでは容量限界で改善しない

## Rust ツール

すべて WSL + rustc 1.85 でビルド (1.95 は ICE バグあり)。共通型・I/O は `tools/datacore/`。

| ツール | 用途 | Python 版との対応 |
|--------|------|------|
| tools/chunk-generator | 文節分割 + 1-3文節ウィンドウ | (なし、Rust が正) |
| tools/postprocess | 品質フィルタ (旧仮名・ト書き・POS 漏れ等) | scripts/postprocess.py |
| tools/build-vocab | 頻度語彙構築 | scripts/build_vocab.py |
| tools/build-train-mix | phase3 train.jsonl 混合 (weighted least-served) | scripts/build_phase3_train.py |
| tools/process-zenz | zenz サブセット整形 (kata2hira, 汚染監査) | scripts/process_zenz_subset.py |
| tools/audit-pools | プール別件数・ライセンス・汚染監査 | scripts/audit_pools.py |
| tools/audit-tokenizer | tokenizer 可逆性・ID 固定検査 | scripts/audit_tokenizer.py |
| tools/mecab-test | mecab-rs 動作確認 | - |
| tools/datacore | 共通型・JSONL/zstd I/O | - |

mecab-rs は NAIST jdic のフォーマットで動作 (unidic-lite とは別)。新規/改修は Rust
ファースト、Python は外部ライブラリ必須局所 (MeCab Python binding, mwparserfromhell 等) のみ。
