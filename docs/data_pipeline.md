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

| ツール | 用途 | ビルド |
|--------|------|--------|
| tools/chunk-generator | 文節チャンク生成 | WSL + rustc 1.85 |
| tools/postprocess | 品質フィルタ | WSL + rustc 1.85 |
| tools/build-vocab | 語彙構築 | WSL + rustc 1.85 |

rustc 1.95 には ICE バグがあるため 1.85 を使用。
mecab-rs は NAIST jdic のフォーマットで動作 (unidic-lite とは別)。
