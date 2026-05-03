# new-ime

mozc 置き換えを目標にした Japanese kana → kanji IME の研究プロトタイプ。
比較対象は `zenz-v2.5-small` (~91M) phrase-level、出口は
`crates/new-ime-interactive` の CLI。

## 現状

- 訓練済みベースライン: **Suiko-v1-small** (CTC-NAT 41M, MaskCTC refine,
  step 100k) — `checkpoints/suiko-v1-small/`
- それ以外の v1.x lineage は退行のため破棄済 (`archive/pre-v2` 参照)
- v2 は本ブランチで再構成中。LLM 訓練・bench は **TOML 駆動**で書き直す

## レイアウト

```
new-ime/
├── crates/         Rust workspace (data pipeline, tokenizer, IME, bench)
├── src/            Python LLM stack (train, eval, model) ※ v2 で構築
├── configs/        実験 TOML (train / bench / data) ※ v2 で構築
├── datasets/       corpus / mixes / eval / tokenizers
├── checkpoints/    学習済モデル (gitignored)
├── assets/         dicts (mozc) / kenlm 言語モデル
├── references/     外部 repo の clone (gitignored, 参照のみ)
├── results/        bench 結果 (一部 tracked)
├── scripts/        補助 shell / data QA
└── docs/
    └── benchmark.md   ← 仕様の真は **コード**。文書は bench protocol だけ
```

## 契約

- **設定は TOML、CLI 引数は非推奨**: `<tool> <config.toml>` の 1 引数のみ
- **実験 = config 1 ファイル**: ファイル名 = 実験名 = 出力ディレクトリ名
- **再現性**: ckpt と並べて使用 TOML を保存。比較は TOML diff
- **ドキュメントは benchmark protocol のみ**。それ以外はコードを読む

## 走らせ方

ベンチ:
```bash
cargo build -p rust-bench --release --features native-tch
# (v2 の TOML 駆動 bench は src/eval/ に実装予定)
```

詳細仕様は [`docs/benchmark.md`](docs/benchmark.md) を参照。
