# new-ime

mozc 置き換えを目標にした Japanese kana → kanji IME の研究プロトタイプ。
比較対象は `zenz-v2.5-small` (~91M) phrase-level、評価出口は
`docs/benchmark.md` 仕様の TOML 駆動 bench、ランタイム動作確認は
`crates/new-ime-tsf` (TSF DLL) を `regsvr32` 登録。

## 現状

- 訓練済みベースライン: **Suiko-v1-small** (CTC-NAT 41M, MaskCTC refine,
  step 100k) — `checkpoints/suiko-v1-small/`
- それ以外の v1.x lineage は退行のため破棄済 (`archive/pre-v2` 参照)
- v2 (本ブランチ): bench は TOML 駆動で実装済 (`src/new_ime/eval/`)。
  訓練ループは scaffold のみで未移植 (`src/new_ime/train/__init__.py`)

## レイアウト

```
new-ime/
├── crates/         Rust workspace (data pipeline, tokenizer, IME, rust-bench)
├── src/new_ime/    Python LLM stack (cli / config / data / eval / model / train)
├── configs/        実験 TOML (configs/bench/, configs/train/)
├── datasets/       corpus / mixes / eval / tokenizers
├── checkpoints/    学習済モデル (gitignored)
├── assets/         dicts (mozc) / kenlm 言語モデル
├── references/     外部 repo の clone (gitignored, 参照のみ)
├── results/        bench 結果 (TOML out_dir で生成、untracked)
├── scripts/        bench 補助 (_uv_env.sh, _canonical_compare.py, ...)
└── docs/
    └── benchmark.md   ← bench protocol。仕様の真はコード
```

## 契約

- **設定は TOML、CLI 引数は非推奨**: `<tool> <config.toml>` の 1 引数のみ
- **実験 = config 1 ファイル**: ファイル名 = 実験名 = 出力ディレクトリ名
- **再現性**: ckpt と並べて使用 TOML を保存。比較は TOML diff
- **ドキュメントは benchmark protocol のみ**。それ以外はコードを読む

## 走らせ方

ベンチ (canonical, WSL CPU):
```bash
source scripts/_uv_env.sh
uv run python -m new_ime.cli.bench         # configs/bench/*.toml を全件実行
uv run python -m new_ime.cli.bench -m suiko-v1-small-greedy -t probe_v3
```

詳細仕様 + 9-model anchor は [`docs/benchmark.md`](docs/benchmark.md)。
