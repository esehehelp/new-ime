# new-ime

mozc 置き換えを目標にした Japanese kana → kanji IME の研究プロトタイプ。
比較対象は `zenz-v2.5-small` (~91M) phrase-level、評価出口は
`docs/benchmark.md` 仕様の TOML 駆動 bench、ランタイム動作確認は
`crates/new-ime-tsf` (TSF DLL) を `regsvr32` 登録。

## 現状

- 訓練済みベースライン: **Suiko-v1-small** (CTC-NAT 41M, MaskCTC refine,
  step 100k) — HF: <https://huggingface.co/esehe/new-ime-suiko-v1-small>
- それ以外の v1.x lineage は退行のため破棄済 (`archive/pre-v2` 参照)
- v2 (本ブランチ): bench は TOML 駆動で実装済 (`src/new_ime/eval/`)。
  訓練ループも `src/new_ime/training/` に移植済み。
- 本番学習 mix / tokenizer は HF dataset repo:
  <https://huggingface.co/datasets/esehe/new-ime-dataset>

## レイアウト

```
new-ime/
├── crates/         Rust workspace (data pipeline, tokenizer, IME, rust-bench)
├── src/new_ime/    Python LLM stack (cli / config / data / eval / model / train)
├── configs/        実験 TOML (configs/bench/, configs/train/)
├── datasets/       corpus / mixes / eval / tokenizers
├── checkpoints/    学習済モデル (gitignored)
├── assets/         dicts (mozc) / kenlm 言語モデル
├── models/         ONNX / runtime KenLM (gitignored)
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
- **大きい artifact は HF**: `datasets/`, `checkpoints/`, `assets/`,
  `models/`, `results/` は原則 git に入れない
- **ドキュメントは benchmark protocol のみ**。それ以外はコードを読む

## Artifacts

Model/runtime artifacts:

- Repo: <https://huggingface.co/esehe/new-ime-suiko-v1-small>
- `checkpoints/checkpoint_step_100000.pt`
- `checkpoints/checkpoint_step_100000_tokenizer.json`
- `onnx/suiko-v1-small-step100000.fp32.onnx`
- `onnx/suiko-v1-small-step100000.int8.onnx`
- `onnx/suiko-v1-small-step100000.fp32.tokenizer.json.vocab.hex.tsv`
- `kenlm/kenlm_general_6gram_q8.bin`
- `kenlm/kenlm_tech_6gram_q8.bin`
- `kenlm/kenlm_entity_4gram.bin`

Dataset artifacts:

- Repo: <https://huggingface.co/datasets/esehe/new-ime-dataset>
- `mixes/student-v1.13-500m.kkc.zst`
- `mixes/student-v1.13-500m.kkc.meta.json`
- `tokenizers/char-jis-24k.json`

Restore production mix:

```powershell
hf download esehe/new-ime-dataset mixes/student-v1.13-500m.kkc.zst --repo-type dataset --local-dir datasets
hf download esehe/new-ime-dataset mixes/student-v1.13-500m.kkc.meta.json --repo-type dataset --local-dir datasets
hf download esehe/new-ime-dataset tokenizers/char-jis-24k.json --repo-type dataset --local-dir datasets
zstd -d datasets\mixes\student-v1.13-500m.kkc.zst -o datasets\mixes\student-v1.13-500m.kkc
```

Restore Suiko-v1-small bench/runtime artifacts:

```powershell
hf download esehe/new-ime-suiko-v1-small checkpoints/checkpoint_step_100000.pt --local-dir .
hf download esehe/new-ime-suiko-v1-small checkpoints/checkpoint_step_100000_tokenizer.json --local-dir .
hf download esehe/new-ime-suiko-v1-small onnx/suiko-v1-small-step100000.int8.onnx --local-dir models
hf download esehe/new-ime-suiko-v1-small onnx/suiko-v1-small-step100000.fp32.onnx --local-dir models
hf download esehe/new-ime-suiko-v1-small onnx/suiko-v1-small-step100000.fp32.tokenizer.json.vocab.hex.tsv --local-dir models
hf download esehe/new-ime-suiko-v1-small kenlm/kenlm_general_6gram_q8.bin --local-dir models
hf download esehe/new-ime-suiko-v1-small kenlm/kenlm_tech_6gram_q8.bin --local-dir models
hf download esehe/new-ime-suiko-v1-small kenlm/kenlm_entity_4gram.bin --local-dir models
```

## 走らせ方

ベンチ (canonical, WSL CPU):
```bash
source scripts/_uv_env.sh
uv run python -m new_ime.cli.bench         # configs/bench/*.toml を全件実行
uv run python -m new_ime.cli.bench -m suiko-v1-small-greedy -t probe_v3
```

詳細仕様 + 9-model anchor は [`docs/benchmark.md`](docs/benchmark.md)。
