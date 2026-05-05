# new-ime

mozc 置き換えを目標にした日本語変換IME．

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
    └── benchmark.md   ← bench protocol
```

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
