# new-ime

Rust-only 方針で再編した実験用ブランチです。active tree には Cargo
workspace の crates、Rust 向け設定、データ置き場、ドキュメント、shell
script だけを残し、Python / C++ / 過去資料は `legacy/` に隔離しています。

## アクティブスコープ

- `crates/*` だけを active なコード配置先とします。
- package prefix は `rust-*`、`data-*`、`new-ime-*` の 3 系統に固定します。
- IME crates は workspace に残しますが、この pass で保証するのは構造整理と
  buildability までで、機能完成は対象外です。
- benchmark の過去結果は active docs に持ち込まず、benchmark contract のみを
  現行文書として扱います。

## レイアウト

```text
new-ime/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── configs/
├── crates/
│   ├── rust-tokenizer/
│   ├── rust-data/
│   ├── rust-model/
│   ├── rust-train/
│   ├── rust-bench/
│   ├── rust-build-vocab/
│   ├── rust-audit-tokenizer/
│   ├── rust-postprocess/
│   ├── data-core/
│   ├── data-mix/
│   ├── data-offset-index/
│   ├── data-audit/
│   ├── data-bench-onnx/
│   ├── data-extract-domain/
│   ├── data-extract-short/
│   ├── data-process-zenz/
│   ├── data-process-whitepaper/
│   ├── data-synth-homophone/
│   ├── data-synth-name/
│   ├── data-synth-numeric-units/
│   ├── data-chunk-generator/
│   ├── new-ime-engine-core/
│   ├── new-ime-tsf/
│   └── new-ime-interactive/
├── datasets/
│   ├── raw/
│   ├── corpus/
│   ├── mixes/
│   ├── eval/
│   ├── tokenizers/
│   └── audits/
├── docs/
│   ├── vision.md
│   ├── repo_layout.md
│   ├── development.md
│   └── benchmark_comparison.md
├── scripts/
│   ├── check-hygiene.sh
│   └── check-docs.sh
└── legacy/
    ├── docs/
    ├── python/
    └── tools/
```

## クイックスタート

```bash
cargo metadata --format-version 1 --no-deps
cargo check --workspace
cargo run -p rust-train -- --help
cargo run -p rust-bench -- --help
cargo run -p new-ime-interactive -- --help
```

Windows 向け TSF の検証は別 target の check として扱います。

```bash
cargo check -p new-ime-tsf --target x86_64-pc-windows-msvc
```

## ルール

- Rust code は `crates/` 配下のみに置きます。
- `datasets/` はデータ置き場専用にします。
- `scripts/` は shell script のみを置きます。
- 廃止した Python / C++ / 過去 workflow は `legacy/` に退避します。

現行ポリシーは `docs/vision.md`、`docs/repo_layout.md`、
`docs/development.md` を参照してください。
