# Rust Training Stack

Python 側の `Dataset` / `DataLoader` / collator が抱えている I/O と RAM の不透明さを潰すため、学習基盤を Rust へ寄せる。

## 追加した crate

- `models/rust/kkc-tokenizer`
  - `SharedCharTokenizer` の Rust 実装
  - default vocab 生成
  - tokenizer JSON の load/save
  - byte fallback を含む encode/decode
- `models/rust/kkc-data`
  - JSONL から学習用 shard へのコンパイル
  - `mmap` で読む shard reader
  - block shuffle / packed batch 構築
  - batch memory 見積もり
- `models/rust/kkc-train`
  - `plan`: shard を読んで batch / step の RAM 上限を先に出す
  - `compile-shard`: JSONL を Rust 側の binary shard に変換
  - `peek-batches`: 実際の packed batch サイズを確認
  - `scan-epoch`: 1 epoch 分の平均/最大 batch サイズを集計
  - `dry-train`: モデルなしで batch path の throughput を測定
  - `init-run`: run manifest / trainer state を先に固定

## 直近の狙い

1. 学習時に JSONL を逐次 `json.loads` しない
2. Python `dict` / `list[int]` を経由しない
3. 実行前に batch RAM を出す
4. block 単位 shuffle で I/O と学習順序を両立する
5. tokenizer / data format を Rust 側で固定する

## 使い方

```bash
cargo run -p kkc-train -- compile-shard --input datasets/mixes/student-20m.jsonl --output datasets/mixes/student-20m.kkc --tokenizer datasets/tokenizers/shared_char.json
cargo run -p kkc-train -- plan --config configs/rust_student.toml
cargo run -p kkc-train -- peek-batches --config configs/rust_student.toml --batches 5
cargo run -p kkc-train -- scan-epoch --config configs/rust_student.toml
cargo run -p kkc-train -- dry-train --config configs/rust_student.toml --steps 1000
cargo run -p kkc-train -- init-run --config configs/rust_student.toml --output models/checkpoints/rust-student-20m
```

## ここから先

次の実装対象は以下。

1. `kkc-data` に block shuffle / prefetch queue / packed batch builder を追加
2. `kkc-train` に CTC-NAT student の trainer を追加
3. teacher / KD は student 完了後に別モジュール化
