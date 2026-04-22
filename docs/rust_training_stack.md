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
  - `record-checkpoint`: checkpoint ledger を更新
  - `show-run`: run manifest / trainer state を確認
  - `check-resume`: 現在 config で resume 可能か検査
  - `fit`: Rust backend で学習ループを実行
  - 現在の `ctc` backend は最小 CPU 実装で、CTC loss / checkpoint / resume の経路確認用
  - backend 共通の optimizer / scheduler state を内蔵

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
cargo run -p kkc-train -- fit --config configs/rust_student.toml --run-dir models/checkpoints/rust-student-20m --steps 100 --checkpoint-every 25
cargo run -p kkc-train -- record-checkpoint --run-dir models/checkpoints/rust-student-20m --step 2000 --epoch 1 --checkpoint models/checkpoints/rust-student-20m/checkpoint_step_2000.safetensors --metric 0.8123 --kind best
cargo run -p kkc-train -- show-run --run-dir models/checkpoints/rust-student-20m
cargo run -p kkc-train -- check-resume --config configs/rust_student.toml --run-dir models/checkpoints/rust-student-20m
```

## CUDA / 非同期 I/O

GPU 学習は `tch` 0.18 (libtorch 2.6 cu124) 経由。デフォルトビルドには含めず、`--features cuda` で optional。

```bash
# build (Windows, .venv の libtorch を再利用)
export LIBTORCH="D:/Dev/new-ime/.venv/Lib/site-packages/torch"
export LIBTORCH_BYPASS_VERSION_CHECK=1
export PATH="$LIBTORCH/lib:$PATH"
cargo build -p kkc-train --features cuda --release

# run
kkc-train fit \
    --config configs/rust_student_cuda.toml \
    --run-dir models/checkpoints/rust-student-cuda \
    --device cuda \
    --steps 200000 \
    --checkpoint-every 2000 \
    --async-ckpt-queue 2
```

`--device` は `cpu` / `cuda` / `cuda:N`。cuda feature 無しで `cuda*` を指定すると開始前に reject される。

### 非同期パイプライン構成

1. **Stage-1 prefetch**: shard → `PackedBatch` (既存の `PrefetchedBatchIter`)
2. **Stage-2 prefetch**: `PackedBatch` → `StagedHostBatch` (i64/f32 連続バッファ)、
   `gpu::StagedBatchPipeline` のワーカースレッド
3. **H2D + compute**: 学習スレッドが stage-2 キューから取り出して tch Tensor へ upload、
   forward / backward
4. **Async checkpoint writer**: `pipeline::AsyncCheckpointWriter` が別スレッドで
   ckpt を書き込み、学習ループは待たない (`--async-ckpt-queue 0` で同期にフォールバック)

tokio は入れない。single-writer / single-reader の bounded `mpsc` で十分なため。

### backend kind

| kind | device | 実装状況 |
|---|---|---|
| `ctc` | cpu | CPU の hand-rolled Transformer、f64 で数値パリティ検証用 |
| `tch-ctc-nat` | cpu / cuda | tch backend。**forward/backward は次パッチ**。現状は H2D + TrainerStep accounting の骨格 |

`tch-ctc-nat` の forward/backward / CTC loss / refine は本ファイルの後続実装でカバー。

## ここから先

次の実装対象は以下。

1. `tch-ctc-nat` の実 forward (embed + encoder + CTC head)
2. mask-CTC refine decoder + remask/stop head
3. AdamW + cosine LR + warmup (tch の `nn::VarStore` + `optim` を利用)
4. safetensors で weights snapshot、`AsyncCheckpointWriter` と接続
5. Python trainer との数値パリティ (same seed, same init, 1 step 一致)
6. AR teacher KD wrapper (student 完了後)
