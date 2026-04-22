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

# run (Suiko-v1-small 次版、phase3_30m)
kkc-train compile-shard \
    --input datasets/mixes/student-300m-short.jsonl \
    --output datasets/mixes/student-300m-short.kkc \
    --tokenizer datasets/tokenizers/char-5k.json

kkc-train init-run \
    --config configs/rust_student_cuda.toml \
    --output models/checkpoints/rust-student-300m-short

kkc-train fit \
    --config configs/rust_student_cuda.toml \
    --run-dir models/checkpoints/rust-student-300m-short \
    --device cuda \
    --steps 200000 \
    --checkpoint-every 2000 \
    --async-ckpt-queue 2 \
    --grad-clip 1.0
```

`--device` は `cpu` / `cuda` / `cuda:N`。cuda feature 無しで `cuda*` を指定すると開始前に reject される。
`--grad-clip` は global grad-norm clip (0.0 で無効)。`tch-ctc-nat` backend のみが honor する。

### 非同期パイプライン構成

1. **Stage-1 prefetch**: shard → `PackedBatch` (既存の `PrefetchedBatchIter`)
2. **Stage-2 prefetch**: `PackedBatch` → `StagedHostBatch` (i64/f32 連続バッファ)、
   `gpu::StagedBatchPipeline` のワーカースレッド
3. **H2D + compute**: 学習スレッドが stage-2 キューから取り出して tch Tensor へ upload、
   forward / backward / AdamW step
4. **Async checkpoint writer**: `pipeline::AsyncCheckpointWriter` が別スレッドで
   safetensors weights + meta.json を書き込み、学習ループは待たない

tokio は入れない。single-writer / single-reader の bounded `mpsc` で十分なため。

### backend kind

| kind | device | 実装状況 |
|---|---|---|
| `ctc` | cpu | CPU の hand-rolled Transformer、f64。smoke / パリティ検証用 |
| `tch-ctc-nat` | cpu / cuda | **本線**。VarStore + tch layers、tied embed、CTC + refine + remask + stop loss、AdamW + warmup-cosine + grad_clip、safetensors ckpt、KD infra |

### tch-ctc-nat の module 構成

- `gpu/batch.rs` — host staging + GPU upload
- `gpu/layers.rs` — MultiHeadAttention (separate q/k/v/out)、EncoderLayer (post-norm)、DecoderLayer (pre-norm)
- `gpu/model.rs` — CtcNatModel、encoder + proposal decoder + refine decoder、tied token_embed (~41M params @ phase3_30m)
- `gpu/loss.rs` — ctc_proposal_loss、build_target_refinement、refine_mlm_loss、remask_loss、stop_loss
- `gpu/optim.rs` — TchOptimizer (AdamW + warmup_cosine + clip_grad_norm)
- `gpu/ckpt.rs` — safetensors weights save/load + meta.json
- `gpu/kd.rs` — KdConfig、alpha_at、hard_example_mask、ArTeacher (TorchScript loader; greedy decode 未実装)
- `gpu/parity.rs` — Python 数値パリティ check (CTC loss + LR schedule)

### パリティ fixture

```bash
.venv/Scripts/python.exe tools/rust/emit_parity_fixture.py
cargo test -p kkc-train --features cuda parity
```

fixture は `parity-fixtures/` に出力、`.gitignore` で除外。

## 残タスク

1. AMP autocast (bf16 on Blackwell、fp16 + dynamic grad scaler は後回し)
2. ArTeacher の TorchScript export script + greedy decode loop
3. Optimizer state (AdamW m/v) を safetensors に persist してフル resume 可能に
   — 現在は resume で AdamW が zero state から再開 (数百 step で収束)
4. 完全な architectural Python parity (weight-for-weight end-to-end)
5. smoke 学習 + Python trainer との loss 曲線比較 (Step 8)
