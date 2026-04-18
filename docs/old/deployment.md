---
status: current
last_updated: 2026-04-18
---

# vast.ai 学習 deploy ガイド

Phase 3 の 90M CTC-NAT 学習を 1 コマンドで rented GPU に立てるための手順と
運用上の注意点。`scripts/deploy_vastai.sh` + `scripts/mirror_checkpoints.sh`
を組み合わせて使う。

## 想定運用

1. ローカルで `datasets/phase3/train.jsonl` を事前ビルド済み (200M 行, 37.5GB)
2. vast.ai で GPU インスタンスをレンタル (5090 32GB / 300GB disk 推奨、
   `feedback_vastai_provisioning.md` 参照)
3. 本手順で deploy、checkpoint は local に自動ミラー、インスタンス落ちても復帰可

## 必要ファイル

- `datasets/phase3/train.jsonl` (または `.zst` 事前圧縮版)
- `datasets/eval_v3/dev.jsonl`, `datasets/eval_v3/test.jsonl`
- `datasets/tokenizers/eval_v3_shared_4k.json`
- `checkpoints/ar_v3_vast/best.pt` (KD 教師)
- `checkpoints/ar_v3_vast/best_vocab.json`
- `configs/train_<name>.env` (以下の bundled configs から選ぶか、新規作成)

## 用意済み config プリセット

| config | 目的 | 想定時間 | 特徴 |
|---|---|---|---|
| `train_phase3_90m_baseline.env` | 本命 | 5090 で ~2-3 時間 | `kd_every=16`, KD 完全有効 |
| `train_phase3_90m_fast.env` | 速度優先 | 5090 で ~1.5 時間 | `kd_every=32`, KD 疎、推移早見 |
| `train_phase3_90m_no_kd.env` | KD 切り分け | 5090 で ~1.3 時間 | KD 無し、baseline 測定用 |
| `train_phase3_20m_smoke.env` | local smoke | 3060 で ~1 時間 | 20M model、2M sample、健全性確認 |

## deploy コマンド

```bash
# 基本 (90M baseline)
bash scripts/deploy_vastai.sh <host> <port>

# 別 config を指定
bash scripts/deploy_vastai.sh <host> <port> --config configs/train_phase3_90m_fast.env

# アップロード・setup だけやって手動 launch したい場合
bash scripts/deploy_vastai.sh <host> <port> --no-train
```

deploy_vastai.sh の内部フェーズ:

1. **pre-flight**: SSH 疎通、GPU (`compute_cap`)、disk、uv 確認
2. **compress**: `datasets/phase3/train.jsonl.zst` が無ければ zstd -9 で圧縮
   (既存 cache が新しければ再利用)
3. **remote setup**: repo clone (or pull)、`uv sync`、Blackwell (`cc=12.x`) なら
   cu128 torch を `uv pip install --upgrade` で入れる
4. **upload**: small files + train.jsonl.zst を 4 分割 parallel scp
5. **remote finalize**: cat + `zstd -d` + サイズ検証 + `wc -l`
6. **run script 生成**: `scripts/run_<NAME>.sh` を remote に emit (config の値で CLI 組み立て)
7. **launch**: `tmux new -d -s train 'bash scripts/run_<NAME>.sh'`
8. **次のステップ表示**: mirror 起動コマンド等の案内

## Checkpoint ミラー (必須)

vast.ai インスタンスは予告なく応答不能になることがある
(`feedback_vastai_checkpoint_sync.md`)。deploy と同時に別シェルで:

```bash
bash scripts/mirror_checkpoints.sh <host> <port> ctc_nat_90m_phase3mix
```

これで `./checkpoints/vast_mirror/ctc_nat_90m_phase3mix/` と
`./logs/vast_mirror/ctc_nat_90m_phase3mix/` に 5 分粒度で同期される。

- `best.pt` / `best_tokenizer.json` は上書き更新のため毎回 scp
- `checkpoint_step_*.pt` は `--ignore-existing` 相当で新規のみ pull
- `logs/train_*.log` も同じ規則で pull
- `--interval 120` で 2 分粒度にも調整可

## 実行中の監視

```bash
# tmux アタッチ
ssh -p <port> root@<host> 'tmux attach -t train'

# ログ tail
ssh -p <port> root@<host> 'tail -f $(ls -t /workspace/new-ime/logs/train_*.log | head -1)'

# GPU 使用率 snapshot
ssh -p <port> root@<host> 'nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader'
```

## 新 config を作るとき

1. `configs/train_phase3_90m_baseline.env` をコピーして `train_<experiment>.env` にする
2. `NAME` / `OUTPUT_SUBDIR` を書き換え (同じ tmux session `train` を別実験で使うなら
   OUTPUT_SUBDIR は必ず変える)
3. 変えたいパラメータだけ編集
4. `bash scripts/deploy_vastai.sh <host> <port> --config configs/train_<experiment>.env`

## よくあるトラブル

| 症状 | 原因 | 対処 |
|---|---|---|
| Phase 4 で scp がタイムアウト | 帯域または vast 側の瞬断 | 再実行 (scp は冪等)、parallel 数を減らす |
| Phase 5 で「SIZE MISMATCH」 | 分割 scp の途中で欠け | 再実行、または `--no-train` で Phase 4 だけ再度 |
| GPU 0% util、rate 急降下 | KD teacher gen の I/O wait | `kd_every` を上げる、または `fast.env` に切替 |
| instance が SSH 応答しない | rented GPU の瞬断 | ローカルミラーに最新 checkpoint あるはず、destroy して再 deploy |
| VRAM > 90% | batch size 過大 | `BATCH_SIZE=96` に下げて GRAD_ACCUM を上げる |

## 将来の改善点 (TODO)

- ARTeacher に KV cache を実装 → KD cost 5-10x 削減、`kd_every=4` でも実用化
- train.jsonl の streaming decompress (`.zst` のまま読む) → decompress 時間・ディスク節約
- deploy_vastai.sh を Python 化 (より構造化したい場合)
