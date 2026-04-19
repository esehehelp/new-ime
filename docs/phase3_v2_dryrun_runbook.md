# phase3_v2 dry-run runbook (30M scratch, 16K step, local)

本番モデル学習前の予行演習と方向性確認。local 実行、checkpoint 1000 ごと、
training 実行と監視は user が担当。

## Mix 比率定義 (統一表記)

**mix pool 比率は pool 内で sum=1.0**。KD は **overlay** (batch レベルで追加計算、
pool の置換ではない) のため、mix pool 比率とは独立。

| プール | pool 内比率 | 備考 |
|---|---:|---|
| sentence-level | **0.555** | `datasets/mixes/scratch-200m.jsonl` (既存 phase3 mix) |
| bunsetsu span=2 | **0.278** | `datasets/corpus/bunsetsu/*.jsonl` (filter span_bunsetsu==2) |
| bunsetsu span=1 | **0.055** | 同上 (filter span_bunsetsu==1) |
| synth_numeric | **0.111** | `datasets/corpus/synth/numeric.jsonl` |
| **sum** | **1.000** | |

(元の user 指定 "50/25/5/10 of all data + 10% KD" を pool 比率にすると
上記。50/0.9 = 55.6% などで正規化。)

KD overlay は `--kd-every 4` → 25% of optimizer steps で発火 (mix pool とは独立)。

## 前提条件

### 1. bunsetsu 化完走確認

`datasets/tools/corpus/run_bunsetsu_all.sh` が全 5 pool を処理済み:

```bash
wc -l datasets/corpus/bunsetsu/*.jsonl
# 想定合計: 約 8-10M 行 (span=1 + span=2 混在)
```

wiktionary_v2.jsonl が書かれていること。

### 2. synth_numeric + synth_numeric_ext 生成済み

```bash
wc -l datasets/corpus/synth/numeric.jsonl
# 36,831 行 (数詞×助数詞)
wc -l datasets/corpus/synth/numeric_ext.jsonl
# 150,000 行 (時刻/日付/通貨/分数/小数/連番)
# 合計 ~187K 行を synth pool として使用
```

### 3. 既存 phase3 sentence mix

```bash
wc -l datasets/mixes/scratch-200m.jsonl
# 200,000,000 行 / ~38 GB
```

## Step 1: 20M training mix 構築

`--max-train-samples 0` で全件ロード (~3GB RAM) できるサイズ。
oversampling は 20M 内で完結 (synth 187K 合算 → 12x 露出)。

sentence pool は既存 phase3 (200M) + v2 sentence sources (1.7M) 合算、
synth pool は synth_numeric (37K) + synth_numeric_ext (150K) 合算。

**注意: sentence pool 実効分布**

既存 `datasets/mixes/scratch-200m.jsonl` (200M) は旧 v1 mix で構築されており内部に
**chunks 60% (短文)** が含まれる。従って pool 内比率 sentence 0.555 は**外見上**
の値で、実効長さ分布では:

- 「sentence」のうち ~60% が chunks (短文、10-30字)
- 「sentence」のうち ~40% が本来の文 (wiki/aozora 等、30-120字)
- 加えて bunsetsu span=2 0.278 / span=1 0.055 も短文

**実効短文比率は ~55-65%** になる見込み。これは意図した設計 (短文特化 IME)
と整合しているが、「sentence vs bunsetsu」の外見比率だけで判断すると誤読する。
本番 training で長文性能が不足する場合、sentence pool 側を v2 系 (chunks
含まない) に入れ替える選択肢あり。

事前に Rust binary をビルド:
```bash
cd tools && cargo build --release --bin build-train-mix-v2 && cd ..
```

```bash
tools/target/release/build-train-mix-v2.exe \
    --output datasets/mixes/student-20m.jsonl \
    --total 20000000 \
    --sentence-src datasets/mixes/scratch-200m.jsonl \
    --sentence-src datasets/corpus/sentence/wikinews.clean.jsonl \
    --sentence-src datasets/corpus/sentence/wikibooks.clean.jsonl \
    --sentence-src datasets/corpus/sentence/wiktionary.clean.jsonl \
    --sentence-src datasets/corpus/sentence/tatoeba.jsonl \
    --sentence-src datasets/corpus/sentence/aozora_dialogue.jsonl \
    --bunsetsu-src datasets/corpus/bunsetsu \
    --synth-src datasets/corpus/synth/numeric.jsonl \
    --synth-src datasets/corpus/synth/numeric_ext.jsonl \
    --ratio-sentence 0.555 \
    --ratio-bunsetsu2 0.278 \
    --ratio-bunsetsu1 0.055 \
    --ratio-synth 0.111 \
    --seed 42
```

所要時間: **15-20 秒** (Rust 版、smoke で 1.28M rows/s 観測)。
出力サイズ ~3-4 GiB。

確認:
```bash
wc -l datasets/mixes/student-20m.jsonl
# 20,000,000 行
```

## Step 2: 30M scratch training

90M step27500 を CTC teacher、cosine→0 LR、**160K step** (local 3060 で ~12h)。

```powershell
.\.venv\Scripts\python.exe -m models.src.training.train_ctc_nat `
    --train datasets/mixes/student-20m.jsonl `
    --dev datasets/eval/general/dev.jsonl `
    --preset phase3_30m `
    --tokenizer-path models/checkpoints/ctc_nat_90m/checkpoint_step_27500_tokenizer.json `
    --batch-size 48 `
    --grad-accum 4 `
    --max-seq-len 128 `
    --max-context 40 `
    --fp16 `
    --num-workers 0 `
    --max-train-samples 0 `
    --max-dev-samples 2000 `
    --max-steps 160000 `
    --epochs 99 `
    --lr 1.5e-4 `
    --warmup-steps 1000 `
    --lr-schedule cosine `
    --weight-decay 0.01 `
    --grad-clip 1.0 `
    --checkpoint-every 1000 `
    --eval-every 500 `
    --log-every 100 `
    --kd-teacher-type ctc `
    --kd-teacher-path models/checkpoints/ctc_nat_90m/checkpoint_step_27500.pt `
    --kd-temperature 2.0 `
    --kd-alpha 0.3 `
    --kd-alpha-final 0.1 `
    --kd-alpha-decay-start 12000 `
    --kd-alpha-decay-steps 20000 `
    --kd-start-step 4000 `
    --kd-warmup-steps 2000 `
    --kd-gate-mode low_conf `
    --kd-hard-threshold 0.92 `
    --kd-every 4 `
    --output models/checkpoints/ctc_nat_30m_v2_dryrun
```

### KD schedule の形 (前寄せ型)

- **0-4000**: scratch 独走 (KD off)
- **4000-6000**: KD warmup、α 0 → 0.3 線形
- **6000-12000**: KD α=0.3 定常、teacher 最大引きこみ期
- **12000-32000**: KD α decay、0.3 → 0.1 線形
- **32000-160000**: KD α=0.1 定常、長い tail で student 自走

### 前寄せ型を選んだ根拠

**CTC teacher の弱点共有問題**: student (CTC-NAT 30M) と teacher (CTC-NAT 90M)
は同じアーキテクチャで、弱点 (numeric ほぼ 0%、homophone ~60%、EM5==EM1 の候補
多様性不足) を共有している。teacher に引きこみ続けても弱点は補正されない。
むしろ student が自由に探索する機会を奪う。

**AR teacher の過学習傾向**: 過去の 30M 50K run (AR teacher、KD α=0.075 → 0.1
long-tail) では eval で過学習傾向が出た。teacher の top-1 を長く模倣することで
student が supervision bias に囚われる。

**前寄せ型の設計思想**:
1. **教師介入期 (4K-12K)**: teacher が student を正しい basin に引き込む。
   scratch からの迷走を防ぎ、早期収束を促す
2. **移行期 (12K-32K)**: α を 0.3 → 0.1 へ decay。teacher から student へ
   主導権を渡す
3. **自走期 (32K-160K)**: 弱い KD (α=0.1) を残し teacher の方向性参照だけ
   維持、student 自身の探索で弱点カテゴリを掘る

long-tail decay より、明示的に「介入期」「自走期」を切り分ける。同構造 teacher
の限界を受け入れた設計。

**Windows 注**: `--num-workers 0` が安全。8 にすると DataLoader が worker に
in-memory dataset を pickle 複製しようとして MemoryError (20M 行 × 8 worker =
RAM 32GB 要求)。

### VRAM 見積もり (3060 12GB、batch 48、seq 128)

| 区分 | 実測 / 推定 |
|---|---:|
| 30M student params + opt/grad | 0.40 GB |
| activation (batch 48, seq 128) | 0.95 GB |
| total estimate | 1.35 GB |
| **cuda peak (測定値)** | **3.50 GB (29% of 12GB)** |

batch 48 でも 3060 12GB の 30% 程度に収まる。余裕あり。OOM 時は
`--batch-size 32` に下げる。

**seq 128 に下げた背景**: teacher (90M step27500) の pos_embedding が
max_positions=128 で学習済み。student 側を 256 に拡張しても teacher 側で
CUDA assert が発生する。将来 teacher を seq_len=256 で再学習するまで、
当面 student と teacher は max_positions=128 で揃える。

**影響**: 長文 sentence (120 字 + context 40) は一部トリムされる可能性。
bunsetsu/synth (大半 < 30 字) は影響なし。

## Step 3: 評価

1000 step checkpoint ごとに外部評価:

```bash
# probe_v2 (467 items, phrase-level EM)
uv run python -m datasets.tools.probe.run_probe_v2 \
    --models ctc_nat_30m \
    --out-dir results/phase3_v2_dryrun/probe_v2
# → datasets/tools/probe/run_probe_v2.py の _ctc factory で新 ckpt path を指すよう編集要

# CVAE probe (188 items, domain 別 EM)
uv run python -m datasets.tools.probe.run_cvae_probe \
    --backend ctc_nat_30m \
    --ckpt models/checkpoints/ctc_nat_30m_v2_dryrun/checkpoint_step_160000.pt \
    --out results/phase3_v2_dryrun/cvae_160k.json
```

## 前回 30M 50k run との差分

| 項目 | 前回 30M | 今回 dry-run |
|---|---|---|
| teacher | AR (ar_v3_vast/best.pt) | **CTC (90M step27500)** |
| KD 損失 | text round-trip + CTC | **直接 logit KL (temp=2.0)** |
| kd-alpha | 0.075 → 0.1 | **0.3 → 0.1** |
| batch-size | 32 | **48** (3060 VRAM 余裕) |
| kd-start-step | 6000 | **4000** (早期から教師引きこみ) |
| kd-every | 32 (3%) | **4 (25%)** |
| LR peak | 3e-4 | **1.5e-4** |
| LR warmup | (暗黙) | **1000 step** |
| LR schedule | warmup→flat | **warmup→cosine→0** |
| max-steps | 50000 | **160000** (~10-12h 3060) |
| seq_len | 128 | 128 (teacher 90M の max_positions が 128 のため一致) |
| max-context | 32 | **40** |
| train 形式 | general/train.jsonl (2M) | **mixes/student-20m.jsonl (20M)** |
| checkpoint-every | 2000 | **1000** (160 ckpt、granular 監視) |
| eval-every | (定例 2000) | **500** (より頻繁) |
| kd-hard-threshold | 0.95 | **0.92** |
| KD decay 設計 | linear long-tail | **前寄せ (12K-32K で 0.3→0.1)** |
| num-workers | 8 | **0** (Windows multiprocessing pickle 爆発回避) |

## 中断判断 (user 監視時の目安)

- **step 0-1000**: LR warmup 期。loss 発散 (>10 持続)、blank_ratio > 0.99
  固着 → ハイパラ致命的齟齬、中断
- **step 1000-4000**: scratch 独走期 (KD off)。loss が単調減少しない or
  blank 固着 → モデル構造 / データ齟齬
- **step 4000-6000**: KD warmup 期 (α 0→0.3 漸増)。kd_loss が CTC loss 比で
  発散 → teacher-student 不整合、KD 設定見直し
- **step 6000-12000**: KD α=0.3 定常、teacher 引きこみ最大。eval 改善幅
  最大期待
- **step 12000-32000**: KD decay 期 (0.3 → 0.1)。student 自走へ遷移、eval
  plateau or 微減は正常 (過学習でなければ OK)
- **step 32000-160000**: KD α=0.1 定常の long tail。eval が plateau なら
  計画通り、monotonic に悪化したら早期終了検討
- **step 160000**: 完走。probe_v2 + cvae_probe で最終評価

## 未実装 / 後工程

- `datasets/tools/probe/run_probe_v2.py` の `--ckpt` 引数対応 (現状 factory 内 hardcode)
- CVAE 実装 (本 dry-run で CTCTeacher の train loop が動くことを先に確認)
