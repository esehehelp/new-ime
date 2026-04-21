---
status: current
last_updated: 2026-04-21
---

# ベンチマーク比較 (living doc)

2026-04-21 時点の canonical モデル群と測定方針。probe_v3 (348 items,
category 付) + AJIMEE JWTD_v2 (200 items, full) が主力 bench。数値は
2026-04-21 更新後 probe_v3 に対する再測定 (注: 本ファイル head の数値表は
再 bench 実行後に埋める。手順は本節末尾参照)。

## 1. Canonical モデル群 (2026-04-21 再構成)

### 1.1 Production 主力: ctc-nat-30m-student step160000

v2 best。v1.0 ship 候補。以下のバリアントを bench 対象とする:

| variant | backend | notes |
|---|---|---|
| `__greedy` | PyTorch fp16, beam=1 | 基準精度 |
| `__kenlm` | PyTorch fp16, beam=5 + single general-LM (α=0.2 β=0.6) | 2026-04-20 canonical |
| `__kenlm-moe` | PyTorch fp16, beam=5 + 3-way MoE (general/tech/entity、α=0.2 β=0.6) | 2026-04-21 追加 |
| `__onnx-fp32-greedy` | onnxruntime CPU, fp32, beam=1 | export parity |
| `__onnx-int8-greedy` | onnxruntime CPU, int8 weight-only, beam=1 | production size/latency 検証 |

ONNX int8 + KenLM 組合わせは `tools/misc/bench_onnx_kenlm_sweep.py` で別
sweep 取得 (本ファイル 4.2 節に結果掲載)。

### 1.2 Reference (競合)

| model | params | 用途 |
|---|---|---|
| zenz-v2.5-xsmall | 30M | 同サイズ帯の主要比較 |
| zenz-v2.5-small | 91M | 3x params 帯 |
| zenz-v2.5-medium | 310M | 10x params 帯 |
| zenz-v3.1-small | 91M | 最新 91M |

全て `num_beams=5, num_return=5` で測定。

### 1.3 Upper bound

| model | params | 備考 |
|---|---|---|
| teacher-150m-teacher step200000 | 150M | AR encoder-decoder、性能上限の目安 |

### 1.4 Legacy (non-canonical, 比較目的のみ)

- `ar-31m-scratch step80000`: 旧 AR baseline
- `ctc-nat-30m-scratch step50000`: Phase 3 初期、KD なし

`--models` フィルタで再生成する用途のみ。canonical 表からは除外。

### 1.5 廃棄 (canonical から除外した記録)

| model | 除外理由 |
|---|---|
| **ctc-nat-30m-bunsetsu-v3** (2026-04-20) | 全 checkpoint (best, step 60000/70000/73000) が v2 step160000 より probe_v3 EM1 -0.086、AJIMEE -0.080 systematic 劣位。詳細 autopsy は §5 |
| **ctc-nat-90m-scratch step27500** | KenLM 付きで EM1 0.500、30M student 0.655 を下回り。step30000 は空出力で完全崩壊 |
| **ctc-nat-90m-scratch step30000** | 全入力で単文字 or 空出力を返す。checkpoint 破損と推定 |

## 2. 前提 (uniform methodology)

### 2.1 計測環境

| 項目 | 値 |
|---|---|
| OS | Windows 11 + WSL (Ubuntu) |
| Python | 3.12 |
| PyTorch | 2.11.0 **+cpu** |
| onnxruntime | 1.24.x (CPUExecutionProvider only) |
| device | **CPU only** — CUDA 不使用を canonical 条件とする |
| thread | intra_op=4 (ONNX)、torch 既定 |

### 2.2 ベンチ

| bench | path | n items | 更新 |
|---|---|---|---|
| probe_v3 | `datasets/eval/probe/probe.json` | 348 (7 categories) | **2026-04-21 更新**、以前の数値は stale |
| AJIMEE JWTD_v2 | `references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json` | 200 (full) | 2026-04-19 版固定 |

入力: katakana → `jaconv.kata2hira` で hiragana 化。正解判定: `references`
list のいずれかと完全一致 → EM1。top-5 内に含まれれば EM5。

### 2.3 デコーディング

| backend | config |
|---|---|
| CTC-NAT (PyTorch, greedy) | beam_width=1, no LM |
| CTC-NAT (PyTorch, KenLM single) | beam=5, `kenlm_general_train_4gram_probing.bin`, α=0.2, β=0.6 |
| CTC-NAT (PyTorch, KenLM MoE) | beam=5, {general / tech / entity} 3-way mixture、α=0.2 β=0.6、`CategoryEstimator` で per-input weight |
| CTC-NAT (ONNX greedy) | onnxruntime CPU, beam=1, fp32 or int8 |
| AR / AR-beam5 | length_penalty=0.6, repetition_penalty=1.2 |
| Teacher (Seq2Seq) | greedy (native) |
| zenz | num_beams=5, num_return_sequences=5, max_new_tokens=80, max_context_chars=40 |

### 2.4 指標

- **EM1**: top-1 完全一致率
- **EM5**: top-5 に正解が含まれる率
- **CharAcc**: top-1 の Levenshtein ベース文字正解率
- **p50 ms**: 1 item あたり wall-clock 中央値 (`backend.convert` 呼び出し)

### 2.5 実行

```bash
cd /mnt/d/Dev/new-ime
PYTHONPATH=. python3 tools/misc/bench_all.py
```

出力: `results/bench_all/{model}__{bench}.json` + `summary.json`。

## 3. 結果: canonical (2026-04-21 post-probe-update)

2026-04-21 probe_v3 更新版での測定。全 14 model_cfg × 2 bench、WSL CPU、35 min。

### 3.1 probe_v3 (348 items)

| model_cfg | params | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|---:|
| zenz-v2.5-medium beam=5 | 310M | **0.747** | **0.876** | **0.966** | 1173 |
| zenz-v3.1-small beam=5 | 91M | 0.718 | 0.856 | 0.959 | 417 |
| zenz-v2.5-small beam=5 | 91M | 0.713 | 0.848 | 0.959 | 376 |
| teacher-150m-teacher step200000 greedy | 150M | 0.698 | 0.698 | 0.950 | 288 |
| zenz-v2.5-xsmall beam=5 | 30M | 0.695 | 0.813 | 0.953 | 118 |
| **ctc-nat-30m-student step160000 kenlm-moe** | **30M** | **0.669** | **0.770** | **0.948** | **22** |
| ctc-nat-30m-student step160000 kenlm (single) | 30M | 0.658 | 0.759 | 0.943 | 17 |
| ctc-nat-30m-student step160000 greedy (PT) | 30M | 0.609 | 0.609 | 0.942 | **10** |
| ctc-nat-30m-student step160000 onnx-fp32 greedy | 30M | 0.609 | 0.609 | 0.942 | 23 |
| ctc-nat-30m-student step160000 onnx-int8 greedy | 30M | 0.603 | 0.603 | 0.940 | 13 |
| ar-31m-scratch step80000 beam5 | 31M | 0.575 | 0.756 | 0.899 | 281 |
| ar-31m-scratch step80000 greedy | 31M | 0.563 | 0.563 | 0.897 | 76 |
| ctc-nat-30m-scratch step50000 kenlm | 30M | 0.540 | 0.655 | 0.900 | 20 |
| ctc-nat-30m-scratch step50000 greedy | 30M | 0.417 | 0.417 | 0.884 | 11 |

### 3.2 AJIMEE JWTD_v2 (200 items, full)

| model_cfg | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|
| zenz-v2.5-medium beam=5 | **0.875** | **0.970** | **0.982** | 1361 |
| zenz-v3.1-small beam=5 | 0.860 | 0.930 | 0.983 | 470 |
| zenz-v2.5-small beam=5 | 0.840 | 0.955 | 0.977 | 418 |
| teacher-150m-teacher step200000 greedy | 0.715 | 0.715 | 0.964 | 322 |
| zenz-v2.5-xsmall beam=5 | 0.695 | 0.845 | 0.953 | 139 |
| **ctc-nat-30m-student step160000 kenlm-moe** | **0.655** | 0.810 | 0.959 | **26** |
| ctc-nat-30m-student step160000 kenlm (single) | 0.650 | 0.815 | 0.959 | 20 |
| ctc-nat-30m-student step160000 greedy (PT) | 0.550 | 0.550 | 0.949 | **11** |
| ctc-nat-30m-student step160000 onnx-fp32 greedy | 0.550 | 0.550 | 0.949 | 22 |
| ctc-nat-30m-student step160000 onnx-int8 greedy | 0.515 | 0.515 | 0.945 | 13 |
| ar-31m-scratch step80000 beam5 | 0.480 | 0.725 | 0.882 | 321 |
| ctc-nat-30m-scratch step50000 kenlm | 0.480 | 0.640 | 0.909 | 23 |
| ar-31m-scratch step80000 greedy | 0.470 | 0.470 | 0.883 | 81 |
| ctc-nat-30m-scratch step50000 greedy | 0.280 | 0.280 | 0.882 | 14 |

### 3.3 Per-category EM1 (probe_v3)

AJIMEE は category 属性を持たないため probe_v3 のみ。

| model_cfg | edge | general | homo | names | numeric | particle | tech |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctc-nat-30m-student greedy | 0.575 | 0.587 | 0.405 | 0.618 | 0.585 | 0.875 | 0.682 |
| ctc-nat-30m-student kenlm | 0.650 | 0.667 | 0.460 | 0.673 | 0.615 | 0.906 | 0.682 |
| **ctc-nat-30m-student kenlm-moe** | **0.675** | 0.667 | 0.460 | 0.673 | **0.631** | 0.906 | **0.727** |
| ctc-nat-30m-student onnx-fp32 | 0.575 | 0.587 | 0.405 | 0.618 | 0.585 | 0.875 | 0.682 |
| ctc-nat-30m-student onnx-int8 | 0.575 | 0.600 | 0.405 | 0.618 | 0.554 | 0.875 | 0.659 |
| zenz-v2.5-xsmall (30M) | 0.825 | 0.667 | 0.460 | 0.709 | 0.661 | 0.875 | 0.727 |
| zenz-v2.5-small (91M) | 0.800 | 0.680 | 0.486 | 0.727 | 0.661 | 0.906 | 0.795 |
| zenz-v2.5-medium (310M) | 0.825 | 0.707 | 0.540 | 0.818 | 0.677 | 0.875 | 0.841 |
| zenz-v3.1-small (91M) | 0.775 | 0.667 | 0.486 | 0.818 | 0.661 | 0.875 | 0.795 |
| teacher-150m-teacher greedy (150M) | 0.775 | 0.667 | 0.486 | 0.727 | 0.661 | 0.906 | 0.727 |

**v2 MoE vs zenz-xsmall (同サイズ) per-cat gap**:

| cat | v2 MoE | xsmall | Δ |
|---|---:|---:|---:|
| **edge** | 0.675 | 0.825 | **-0.150** (名寄せの差、最大 gap) |
| names | 0.673 | 0.709 | -0.036 |
| numeric | 0.631 | 0.661 | -0.030 |
| **general** | 0.667 | 0.667 | **0 (同値)** |
| tech | 0.727 | 0.727 | 0 (同値) |
| particle | 0.906 | 0.875 | **+0.031 (v2 勝ち)** |
| homophone | 0.460 | 0.460 | 0 |

**edge -0.15** が最大の欠損領域。zenz xsmall は edge category (英字/カタカナ/IT 用語等) に強く、v2 はここで大敗。v4 で edge 強化が効きそう。

### 3.4 主要所見

**v2 decoder 比較 (30M、同一 ckpt)**:
- greedy → single KenLM: probe +0.049 / AJIMEE +0.100
- single KenLM → **KenLM-MoE**: probe **+0.011** / AJIMEE **+0.005**
- fp32 ONNX = PyTorch fp16 greedy: **完全一致** (export parity 確認)
- **int8 ONNX greedy**: probe -0.006 / **AJIMEE -0.035** (numeric/tech で quantization 影響大)
  - int8 + KenLM の結果は別ランナー §4.2 参照 (LM が int8 の劣化を吸収)

**同サイズ帯 (30M) 競合比較**:
- probe: zenz-xsmall 0.695 vs v2 MoE 0.669 → **-0.026**
- AJIMEE: zenz-xsmall 0.695 vs v2 MoE 0.655 → **-0.040**
- 速度: **v2 MoE は xsmall の 5.4x 速い** (22ms vs 118ms probe)
- v1.0 位置付け: **速度一手、精度やや劣後** という profile 継続

**Upper bound (teacher)**:
- teacher-150m probe 0.698 / AJIMEE 0.715 → student とのギャップ probe 0.03 / AJIMEE 0.06
- KD teacher としての有用性あり、ただし学生に転移しきれていない

## 4. KenLM 関連の付加情報

### 4.1 KenLM 訓練 corpus 汚染チェック (2026-04-21)

`datasets/eval/general/train.jsonl` (KenLM 訓練 corpus、20.1M rows) に
probe_v3 + AJIMEE (548 items) の n-gram 混入率:

| n | hit 率 | 判定 |
|---|---|---|
| 6 | 16.32% | 偽陽性多数 (「されている。」等の generic phrase) |
| 10 | 0.18% | ほぼノイズ |
| **20** | **0.0003% (63/20M)** | 真の verbatim overlap、無視可 |

→ KenLM 訓練 corpus は **実質的に clean**。現 KenLM (`kenlm_general_train_4gram_probing.bin`) の数値は honest。

### 4.2 ONNX × KenLM / KenLM-MoE bench (2026-04-21 updated probe)

`tools/misc/bench_onnx_kenlm_moe.py` 出力。v2 step160000 の fp32/int8 ONNX 上で、
single KenLM と 3-way KenLM-MoE を比較 (α=0.2 β=0.6 beam=5):

#### probe_v3 (348 items)

| config | EM1 | EM5 | CharAcc | p50 ms | size |
|---|---|---|---|---|---|
| fp32 + KenLM single | 0.658 | 0.759 | 0.943 | 35 | 110 MB |
| int8 + KenLM single | 0.658 | 0.753 | 0.943 | 23 | 36 MB |
| **fp32 + KenLM-MoE** | **0.669** | **0.770** | **0.948** | 43 | 110 MB |
| **int8 + KenLM-MoE** (★production 候補) | **0.667** | 0.761 | 0.947 | **29** | **36 MB** |

#### AJIMEE (200 items, full)

| config | EM1 | EM5 | CharAcc | p50 ms |
|---|---|---|---|---|
| fp32 + KenLM single | 0.650 | 0.815 | 0.959 | 37 |
| int8 + KenLM single | 0.645 | **0.830** | 0.958 | 25 |
| **fp32 + KenLM-MoE** | **0.655** | 0.810 | 0.959 | 45 |
| **int8 + KenLM-MoE** | 0.645 | 0.820 | 0.958 | 32 |

#### 重要所見

- **int8 + MoE vs fp32 + MoE**: probe -0.002 / AJIMEE -0.010。**MoE の効果は int8 でも維持**
- **KenLM が int8 の量子化 noise を吸収**: greedy 単独での int8 損失 (probe -0.006 / AJIMEE -0.035) は LM 合成でほぼゼロに
- **AJIMEE EM5 で int8 single が fp32 single を上回る** (0.830 vs 0.815、+0.015)。quantization noise が beam 多様性を増やす副産物
- production 推奨: **int8 ONNX + KenLM-MoE (α=0.2 β=0.6 beam=5)** — probe 0.667、AJIMEE 0.645、p50 29ms、size 36MB

### 4.3 KenLM-MoE (3-way mixture、2026-04-21 追加)

ctc-nat-30m-student step160000 ckpt 上で single vs MoE (probe_v3、**更新前** probe):

| config | probe EM1 | probe EM5 | ajimee EM1 | ajimee EM5 |
|---|---|---|---|---|
| single general α=0.2 β=0.6 | 0.655 | 0.753 | 0.650 | 0.815 |
| **MoE α=0.2 β=0.6** | **0.664 (+0.009)** | **0.761** | 0.655 | 0.810 |
| MoE α=0.4 β=0.6 | 0.644 | 0.759 | **0.670 (+0.020)** | **0.835 (+0.020)** |

category 別では **tech +0.045** (0.682 → 0.727)、edge +0.025、names +0.018 が
有意。probe は α=0.2 が、AJIMEE は α=0.4 が最適で bench 毎に差。

## 5. 廃棄モデル autopsy

### 5.1 ctc-nat-30m-bunsetsu-v3 (2026-04-20 失敗)

| bench | config | v3 best EM1 | v2 step160k EM1 | Δ |
|---|---|---|---|---|
| probe_v3 | KenLM best (α=0.2 β=0.3) | 0.569 | 0.655 (α=0.2 β=0.6) | **-0.086** |
| probe_v3 | greedy | 0.454 | 0.603 | -0.149 |
| AJIMEE | KenLM best (α=0.4 β=0.6) | 0.570 | 0.650 | -0.080 |
| AJIMEE | greedy | 0.340 | 0.550 | **-0.210** |

probe trajectory (step): 4k 0.158 → 10k 0.259 → 20k 0.353 → 40k 0.408 →
60k/68k **peak 0.454** (greedy) → 73k 0.434。step 60000 以降 plateau、v2
超えの兆候なし。

**仮説**:
1. bunsetsu-heavy mix が names / numeric の surface 分布を薄めた
   (names -0.18, numeric -0.15 vs v2)
2. Seq2Seq teacher の confidence が高すぎ (kd_conf ≈ 0.97)、
   hard_threshold=0.85 で kd_hard=0.02-0.03 しか発火 → KD 信号が薄すぎ実質 CE 単独
3. 200M subset=60M による data variety vs v2 の 20M rows × 1.5 epoch
   での暗記効果の trade が裏目

### 5.2 ctc-nat-90m-scratch step30000 (2026-04-20 崩壊)

全入力で "を" 単文字や空出力。step27500 までは正常、step30000 で完全破綻。
原因未特定、推定される理由: optimizer state の数値不安定化 or save 途中の
ファイル破損。step27500 を 90M series の最後の有効 checkpoint として保存。

## 6. 過去スナップショット (参考値、canonical ではない)

### 6.1 probe_v3 バージョン注意

- 2026-04-19 〜 2026-04-20 測定の probe_v3 数値は旧 probe (same 348 items)
  での結果。**2026-04-21 の更新版 probe に対しては stale**
- 過去数値を比較に使う場合は「旧 probe 版」であることを明記

### 6.2 履歴 bench (docs/old/ 参照)

- 2026-04-19: probe_v2 (467 items) → probe_v3 に canonical 移行で降格
- 2026-04-18: 90M step15000 + KenLM sweep → 90M scratch 廃棄に伴い参照のみ

## 更新履歴

- **2026-04-21**: canonical モデル群を ONNX fp32/int8 + KenLM-MoE を含む
  構成に再整理。bunsetsu-v3 を廃棄へ、90M-scratch も canonical から除外。
  probe_v3 更新に伴い §3 数値表は再 bench 後に埋める
- 2026-04-20: 9 モデル × 2 bench の uniform methodology を初版 canonical 化、
  EM5 構造問題 (CTCNATBackend が beam[0] のみ返却) を surface top-5 dedup で修正、
  ctc-nat-90m-scratch step30000 崩壊を検出
- 2026-04-19: probe_v2 → probe_v3 canonical 移行、30m-student step49000 +
  teacher-150m step100000 の 3-bench 評価
- 2026-04-18: CTC-NAT 90M + KenLM sweep 初期、zenz-v2.5 3 sizes 初対比
