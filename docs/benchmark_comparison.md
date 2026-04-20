---
status: current
last_updated: 2026-04-20
---

# ベンチマーク比較 (living doc)

2026-04-20 の全モデル × 2-bench 画一計測を canonical とする。過去スナップショットは
末尾「履歴」節に保存。

## 前提 (uniform methodology)

### 計測環境

| 項目 | 値 |
|---|---|
| OS | Windows 11 + WSL (Ubuntu) |
| Python | 3.12 |
| PyTorch | 2.11.0 **+cpu** (torch.cuda.is_available() == False) |
| device | CPU のみ (明示的に `device="cpu"`) |
| thread 制御 | 明示設定なし (WSL 既定) |

### ベンチ

| bench | path | n items |
|---|---|---|
| probe_v3 | `datasets/eval/probe/probe.json` | 348 (7 categories, all items) |
| AJIMEE JWTD_v2 | `references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json` | 200 (no sampling) |

- 両 bench とも full evaluation (サンプリング・切出しなし)
- 入力: katakana → `jaconv.kata2hira` で hiragana 化
- context: `context_text` フィールド (空の item はそのまま空)
- 正解判定: `references` list のいずれかと完全一致で EM1、top-5 内に含まれれば EM5

### モデル選択基準

各 checkpoint dir の **最新 step** を採用。ただし `ctc-nat-90m-scratch` のみ
**step30000 が全入力で崩壊** (空文字 / 単一字) のため、最後に正常だった step27500 を
採用。

| model_cfg | checkpoint |
|---|---|
| ctc-nat-30m-student | checkpoint_step_160000.pt |
| ctc-nat-30m-scratch | checkpoint_step_50000.pt |
| ctc-nat-90m-scratch | checkpoint_step_27500.pt (step30000 broken) |
| ar-31m-scratch | checkpoint_step_80000.pt |
| teacher-150m-teacher | checkpoint_step_200000.pt |
| zenz-v2.5-xsmall | `references/zenz-v2.5-xsmall` (HF) |
| zenz-v2.5-small | `references/zenz-v2.5-small` (HF) |
| zenz-v2.5-medium | `references/zenz-v2.5-medium` (HF) |
| zenz-v3.1-small | `references/zenz-v3.1-small` (HF) |

### デコーディング設定

| backend | config | 値 |
|---|---|---|
| CTC-NAT (greedy) | beam_width=1, no LM | - |
| CTC-NAT (kenlm) | beam_width=5, α=0.2, β=0.6, top_k_per_step=16 | KenLM: `models/kenlm/kenlm_general_train_4gram_probing.bin` |
| CTC-NAT (surface dedup) | top-5 surface (dedup) 返却 | 2026-04-20 fix: 旧 backend は beam[0] のみ返し EM5=EM1 だった |
| AR (greedy) | beam_width=1 | length_penalty=0.6, repetition_penalty=1.2 |
| AR (beam5) | beam_width=5 | 同上 |
| Teacher (AR ed) | greedy | TeacherBackend 既定 |
| zenz (GPT-2 AR) | num_beams=5, num_return_sequences=5, max_new_tokens=80, max_context_chars=40 | prompt: `INPUT + reading + [CTX+ctx] + OUTPUT` (AzooKey 準拠) |

### 指標

- **EM1**: top-1 と `references` の完全一致率
- **EM5**: top-5 のいずれかが `references` と完全一致する率
- **CharAcc**: top-1 とのレーベンシュタイン正規化文字正解率
- **p50 ms**: 1 item あたり wall-clock レイテンシ中央値 (backend.convert 呼び出し範囲)

### 実行

```bash
cd /mnt/d/Dev/new-ime
PYTHONPATH=. python3 tools/misc/bench_all.py
```

出力: `results/bench_all/{model_cfg}__{bench}.json` + `summary.json`

---

## 結果: probe_v3 (348 items)

| model_cfg | params | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|---:|
| zenz-v2.5-medium beam=5 | 310M | **0.727** | **0.862** | **0.958** | 1162 |
| zenz-v3.1-small beam=5 | 91M | 0.695 | 0.842 | 0.952 | 423 |
| zenz-v2.5-small beam=5 | 91M | 0.693 | 0.833 | 0.951 | 403 |
| teacher-150m-teacher step200000 greedy | 150M | 0.681 | 0.681 | 0.944 | 253 |
| zenz-v2.5-xsmall beam=5 | 30M | 0.675 | 0.802 | 0.946 | 120 |
| ctc-nat-30m-student step160000 kenlm | 30M | 0.655 | 0.753 | 0.940 | **17** |
| ctc-nat-30m-student step160000 greedy | 30M | 0.603 | 0.603 | 0.938 | **9** |
| ar-31m-scratch step80000 beam5 | 31M | 0.575 | 0.764 | 0.902 | 270 |
| ar-31m-scratch step80000 greedy | 31M | 0.563 | 0.563 | 0.900 | 72 |
| ctc-nat-30m-scratch step50000 kenlm | 30M | 0.543 | 0.658 | 0.903 | 17 |
| ctc-nat-90m-scratch step27500 kenlm | 90M | 0.500 | 0.618 | 0.900 | 30 |
| ctc-nat-30m-scratch step50000 greedy | 30M | 0.419 | 0.419 | 0.887 | 10 |
| ctc-nat-90m-scratch step27500 greedy | 90M | 0.365 | 0.365 | 0.887 | 22 |

### per-category EM1 (probe_v3)

| model_cfg | edge | general | homophone | names | numeric | particle | tech |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctc-nat-30m-student step160000 kenlm | 0.650 | 0.667 | 0.460 | 0.673 | 0.600 | 0.906 | 0.682 |
| ctc-nat-30m-student step160000 greedy | 0.575 | 0.587 | 0.405 | 0.618 | 0.554 | 0.875 | 0.682 |

(他モデルの per-category は `results/bench_all/{model_cfg}__probe.json` 参照)

---

## 結果: AJIMEE JWTD_v2 (200 items, full)

| model_cfg | EM1 | EM5 | CharAcc | p50 ms |
|---|---:|---:|---:|---:|
| zenz-v2.5-medium beam=5 | **0.875** | **0.970** | **0.982** | 1281 |
| zenz-v3.1-small beam=5 | 0.860 | 0.930 | 0.983 | 432 |
| zenz-v2.5-small beam=5 | 0.840 | 0.955 | 0.977 | 427 |
| teacher-150m-teacher step200000 greedy | 0.715 | 0.715 | 0.964 | 299 |
| zenz-v2.5-xsmall beam=5 | 0.695 | 0.845 | 0.953 | 131 |
| ctc-nat-30m-student step160000 kenlm | 0.650 | 0.815 | 0.959 | **22** |
| ctc-nat-30m-student step160000 greedy | 0.550 | 0.550 | 0.949 | **10** |
| ctc-nat-90m-scratch step27500 kenlm | 0.535 | 0.675 | 0.939 | 38 |
| ar-31m-scratch step80000 beam5 | 0.480 | 0.725 | 0.882 | 289 |
| ctc-nat-30m-scratch step50000 kenlm | 0.480 | 0.640 | 0.909 | 22 |
| ar-31m-scratch step80000 greedy | 0.470 | 0.470 | 0.883 | 78 |
| ctc-nat-90m-scratch step27500 greedy | 0.295 | 0.295 | 0.906 | 26 |
| ctc-nat-30m-scratch step50000 greedy | 0.280 | 0.280 | 0.882 | 10 |

---

## 付記: 2026-04-20 backend fix

- `CTCNATBackend._decode_one` は旧実装で beam 探索結果のうち `beam[0]` 1 件のみ返していた
  → `convert()` が常に長さ 1 の list を返し、CTC モデル全ての EM5 が構造的に EM1 と同値
- 本版では beam 全件を surface 化し dedup した top-5 を返す。2026-04-20 計測はすべて修正後
- 旧 backend で取得した probe_v3 の数字 (EM5=EM1 固定) と比較する場合は注意

---

## 過去スナップショット

### 2026-04-19: probe_v2 (467 項目, 7 category) — 旧 canonical

> probe_v3 (348 items, AJIMEE 互換 + category 付き長文) 導入により canonical から降格。
> probe_v2 は AJIMEE に寄せた短句中心なので直接比較不可。

| モデル | 設定 | EM1 | EM5 | p50 ms |
|---|---|---:|---:|---:|
| CTC-NAT 30m_v2 step49000 | α=0.2, β=0.6, beam=5 + KenLM | 0.779 | 0.779 | 12 |
| CTC-NAT 30m_v2 step49000 | greedy | 0.739 | 0.739 | 8 |
| teacher-150m step100000 | greedy (AR) | 0.739 | 0.739 | 40 |
| zenz-v3.1-small | beam=5 | 0.715 | 0.925 | 274 |
| zenz-v2.5-small | beam=5 | 0.700 | 0.916 | 266 |

### 2026-04-19: 3-bench (manual/ajimee-80/general-80) — 廃止

> AJIMEE を full 200 に、general は別 bench 化したため廃止。参考値として残置。

| model | manual EM | ajimee-80 EM | general-80 EM |
|---|---:|---:|---:|
| zenz-v2.5-small | 0.890 | 0.750 | 0.375 |
| teacher-150m step100000 | 0.850 | 0.588 | 0.363 |
| ctc-nat-30m-student step49000 greedy | 0.830 | 0.338 | 0.263 |
| ctc-nat-30m-student step49000 + α=0.4 β=0.6 | 0.870 | 0.575 | 0.300 |

### 2026-04-18: CTC-NAT 90M step15000 + KenLM sweep — 廃止

> 90M step15000 checkpoint は対象外 (step30000 崩壊、step27500 を現 canonical に)。

| bench | baseline EM | best EM (α=0.80 β=1.0 beam=8) |
|---|---:|---:|
| manual_test (100) | 0.700 | 0.870 |
| ajimee_jwtd (80) | 0.225 | 0.413 |
| general_dev (200) | 0.115 | 0.255 |

---

## 更新履歴

- 2026-04-20: 9 モデル × 2 bench (probe_v3 full 348 / AJIMEE full 200) の画一計測を
  canonical 化。CTCNATBackend の EM5 構造問題 (beam top-1 のみ返却) を修正、
  surface top-5 dedup 実装。ctc-nat-90m-scratch step30000 崩壊を検出
- 2026-04-19: ctc-nat-30m-student step49000 + teacher-150m step100000 の 3-bench
  評価、30m_v2 KenLM α×β sweep 実施
- 2026-04-18: CTC-NAT 90M + KenLM sweep、zenz-v2.5 3 sizes 初回対比
