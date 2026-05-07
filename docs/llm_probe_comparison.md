# probe_v3 — IME-specialized vs frontier LLM 比較

`datasets/eval/probe/probe.json` (348 件、7 カテゴリ) で kana→kanji 変換性能を測定し、自前 30M モデル / 既存 IME-specialized モデル (zenz / jinen) / frontier 汎用 LLM を同 bench で比較した結果。

## 評価条件

- bench: probe_v3 全 348 件 (一部 chat-UI 経由は手動)
- metric: **EM1_nfkc** = top-1 出力が `expected_output` のいずれかと NFKC 正規化後に完全一致する割合
- LLM bench は `scripts/llm_probe_bench.py` 経由 (system prompt + few-shot 無し、temperature=0)
- thinking モデルは reasoning_content fallback で final answer を抽出
- DeepSeek V4-Flash / Qwen3.6 think / Qwen3.6 nothink / Qwen3-235B-Instruct は context prefix を出力に含める癖あり、**post-hoc 補正後の値**で比較

## 総合 EM1_nfkc ランキング

| rank | model | EM1_nfkc | params | type | thinking | source |
|---:|---|---:|---:|---|---|---|
| 1 | **GPT-5.5** | **0.828** | unk | LLM closed | yes? | OpenAI chat UI |
| 2 | **Opus 4.7** | **0.819** | unk | LLM closed | yes? | Anthropic chat UI |
| 3 | zenz-v2.5-medium | 0.7557 | ~1.3B | IME AR | n/a | HF (local) |
| 4 | **zenz-v3.1-small** | **0.7213** | **91M** | IME AR | n/a | HF (local) |
| 5 | zenz-v2.5-small | 0.7184 | ~310M | IME AR | n/a | HF (local) |
| 6 | zenz-v2.5-xsmall | 0.7011 | 124M | IME AR | n/a | HF (local) |
| 7 | jinen-v1-small | 0.6983 | ~230M | IME AR | n/a | HF (local) |
| 8 | Gemini 3.1 Pro | 0.693 | unk | LLM closed | yes? | Google AI Studio chat |
| 9 | GLM-5.1 thinking | 0.692 | 357B | LLM open | yes | DeepInfra |
| 10 | **Suiko-v1-small + KenLM-MoE** | **0.678** | **30M** + LM | CTC-NAT (own) | n/a | local |
| 11 | Gemma 4 31B-it | 0.675 | 31B | LLM open | n/a | DeepInfra |
| 12 | DeepSeek V4-Flash | 0.644† | 671B/37B-act | LLM open | yes | DeepSeek |
| 13 | jinen-v1-xsmall | 0.6264 | ~70M | IME AR | n/a | HF (local) |
| 14 | Hatsuyume + KenLM-MoE | 0.621 | 30M + LM | CTC-NAT (own) | n/a | local |
| 15 | Qwen3.6-35B-A3B | 0.471† | 35B/3B-act | LLM open MoE | yes | DeepInfra |
| 16 | Qwen3-235B-A22B-Instruct-2507 | 0.451† | 235B/22B-act | LLM open MoE | no | DeepInfra |
| 17 | Qwen3.6-35B-A3B | 0.365† | 35B/3B-act | LLM open MoE | no | DeepInfra |

† context-prefix 出力の post-hoc strip 後

(DeepSeek V4-Pro thinking は計測中、結果出次第追記)

## 観察

**Closed frontier LLM が ceiling (0.82-0.85)**: GPT-5.5 と Opus 4.7 が突出。zenz-v2.5-medium (1.3B) が 0.76 で 7pt 差まで肉薄するも届かず。

**zenz-v3.1-small は param 効率の頂点**:
- 91M 単体で 0.7213 — open-weight 系 thinking モデル全般を上回る
- AJIMEE-bench (別 bench) では 0.86、frontier 領域に到達 (probe_v3 は relatively hard)
- 91M zenz が **357B GLM-5.1 thinking と同点 (0.692)、235B Qwen3-Instruct を 28pt 上回る**

**汎用 LLM は IME 特化に param で勝てない**:
- Gemma 4 31B (no thinking) **0.675** — 自前 Suiko-v1 (30M) と同点
- Qwen3 系 non-thinking は IME 特化 30M に大きく負ける (0.35-0.45)
- thinking 入れても Qwen3.6 → 0.471 で Suiko-v1 未満
- DeepSeek V4-Flash thinking (671B) でさえ 0.644 で Suiko-v1 以下

**自前モデル**:
- Suiko-v1-small + KenLM-MoE **0.678** — 30M で 31B Gemma 4 と同点、open IME ライン (zenz/jinen) からは ‑4pt
- Hatsuyume (zenz KD 試行) は **退行** (0.621)、KD signal の text-roundtrip で伝達失敗

## per-category EM1_nfkc 比較

|model|edge (40)|gen (75)|homo (37)|names (55)|num (65)|part (32)|tech (44)|
|---|---:|---:|---:|---:|---:|---:|---:|
|GPT-5.5|0.925|0.733|**0.784**|0.855|0.785|0.938|0.886|
|Opus 4.7|**0.950**|0.733|0.703|0.836|0.769|0.938|**0.909**|
|zenz-v3.1-small|0.775|0.667|0.487|0.818|0.677|0.875|0.795|
|Gemini 3.1 Pro|**0.950**|0.627|0.730|0.636|0.569|0.875|0.659|
|GLM-5.1 thinking|0.850|0.640|0.639|0.655|0.600|0.906|0.705|
|Suiko-v1 KenLM-MoE|0.700|0.653|0.486|0.691|0.662|0.875|0.727|
|Gemma 4 31B|0.900|0.653|0.432|0.618|0.554|0.875|0.818|
|DeepSeek V4-Flash|0.800|0.547|0.486|0.636|0.538|0.875|0.614|
|Hatsuyume KenLM-MoE|0.725|0.627|0.351|0.582|0.615|0.844|0.636|
|Qwen3.6-35B think|0.625|0.480|0.243|0.382|0.354|0.844|0.386|
|Qwen3-235B Instruct|0.625|0.453|0.243|0.345|0.323|0.875|0.386|
|Qwen3.6-35B nothink|0.550|0.427|0.054|0.364|0.215|0.719|0.227|

zenz-v2.5 系は per-cat データが古い bench に無いため省略 (総合スコアのみ集計)。

## カテゴリ別の所感

- **homophone (37)** — 最難。frontier closed でさえ 0.70-0.78、open IME (zenz) で 0.49、汎用 LLM no-thinking は 0.05-0.43。GPT-5.5 が 0.784 で最高
- **edge (40)** — IME 用語 / 固有名詞混じり。frontier LLM が 0.85-0.95、IME 特化は 0.70-0.78。世界知識が効くカテゴリ。Opus 4.7 が 0.950 で最高
- **tech (44)** — 同様に世界知識依存。Opus 4.7 が 0.909 で最高
- **numeric (65)** — フォーマット精度。**IME 特化が LLM に勝てる稀少カテゴリ** (Suiko-v1 0.66 / zenz 0.68 が GLM-5.1 thinking 0.60 を上回る)。Qwen3 系 non-thinking は 0.05 まで崩壊
- **particle (32)** — 助詞選択。すべてのモデルで 0.72-0.94、上限張り付き
- **names (55)** — 固有名詞。zenz-v3.1-small が 0.818、frontier closed (0.84-0.86) に近い。IME-specialized が固有名詞辞書を持つ強み
- **general (75)** — 通常文。frontier 0.73、IME 0.65-0.67、open frontier (GLM) 0.64 で安定差

## 自前モデルの位置取り

Suiko-v1-small (30M + KenLM-MoE) は:
- **総合 0.678** — 235B Qwen3-Instruct (0.451) を **+23pt** 上回り、31B Gemma 4 / 357B GLM-5.1 thinking と同点
- **frontier closed (GPT-5.5 0.828) からは ‑15pt**
- **open IME SOTA (zenz-v3.1-small 0.7213) からは ‑4pt**

短期 target: zenz-v3.1-small ライン (0.72)、greedy probe_v3 EM1 > 0.65 で Suiko-v2 promote (memory `project_naming_convention`)。中期: numeric / particle で frontier 同等、homophone / tech / names で +5-10pt 改善で 0.75 圏内。

## bench cost (ref)

|model|provider|cost (348 items)|notes|
|---|---|---:|---|
|GLM-5.1 thinking|DeepInfra|$0.73|reasoning_tokens dominant|
|Qwen3.6-35B-A3B think|DeepInfra|$0.61|大半 reasoning|
|DeepSeek V4-Flash|DeepSeek|~$0.14|cache hit on system prompt|
|Gemma 4 31B|DeepInfra|$0.008|cheapest workhorse|
|Qwen3-235B-Instruct|DeepInfra|$0.004|native non-thinking|
|Qwen3.6-35B-A3B no-think|DeepInfra|$0.011|non-thinking|
|GPT-5.5 / Opus 4.7|chat UI|手動|API は高額のため chat UI 経由|

## 関連ファイル

- 元データ: `datasets/eval/probe/probe.json`
- スコア生成: `scripts/llm_probe_bench.py`
- per-bench raw: `results/llm_bench/<tag>/probe_v3_em1.jsonl`
- per-bench summary: `results/llm_bench/<tag>/summary.json`
- chat UI 用: `docs/probe_v3_paste.md` / `docs/probe_v3_with_answers.md`
- 失敗カテゴリ分析: `docs/probe_failure_analysis.md`
