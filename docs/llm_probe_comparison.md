# probe_v3 — IME-specialized vs frontier LLM 比較

`datasets/eval/probe/probe.json` (348 件、7 カテゴリ) で kana→kanji 変換性能を測定し、自前 30M モデル / 既存 IME-specialized モデル (zenz / jinen) / frontier 汎用 LLM を同 bench で比較した結果。

## 評価条件

- bench: probe_v3 全 348 件
- metric: **EM1_nfkc** = top-1 出力が `expected_output` のいずれかと NFKC 正規化後に完全一致する割合
- LLM bench は `scripts/llm_probe_bench.py` 経由 (system prompt + few-shot 無し)
- thinking モデルは reasoning_content fallback で final-answer を抽出
- DeepSeek / Qwen3.6-think は context prefix を出力に含む傾向あり、**post-hoc 補正後の数値**で比較

## 総合 EM1_nfkc ランキング

| rank | model | EM1_nfkc | params | type | thinking | source |
|---:|---|---:|---:|---|---|---|
| 1 | **GLM-5.1** | **0.844**\* | 357B | LLM | yes | DeepInfra |
| 2 | **GPT-5.5** | **0.828** | unk | LLM | yes? | OpenAI chat |
| 3 | **Opus 4.7** | **0.819** | unk | LLM | yes? | Anthropic chat |
| 4 | zenz-v2.5-medium | 0.7557 | ~1.3B | IME AR | n/a | HF (local) |
| 5 | zenz-v3.1-small | 0.7213 | 91M | IME AR | n/a | HF (local) |
| 6 | zenz-v2.5-small | 0.7184 | ~310M | IME AR | n/a | HF (local) |
| 7 | zenz-v2.5-xsmall | 0.7011 | 124M | IME AR | n/a | HF (local) |
| 8 | jinen-v1-small | 0.6983 | ~230M | IME AR | n/a | HF (local) |
| 9 | **Suiko-v1-small + KenLM-MoE** | **0.678** | **30M** + LM | CTC-NAT (own) | n/a | local |
| 10 | Gemma 4 31B-it | 0.675 | 31B | LLM | n/a | DeepInfra |
| 11 | DeepSeek V4-Flash | 0.644† | 671B/37B-act | LLM | yes | DeepSeek |
| 12 | jinen-v1-xsmall | 0.6264 | ~70M | IME AR | n/a | HF (local) |
| 13 | Hatsuyume + KenLM-MoE | 0.621 | 30M + LM | CTC-NAT (own) | n/a | local |
| 14 | Qwen3.6-35B-A3B | 0.471† | 35B/3B-act | LLM MoE | yes | DeepInfra |
| 15 | Qwen3-235B-A22B-Instruct-2507 | 0.451† | 235B/22B-act | LLM MoE | no | DeepInfra |
| 16 | Qwen3.6-35B-A3B | 0.365† | 35B/3B-act | LLM MoE | no | DeepInfra |

\* GLM-5.1 は full bench 進行中 (n=230/348 時点)。final で大きくは振れない見込み
† context-prefix 出力の post-hoc strip 後

## 観察

**Frontier LLM ceiling は 0.82-0.85 帯**。GLM-5.1 / GPT-5.5 / Opus 4.7 がほぼ同点で並ぶ。

**zenz / jinen 系 (IME-specialized)**:
- zenz-v3.1-small **91M で 0.72** — frontier から ‑10pt、open-weight non-thinking 最強
- AJIMEE-bench (別 bench) では zenz-v3.1-small が 0.86、zenz-v2.5-medium が 0.875 と frontier 領域に到達 (probe_v3 の方が hard)
- param 効率突出: 91M zenz が 235B Qwen を 28pt 上回る

**汎用 LLM**:
- Gemma 4 31B (no thinking, $0.013/1k tok) **0.675** — 自前 Suiko-v1 (30M) と同点
- Qwen3 系 non-thinking は IME 特化 30M に大きく負ける (0.35-0.45)
- thinking 入れても Qwen3.6 → 0.471 で Suiko-v1 未満

**自前モデル**:
- Suiko-v1-small + KenLM-MoE **0.678** — 30M で 31B Gemma 4 と同点、open IME ライン (zenz/jinen) からは ‑4pt
- Hatsuyume (zenz KD 試行) は **退行** (0.621)、KD signal の text-roundtrip で伝達失敗

## per-category EM1_nfkc 比較 (full 348 完了モデルのみ)

|model|edge (40)|gen (75)|homo (37)|names (55)|num (65)|part (32)|tech (44)|
|---|---:|---:|---:|---:|---:|---:|---:|
|GPT-5.5|0.925|0.733|**0.784**|0.855|0.785|0.938|0.886|
|Opus 4.7|**0.950**|0.733|0.703|0.836|0.769|0.938|**0.909**|
|zenz-v3.1-small|0.775|0.667|0.487|0.818|0.677|0.875|0.795|
|Suiko-v1 KenLM-MoE|0.700|0.653|0.486|0.691|0.662|0.875|0.727|
|Gemma 4 31B|0.900|0.653|0.432|0.618|0.554|0.875|0.818|
|DeepSeek V4-Flash|0.800|0.547|0.486|0.636|0.538|0.875|0.614|
|Hatsuyume KenLM-MoE|0.725|0.627|0.351|0.582|0.615|0.844|0.636|
|Qwen3.6-35B think|0.625|0.480|0.243|0.382|0.354|0.844|0.386|
|Qwen3-235B Instruct|0.625|0.453|0.243|0.345|0.323|0.875|0.386|
|Qwen3.6-35B nothink|0.550|0.427|0.054|0.364|0.215|0.719|0.227|

## カテゴリ別の所感

- **homophone (37)** — 最難。frontier でさえ 0.70-0.78、open IME (zenz) でも 0.49、汎用 LLM no-thinking は 0.05-0.43。GPT-5.5 が 0.784 で最高
- **edge (40)** — IME 用語 / 固有名詞混じり。frontier LLM が 0.90+、IME 特化は 0.70-0.78。世界知識が効くカテゴリ
- **tech (44)** — 同様に世界知識依存。Opus 4.7 が 0.909 で最高
- **numeric (65)** — フォーマット精度。**IME 特化が LLM に勝てる稀少カテゴリ** (Suiko-v1 0.66、zenz 0.68 が GPT/Opus 0.77-0.79 と接近)。Qwen3 系 non-thinking は 0.05 まで崩壊
- **particle (32)** — 助詞選択。すべてのモデルで 0.72-0.94、上限張り付き
- **names (55)** — 固有名詞。zenz-v3.1-small が 0.818 で frontier に近い、open-weight ベンダー知識を持つモデルが強い
- **general (75)** — 通常文。frontier 0.73、IME 0.65-0.67 で安定差

## 自前モデルの位置取り

Suiko-v1-small (30M + KenLM-MoE) は:
- **総合 0.678** — 235B Qwen3-Instruct (0.451) を **+23pt** 上回り、31B Gemma 4 と同点
- **frontier (GPT-5.5 0.828) からは ‑15pt**
- **open IME SOTA (zenz-v3.1-small 0.7213) からは ‑4pt**

短期 target: zenz-v3.1-small ライン (0.72)、greedy probe_v3 EM1 > 0.65 で Suiko-v2 promote (memory `project_naming_convention`)。中期: numeric / particle で frontier 同等、homophone / tech / names で +5-10pt 改善。

## bench cost (ref)

|model|provider|cost (348 items)|notes|
|---|---|---:|---|
|GLM-5.1|DeepInfra|~$0.40|reasoning_tokens dominant|
|DeepSeek V4-Pro|DeepSeek|~$0.55|measuring|
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
