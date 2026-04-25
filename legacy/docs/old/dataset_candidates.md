---
status: current
last_updated: 2026-04-18
---

# データセット候補一覧 (ライセンス整理)

Phase 2.5 以降の学習データ拡充候補。

**方針 (2026-04-18 改訂)**: 本プロジェクトは研究プロトタイプ検証に舵を切った結果、
当初の「MIT 互換のみ」制約を緩和し、**コード (MIT) と成果物 (モデル重み・混合 JSONL・
蒸留出力) のライセンスを分離** する。成果物は `MODEL_LICENSE` (CC BY-SA 4.0) の下で管理し、
ShareAlike 系 (Wikipedia 派生) ソースも学習混合に投入可能。詳細は `DATA_LICENSES.md` と
`ATTRIBUTION.md` 参照。

その上でライセンスは 3 段階で扱う:

- **寛容系** (MIT/Apache-2.0/BSD/CC-BY/ODC-BY/CC0/PDDL/PD/政府標準利用規約2.0):
  成果物側に追加制約をかけずに混合可能。attribution 義務のみ。
- **ShareAlike 系** (CC-BY-SA 3.0/4.0, CDLA-Sharing): 混合可能だが派生成果物は
  CC-BY-SA (またはそれ以上の互換) での配布に縛られる。`MODEL_LICENSE` の既定と整合。
- **非商用・改変禁止系** (CC-BY-NC, CC-BY-ND): 研究プロトタイプでも採用しない。

調査日: 2026-04-18。HF dataset ID, 概算サイズ, 用途的価値を記載。

## 寛容系 (追加制約なしで混合可) — 大規模 web 系

| # | データセット | ライセンス | 入手先 | ja サイズ | 用途的価値 |
|---|------------|-----------|--------|-----------|------|
| 1 | **CulturaX (ja)** | mC4 (ODC-BY) + OSCAR (CC0) | HF: `uonlp/CulturaX` | 約 111M docs / **107.8B tokens** | フィルタ済みCC、ABEJAより質高い可能性、ライセンス2系統明示 |
| 2 | **HPLT v2 (ja)** | **CC0-1.0** | HF: `HPLT/HPLT2.0_cleaned`, `HPLT/HPLT3.0` | 約 418M docs / 23.3B segments / 901B chars | 完全PD、リスク最小、HPLT3.0 が最新 |
| 3 | **FineWeb-2 (jpn_Jpan)** | **ODC-By v1.0** | HF: `HuggingFaceFW/fineweb-2` (config: `jpn_Jpan`) | 約 400M docs / 331B words / 1.50TB UTF-8 | HF最新の品質フィルタCC派生 |
| 4 | **ABEJA-CC-JA** | **PDDL** (Public Domain) | HF: `kajuma/ABEJA-CC-JA`, AWS Open Data Registry | 約 588 GB / 241M 行 | Common Crawl 派生、商用可、最大規模クラス |
| 5 | mC4 (ja) | **ODC-BY** | HF: `allenai/c4` (ja config) | ja 数百 GB | Google が mT5 用に整備、CulturaX/FineWeb-2 の元 |
| 6 | OSCAR-2301 (ja) | **CC0-1.0** (gated=manual 申請要) | HF: `oscar-corpus/OSCAR-2301` | ja 数百 GB | Common Crawl ToU 遵守が前提 |

## 寛容系 — 既存パイプライン乗せやすい

| # | データセット | ライセンス | 入手先 | サイズ | 用途的価値 |
|---|------------|-----------|--------|--------|------|
| 7 | **zenz-v2.5-dataset / `train_llm-jp-corpus-v3.jsonl` サブセット** | **ODC-BY** (リポ全体は CC-BY-SA 4.0、サブセット由来は ODC-BY) | HF: `Miwa-Keita/zenz-v2.5-dataset` | **32.4 GB** | 既に `(input, output, left_context)` 形式に整形済み、即座に学習投入可。サブセット単独 ODC-BY 解釈。慎重なら△扱い |
| 8 | おーぷん2ちゃんねる対話コーパス | **Apache-2.0** | https://github.com/1never/open2ch-dialogue-corpus | 約 8.14M 対話 | ネット口語、品質ばらつき大、フィルタ必要 |
| 9 | 青空文庫 (PD作品のみ) | **Public Domain** | https://www.aozora.gr.jp/ | 約 17,000 作品 | 既に使用中、近代論説文の追加抽出に価値 |
| 10 | Common Voice Japanese | **CC0-1.0** | Mozilla Data Collective (2025/10〜) | ja validated 223h | 音声転写テキスト、話し言葉 |

## 寛容系 — 中小規模・特殊文体

| # | データセット | ライセンス | 入手先 | サイズ | 用途的価値 |
|---|------------|-----------|--------|--------|------|
| 11 | **Wikinews (ja)** | **CC-BY 2.5** | Wikimedia dump: `dumps.wikimedia.org/jawikinews/latest/` | 圧縮 9.9MB / 展開数十-100MB級 | ニュース文体、量は少ないが質高い |
| 12 | **政府白書 (経済財政・防衛・厚労等)** | **政府標準利用規約 第2.0版** (CC-BY 4.0 互換) | e-Gov: `e-gov.go.jp/about-government/white-papers.html`, 各府省 | 年次合計で数十-数百MB級 | 硬めビジネス文体、現状欠落分野。第三者図表は要確認 |
| 13 | **国会会議録** | **著作権法 40 条 1 項** (政治上の演説は自由利用可、機械学習目的含む) | https://kokkai.ndl.go.jp/ (API あり) | 1947年〜現在で **数十GB級** | 政治演説の口語+硬文。「同一著作者の編集」(特定議員集) は適用外、雑多な発言の集合は問題なし |

## 要確認 / 留保

| # | データセット | ライセンス | メモ |
|---|------------|-----------|------|
| 14 | Swallow Corpus v3 | **要確認** (コードは MIT、コーパス本体 HF 未公開) | https://github.com/swallow-llm/swallow-corpus — 公式にコーパス本体が配布されていない可能性。v2 は ~3.2 兆文字規模 |
| 15 | LLM-jp Corpus v4 | サブセット毎に異なる (v3 に Apache-2.0 タグあり) | https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4, ja 3.4 TB |
| 16 | CC-100 (ja) | **Unknown** (HF 記載) | `statmt/cc100`, ja 15 GB。明示ライセンスなし → MIT 互換主張不可 |

## ShareAlike 系 (混合可、成果物は CC-BY-SA)

既存の自前 Wikipedia 抽出パイプラインも本カテゴリ。成果物は `MODEL_LICENSE` の範囲で配布する前提。

| # | データセット | ライセンス | 理由・メモ |
|---|------------|-----------|------|
| 17 | **Wiki40B (ja)** | **CC-BY-SA 3.0** (原典 Wikipedia 由来) | 自前 Wikipedia 抽出と重複多いので優先度低。採用時は attribution を `ATTRIBUTION.md` に追記 |
| 18 | zenz-v2.5-dataset / `train_wikipedia.jsonl` サブセット | **CC-BY-SA 4.0** | Wikipedia 由来。混合する場合は自前 wiki 抽出との重複で水増しにならないよう dedup 要 |
| 20 | AJIMEE JWTD (評価で使用中) | **CC-BY-SA 3.0** | 学習投入は**テスト汚染リスクで禁止**。評価専用で使う |
| 21 | ReazonSpeech | **CDLA-Sharing-1.0** | 著作権法 30 条の 4 限定、派生物も同ライセンス要求。音声転写が必要な場合のみ |

## 非採用 (非商用・改変禁止)

| # | データセット | ライセンス | 理由 |
|---|------------|-----------|------|
| 19 | 日本語日常対話コーパス | **CC-BY-NC-ND 4.0** | 非商用 + 改変禁止。研究プロトタイプでも採用しない |

## 推奨優先度 (現時点の判断)

### 即着手すべき (寛容系)
1. **zenz-v2.5-dataset の `train_llm-jp-corpus-v3.jsonl` (32.4GB, ODC-BY)** — **すでに kana-kanji 形式に整形済み**、最小工数で最大効果。サブセット単独は ODC-BY 解釈 (ATTRIBUTION.md の慎重運用を参照)
2. **CulturaX (ja) または FineWeb-2 (jpn_Jpan)** — 高品質フィルタ済み CC、サブセット (5-20GB) で開始
3. **HPLT v2 / v3 (ja)** — CC0 で完全クリーン、リスクゼロ

### 多様性追加
4. **おーぷん2ちゃんねる (Apache-2.0)** — 口語カバー
5. **政府白書 (政府標準利用規約2.0)** — 硬文/ビジネス文体カバー
6. **国会会議録 (著作権法40条1項)** — 政治演説、硬文

### 後回しで良い
7. **ABEJA-CC-JA (588GB, PDDL)** — 規模あるが CulturaX/FineWeb-2 と重複多い、優先度下げ
8. **Wikinews ja (CC-BY 2.5)** — 量少ないが追加コスト低い
9. **OSCAR-2301 (CC0)** — gated 申請の手間 vs 同種代替あり、優先度下げ

## 運用上の注意

- **Common Crawl 派生** (CulturaX, FineWeb-2, ABEJA, mC4, OSCAR) は **Common Crawl 利用規約**にも従う必要あり
- **CC-BY 系** は配布時に **出典明記義務** が生じる (CulturaX, Wikinews, 政府白書)。zenz-llm-jp サブセット (ODC-BY) も attribution 必要
- **ShareAlike 系** (自前 Wikipedia 抽出、Wiki40B、zenz の `train_wikipedia.jsonl`) を混合した成果物は `MODEL_LICENSE` (CC BY-SA 4.0) 配布に縛られる
- 多くのコーパスは生テキスト → 既存パイプライン (MeCab + NEologd, Rust chunk-generator) への投入に **読み付与工程** が必要
  - ただし zenz-v2.5-dataset は既に `(input, output, left_context)` 形式 → そのまま使える
- `datasets/src/` 以下にソース別サブディレクトリを切る (例: `datasets/src/zenz_llmjp/`, `datasets/src/culturax_ja/`)
- **テスト汚染防止**: AJIMEE / 自前 eval_v3 と語彙的に重複するソースを避けるか、後処理で除外

## 参考 URL

- CulturaX: https://huggingface.co/datasets/uonlp/CulturaX
- HPLT v2: https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned (v3: `HPLT/HPLT3.0`)
- FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- ABEJA-CC-JA: https://huggingface.co/datasets/kajuma/ABEJA-CC-JA
- OSCAR-2301: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
- mC4 / c4: https://huggingface.co/datasets/allenai/c4
- CC-100: https://huggingface.co/datasets/statmt/cc100
- Swallow Corpus (code): https://github.com/swallow-llm/swallow-corpus
- llm-jp-corpus-v4: https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4
- zenz-v2.5-dataset: https://huggingface.co/datasets/Miwa-Keita/zenz-v2.5-dataset
- open2ch: https://github.com/1never/open2ch-dialogue-corpus
- 日常対話: https://github.com/jqk09a/japanese-daily-dialogue
- ReazonSpeech: https://huggingface.co/datasets/reazon-research/reazonspeech
- 青空文庫: https://www.aozora.gr.jp/
- Wikinews: https://dumps.wikimedia.org/jawikinews/latest/
- 国会議事録: https://kokkai.ndl.go.jp/
- 政府白書ポータル: https://www.e-gov.go.jp/about-government/white-papers.html
- Common Voice: https://commonvoice.mozilla.org/
