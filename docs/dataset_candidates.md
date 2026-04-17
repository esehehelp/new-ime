# データセット候補一覧 (ライセンス整理)

Phase 2.5 以降の学習データ拡充候補。評価は 2026-04-18 時点、プロジェクトは MIT を想定しているため **MIT 互換ライセンスのみ使用可**。CC-BY-SA / CC-BY-NC / CC-BY-ND / CDLA-Sharing は非互換。

## MIT 互換 (使用可)

| # | データセット | ライセンス | 入手先 | 概算サイズ | メモ |
|---|------------|-----------|--------|-----------|------|
| 1 | ABEJA-CC-JA | **PDDL** (Public Domain Dedication) | `kajuma/ABEJA-CC-JA` (HF), 本体は AWS Open Data Registry | 約 588 GB / 241M 行 | Common Crawl 派生、商用可、最大規模クラス |
| 2 | mC4 (ja) | **ODC-BY** | `allenai/c4` (ja config) | ja 数百 GB | Google が mT5 用に整備、古いが安定 |
| 3 | OSCAR-2301 (ja) | **CC0-1.0** (HF タグ, gated=manual 申請要) | `oscar-corpus/OSCAR-2301` | ja 数百 GB | Common Crawl ToU 遵守が前提 |
| 4 | おーぷん2ちゃんねる対話コーパス | **Apache-2.0** | https://github.com/1never/open2ch-dialogue-corpus | 約 8.14M 対話 | ネット口語、品質ばらつき大、フィルタ必要 |
| 5 | 青空文庫 (Public Domain 作品のみ) | **Public Domain** | https://www.aozora.gr.jp/ (全集 zip あり) | 約 17,000 作品 | 既に使用中。近代論説文のフィルタ抽出で追加価値 |
| 6 | Common Voice Japanese | **CC0-1.0** | Mozilla Data Collective (2025/10〜) | ja validated 223h | 音声転写テキスト部分、話し言葉 |

## 要確認 / グレー (留保)

| # | データセット | ライセンス | メモ |
|---|------------|-----------|------|
| 7 | Swallow Corpus v3 | **要確認** (コードは MIT、コーパス本体 HF 未公開) | https://github.com/swallow-llm/swallow-corpus — 公式にコーパス本体が配布されていない可能性。v2 は ~3.2 兆文字規模、v3 は更に大きい想定 |
| 8 | LLM-jp Corpus v4 | サブセット毎に異なる (v3 に Apache-2.0 タグあり) | https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4, ja 3.4 TB |
| 9 | CC-100 (ja) | **Unknown** (HF 記載) | `statmt/cc100`, ja 15 GB。明示ライセンスなし → MIT 互換主張不可 |
| 10 | 国会議事録 | 明示ライセンスなし (著作権法 30 条の 4 / 40 条 1 項で機械学習利用は広く可と解釈) | https://kokkai.ndl.go.jp/ — 個々の発言は発言者著作権、ただし政治上の演説は自由利用可 |

## MIT 非互換 (使用不可)

| # | データセット | ライセンス | 理由 |
|---|------------|-----------|------|
| 11 | 日本語日常対話コーパス | CC-BY-NC-ND 4.0 | 非商用 + 改変禁止 |
| 12 | AJIMEE JWTD (評価で使用中) | CC-BY-SA 3.0 | 評価目的の参照のみ可、学習データとしての再配布不可 |
| 13 | ReazonSpeech | CDLA-Sharing-1.0 | 著作権法 30 条の 4 限定、派生物も同ライセンス要求 |

## 推奨優先度 (現時点の判断)

1. **ABEJA-CC-JA (PDDL)** — MIT 互換で最大規模、Common Crawl 由来で多様性あり。サブセット (5-10GB) から始めるのが現実的
2. **OSCAR-2301 ja (CC0)** — 申請手続き必要だが純粋な CC0 なのでリスク最小
3. **mC4 ja (ODC-BY)** — HF 経由で最も手軽。量の割に品質管理は ABEJA/Swallow に劣る
4. **おーぷん2ちゃんねる (Apache-2.0)** — 口語カバー、サイズ中程度。NSFW/俗語フィルタ必要
5. **Common Voice (CC0)** — 話し言葉の書き起こし、小規模だが質高い

## 運用上の注意

- Common Crawl 派生 (ABEJA, mC4, OSCAR) は **Common Crawl 利用規約**にも従う必要あり
- モデルの学習データに CC-BY を含めると、モデル自体の配布時に **出典明記義務**が生じる可能性 (判例未確定)
- 多くのコーパスは生テキスト形式 → 既存パイプライン (MeCab + NEologd, chunk-generator) への投入に読み付与工程が必要
- `datasets/src/` 以下にソース種別ごとサブディレクトリを切って管理すると混乱しにくい (例: `datasets/src/abeja_cc_ja/`)

## 参考 URL

- ABEJA-CC-JA: https://huggingface.co/datasets/kajuma/ABEJA-CC-JA
- OSCAR-2301: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
- mC4 / c4: https://huggingface.co/datasets/allenai/c4
- CC-100: https://huggingface.co/datasets/statmt/cc100
- Swallow Corpus (code): https://github.com/swallow-llm/swallow-corpus
- open2ch: https://github.com/1never/open2ch-dialogue-corpus
- 日常対話: https://github.com/jqk09a/japanese-daily-dialogue
- ReazonSpeech: https://huggingface.co/datasets/reazon-research/reazonspeech
- 青空文庫: https://www.aozora.gr.jp/
- 国会議事録: https://kokkai.ndl.go.jp/
- llm-jp-corpus-v4: https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4
