---
status: proposed
last_updated: 2026-04-18
---

# Corpus candidates — tech + colloquial expansion

Probe_v1 sweep pointed at two concrete gaps in the current training mix:

- tech (IT / software) 0.45 top-1 — no IT-specific corpus in phase3 mix
- general conversation 0.95 (strong) / homophone 0.50 (weak) — colloquial register thin

This doc lists permissive-license sources for each gap, ranked by
feasibility × expected impact × license cleanliness. "Feasibility" is
cost-to-ingest on our side (kana-reading + surface pairs need to be
derivable or already present).

## Technical / IT documentation

| source | license | size (raw) | how to ingest | feasibility | notes |
|---|---|---|---|---|---|
| **Wikibooks 日本語** | CC-BY-SA 3.0 | ~50MB XML | dump → mwxml → sentence split → yomi gen | high | Teaching-register tech + math content, good fit |
| **Wiktionary 日本語 例文** | CC-BY-SA 3.0 | ~200MB XML | extract example sentences from usage notes | high | Short sentences with reading hints embedded |
| **JM Project man pages (日本語)** | BSD / GPL | 数MB | clone jm.osdn.jp pages | medium | Classic Unix tech, ~2000 man pages |
| **Debian manpages ja** | 多くが GPL | 数十MB | apt source + translation files | medium | Similar to JM, more modern |
| **IETF RFC (英語)** | "IETF Trust License" (基本的に自由利用) | 200MB | https://www.rfc-editor.org/rfc/ | **low value** | 英語なので reading は無い — 日本語翻訳が必要 |
| **RFC 日本語訳 (JPNIC / IPA)** | 翻訳者ごとに異なる、JPNIC 公式は自由利用 | ~5MB | scrape jpnic.ad.jp/rfc/ | medium | 100-200 RFCs available |
| **IPA 情報セキュリティ白書** | 引用可・出典明記で利用可 | 数十MB | PDF → text | low | 政府刊行物、OCR 必要 |
| **Qiita API** | 各記事 CC-BY-SA 3.0 (作者次第)、CC-BY-SA article flag あり | 数GB (全記事) | GET /api/v2/items | **avoid** | 個別記事のライセンス確認が必要、scale しにくい |
| **Zenn** | CC 指定なし (全著作権保留が default) | 数GB | n/a | **avoid** | 記事ごと明示許諾が必要 |
| **GitHub READMEs (日本語)** | リポジトリの LICENSE 次第 (MIT/Apache/BSD ならOK) | 膨大 | gh-search by license + path | medium | フィルタが重い、リポジトリ選別が必要 |
| **Linux カーネルドキュメント ja 翻訳** | GPL | 数MB | kernel.org Documentation/translations/ja_JP | medium | 技術レジスタ文書として貴重 |

### Top-3 tech picks for immediate ingest
1. **Wikibooks 日本語** — 最大の容量、構造化済み、ライセンス明確
2. **Wiktionary 日本語** (例文部分) — 短文 + 読み情報込み
3. **JM / Linux ja 翻訳** — マニュアル調の技術レジスタ

## Colloquial / conversational

| source | license | size (raw) | how to ingest | feasibility | notes |
|---|---|---|---|---|---|
| **Wikinews 日本語** | **CC-BY 2.5 (CC-BY 4.0 for 新記事)** | ~20MB XML | dump → extract article body | **high** | News register, Wiki より口語寄り、形式明確 |
| **OpenSubtitles 2024 JP** | 字幕のライセンスは作品ごと、集約は CC-BY-SA 2.0 (opus.nlpl.eu) | 数百MB | opus-tools / HuggingFace datasets | **high** | 映画/TV の会話字幕、colloquial の王様 |
| **Tatoeba (全量)** | CC-BY 2.0 FR | 数百MB | tatoeba.org/ja-en/downloads | high | 現在 37MB サブセットのみ使用、全量化可能 |
| **JESC (Japanese-English Subtitle Corpus)** | CC-BY-SA 4.0 | 数百MB | http://nlp.stanford.edu/projects/jesc/ | high | OpenSubtitles ベース、正規化済み |
| **aozora 会話抽出** | PD | (既存から) | 既存 aozora を rule-based で dialogue line 抽出 | high | 既に手元にあるコーパスを再利用 |
| **Common Voice 日本語 (transcripts)** | CC0 | ~50MB | Mozilla Common Voice ja | medium | 音声を書き起こした colloquial 文、短い |
| **mC4 / CC-100 Japanese subsets** | Common Crawl TOS | 既存 fineweb2 / hplt3 でカバー済 | — | — | 既にあるため skip |
| **ASPEC-JE (NTCIR), NICT conversation corpora** | 各種、研究者申請必要 | — | — | low | ライセンス面倒、skip |
| **Reddit 日本語 subreddit ダンプ** | Reddit TOS | 数GB | pushshift.io (history) | **avoid** | Reddit の API ポリシー変更で商用利用不可 |
| **2ch/5ch pastes** | TOS 上 scrape 禁止 | — | — | **avoid** | 権利面で NG |
| **Yahoo 知恵袋** | TOS 制限 | — | — | **avoid** | 個別許諾 必要 |

### Top-3 colloquial picks for immediate ingest
1. **Wikinews 日本語** — clean dump, 明確 license
2. **OpenSubtitles ja** — dialogue の量圧倒的、license 扱いやすい
3. **Tatoeba 全量** — 既に一部使用中、拡大容易

## Out-of-scope (do not ingest)

- Swallow Corpus v3: license 未確認 (プラン明示 out of scope)
- CC-BY-NC/NC-ND 系: モデル配布に支障
- Reddit / Twitter / 5ch / Yahoo 知恵袋: TOS 制約
- Zenn / note.com 一般記事: 個別著作権

## Immediate action plan

Step 1 (今セッションで): 以下 3 つを fetch + extract script 作成
- `tools/corpus_v2/fetch_wikibooks.sh`
- `tools/corpus_v2/fetch_wikinews.sh`
- `tools/corpus_v2/fetch_tatoeba_full.sh`

Step 2: yomi 生成 (MeCab or existing kana pipeline)
- 既存の `tools/chunk-generator` / `tools/build-train-mix` (Rust) を流用

Step 3: phase3 混合レシピに tech 10-15% + colloquial 10% を追加、既存
chunks_main を 50% → 30% に圧縮

Step 4: next training run (local 3060 for smoke + 5090 for full)

## Note on RFC

RFC は英語原文なので単体では使えない (reading が無い)。日本語訳 RFC
を扱うには個別翻訳プロジェクト (JPNIC、IPA など) をクロールする必要が
あり、数が限られる (~200 本)。ROI 低めなので本計画では後回し。
