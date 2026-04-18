# corpus_v2 — permissive-license tech + colloquial expansion

Per `docs/corpus_candidates_v2.md` and the probe_v1 sweep findings,
these are the raw sources and the per-source processing plan. All
sources were fetched by `fetch_all.sh`; run that first.

## What's in `datasets/raw_v2/`

| file | license | next step |
|---|---|---|
| `jawikibooks-latest-pages-articles.xml` | CC-BY-SA 3.0 | mwxml extract article bodies → sentence split → yomi gen |
| `jawikinews-latest-pages-articles.xml` | CC-BY 2.5 | same as Wikibooks |
| `jawiktionary-latest-pages-articles.xml` | CC-BY-SA 3.0 | extract usage example sentences (`# 用例` etc), ignore headword tables |
| `tatoeba/sentences.csv` | CC-BY 2.0 FR | filter rows where lang == `jpn`, keep only the text column |
| `opensubtitles/ja.txt` | CC-BY-SA 2.0 | one sentence per line already; dedupe + drop sub numbers + drop lines < 4 chars |
| `linux_kernel_ja/` | GPL-2.0 | `.rst` / `.txt` → sentence split |

## yomi generation

None of these sources ship with phonetic readings, so every one has to
go through a yomi pass before it's usable by the CTC-NAT trainer. The
existing `wiki_clean_v3.jsonl` pipeline uses mecab+unidic — rerun the
same stages under `tools/datacore` against the v2 sources, then join
the outputs into the phase3 mix.

## proposed phase3_v2 mix (not yet wired)

Based on probe_v1 weak categories (tech 0.45, homophone 0.50, edge 0.53):

| pool | current | proposed | rationale |
|---|---|---|---|
| chunks_main / chunks_super | 60% | 25% | Over-represented, short fragments hurt sentence structure |
| wiki_aozora | 10% | 20% | Keep core strength |
| zenz_llmjp | 15% | 15% | Baseline quality coverage |
| fineweb2_ja / hplt3_ja | 15% | 10% | Generic web, diminishing returns |
| **wikibooks_v2** | 0% | 8% | Tech / teaching register |
| **wiktionary_v2** | 0% | 5% | Short usage examples, homophone coverage |
| **wikinews_v2** | 0% | 5% | News register, more colloquial |
| **opensubtitles_v2** | 0% | 8% | Dialogue, addresses the general conversation gap |
| **tatoeba_v2** | 0% | 2% | High-quality short sentences |
| **linux_kernel_ja_v2** | 0% | 2% | Pure tech docs |

Total new tech (wikibooks + wiktionary + linux): 15%.
Total new colloquial (wikinews + opensubtitles + tatoeba): 15%.
