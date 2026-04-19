# corpus_v2 — permissive-license tech + colloquial expansion

Per `docs/corpus_candidates_v2.md` and the probe_v1 sweep findings,
these are the raw sources and the per-source processing plan. All
sources were fetched by `fetch_all.sh`; run that first.

## What's in `datasets/raw_current/`

| file | license | next step |
|---|---|---|
| `jawikibooks-latest-pages-articles.xml` | CC-BY-SA 3.0 | mwxml extract article bodies → sentence split → yomi gen |
| `jawikinews-latest-pages-articles.xml` | CC-BY 2.5 | same as Wikibooks |
| `jawiktionary-latest-pages-articles.xml` | CC-BY-SA 3.0 | extract usage example sentences (`# 用例` etc), ignore headword tables |
| `tatoeba/sentences.csv` | CC-BY 2.0 FR | filter rows where lang == `jpn`, keep only the text column |

Dropped after license review:

- OpenSubtitles — individual subtitles retain studio / translator copyright
- Linux kernel ja — GPL output-contamination risk, tiny volume

Replacement covering the colloquial register: `datasets/v2/aozora_dialogue.jsonl`
produced by `extract_aozora_dialogue.py` (PD, 134k utterances, readings
already present).

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
| wiki_aozora | 10% | 22% | Keep core strength, slight bump |
| zenz_llmjp | 15% | 15% | Baseline quality coverage |
| fineweb2_ja / hplt3_ja | 15% | 10% | Generic web, diminishing returns |
| **wikibooks_v2** | 0% | 10% | Tech / teaching register (CC-BY-SA 3.0) |
| **wiktionary_v2** | 0% | 5% | Short usage examples, homophone coverage (CC-BY-SA 3.0) |
| **wikinews_v2** | 0% | 3% | News register, more colloquial (CC-BY 2.5) |
| **aozora_dialogue** | 0% | 7% | Colloquial 「…」 utterances (PD) |
| **tatoeba_v2** | 0% | 3% | High-quality short sentences (CC-BY 2.0 FR) |

Total new tech (wikibooks + wiktionary): 15%.
Total new colloquial (wikinews + aozora_dialogue + tatoeba): 13%.
All licenses are attribution-style (no GPL, no NC / ND, no TOS-bound sources).
