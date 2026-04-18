---
status: current
last_updated: 2026-04-18
---

# α/β sweep results on probe_v1 — dict integration post-mortem

## TL;DR

The longest-prefix-match dict architecture — even with kanji-ratio
filters and cost thresholds — **hurts accuracy in every category** on
the 116-item multi-domain probe set. CTC-NAT + KenLM with the dict
completely disabled scored **macro top-1 0.710** at α=0.6 β=0.6;
adding just `user_dict.tsv` dropped it to 0.568; adding
`fixed_dict_mozc.tsv + fixed_dict_mozc_ut.tsv` collapsed it to 0.324.

The intuition that mozc-ut would rescue medical / legal / named-entity
coverage is **wrong** on these probes: the model + LM already gets 87 %
of medical and 100 % of names right without any dict. Every dict tier
drops those numbers. Even homophone disambiguation — the original
motivation for the dict — is worse with dict than without (0.50 → 0.25).

The α/β sweep used macro-average top-1 EM over categories so the
dominant category doesn't mask per-domain damage.

## Configs swept (probe_v1, 116 items, 7 categories)

Model: `ctc_nat_30m_best_latest.onnx` (step 28000).
LM: `kenlm_eval_v3_train_4gram_probing.bin` (4-gram on eval_v3/train).

| config | α | β | macro top-1 | macro top-3 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| **no dict** | 0.60 | 0.60 | **0.710** | **0.764** | 29 | 35 |
| user dict only | 0.60 | 0.60 | 0.568 | 0.636 | 38 | 74 |
| user + mozc + mozc-ut | 0.40 | 0.30 | 0.324 | 0.380 | 28 | 75 |

## Per-category comparison at each config's best α/β

| category | n | no dict | user dict | full dict |
|---|---|---|---|---|
| general | 20 | **0.950** | 0.750 | 0.050 |
| tech | 20 | **0.450** | 0.250 | 0.100 |
| medical | 15 | **0.867** | 0.533 | 0.600 |
| legal | 15 | **0.667** | 0.600 | 0.267 |
| names | 15 | **1.000** | 0.933 | 0.600 |
| homophone | 16 | **0.500** | 0.375 | 0.250 |
| edge | 15 | 0.533 | 0.533 | 0.400 |

Every category's best result is in the **no-dict** column.

## Why the dict hurts

The longest-prefix match greedily picks the longest reading that
happens to match the current kana position. For a 1.4M-entry UT dict
that includes Wikipedia titles like `鏡背` (きょうはい) and `吉都`
(きっと), common kana sequences collide with rare compound titles:

- `きょうはいいてんきだ` → `鏡背移転キダ` (`きょうはい` → 鏡背 from UT)
- `あしたもきっとはれる` → `明日も吉都貼れる` (`きっと` → 吉都)
- `ともだちとしょくじした` → `友達塗色字した` (`とし` → 塗色)

Once the prefix locks in a wrong long match, the rest of the sentence
is stitched from leftover fragments and the LM-contextual rescoring
can't undo the damage because all candidates are derived from the same
broken segmentation.

The `--min-kanji-ratio` and `--max-kana-len` filters removed the
obvious mojibake but not the compound-title class of failure, because
those entries are 100 % kanji and 6–10 chars in kana.

## Implications for corpus / architecture strategy

1. **Don't stack dicts on top of the model.** The CTC-NAT encoder plus
   KenLM 4-gram fusion already handles the common path better than any
   longest-prefix dict we can assemble. The dict's role should be
   *residual*, not primary.

2. **Dict lookup should be gated by model confidence.** The current
   architecture consults the dict unconditionally at every kana-run
   position. A useful dict layer would only fire when the model's
   top-1 has low logp (or when CTC + LM top-K collapse to the same
   surface, meaning the model is uncertain about the kanji choice).

3. **mozc / mozc-ut can't be imported wholesale.** ~1.4M entries in a
   single longest-prefix table degrade the mean case more than they
   lift the long tail. If we want the rare-term coverage, we need a
   different shape — e.g. mecab-style segmentation to pick the right
   dict entry given surrounding context.

4. **CVAE adaptation is no substitute for this fix.** CVAE tunes the
   model's output distribution per user / domain; it doesn't help if
   a dict layer sitting above the model is outputting 鏡背 regardless
   of what the model would have said.

5. **Corpus priority order revised.** Earlier plan was to grow corpus
   diversity to fill dict gaps. The sweep shows the model already
   covers general / tech / medical / legal / names to 45–100 %
   without any dict. Training-time investment should go into the
   categories that stay weak even without dict interference — tech
   45 %, homophone 50 %, edge 53 % — rather than into loading more
   dict entries to compensate.

## Data for reproducibility

- Sweep driver: `scripts/sweep_interactive_ctc.py`
- Probe set: `datasets/probe_v1/probe.tsv`
- Raw per-config JSONs: `results/sweep_*/`
- Sweep grids this run:
  - full dict: `results/sweep_20260418_170409/`
  - user dict: `results/sweep_user_only/`
  - no dict:   `results/sweep_nodict/`
