#!/usr/bin/env bash
# Build student-v1.13-500m.kkc with the char-jis-24k tokenizer.
#
# v1.13 = v1.7 mix recipe (chunks/zenz/wiki/bunsetsu/fineweb2/hplt/synth)
#         + new wider tokenizer (char-jis-24k, vocab=24224 = char + JIS全 +
#         ASCII救済 + 記号 block 拡張). Mix args are intentionally identical
#         to v1.7 so the only changed variable is the tokenizer; that lets
#         training-time deltas be attributed to vocab expansion alone.
#
# Reproducibility:
#   - All data-mix args (ratios, seed, contamination refs) are pinned below.
#   - Source corpus paths use the data-mix defaults; corpus files live under
#     datasets/corpus/{legacy,bunsetsu,synth}.
#   - Output build log is committed to git alongside the .meta.json sidecar.

set -euo pipefail

OUTDIR="datasets/mixes"
NAME="student-v1.13-500m"
TOKENIZER="datasets/tokenizers/char-jis-24k.json"

JSONL_PATH="${OUTDIR}/${NAME}.jsonl"
SHARD_PATH="${OUTDIR}/${NAME}.kkc"
LOG_PATH="${OUTDIR}/${NAME}.build.log"

mkdir -p "${OUTDIR}"

echo "=== Stage 1: data-mix → ${JSONL_PATH}" | tee "${LOG_PATH}"
./build/release/data-mix.exe \
  --output "${JSONL_PATH}" \
  --total 500000000 \
  --ratio-chunks 0.20 \
  --ratio-zenz 0.15 \
  --ratio-wiki 0.15 \
  --ratio-bunsetsu 0.20 \
  --ratio-fineweb2 0.10 \
  --ratio-hplt 0.10 \
  --ratio-synth 0.10 \
  --filter-chunks \
  --filter-wiki \
  --filter-bunsetsu \
  --seed 42 \
  2>&1 | tee -a "${LOG_PATH}"

echo "=== Stage 2: rust-data compile → ${SHARD_PATH}" | tee -a "${LOG_PATH}"
./build/release/rust-data.exe compile \
  --input "${JSONL_PATH}" \
  --output "${SHARD_PATH}" \
  --tokenizer "${TOKENIZER}" \
  --max-context-chars 40 \
  --max-reading-tokens 128 \
  --max-surface-tokens 128 \
  2>&1 | tee -a "${LOG_PATH}"

echo "=== Stage 3: cleanup intermediate JSONL" | tee -a "${LOG_PATH}"
rm -f "${JSONL_PATH}"
echo "Done." | tee -a "${LOG_PATH}"
ls -la "${SHARD_PATH}"* | tee -a "${LOG_PATH}"
