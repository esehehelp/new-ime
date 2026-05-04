#!/usr/bin/env bash
# Run 5 ime-bench scenarios in REVERSE order, all with -v.
# Each scenario's output is moved to results/test_scenarios/sN_*/ so
# subsequent scenarios don't overwrite.
#
# -m target = suiko-v1-small-greedy
# -t target = probe_v3
#
# Compares the greedy outputs against archive/pre-v2 anchors at
# results/bench_v1_vs_v1_2/Suiko-v1-small__greedy__*.json.
#
# Picks .venv-windows or .venv-linux automatically; canonical bench
# environment is WSL (Linux) per project policy.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source "$(dirname "$0")/_uv_env.sh"
echo "[env] UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"

OUT="results/test_scenarios"
rm -rf "$OUT"
mkdir -p "$OUT"

run_scenario() {
    local label="$1"
    local out_subdir="$2"
    shift 2
    echo ""
    echo "===== [$(date +%T)] $label ====="
    echo "[cmd] ime-bench $*"
    rm -rf results/bench
    PYTHONPATH=src uv run python -m new_ime.cli.bench "$@"
    if [[ -d results/bench ]]; then
        mv results/bench "$OUT/$out_subdir"
        echo "[done] -> $OUT/$out_subdir"
    else
        echo "[warn] no results/bench produced"
    fi
}

# REVERSE order: 5 -> 1
run_scenario "Scenario 5: -v -m suiko-v1-small-greedy -t probe_v3" \
    "s5_m_and_t" \
    -v -m suiko-v1-small-greedy -t probe_v3

run_scenario "Scenario 4: -v -t probe_v3" \
    "s4_t_only" \
    -v -t probe_v3

run_scenario "Scenario 3: -v -m suiko-v1-small-greedy" \
    "s3_m_only" \
    -v -m suiko-v1-small-greedy

run_scenario "Scenario 2: -v -c configs/bench/" \
    "s2_c_explicit" \
    -v -c configs/bench/

run_scenario "Scenario 1: -v (no other args)" \
    "s1_no_args" \
    -v

echo ""
echo "===== [$(date +%T)] ALL SCENARIOS DONE ====="
echo "outputs under: $OUT"
ls -la "$OUT/" 2>&1
