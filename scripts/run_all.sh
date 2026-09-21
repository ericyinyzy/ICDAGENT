#!/usr/bin/env bash
# Full K=2 ICDAGENT pipeline over one set of clinical notes.
#
#   bash scripts/run_all.sh <notes.json> <benchmark> <output_dir>
#
# e.g. bash scripts/run_all.sh notes.json mimic4-icd9 runs/icd9
#
# Each stage runs as its own process: R_code and R_crit both take 90% of GPU
# memory, so they cannot be co-resident.
set -euo pipefail

NOTES=${1:?usage: run_all.sh <notes.json> <benchmark> <output_dir>}
BENCH=${2:?usage: run_all.sh <notes.json> <benchmark> <output_dir>}
OUT=${3:?usage: run_all.sh <notes.json> <benchmark> <output_dir>}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$OUT"

python "$ROOT/scripts/stage1_coding.py" \
    --notes "$NOTES" --benchmark "$BENCH" --output "$OUT/coding_round1.json"

python "$ROOT/scripts/stage2_critic.py" \
    --coding "$OUT/coding_round1.json" --benchmark "$BENCH" \
    --output "$OUT/critic_round1.json" --rejected "$OUT/rejected_round1.json"

python "$ROOT/scripts/stage3_reselect.py" \
    --rejected "$OUT/rejected_round1.json" --benchmark "$BENCH" \
    --output "$OUT/coding_round2.json"

python "$ROOT/scripts/stage2_critic.py" \
    --coding "$OUT/coding_round2.json" --benchmark "$BENCH" \
    --output "$OUT/critic_round2.json" --rejected "$OUT/rejected_round2.json"

python "$ROOT/scripts/merge_rounds.py" \
    --rounds "$OUT/critic_round1.json" "$OUT/critic_round2.json" \
    --output "$OUT/predictions.json"

python "$ROOT/scripts/evaluate.py" \
    --predictions "$OUT/predictions.json" --benchmark "$BENCH"
