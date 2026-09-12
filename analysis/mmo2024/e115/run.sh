#!/usr/bin/env bash
# Entry 115 -- re-run of entry 110 seed 0 with landing COORDINATES in the dump.
# estimate.  Same driver, same budget, same scorer as entries 106 (MC-ESO) and
# 109 (NMMSO), so the three are paired problem by problem.
#
#   analysis/mmo2024/e115/run.sh <seeds> [seed-offset] [methods]
#
# One CSV per problem so a run that is cut short still leaves finished problems
# behind (entry 85's rule).  Sharded 4-wide, as entries 106/107.
# RESTART_LANDER_DUMP collects one row per descent (landing, best_f, evals, CMA
# stop reason) -- that dump is what separates "the estimate was right" from
# "the estimate was right for the wrong reason".
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-1}"
OFF="${2:-0}"
METHODS="${3:-Restart-Lander}"
OUT="analysis/mmo2024/e115"
export PYTHONPATH="${PYTHONPATH:-}:${STUB:-/tmp/pystub}"
export RESTART_LANDER_DUMP="$OUT/descents"

run_one() {
  name="$1"
  python3 scripts/niching_baseline.py --funcs "$name" --methods "$METHODS" \
    --evals-frac 1.0 --seeds "$SEEDS" --seed-offset "$OFF" \
    --report-rule current \
    --csv "$OUT/by_problem/${name}.csv" > "$OUT/by_problem/${name}.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
}
export -f run_one
export SEEDS OFF METHODS OUT RESTART_LANDER_DUMP

mkdir -p "$OUT/by_problem" "$OUT/descents"
for p in $(seq -w 1 16); do echo "M${p}-D10-PIN01"; done \
  | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "=== all done $(date -u +%H:%M:%S)"
