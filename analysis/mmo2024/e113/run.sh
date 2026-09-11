#!/usr/bin/env bash
# Entry 113 -- queue 2: the breakdown of the 4.5x (MPR) / 5.7x (Score) loss to
# the memoryless null.  Same driver, budget and scorer as entries 106 / 109 /
# 110, so this pairs problem by problem with all three.
#
#   analysis/mmo2024/e113/run.sh <seeds> [seed-offset]
#
# MC-ESO-traced is bit-identical to MC-ESO at the same seed (identity check in
# identity_check.py); MCESO_HUNT_DUMP collects one row per spillover segment and
# one row per re-seed draw, mirroring entry 110's RESTART_LANDER_DUMP columns.
# One CSV per problem so a run cut short still leaves finished problems behind.
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-1}"
OFF="${2:-0}"
OUT="analysis/mmo2024/e113"
export PYTHONPATH="${PYTHONPATH:-}:${STUB:-/tmp/pystub}"
export MCESO_HUNT_DUMP="$OUT/hunts"

run_one() {
  name="$1"
  python3 scripts/niching_baseline.py --funcs "$name" --methods MC-ESO-traced \
    --evals-frac 1.0 --seeds "$SEEDS" --seed-offset "$OFF" \
    --report-rule current \
    --csv "$OUT/by_problem/${name}.csv" > "$OUT/by_problem/${name}.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
}
export -f run_one
export SEEDS OFF OUT MCESO_HUNT_DUMP

mkdir -p "$OUT/by_problem" "$OUT/hunts"
for p in $(seq -w 1 16); do echo "M${p}-D10-PIN01"; done \
  | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "=== all done $(date -u +%H:%M:%S)"
