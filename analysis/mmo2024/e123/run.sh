#!/usr/bin/env bash
# Entry 123 -- is the per-problem spread seed or instance?  Same driver as
# entry 122 (Restart-Lander, D=10, full 500k budget, coordinates in the dump);
# the only thing that differs is the seed.  seed 0 already exists for all three
# instances (PIN01 in e115/descents, PIN02/PIN03 in e122/descents), so this run
# adds seeds 1 and 2 only and the three together make a full 3x3 grid.
#
#   analysis/mmo2024/e123/run.sh [seeds] [seed-offset] [methods]
#
# The queue is ordered problem by problem (all 6 runs of M01, then M02, ...) so
# that a run cut short leaves COMPLETE 3x3 grids behind rather than six partial
# ones (entry 85's rule; the prereg's SD ratio needs a complete grid per
# problem, and problems are independent of each other).
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-2}"
OFF="${2:-1}"
METHODS="${3:-Restart-Lander}"
OUT="analysis/mmo2024/e123"
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
{ for p in 01 02 03 05 15 16; do
    for i in 01 02 03; do echo "M${p}-D10-PIN${i}"; done
  done; } \
  | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "=== all done $(date -u +%H:%M:%S)"
