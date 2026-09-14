#!/usr/bin/env bash
# Entry 124 -- close the two grids entry 123 left open (M15, M16).  Same driver
# as entries 122/123 (Restart-Lander, D=10, full 500k budget, coordinates in the
# dump); the only things that differ are the problem and the seed.
#
# Existing cells were counted BEFORE running anything (entry 123's lesson: it
# re-ran six runs it already had and threw away nine minutes).  Present already:
#   M15: PIN01 seed0 (e115), PIN01 seed100 (e115/s1), PIN02/03 seed0 (e122)  -> 4/9
#   M16: the same four, plus PIN02/03 seed100 (e123)                         -> 6/9
# So this entry runs exactly the eight missing cells, one run per task so that
# the four cores stay evenly loaded.
#
#   analysis/mmo2024/e124/run.sh
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e124"
export PYTHONPATH="${PYTHONPATH:-}:${STUB:-/tmp/pystub}"
export RESTART_LANDER_DUMP="$OUT/descents"

run_one() {
  name="$1"; idx="$2"
  python3 scripts/niching_baseline.py --funcs "$name" --methods Restart-Lander \
    --evals-frac 1.0 --seeds 1 --seed-offset "$idx" \
    --report-rule current \
    --csv "$OUT/by_problem/${name}_s${idx}.csv" \
    > "$OUT/by_problem/${name}_s${idx}.log" 2>&1
  echo "done $name seed_index=$idx $(date -u +%H:%M:%S)"
}
export -f run_one
export OUT RESTART_LANDER_DUMP

mkdir -p "$OUT/by_problem" "$OUT/descents"
# The five M15 cells first: M15 is the problem entry 122 found the largest
# instance range on (0.34), so a run cut short should leave M15 closed, not M16.
{ echo "M15-D10-PIN02 1"; echo "M15-D10-PIN03 1"
  echo "M15-D10-PIN02 2"; echo "M15-D10-PIN03 2"
  echo "M15-D10-PIN01 2"
  echo "M16-D10-PIN01 2"; echo "M16-D10-PIN02 2"; echo "M16-D10-PIN03 2"; } \
  | xargs -P 4 -L1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
