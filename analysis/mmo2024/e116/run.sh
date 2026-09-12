#!/usr/bin/env bash
# Entry 116 -- queue 1's next step: give MC-ESO and NMMSO the SAME legal
# reporting rule entry 115 found for the null, and re-read the ranking.
#
#   analysis/mmo2024/e116/run.sh [seeds] [seed-offset]
#
# Same driver, same budget (--evals-frac 1.0 = the suite's own floor(50000*D)),
# same problems and same seed 0 as entries 106/109/110/115, so the three methods
# pair problem by problem.  REPORT_SET_DUMP writes each run's *uncapped*
# reported set (f + coordinates) -- the candidate set a reporting rule chooses
# from.  No oracle column: nearest-optimum attribution is recomputed at analysis
# time and used for scoring only (entry 115's lesson).
#
# One CSV per problem so a run that is cut short still leaves finished problems
# behind (entry 85's rule).  The two methods are interleaved by problem in one
# 4-wide pool, so a cut-short run leaves *matched pairs* rather than all of one
# method -- the comparison is paired, so a matched prefix is analysable and an
# unmatched one is not.
#
# MC-ESO defaults are untouched: this adds a dump and changes nothing else.
#
# After the run, by_problem/ was folded into by_problem.csv and its .log files
# deleted (retention rule: a .log is not evidence, and 32 one-row CSVs are file
# count for nothing).  Re-running this recreates the directory; fold it again.
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-1}"
OFF="${2:-0}"
OUT="analysis/mmo2024/e116"
export REPORT_SET_DUMP="$OUT/dumps"

run_one() {
  name="${1%%:*}"; meth="${1##*:}"
  python3 scripts/niching_baseline.py --funcs "$name" --methods "$meth" \
    --evals-frac 1.0 --seeds "$SEEDS" --seed-offset "$OFF" \
    --report-rule current \
    --csv "$OUT/by_problem/${name}_${meth}.csv" \
    > "$OUT/by_problem/${name}_${meth}.log" 2>&1
  echo "done $name $meth $(date -u +%H:%M:%S)"
}
export -f run_one
export SEEDS OFF OUT REPORT_SET_DUMP

mkdir -p "$OUT/by_problem" "$OUT/dumps"
for p in $(seq -w 1 16); do
  echo "M${p}-D10-PIN01:MC-ESO"
  echo "M${p}-D10-PIN01:NMMSO"
done | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "=== all done $(date -u +%H:%M:%S)"
