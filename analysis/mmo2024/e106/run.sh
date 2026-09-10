#!/usr/bin/env bash
# Entry 106 -- first optimisation runs on the GECCO'2024 suite.
#
#   analysis/mmo2024/e106/run.sh <seeds> [methods]
#
# One CSV per problem so a run that is cut short still leaves finished problems
# behind (entry 85's rule). Sharded 4-wide; entry 25's "3 at a time" cap is for
# the memory-heavy CEC2013 scoring, and MC-ESO on D=10 stays under 300 MB.
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-3}"
METHODS="${2:-MC-ESO}"
OUT="analysis/mmo2024/e106"
export PYTHONPATH="${PYTHONPATH:-}:${STUB:-}"

run_one() {
  name="$1"
  python3 scripts/niching_baseline.py --funcs "$name" --methods "$METHODS" \
    --evals-frac 1.0 --seeds "$SEEDS" --report-rule current \
    --csv "$OUT/by_problem/${name}.csv" > "$OUT/by_problem/${name}.log" 2>&1
  echo "done $name"
}
export -f run_one
export SEEDS METHODS OUT

mkdir -p "$OUT/by_problem"
for p in $(seq -w 1 16); do echo "M${p}-D10-PIN01"; done \
  | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
