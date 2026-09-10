#!/usr/bin/env bash
# Entry 107 -- is MC-ESO's 0.56 gap on the new suite the reporting rule or the
# search? Same runs as entry 106, scored three ways, zero extra evaluations.
#
#   analysis/mmo2024/e107/run.sh <seeds> [methods]
#
# One CSV per problem so a run that is cut short still leaves finished problems
# behind (entry 85's rule). Sharded 4-wide, as entry 106.
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-1}"
METHODS="${2:-MC-ESO}"
OUT="analysis/mmo2024/e107"
export PYTHONPATH="${PYTHONPATH:-}:${STUB:-}"

run_one() {
  name="$1"
  python3 scripts/niching_baseline.py --funcs "$name" --methods "$METHODS" \
    --evals-frac 1.0 --seeds "$SEEDS" --report-rule all \
    --csv "$OUT/by_problem/${name}.csv" > "$OUT/by_problem/${name}.log" 2>&1
  echo "done $name"
}
export -f run_one
export SEEDS METHODS OUT

mkdir -p "$OUT/by_problem"
for p in $(seq -w 1 16); do echo "M${p}-D10-PIN01"; done \
  | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
