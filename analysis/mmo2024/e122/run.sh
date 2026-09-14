#!/usr/bin/env bash
# Entry 122 -- is PIN01 representative?  Same run as entry 115 (Restart-Lander,
# seed 0, full 500k budget, landing COORDINATES in the dump) on two OTHER
# instances of the same 16 problems.  The only thing that differs from e115 is
# the problem name: PIN01 -> PIN02 / PIN03.  Same driver, same budget, same
# report rule, so the three instances are paired problem by problem.
#
#   analysis/mmo2024/e122/run.sh [seeds] [seed-offset] [methods]
#
# One CSV per problem so a run that is cut short still leaves finished problems
# behind (entry 85's rule).  PIN02's 16 problems are queued before PIN03's, so
# a cut-off leaves PIN02 complete rather than both half-done (the prereg's
# stopping rule needs one complete instance, not two partial ones).
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-1}"
OFF="${2:-0}"
METHODS="${3:-Restart-Lander}"
OUT="analysis/mmo2024/e122"
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
{ for p in $(seq -w 1 16); do echo "M${p}-D10-PIN02"; done
  for p in $(seq -w 1 16); do echo "M${p}-D10-PIN03"; done; } \
  | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "=== all done $(date -u +%H:%M:%S)"
