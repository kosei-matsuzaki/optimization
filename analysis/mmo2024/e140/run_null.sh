#!/usr/bin/env bash
# その140 — 対の null 側。`Restart-Lander` を NMMSO と同じ 4 問・同じ 3 本の追加 seed で回す。
#
#   analysis/mmo2024/e140/run_null.sh
#
# 条件は e115/run.sh と 1 つも変えない（同じ driver・同じ予算・同じ報告規則・同じ `RESTART_LANDER_DUMP`）。
# 変えるのは seed と問題の本数だけ。**投入順は seed major**（切れても揃った seed が残る）。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e140"
export RESTART_LANDER_DUMP="$OUT/descents"

run_one() {
  spec="$1"; name="${spec%%:*}"; sd="${spec##*:}"
  python3 scripts/niching_baseline.py --funcs "$name" --methods Restart-Lander \
    --evals-frac 1.0 --seeds 1 --seed-offset "$sd" \
    --report-rule current \
    --csv "$OUT/by_problem_null/${name}_s${sd}.csv" \
    > "$OUT/by_problem_null/${name}_s${sd}.log" 2>&1
  echo "done $name seed$sd $(date -u +%H:%M:%S)"
}
export -f run_one
export OUT RESTART_LANDER_DUMP

mkdir -p "$OUT/by_problem_null" "$OUT/descents"
for sd in 2 3 4; do
  for p in M01 M02 M04 M10; do echo "${p}-D10-PIN01:${sd}"; done
done | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "=== all done $(date -u +%H:%M:%S)"
