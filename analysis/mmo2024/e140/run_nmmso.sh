#!/usr/bin/env bash
# その140 — キュー 1 の残り: その139 が見つけた 4 問（M01 / M02 / M04 / M10）で
# **直した NMMSO（`swarm_size` 既定 = `10·D`）が記憶なし多スタートを本当に上回るのか**を
# seed 2/3/4 で確かめる（既存の seed 0/1 と合わせて 1 問あたり 5 seed）。
#
#   analysis/mmo2024/e140/run_nmmso.sh
#
# 条件は e139/run.sh と 1 つも変えない（同じ driver・同じ予算 `--evals-frac 1.0`・
# 同じ報告規則 `--report-rule current`・同じ `REPORT_SET_DUMP`）。変えるのは seed だけ。
#
# **投入順は seed major**（seed2 の 4 問 → seed3 の 4 問 → seed4 の 4 問）。
# 40 分枠で切れても「揃った seed」が残る ＝ 対が壊れない（問題を落とすと対が壊れる）。
#
# MC-ESO は 1 ビットも触らない（案 (C)・方針欄 (1)）。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e140"
export REPORT_SET_DUMP="$OUT/dumps"

run_one() {
  spec="$1"; name="${spec%%:*}"; sd="${spec##*:}"
  python3 scripts/niching_baseline.py --funcs "$name" --methods NMMSO \
    --evals-frac 1.0 --seeds 1 --seed-offset "$sd" \
    --report-rule current \
    --csv "$OUT/by_problem/${name}_NMMSO_s${sd}.csv" \
    > "$OUT/by_problem/${name}_NMMSO_s${sd}.log" 2>&1
  echo "done $name seed$sd $(date -u +%H:%M:%S)"
}
export -f run_one
export OUT REPORT_SET_DUMP

mkdir -p "$OUT/by_problem" "$OUT/dumps"
for sd in 2 3 4; do
  for p in M01 M02 M04 M10; do echo "${p}-D10-PIN01:${sd}"; done
done | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "=== all done $(date -u +%H:%M:%S)"
