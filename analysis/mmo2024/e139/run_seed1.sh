#!/usr/bin/env bash
# その139 の確認 run — seed 0 で符号が反転した（または反転しかけた）5 問だけ seed 1 で回す。
#
# 理由: **NMMSO は seed を固定しても run 間で完全再現しない**（set のイテレーション順が
# object id 依存。`acceptance_topology.md` の環境節）ので、**主判定が +0.0007 のような
# 薄い差に乗っていると、1 seed では符号を主張できない。**
# 対象は seed 0 で null を上回った 4 問（M01 +0.0007 / M02 +0.0351 / M04 +0.0428 / M10 +0.1071）
# と、**逆向きに薄かった 1 問**（M03 −0.0122）。
#
# **null 側は追加評価ゼロ** —— seed 1 の `Restart-Lander` は `e115/s1/descents/` に 16 本とも保存済み。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e139"
export REPORT_SET_DUMP="$OUT/dumps_s1"

run_one() {
  name="$1"
  python3 scripts/niching_baseline.py --funcs "$name" --methods NMMSO \
    --evals-frac 1.0 --seeds 1 --seed-offset 1 --report-rule current \
    --csv "$OUT/by_problem_s1/${name}_NMMSO.csv" \
    > "$OUT/by_problem_s1/${name}_NMMSO.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
}
export -f run_one
export OUT REPORT_SET_DUMP

mkdir -p "$OUT/by_problem_s1" "$OUT/dumps_s1"
for n in M01 M02 M03 M04 M10; do echo "${n}-D10-PIN01"; done \
  | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "=== all done $(date -u +%H:%M:%S)"
