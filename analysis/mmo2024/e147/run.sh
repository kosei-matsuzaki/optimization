#!/usr/bin/env bash
# その147 — キュー 2: **2 つ目のつまみ `sigma_ratio` を振る**。
# `Restart-Lander` の `sigma_ratio` を 0.1（`restart_lander.py:56` の直書き ＝ その103 の値）から
# **0.2（MC-ESO の σ_init、`mceso.py:216`）** と **0.05（対称を取る反対側の点）** にした 2 腕を、
# 新 suite M01-M16（D=10、PIN01、正規予算 50 万、seed 0）で 1 本ずつ回す。
#
#   analysis/mmo2024/e147/run.sh [seeds] [seed-offset]
#
# 条件は その146/run.sh と **1 つを除いて同一**（同じ driver・同じ予算 `--evals-frac 1.0`・
# 同じ問題・同じ seed 0・同じ `RESTART_LANDER_DUMP`・同じ `--report-rule current`）。
# 違うのは `_METHODS` の `sigma_ratio` だけ（`descent_budget` は 12500 のまま）
# ＝ **対は条件 1 個の差で取れる。**
# `core/` は 1 ビットも触っていない。既存の `Restart-Lander` / `-1540` / `-3000` / `-6000` も不変。
# 現行 null（σ=0.1）は回さない（`e115/descents` と対にする）＝ **新規 run は 32 本だけ。**
#
# 降下ダンプの名前は `{問題}_seed{seed}.csv` で手法名が入らない（`restart_lander.py:118`）ので、
# **腕ごとに別ディレクトリに吐く**（同じディレクトリに出すと上書きで混ざる）。
#
# 投入順は 1 問につき 2 腕を隣り合わせ、群 A（M01-M08、K=20）と群 B（M09-M16、K=10）を交互
# （その133 の教訓 ＋ 打ち切り時に片腕だけの問題を作らないため）。
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-1}"
OFF="${2:-0}"
OUT="analysis/mmo2024/e147"
export PYTHONPATH="${PYTHONPATH:-}:${STUB:-/tmp/pystub}"

run_one() {
  name="$1"; arm="$2"
  RESTART_LANDER_DUMP="$OUT/descents_${arm##*-}" \
  python3 scripts/niching_baseline.py --funcs "$name" --methods "$arm" \
    --evals-frac 1.0 --seeds "$SEEDS" --seed-offset "$OFF" \
    --report-rule current \
    --csv "$OUT/by_problem/${arm}_${name}.csv" > "$OUT/by_problem/${arm}_${name}.log" 2>&1
  echo "done $arm $name $(date -u +%H:%M:%S)"
}
export -f run_one
export SEEDS OFF OUT

mkdir -p "$OUT/by_problem" "$OUT/descents_s020" "$OUT/descents_s005"
for p in $(seq 1 8); do
  for q in "$p" "$((p + 8))"; do
    printf 'M%02d-D10-PIN01 Restart-Lander-s020\nM%02d-D10-PIN01 Restart-Lander-s005\n' "$q" "$q"
  done
done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
