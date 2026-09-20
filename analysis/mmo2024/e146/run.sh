#!/usr/bin/env bash
# その146 — キュー 1: **順位が入れ替わる `descent_budget` の水準を挟む**。
# その145 が 12500 → 1540 で null の Score を 0.6284 → 0.3284（＝ 直した NMMSO の
# 0.4144 の下）に落とした。境界はこの間にある。**2 点だけ置いて括る**:
# `Restart-Lander-3000` と `Restart-Lander-6000`（`sigma_ratio=0.1` は据え置き）。
#
#   analysis/mmo2024/e146/run.sh [seeds] [seed-offset]
#
# 条件は その145/run.sh と **1 つを除いて同一**（同じ driver・同じ予算 `--evals-frac 1.0`・
# 同じ問題・同じ seed 0・同じ `RESTART_LANDER_DUMP`・同じ `--report-rule current`）。
# 違うのは `_METHODS` の `descent_budget` だけ ＝ **対は条件 1 個の差で取れる。**
# `core/` は 1 ビットも触っていない。既存の `Restart-Lander` / `-1540` の行も不変。
# MC-ESO / NMMSO / RR は回さない（保存値と対にする）＝ **新規 run は 32 本だけ。**
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
OUT="analysis/mmo2024/e146"
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

mkdir -p "$OUT/by_problem" "$OUT/descents_3000" "$OUT/descents_6000"
for p in $(seq 1 8); do
  for q in "$p" "$((p + 8))"; do
    printf 'M%02d-D10-PIN01 Restart-Lander-3000\nM%02d-D10-PIN01 Restart-Lander-6000\n' "$q" "$q"
  done
done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
