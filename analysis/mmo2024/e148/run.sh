#!/usr/bin/env bash
# その148 — キュー 1: **崖の位置を 1 段詰める**。その146 が括った 1540-3000（1.95 倍幅）を
# 幾何中点 2200 で二分する。腕は `Restart-Lander-2200` 1 本（`sigma_ratio=0.1` 据え置き）。
#
#   analysis/mmo2024/e148/run.sh <arm-budget> [seeds] [seed-offset]
#
# 条件は その146/run.sh と **`descent_budget` 以外すべて同一**（同じ driver・同じ予算
# `--evals-frac 1.0`・同じ 16 問・同じ seed 0・同じ `RESTART_LANDER_DUMP`・同じ
# `--report-rule current`）＝ **対は条件 1 個の差。** `core/` は 1 ビットも触っていない。
# MC-ESO / NMMSO / RR は回さない（保存値と対にする）＝ **1 段あたり新規 run は 16 本だけ。**
#
# 降下ダンプの名前は `{問題}_seed{seed}.csv` で手法名が入らない（`restart_lander.py:118`）ので、
# **腕ごとに別ディレクトリに吐く**（同じディレクトリに出すと上書きで混ざる）。
#
# 投入順は群 A（M01-M08、K=20）と群 B（M09-M16、K=10）を交互（その133 の教訓）。
set -u
cd "$(dirname "$0")/../../.."
BUD="${1:?usage: run.sh <arm-budget e.g. 2200>}"
SEEDS="${2:-1}"
OFF="${3:-0}"
OUT="analysis/mmo2024/e148"
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

mkdir -p "$OUT/by_problem" "$OUT/descents_${BUD}"
for p in $(seq 1 8); do
  for q in "$p" "$((p + 8))"; do
    printf 'M%02d-D10-PIN01 Restart-Lander-%s\n' "$q" "$BUD"
  done
done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
