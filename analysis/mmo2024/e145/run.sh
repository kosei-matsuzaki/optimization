#!/usr/bin/env bash
# その145 — キュー 1: **勝っている null 自身のつまみを振る**。
# `Restart-Lander` の `descent_budget` を 12500（その103 の値 ＝ 50 万 / 40）から
# **MC-ESO の 1 segment の実測 1540** に下げた腕 `Restart-Lander-1540` を、
# 新 suite M01-M16（D=10、PIN01、正規予算 50 万、seed 0）で 1 本ずつ回す。
#
#   analysis/mmo2024/e145/run.sh [seeds] [seed-offset]
#
# 条件は その115/run.sh と **1 つを除いて同一**（同じ driver・同じ予算 `--evals-frac 1.0`・
# 同じ問題・同じ seed 0・同じ `RESTART_LANDER_DUMP`・同じ `--report-rule current`）。
# 違うのは driver の `_METHODS` の 1 行 = `descent_budget` だけ ＝ **対は条件 1 個の差で取れる。**
# `core/` は 1 ビットも触っていない。既存の `Restart-Lander` の行も不変。
# MC-ESO は回さない（その113 の保存ダンプと対にする）＝ **新規 run は 16 本だけ。**
#
# 投入順は群 A（M01-M08、K=20）と群 B（M09-M16、K=10）を交互（その133 の教訓）。
# 枠に入らず途中で切れても残りが群 A に偏らない。
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-1}"
OFF="${2:-0}"
OUT="analysis/mmo2024/e145"
export PYTHONPATH="${PYTHONPATH:-}:${STUB:-/tmp/pystub}"
export RESTART_LANDER_DUMP="$OUT/descents"

run_one() {
  name="$1"
  python3 scripts/niching_baseline.py --funcs "$name" --methods Restart-Lander-1540 \
    --evals-frac 1.0 --seeds "$SEEDS" --seed-offset "$OFF" \
    --report-rule current \
    --csv "$OUT/by_problem/${name}.csv" > "$OUT/by_problem/${name}.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
}
export -f run_one
export SEEDS OFF OUT RESTART_LANDER_DUMP

mkdir -p "$OUT/by_problem" "$OUT/descents"
for p in $(seq 1 8); do
  printf 'M%02d-D10-PIN01\nM%02d-D10-PIN01\n' "$p" "$((p + 8))"
done | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "=== all done $(date -u +%H:%M:%S)"
