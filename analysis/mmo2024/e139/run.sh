#!/usr/bin/env bash
# その139 — キュー 1 の残り (c): **直した NMMSO**（`swarm_size` 既定 = `10·D`、その137）で
# 新 suite M01-M16（D=10、PIN01、正規予算 50 万、seed 0）を測り直し、その116 の Score 0.1548
# （＝ 旧既定 `swarm_size=10` ＝ 公表設定の 1/D 倍で取った数値）を置き換える。
#
#   analysis/mmo2024/e139/run.sh [seeds] [seed-offset]
#
# 条件は その116/run.sh と 1 つを除いて同一（同じ driver・同じ予算 `--evals-frac 1.0`・
# 同じ問題・同じ seed 0・同じ `REPORT_SET_DUMP`）。違うのは `core/optimizers/nmmso.py` の
# 既定が `10` から `10·D` に直っていることだけ ＝ **対は条件 1 個の差で取れる。**
#
# **投入順は群 A（M01-M08、K=20）と群 B（M09-M16、K=10）を交互にする**（その133 の教訓）。
# 枠に入らず途中で切れても、残るのが群 A に偏らない。
#
# MC-ESO は 1 ビットも触らない。新規 run は NMMSO の 16 本だけ。
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-1}"
OFF="${2:-0}"
OUT="analysis/mmo2024/e139"
export REPORT_SET_DUMP="$OUT/dumps"

run_one() {
  name="$1"
  python3 scripts/niching_baseline.py --funcs "$name" --methods NMMSO \
    --evals-frac 1.0 --seeds "$SEEDS" --seed-offset "$OFF" \
    --report-rule current \
    --csv "$OUT/by_problem/${name}_NMMSO.csv" \
    > "$OUT/by_problem/${name}_NMMSO.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
}
export -f run_one
export SEEDS OFF OUT REPORT_SET_DUMP

mkdir -p "$OUT/by_problem" "$OUT/dumps"
for p in $(seq 1 8); do
  printf 'M%02d-D10-PIN01\nM%02d-D10-PIN01\n' "$p" "$((p + 8))"
done | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "=== all done $(date -u +%H:%M:%S)"
