#!/usr/bin/env bash
# その165 — キュー 1: **降下長の最適水準は K に依存するか**。
#
#   bash analysis/mmo2024/e165/run.sh [deadline HH:MM UTC]
#
# 回すのは **`Restart-Lander-100000` を M01-M16 / D=20 / PIN01 / seed 0 / 正規予算 floor(50000x20) = 100 万 = 16 run**。
# 対照（その164 の 50000、その163 の 25000、その151 の既定 12500）は **追加評価ゼロ**
# （`analysis/mmo2024/e164/descents.csv.gz` ほかの保存物を読むだけ）。
#
# 条件は その164 / その163 / その151 と **`--methods` 以外すべて同一**:
#   * `scripts/niching_baseline.py --evals-frac 1.0 --report-rule current --seeds 1 --seed-offset 0`
#   * 腕は driver 側のパラメータ 1 個（`descent_budget`）だけ違う。`sigma_ratio` は 0.1 のまま。
#   * `core/` は 1 行も触っていない。MC-ESO の既定は変えない。
#
# 投入順は **その164 と違い 群 B（M09-M16、K=10）を先、群 A（M01-M08、K=20）を後**。
# 理由: キューが要求する deliverable は **群 B の 1 水準**（8 run ≒ 19 分）で、群 A は
# 「余裕があれば 4 点目を揃える」任意分だから ＝ **打ち切りが起きたとき失うのは任意分の側にする。**
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e165"
DEADLINE="${1:-}"
PYSTUB="${PYSTUB:-/tmp/pystub}"
export PYTHONPATH="${PYTHONPATH:-}:$PYSTUB"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/descents" "$OUT/by_problem"

run_one() {
  name="$1"
  if [ -n "${DEADLINE:-}" ] && [ "$(date -u +%H:%M)" \> "$DEADLINE" ]; then
    echo "skip $name (past deadline $DEADLINE)"; return 0
  fi
  RESTART_LANDER_DUMP="$OUT/descents" \
  python3 scripts/niching_baseline.py --funcs "$name" --methods Restart-Lander-100000 \
    --evals-frac 1.0 --seeds 1 --seed-offset 0 --report-rule current \
    --csv "$OUT/by_problem/RL100000_${name}.csv" > "$OUT/by_problem/RL100000_${name}.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
}
export -f run_one
export OUT DEADLINE

# 群 B（K=10）が先、群 A（K=20）が後
for q in $(seq 9 16) $(seq 1 8); do
  printf 'M%02d-D20-PIN01\n' "$q"
done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
