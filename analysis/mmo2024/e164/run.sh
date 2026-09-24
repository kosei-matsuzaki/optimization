#!/usr/bin/env bash
# その164 — キュー 1: **D=20 で `descent_budget` をもう 1 段（25000 -> 50000）上げた 1 水準を 16 問回す**。
#
#   bash analysis/mmo2024/e164/run.sh [deadline HH:MM UTC]
#
# 回すのは **M01-M16 / D=20 / PIN01 / seed 0 / 正規予算 floor(50000x20) = 100 万 = 16 run**。
# 対照（その163 の 25000 と その151 の既定 12500、同じ 16 問・同じ次元・同じ instance・同じ seed）は
# **追加評価ゼロ**（`analysis/mmo2024/e163/descents.csv.gz` と `e151/descents.csv.gz` の保存物を読むだけ）。
#
# 条件は その163 / その151 と **`--methods` 以外すべて同一**:
#   * `scripts/niching_baseline.py --evals-frac 1.0 --report-rule current --seeds 1 --seed-offset 0`
#   * 腕は driver 側のパラメータ 1 個（`descent_budget`）だけ違う。`sigma_ratio` は 0.1 のまま。
#   * `core/` は 1 行も触っていない。MC-ESO の既定は変えない。
#
# 投入順は **群 A（M01-M08、K=20）／群 B（M09-M16、K=10）交互**（その133 の教訓）＝
# -P 4 なので**どの波も A 2 本 / B 2 本**で、打ち切っても群に偏らない。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e164"
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
  python3 scripts/niching_baseline.py --funcs "$name" --methods Restart-Lander-50000 \
    --evals-frac 1.0 --seeds 1 --seed-offset 0 --report-rule current \
    --csv "$OUT/by_problem/RL50000_${name}.csv" > "$OUT/by_problem/RL50000_${name}.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
}
export -f run_one
export OUT DEADLINE

for p in $(seq 1 8); do
  for q in "$p" "$((p + 8))"; do
    printf 'M%02d-D20-PIN01\n' "$q"
  done
done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
