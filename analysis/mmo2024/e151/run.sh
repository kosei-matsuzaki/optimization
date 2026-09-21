#!/usr/bin/env bash
# その151 — キュー 1: **`Restart-Lander` を D=20 で 16 問回す（安いスクリーン、16 run）**。
#
#   bash analysis/mmo2024/e151/run.sh [deadline HH:MM UTC]
#
# 回すのは **M01-M16 / D=20 / PIN01 / seed 0 / 正規予算 floor(50000x20) = 100 万 = 16 run**。
# D=10 の対照（その115・その116・その131 の保存物 = MPR 0.5644 / Score 0.6284）は**追加評価ゼロ**。
#
# 条件は その149 の RL 側と **次元（D10 -> D20）以外すべて同一**:
#   * `scripts/niching_baseline.py --evals-frac 1.0 --report-rule current`
#   * `Restart-Lander` の既定（`sigma_ratio=0.1` / `descent_budget=12500`）は **1 ビットも変えない**
#   * `core/` は 1 行も触らない。腕は作らない。MC-ESO / NMMSO / RR は回さない。
#
# 投入順は **群 A（M01-M08、K=20）／群 B（M09-M16、K=10）交互**（その133 の教訓）＝
# -P 4 なので**どの波も A 2 本 / B 2 本**で、打ち切っても群に偏らない。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e151"
DEADLINE="${1:-}"
PYSTUB="${PYSTUB:-/tmp/pystub}"
export PYTHONPATH="${PYTHONPATH:-}:$PYSTUB"
mkdir -p "$OUT/descents" "$OUT/by_problem"

run_one() {
  name="$1"
  if [ -n "${DEADLINE:-}" ] && [ "$(date -u +%H:%M)" \> "$DEADLINE" ]; then
    echo "skip $name (past deadline $DEADLINE)"; return 0
  fi
  RESTART_LANDER_DUMP="$OUT/descents" \
  python3 scripts/niching_baseline.py --funcs "$name" --methods Restart-Lander \
    --evals-frac 1.0 --seeds 1 --seed-offset 0 --report-rule current \
    --csv "$OUT/by_problem/RL_${name}.csv" > "$OUT/by_problem/RL_${name}.log" 2>&1
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
