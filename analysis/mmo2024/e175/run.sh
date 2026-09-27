#!/usr/bin/env bash
# その175 — キュー 1: **論文の背骨を D=5 / D=20 に広げる ＝ NMMSO を新 suite で測る。**
#
#   bash analysis/mmo2024/e175/run.sh <D05|D20> [deadline HH:MM UTC]
#
# 回すのは **NMMSO（公表設定 `swarm_size=10·D`、既定）× M01-M16 × PIN01 × seed 0 ×
# 正規予算 `50000·D`**（D=05 は 25 万、D=20 は 100 万）。
# **対照はすべて保存物 ＝ 追加評価ゼロ**:
#   * D=5 の `Restart-Lander` -> `analysis/mmo2024/e153/descents.csv.gz`（PIN01 seed 0、16 問）
#   * D=20 の `Restart-Lander` -> `analysis/mmo2024/e151/descents.csv.gz`（PIN01 seed 0、16 問）
#
# 条件は その139（D=10 の NMMSO ＝ 配線の見本）と **次元以外すべて同一**:
#   * `scripts/niching_baseline.py --evals-frac 1.0 --report-rule current`
#   * `REPORT_SET_DUMP` で報告集合（f ＋ 座標、上限なし）を出し、採点は e115 の腕で後から当てる
#   * `core/` は 1 行も触らない。腕は作らない。MC-ESO / RR は回さない。既定は変えない。
#
# 投入順は **群 A（M01-M08、K=20）／群 B（M09-M16、K=10）交互**（その133 の教訓）＝
# -P 4 なのでどの波も A 2 本 / B 2 本で、打ち切っても群に偏らない。
set -u
cd "$(dirname "$0")/../../.."
DIM="${1:?usage: run.sh <D05|D20> [deadline HH:MM]}"
DEADLINE="${2:-}"
OUT="analysis/mmo2024/e175"
: "${PYNMMSO_PATH:?set PYNMMSO_PATH to the patched pynmmso dir (env 手順 2-3)}"
export PYTHONPATH="${PYTHONPATH:-}:$PYNMMSO_PATH"
mkdir -p "$OUT/by_problem" "$OUT/dumps/$DIM"

run_one() {
  name="$1"
  if [ -n "${DEADLINE:-}" ] && [ "$(date -u +%H:%M)" \> "$DEADLINE" ]; then
    echo "skip $name (past deadline $DEADLINE)"; return 0
  fi
  REPORT_SET_DUMP="$OUT/dumps/$DIM" \
  python3 scripts/niching_baseline.py --funcs "$name" --methods NMMSO \
    --evals-frac 1.0 --seeds 1 --seed-offset 0 --report-rule current \
    --csv "$OUT/by_problem/NMMSO_${name}.csv" > "$OUT/by_problem/NMMSO_${name}.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
}
export -f run_one
export OUT DEADLINE DIM

for p in $(seq 1 8); do
  for q in "$p" "$((p + 8))"; do
    printf 'M%02d-%s-PIN01\n' "$q" "$DIM"
  done
done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
