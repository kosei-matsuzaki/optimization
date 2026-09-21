#!/usr/bin/env bash
# その153 — キュー 1: **`Restart-Lander` と RR-CMA-ES を D=5 で 16 問ずつ回す（上下から挟む、32 run）**。
#
#   bash analysis/mmo2024/e153/run.sh [deadline HH:MM UTC]
#
# 回すのは **M01-M16 / D=5 / PIN01 / seed 0 / 正規予算 floor(50000x5) = 25 万 = 32 run**。
# D=10（その131 §7 の表）と D=20（その151 / その152 の保存物）は**追加評価ゼロ**。
#
# 条件は その149（D=10）・その151/その152（D=20）と **次元以外すべて同一**:
#   * RL: `scripts/niching_baseline.py --evals-frac 1.0 --report-rule current`、既定は 1 ビットも変えない
#   * RR: `run_rrcma.py`（その152 の写し。出力先以外 1 文字も変えていない）
#   * `core/` は 1 行も触らない。腕は作らない。MC-ESO / NMMSO / `r3pso` / `NCDE` は回さない。
#
# 投入順は **同じ問題の RL と RR を隣り合わせ、群 A（M01-M08、K=20）／群 B（M09-M16、K=10）交互** ＝
# -P 4 なので**どの波も手法 2 本ずつ・群も偏らない**（打ち切っても対が壊れない。その133・その152 の教訓）。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e153"
DEADLINE="${1:-}"
PYSTUB="${PYSTUB:-/tmp/pystub}"
export PYTHONPATH="${PYTHONPATH:-}:$PYSTUB"
mkdir -p "$OUT/dumps" "$OUT/descents" "$OUT/by_problem"

run_one() {
  meth="$1"; name="$2"
  if [ -n "${DEADLINE:-}" ] && [ "$(date -u +%H:%M)" \> "$DEADLINE" ]; then
    echo "skip $meth $name (past deadline $DEADLINE)"; return 0
  fi
  if [ "$meth" = RR ]; then
    python3 "$OUT/run_rrcma.py" "$name" 0 >> "$OUT/run_rr.log" 2>&1
  else
    RESTART_LANDER_DUMP="$OUT/descents" \
    python3 scripts/niching_baseline.py --funcs "$name" --methods Restart-Lander \
      --evals-frac 1.0 --seeds 1 --seed-offset 0 --report-rule current \
      --csv "$OUT/by_problem/RL_${name}.csv" > "$OUT/by_problem/RL_${name}.log" 2>&1
  fi
  echo "done $meth $name $(date -u +%H:%M:%S)"
}
export -f run_one
export OUT DEADLINE

for p in $(seq 1 8); do
  for q in "$p" "$((p + 8))"; do
    printf 'RL M%02d-D05-PIN01\nRR M%02d-D05-PIN01\n' "$q" "$q"
  done
done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
