#!/usr/bin/env bash
# その149 — キュー 1: **instance 軸で直接対決の n を増やす（PIN01 → PIN01+PIN02）**。
#
#   bash analysis/mmo2024/e149/run.sh [deadline HH:MM UTC]
#
# 回すのは **PIN02 の 16 問 × 2 手法 × seed 0 = 32 run**（D=10・正規予算 50 万）。
# PIN01 側は その131 の §7 の表（本文）が正本なので **追加評価ゼロ**。
#
# 条件は その127/129 と **instance（PIN01 -> PIN02）以外すべて同一**:
#   * RR-CMA-ES  … `run_rrcma.py`（その127 のものを git から取り出した。2 フラグだけ）
#   * Restart-Lander … `scripts/niching_baseline.py --evals-frac 1.0 --report-rule current`
#   * `core/` は 1 ビットも触らない。腕は作らない。MC-ESO / NMMSO は回さない。
#
# 投入順は **問題ごとに 2 手法を隣り合わせ**（打ち切っても対が壊れない）＋
# 群 A（M01-M08、K=20）／群 B（M09-M16、K=10）交互（その133 の教訓）。
# -P 4 なので**常に 2 問ぶんが在庫**になる。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e149"
DEADLINE="${1:-}"
PYSTUB="${PYSTUB:-/tmp/pystub}"
export PYTHONPATH="${PYTHONPATH:-}:$PYSTUB"
mkdir -p "$OUT/dumps" "$OUT/descents" "$OUT/by_problem"

run_one() {
  meth="$1"; name="$2"
  if [ -n "${DEADLINE:-}" ] && [ "$(date -u +%H:%M)" \> "$DEADLINE" ]; then
    echo "skip $meth $name (past deadline $DEADLINE)"; return 0
  fi
  if [ "$meth" = "RR" ]; then
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
    printf 'RR M%02d-D10-PIN02\n' "$q"
    printf 'RL M%02d-D10-PIN02\n' "$q"
  done
done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
