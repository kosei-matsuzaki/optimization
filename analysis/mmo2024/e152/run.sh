#!/usr/bin/env bash
# その152 — キュー 1: **RR-CMA-ES を D=20 で 16 問回す（直接対決の 2 次元目、16 run）**。
#
#   bash analysis/mmo2024/e152/run.sh [deadline HH:MM UTC]
#
# 回すのは **M01-M16 / D=20 / PIN01 / seed 0 / 正規予算 floor(50000x20) = 100 万 = 16 run**。
# 対の相手（`Restart-Lander` D=20）は その151 の保存物、D=10 の 16 対は その131 §7 の表 ＝ **追加評価ゼロ**。
#
# 条件は その149 の RR 側と **次元（D10 -> D20）以外すべて同一**:
#   * `run_rrcma.py`（その127 のものを その149 経由で写した。出力先以外 1 文字も変えていない）
#   * 既定から動かすのは 2 つのモジュールフラグだけ（`restart_strategy=RESTART` / `repelling_restart=True`）
#   * `core/` は 1 行も触らない。腕は作らない。MC-ESO / NMMSO / `r3pso` / `Restart-Lander` は回さない。
#
# 投入順は **群 A（M01-M08、K=20）／群 B（M09-M16、K=10）交互**（その133 の教訓）＝
# -P 4 なので**どの波も A 2 本 / B 2 本**で、打ち切っても群に偏らない。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e152"
DEADLINE="${1:-}"
PYSTUB="${PYSTUB:-/tmp/pystub}"
export PYTHONPATH="${PYTHONPATH:-}:$PYSTUB"
mkdir -p "$OUT/dumps"

run_one() {
  name="$1"
  if [ -n "${DEADLINE:-}" ] && [ "$(date -u +%H:%M)" \> "$DEADLINE" ]; then
    echo "skip $name (past deadline $DEADLINE)"; return 0
  fi
  python3 "$OUT/run_rrcma.py" "$name" 0 >> "$OUT/run_rr.log" 2>&1
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
