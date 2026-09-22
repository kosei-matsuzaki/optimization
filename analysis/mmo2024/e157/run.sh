#!/usr/bin/env bash
# その157 — キュー 1: **D=5 の instance 間ばらつき**。
#
#   bash analysis/mmo2024/e157/run.sh [deadline HH:MM UTC]
#
# 回すのは **M01-M16 / D=5 / PIN02・PIN03・PIN04 / seed 0 / 正規予算 floor(50000x5)=25 万**。
# PIN01 の 16 対は その153 の保存物から読むので**追加評価ゼロ**。
#
# **投入順が枠の binding 規則そのもの**（その142）:
#   1. RR-CMA-ES × 48 run を先に全部入れる（PIN02 → PIN03 → PIN04、各 instance 内は群 A/群 B 交互）
#   2. その後ろに `Restart-Lander` × 48 run を同じ順で並べる
#   ＝ **枠が足りなければ落ちるのは RL 側**で、判定の主軸（RR の 4 instance 平均 MPR）は必ず揃う。
# 各ジョブは起動時に deadline を見て、過ぎていれば回さず skip する。
#
# 配線は その153（PIN01）と instance 以外すべて同一:
#   * RR: `run_rrcma.py`（その155 の写し、`cov` は既定の 20）
#   * RL: `scripts/niching_baseline.py --evals-frac 1.0 --report-rule current`、既定は 1 ビットも変えない
#   * `core/` は 1 行も触らない。腕は作らない。MC-ESO / NMMSO / `r3pso` / `NCDE` は回さない。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e157"
DEADLINE="${1:-}"
PYSTUB="${PYSTUB:-/tmp/pystub}"
export PYTHONPATH="${PYTHONPATH:-}:$PYSTUB"
mkdir -p "$OUT/dumps" "$OUT/descents" "$OUT/by_problem" "$OUT/restarts"

run_one() {
  meth="$1"; name="$2"
  if [ -n "${DEADLINE:-}" ] && [ "$(date -u +%H:%M)" \> "$DEADLINE" ]; then
    echo "skip $meth $name (past deadline $DEADLINE)"; return 0
  fi
  if [ "$meth" = RR ]; then
    python3 "$OUT/run_rrcma.py" "$name" 20 0 >> "$OUT/run_rr.log" 2>&1
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

for meth in RR RL; do
  for pin in 02 03 04; do
    for p in $(seq 1 8); do
      for q in "$p" "$((p + 8))"; do
        printf '%s M%02d-D05-PIN%s\n' "$meth" "$q" "$pin"
      done
    done
  done
done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
