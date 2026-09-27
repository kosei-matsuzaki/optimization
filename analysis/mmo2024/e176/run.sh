#!/usr/bin/env bash
# その176 — キュー 1 の残り: **NMMSO × 新 suite D=20 の残り 14 問**（M09 / M10 は その175 が実走済み）。
#
#   PYNMMSO_PATH=... bash analysis/mmo2024/e176/run.sh [deadline HH:MM UTC]
#
# 条件は その175 / その139 と **次元以外すべて同一**:
#   NMMSO（公表設定 `swarm_size=10·D`、既定）× PIN01 × seed 0 × 正規予算 100 万
#   `scripts/niching_baseline.py --evals-frac 1.0 --report-rule current`
#   `REPORT_SET_DUMP` で報告集合（f ＋ 座標、上限なし）を出し、採点は e115 の腕で後から当てる。
#   `core/` は 1 行も触らない。腕は作らない。MC-ESO / RR は回さない。既定は変えない。
#
# **投入順は「群 A / 群 B 交互、各群内は その175 の D=5 実測コスト降順」**（LPT）。
#   * 交互投入は その133 の教訓（打ち切っても群に偏らない）。
#   * 群内降順は makespan を詰めるため（最遅 M07 は単独で約 15 分の外挿なので最初に入れる）。
#   * **安い問題を拾って部分集合にはしない** —— 落ちるのは「入らなかった末尾」であって選抜ではない。
set -u
cd "$(dirname "$0")/../../.."
DEADLINE="${1:-}"
OUT="analysis/mmo2024/e176"
DIM="D20"
: "${PYNMMSO_PATH:?set PYNMMSO_PATH to the patched pynmmso dir}"
export PYTHONPATH="${PYTHONPATH:-}:$PYNMMSO_PATH"
mkdir -p "$OUT/by_problem" "$OUT/dumps/$DIM"

run_one() {
  name="$1"
  if [ -n "${DEADLINE:-}" ] && [ "$(date -u +%H:%M)" \> "$DEADLINE" ]; then
    echo "skip $name (past deadline $DEADLINE)"; return 0
  fi
  t0=$(date -u +%s)
  REPORT_SET_DUMP="$OUT/dumps/$DIM" \
  python3 scripts/niching_baseline.py --funcs "$name" --methods NMMSO \
    --evals-frac 1.0 --seeds 1 --seed-offset 0 --report-rule current \
    --csv "$OUT/by_problem/NMMSO_${name}.csv" > "$OUT/by_problem/NMMSO_${name}.log" 2>&1
  rc=$?
  echo "done $name rc=$rc secs=$(( $(date -u +%s) - t0 )) $(date -u +%H:%M:%S)"
}
export -f run_one
export OUT DEADLINE DIM

for q in 07 15 08 16 03 13 05 11 04 12 01 14 02 06; do
  printf 'M%s-%s-PIN01\n' "$q" "$DIM"
done | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
