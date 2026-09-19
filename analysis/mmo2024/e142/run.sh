#!/usr/bin/env bash
# その142 — キュー 1: 外部 niching 手法を新 suite（M01-M16、D=10、PIN01、正規予算 50 万、seed 0）
# で回し、頭の主張の手法軸を n=2 から n=4 にする。
#
# 腕は 3 本: NCDE（出荷既定）/ r3pso（出荷既定 30）/ r3pso-p400（Li 2010 の公表帯を D=10 で読んだ 400）。
# `core/optimizers/` は 1 行も触っていない ＝ 既存の記録値の意味は変わらない。
#
# **投入順**: 先に {NCDE, r3pso} の 16 問（32 run）を全部出し、そのあと r3pso-p400 の 16 問。
# 枠が binding したときに落ちるのが「腕」であって「問題」ではないようにする（Score の比較可能性）。
# 群 A（M01-M08）と群 B（M09-M16）は交互（その133）。
set -u
cd "$(dirname "$0")/../../.."
export PYTHONPATH=/tmp/pystub
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="analysis/mmo2024/e142"
export REPORT_SET_DUMP="$OUT/dumps"
mkdir -p "$OUT/by_problem" "$OUT/dumps"

run_one() {
  name="$1"; meth="$2"
  python3 scripts/niching_baseline.py --funcs "$name" --methods "$meth" \
    --evals-frac 1.0 --seeds 1 --seed-offset 0 --report-rule current \
    --csv "$OUT/by_problem/${name}_${meth}.csv" \
    > "$OUT/by_problem/${name}_${meth}.log" 2>&1 \
    && echo "done $name $meth $(date -u +%H:%M:%S)" \
    || echo "FAILED $name $meth $(date -u +%H:%M:%S)"
}
export -f run_one
export OUT REPORT_SET_DUMP

{
  for m in NCDE r3pso; do
    for p in $(seq 1 8); do
      printf 'M%02d-D10-PIN01 %s\nM%02d-D10-PIN01 %s\n' "$p" "$m" "$((p + 8))" "$m"
    done
  done
  for p in $(seq 1 8); do
    printf 'M%02d-D10-PIN01 r3pso-p400\nM%02d-D10-PIN01 r3pso-p400\n' "$p" "$((p + 8))"
  done
} | xargs -P 4 -n 2 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
