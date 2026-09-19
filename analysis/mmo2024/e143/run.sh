#!/usr/bin/env bash
# その143 — キュー 1 の残り: 公表帯の `r3pso`（`n_particles=400`）で 16 問を揃える。
#
# その142 は枠が binding して `r3pso-p400` を M01 / M09 / M10 の 3 問しか取れず、
# 事前登録どおり「腕を落とし、問題は落とさない」で判定から外した。
# この回はその残り 13 問（M02-M08、M11-M16）だけを同じ条件で取る。
#
# 条件はその142 と 1 文字も変えない: D=10・PIN01・正規予算 50 万（--evals-frac 1.0）・seed 0・
# --report-rule current（報告規則は e115/analyze.py の `eps_loose+dedup` r=0.05*span で offline 適用）。
# `core/optimizers/` は 1 行も触らない。MC-ESO も 1 ビット触らない。
set -u
cd "$(dirname "$0")/../../.."
export PYTHONPATH=/tmp/pystub
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="analysis/mmo2024/e143"
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

# 群 A（M02-M08、K=20）と群 B（M11-M16、K=10）を交互に投入する（その133 の規則）。
# 枠が binding しても片群に偏って落ちないようにする。
{
  for p in 02 03 04 05 06 07 08; do printf 'M%s-D10-PIN01 r3pso-p400\n' "$p"; done
} > /tmp/jobs_a.txt
{
  for p in 11 12 13 14 15 16; do printf 'M%s-D10-PIN01 r3pso-p400\n' "$p"; done
} > /tmp/jobs_b.txt
paste -d'\n' /tmp/jobs_a.txt /tmp/jobs_b.txt | grep -v '^$' \
  | xargs -P 4 -n 2 bash -c 'run_one "$@"' _
echo "=== all done $(date -u +%H:%M:%S)"
