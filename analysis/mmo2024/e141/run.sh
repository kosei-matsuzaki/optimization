#!/bin/bash
# Entry 141: population-size audit of the two external niching baselines.
# 5 CEC2013 functions x 4 arms x 10 seeds, the suite's own 400k budget.
# One process per (function, arm); 4 workers on this 4-core box.
cd "$(dirname "$0")/../../.."
export PYTHONPATH=/tmp/pystub
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=analysis/mmo2024/e141/raw
mkdir -p "$OUT"
for f in N15-CF4-3D N17-CF4-5D N18-CF3-10D N19-CF4-10D N20-CF4-20D; do
  for m in NCDE NCDE-p300 r3pso r3pso-p300; do
    echo "$f $m"
  done
done | xargs -P 4 -n 2 bash -c '
  python3 scripts/niching_baseline.py --funcs "$0" --methods "$1" \
    --seeds 10 --evals-frac 1.0 --report-rule current \
    --csv '"$OUT"'/"$0"_"$1".csv > '"$OUT"'/"$0"_"$1".log 2>&1 \
    && echo "done $0 $1" || echo "FAILED $0 $1"'
