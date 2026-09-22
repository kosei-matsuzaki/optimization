#!/bin/bash
# その155 — c ∈ {2,20,200} × 16 問（D=5, PIN01, seed 0）。4 並列。
cd "$(dirname "$0")/../../.."
export PYTHONPATH=/tmp/pystub
for c in 20 2 200; do
  for i in $(seq -w 1 16); do
    echo "M${i}-D05-PIN01 $c"
  done
done | xargs -P 4 -n 2 sh -c 'python3 analysis/mmo2024/e155/run_rrcma_cov.py "$0" "$1" 0'
