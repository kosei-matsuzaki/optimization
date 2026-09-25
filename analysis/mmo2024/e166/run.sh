#!/bin/bash
# その166 — キュー 1: CEC2013 F14-F20 で `Restart-Lander` 対 NMMSO（seed 0、suite 既定予算）
# 3 シャード並列（このコンテナは 4 コア。同時 3 本まで＝ acceptance_topology.md の環境節）
set -u
cd "$(dirname "$0")/../../.."
D=analysis/mmo2024/e166

shard () {   # $1=tag  $2=funcs
  RESTART_LANDER_DUMP=$D/descents \
  python3 scripts/niching_baseline.py --funcs "$2" \
      --methods Restart-Lander,NMMSO --evals-frac 1.0 --seeds 1 \
      --report-rule current --csv $D/baseline_$1.csv > $D/run_$1.log 2>&1
  echo "shard $1 exit=$?"
}

shard c N20-CF4-20D &
shard b N18-CF3-10D,N19-CF4-10D &
shard a N14-CF3-3D,N15-CF4-3D,N16-CF3-5D,N17-CF4-5D &
wait
echo "all shards done"
