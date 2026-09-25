#!/bin/bash
# その167 — キュー 1: CEC2013 F11-F13（2D）で `Restart-Lander` を seed 0/1/2。
# NMMSO は このコンテナで回せない（prereg.md §5）ので RL 側のみ。
# 3 シャード並列（このコンテナは 4 コア。同時 3 本まで＝ acceptance_topology.md の環境節）
set -u
cd "$(dirname "$0")/../../.."
D=analysis/mmo2024/e167
export PYTHONPATH="${PYNMMSO_STUB:-}"

shard () {   # $1=tag  $2=funcs
  RESTART_LANDER_DUMP=$D/descents_$1 \
  python3 scripts/niching_baseline.py --funcs "$2" \
      --methods Restart-Lander --evals-frac 1.0 --seeds 3 \
      --report-rule current --csv $D/baseline_$1.csv > $D/run_$1.log 2>&1
  echo "shard $1 exit=$?"
}

shard a N11-CF1-2D &
shard b N12-CF2-2D &
shard c N13-CF3-2D &
wait
echo "all shards done"
