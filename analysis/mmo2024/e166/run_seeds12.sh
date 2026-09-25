#!/bin/bash
# その166 追加 —— seed 1,2（事前登録 §5 の「落とす順」の逆 ＝ 枠に余裕が出たので足した分）
set -u
cd "$(dirname "$0")/../../.."
D=analysis/mmo2024/e166
shard () {
  RESTART_LANDER_DUMP=$D/descents \
  python3 scripts/niching_baseline.py --funcs "$2" \
      --methods Restart-Lander,NMMSO --evals-frac 1.0 --seeds 2 --seed-offset 1 \
      --report-rule current --csv $D/baseline_s12_$1.csv > $D/run_s12_$1.log 2>&1
  echo "shard $1 exit=$?"
}
shard c N20-CF4-20D &
shard b N18-CF3-10D,N19-CF4-10D &
shard a N14-CF3-3D,N15-CF4-3D,N16-CF3-5D,N17-CF4-5D &
wait
echo "all shards done"
