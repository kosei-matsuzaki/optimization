#!/bin/bash
# e172: 新 suite D=10 のクラス上限を `Restart-Lander` の降下設定（σ₀ = 0.1×span）で引き直す。
# 対照は再走させない —— 凍結表 e94/ceiling_mpr_d10.csv（σ=0.2、同じ 200 抽選 k=0..199）が抽選単位の対。
# 最適化 run はゼロ。core/ は触らない。NMMSO は回さない（stub）。
# 使い方: PYNMMSO_STUB=<stub dir> ./analysis/mmo2024/e172/run.sh
#
# 4 コアを埋めるのは「波」ではなく作業プール（xargs -P 4）。1 問 1 proc で、
# **1 降下コストの重い順**に投入する（longest-processing-time-first ＝ makespan 最小）。
# 投入順はコストだけで決めており、結果を見て決めない（prereg.md に登録済み）。
# 群 B の M09 / M10 / M11（1 降下 7.8k-11.4k 評価）は 40 分枠に入らないので今回は回さない。
set -u
cd "$(dirname "$0")/../../.."
OUT=analysis/mmo2024/e172
export PYTHONPATH=${PYNMMSO_STUB:?set PYNMMSO_STUB to the stub dir}
mkdir -p $OUT/null
N=200          # e92 の運用規則（この suite の K=20 群は 40 本では足りない。200 本を既定にする）

echo "=== start $(date -u +%H:%M:%S): 13 問 x $N 抽選, sigma0=0.1*span, 降下上限 12500 ==="
printf '%s\n' M03-D10-PIN01 M02-D10-PIN01 M01-D10-PIN01 M13-D10-PIN01 \
              M07-D10-PIN01 M15-D10-PIN01 M05-D10-PIN01 M16-D10-PIN01 \
              M08-D10-PIN01 M12-D10-PIN01 M14-D10-PIN01 M06-D10-PIN01 \
              M04-D10-PIN01 \
  | xargs -P 4 -I{} sh -c "python3 scripts/hunt_coverage.py --null --func {} \
      --descents $N --budget 12500 --sigma-ratio 0.1 --geo-draws 1000 --procs 1 \
      --csv $OUT/null/{}_rl.csv > $OUT/null/{}_rl.log 2>&1; \
      echo \"  done {} \$(date -u +%H:%M:%S)\""

gzip -f $OUT/null/*.csv 2>/dev/null
echo "=== done $(date -u +%H:%M:%S): $(ls $OUT/null/*.gz 2>/dev/null | wc -l) null dumps (期待 13) ==="
