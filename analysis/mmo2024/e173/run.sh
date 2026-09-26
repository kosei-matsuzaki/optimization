#!/bin/bash
# e173 第 2 部: M16 / M01 / M13 の null 抽選を 200 -> 600 に上積みする（draws 200-599）。
# `--null` の抽選 k は k だけに依存するので 600 抽選は 200 抽選の入れ子 ＝ e172 の 200 本と paired。
# 最適化 run はゼロ。core/ は触らない。NMMSO は回さない（stub）。
# 使い方: PYNMMSO_STUB=<stub dir> ./analysis/mmo2024/e173/run.sh
#
# 4 コアを埋めるのは作業プール（xargs -P 4）。1 問を 200 抽選ずつ 2 チャンクに割って
# LPT（重い順）で投入する —— 1 問 1 proc だと M01（200 抽選 8.1 分）が makespan を決めてしまう。
# 投入順はコスト（e172 の実測壁時計）だけで決めており、結果を見て決めない（prereg.md 登録済み）。
set -u
cd "$(dirname "$0")/../../.."
OUT=analysis/mmo2024/e173
export PYTHONPATH=${PYNMMSO_STUB:?set PYNMMSO_STUB to the stub dir}
mkdir -p $OUT/null

echo "=== start $(date -u +%H:%M:%S): 3 問 x 抽選 200-599 (400 本上積み), sigma0=0.1*span, 降下上限 12500 ==="
printf '%s\n' "M01-D10-PIN01 200" "M01-D10-PIN01 400" \
              "M13-D10-PIN01 200" "M13-D10-PIN01 400" \
              "M16-D10-PIN01 200" "M16-D10-PIN01 400" \
  | xargs -P 4 -I{} sh -c 'set -- {}; \
      python3 scripts/hunt_coverage.py --null --func $1 \
        --descent-start $2 --descents 200 --budget 12500 --sigma-ratio 0.1 \
        --geo-draws 1000 --procs 1 \
        --csv '"$OUT"'/null/$1_s$2_rl.csv > '"$OUT"'/null/$1_s$2_rl.log 2>&1; \
      echo "  done $1 start=$2 $(date -u +%H:%M:%S)"'

gzip -f $OUT/null/*.csv 2>/dev/null
echo "=== done $(date -u +%H:%M:%S): $(ls $OUT/null/*.gz 2>/dev/null | wc -l) dumps (期待 6) ==="
