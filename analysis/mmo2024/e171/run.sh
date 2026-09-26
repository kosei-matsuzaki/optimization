#!/bin/bash
# e171: その88・その89 のクラス上限表を Restart-Lander の実際の降下で引き直す。
# 最適化 run はゼロ（撒いて降ろす null だけ）。core/ は触らない。NMMSO は回さない。
# 使い方: PYNMMSO_STUB=<stub dir> ./analysis/mmo2024/e171/run.sh
set -u
cd "$(dirname "$0")/../../.."
OUT=analysis/mmo2024/e171
export PYTHONPATH=${PYNMMSO_STUB:?set PYNMMSO_STUB to the stub dir}
mkdir -p $OUT/null
N=600          # キューの指示（2000 本ではなく 500-800 本）

# 1 降下のコストが 8 倍動くので（その88 §3）、重い順に波を組んで 4 コアを均す。
# --geo-draws は幾何 null を使わないので最小にする（その170 と同じ）。
run() {  # $1 func  $2 tag  $3 sigma  $4 budget  $5 procs
  python3 scripts/hunt_coverage.py --null --func "$1" --descents $N \
    --budget "$4" --sigma-ratio "$3" --geo-draws 1000 --procs "$5" \
    --csv $OUT/null/"$1"_"$2".csv > $OUT/null/"$1"_"$2".log 2>&1
}

echo "=== wave 1: N20-20D (RL 設定, 4 procs) ==="
run N20-CF4-20D rl 0.1 12500 4

echo "=== wave 2: 10D 2 本 + 5D 2 本 (RL 設定, 1 proc ずつ) ==="
for F in N19-CF4-10D N18-CF3-10D N17-CF4-5D N16-CF3-5D; do run $F rl 0.1 12500 1 & done
wait

echo "=== wave 3: 3D 2 本 (RL 設定) + 対照（既定 σ0.2/1499）7 本 ==="
run N15-CF4-3D rl 0.1 12500 1 &
run N14-CF3-3D rl 0.1 12500 1 &
run N20-CF4-20D base 0.2 1499 2 &
wait
for F in N19-CF4-10D N18-CF3-10D N17-CF4-5D N16-CF3-5D; do run $F base 0.2 1499 1 & done
wait
run N15-CF4-3D base 0.2 1499 2 &
run N14-CF3-3D base 0.2 1499 2 &
wait

# `--csv` は素の CSV を書く（拡張子を見ない）ので自分で圧縮する。行単位ダンプの規約。
gzip -f $OUT/null/*.csv 2>/dev/null
echo "=== done: $(ls $OUT/null/*.gz 2>/dev/null | wc -l) null dumps (期待 14) ==="
echo "次: python3 analysis/mmo2024/e171/analyze.py"
