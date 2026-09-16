#!/usr/bin/env bash
# その133 — 重複を潰した半径での再投入腕と素の basin-hopping 対照を 16 問 × seed 0 × 正規予算 50 万で回す。
#
# その125 の run.sh と違うのは **問題の投入順だけ** ——
# **群 A（M01-M08）と群 B（M09-M16）が交互になる順**にしてある。その125 は M01→M16 の順だったので、
# 打ち切ると群 A に偏る（群別の判定が片側だけ揃う）。この順なら打ち切っても両群が同数に近く残る。
# **2 腕は 1 つの 4 並列プールに交互に入れる**（腕ごとに固めると打ち切り時に対が 1 つも作れない。その125 の実証）。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e133/descents"
export PYTHONPATH="${PYTHONPATH:-}:${STUB:-/tmp/pystub}"
mkdir -p "$OUT"

for p in 01 09 02 10 03 11 04 12 05 13 06 14 07 15 08 16; do
  for a in reseed bhop; do echo "M${p}-D10-PIN01 $a"; done
done | xargs -P 4 -L 1 bash -c 'python3 analysis/mmo2024/e133/arm.py "$0" "$1" '"$OUT"
echo "=== all done $(date -u +%H:%M:%S)"
