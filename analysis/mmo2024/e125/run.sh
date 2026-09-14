#!/usr/bin/env bash
# その125 — 再投入腕と素の basin-hopping 対照を 16 問 × seed 0 × 正規予算 50 万で回す。
#
# **2 腕を 1 つの 4 並列プールに交互に入れる**（腕ごとに順に回すと、打ち切り時に
# 片腕だけ 16 問・もう片腕 0 問になって対が 1 つも作れない。事前登録の打ち切り規則）。
set -u
cd "$(dirname "$0")/../../.."
OUT="analysis/mmo2024/e125/descents"
export PYTHONPATH="${PYTHONPATH:-}:${STUB:-/tmp/pystub}"
mkdir -p "$OUT"

for p in $(seq -w 1 16); do
  for a in reseed bhop; do echo "M${p}-D10-PIN01 $a"; done
done | xargs -P 4 -L 1 bash -c 'python3 analysis/mmo2024/e125/arm.py "$0" "$1" '"$OUT"
echo "=== all done $(date -u +%H:%M:%S)"
