#!/bin/sh
# Entry 88 budget control, the same one entry 87 ran on N18: 10x the evaluations
# per descent, so a "never reached" verdict cannot be descent-budget starvation.
set -e
cd "$(dirname "$0")/../../.."
for F in N14-CF3-3D N15-CF4-3D N16-CF3-5D N17-CF4-5D N19-CF4-10D N20-CF4-20D; do
  echo "########## $F"
  python3 scripts/hunt_coverage.py --null --func "$F" \
      --descents 300 --budget 14990 --procs 4 --geo-draws 1000 \
      --csv "analysis/hm/e88/null_iso_b14990_${F}.csv"
done
