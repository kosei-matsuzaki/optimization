#!/bin/sh
# Entry 88: is the F14-F20 coverage ceiling a property of the
# "uniform restart + isotropic local descent" class, on every function of the
# set, or only on N18 (entry 87)?  Runs the same two nulls entry 87 ran, on the
# six functions entry 87 did not touch.  No optimizer runs at all.
set -e
cd "$(dirname "$0")/../../.."
for F in N14-CF3-3D N15-CF4-3D N16-CF3-5D N17-CF4-5D N19-CF4-10D N20-CF4-20D; do
  echo "########## $F"
  python3 scripts/hunt_coverage.py --null --func "$F" \
      --descents 2000 --budget 1499 --procs 4 \
      --geo-csv "analysis/hm/e88/geo_${F}.csv" \
      --csv "analysis/hm/e88/null_iso_b1499_${F}.csv"
done
