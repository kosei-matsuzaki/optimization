#!/usr/bin/env bash
# entry 100: extend every sigma0 = 0.1 dump of the new suite from 200 to 400
# draws, by computing only draws 200-399 (`--descent-start`, added to
# hunt_coverage.py this cycle and identity-checked against the saved dumps
# before use).  Group B's 200-draw dumps are entry 98's, group A's entry 99's.
#
# --geo-draws is cut to 20000: the geometric (Voronoi) null is written to a
# separate file, is not read by the ceiling estimators, and draws from its own
# generator (`default_rng(0)`), so it cannot touch the descent dump.
set -u
cd "$(dirname "$0")/../../.."
export PYTHONPATH=/tmp/pystub
OUT=analysis/mmo2024/e100
mkdir -p "$OUT"
# group B first (K=10, ~17 min for 200 draws) then group A (K=20, ~21 min), so
# that a cycle that runs out of clock leaves whole problems finished, not halves.
for P in 09 10 11 12 13 14 15 16 01 02 03 04 05 06 07 08; do
  F="M${P}-D10-PIN01"
  D="$OUT/${F}_sig100_draws200to399.csv"
  [ -f "${D}.gz" ] && { echo "skip $F (done)"; continue; }
  echo "=== $F  $(date -u +%H:%M:%S)"
  python3 scripts/hunt_coverage.py --null --func "$F" --geo-draws 20000 \
      --descents 200 --descent-start 200 --budget 12500 --sigma-ratio 0.1 \
      --procs 4 --csv "$D" 2>&1 | grep -E 'descents in|K = '
  gzip -f "$D"
done
echo "=== all done $(date -u +%H:%M:%S)"
