#!/usr/bin/env bash
# entry 105: the sigma side of queue item 2 -- group A (M01-M08) at
# sigma0 = 0.05 x span, 200 draws, D=10, PIN=1, budget 12500.  Same
# configuration as the saved dumps (e91/e92 at 0.2, e99 at 0.1) so every draw
# pairs across all three sigma points.
#
# --geo-draws is cut to 20000: the geometric (Voronoi) null goes to a separate
# file, is not read by the ceiling estimators, and draws from its own generator
# (`default_rng(0)`), so it cannot touch the descent dump.
#
# Optional second argument selects the sigma label/value; default 050 / 0.05.
# Usage: run.sh [problem list] ; e.g. run.sh "01 02 03"
set -u
cd "$(dirname "$0")/../../.."
export PYTHONPATH=/tmp/pystub
OUT=analysis/mmo2024/e105
SIG=${SIG:-0.05}
TAG=${TAG:-sig050}
PROBS=${1:-"01 02 03 04 05 06 07 08"}
mkdir -p "$OUT"
for P in $PROBS; do
  F="M${P}-D10-PIN01"
  D="$OUT/${F}_${TAG}200.csv"
  [ -f "${D}.gz" ] && { echo "skip $F (done)"; continue; }
  echo "=== $F  $(date -u +%H:%M:%S)"
  python3 scripts/hunt_coverage.py --null --func "$F" --geo-draws 20000 \
      --descents 200 --budget 12500 --sigma-ratio "$SIG" \
      --procs 4 --csv "$D" 2>&1 | grep -E 'descents in|K = '
  gzip -f "$D"
done
echo "=== all done $(date -u +%H:%M:%S)"
# completion is judged by file count, never by the log lines (entry 104's trap):
ls "$OUT"/*.gz 2>/dev/null | wc -l
