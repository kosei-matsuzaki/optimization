#!/usr/bin/env bash
# entry 103: extend every sigma0 = 0.1 dump of the new suite past 400 draws, in
# chunks of 100, so that the rarefaction curve gains points at the SAME 100-draw
# spacing as the increments entry 101 read (100->200 +0.0499, 200->300 +0.0263,
# 300->400 +0.0263).  The queue asks whether the increment finally shrinks.
#
# Usage: analysis/mmo2024/e103/run.sh <start>      e.g. 400  -> draws 400-499
#
# Each chunk is a separate per-problem dump; `analyze.py` concatenates the saved
# 200-draw dumps (e98/e99), e100's 200-399 chunk, and every e103 chunk it finds.
# Chunks are run whole-set-at-a-time (all 16 problems at one start) so the
# 16-problem mean never mixes two draw counts again -- entry 101 §1 fixed that
# and this cycle must not undo it.
#
# Every parameter other than the draw window is entry 100's, unchanged:
# D=10, PIN01, budget 12500, --sigma-ratio 0.1, --procs 4, --geo-draws 20000.
set -u
START="${1:?usage: run.sh <start draw>}"
N=100
cd "$(dirname "$0")/../../.."
export PYTHONPATH=/tmp/pystub
OUT=analysis/mmo2024/e103
mkdir -p "$OUT"
END=$((START + N - 1))
for P in 09 10 11 12 13 14 15 16 01 02 03 04 05 06 07 08; do
  F="M${P}-D10-PIN01"
  D="$OUT/${F}_sig100_draws${START}to${END}.csv"
  [ -f "${D}.gz" ] && { echo "skip $F (done)"; continue; }
  echo "=== $F  $(date -u +%H:%M:%S)"
  python3 scripts/hunt_coverage.py --null --func "$F" --geo-draws 20000 \
      --descents "$N" --descent-start "$START" --budget 12500 --sigma-ratio 0.1 \
      --procs 4 --csv "$D" 2>&1 | grep -E 'descents in|K = '
  gzip -f "$D"
done
echo "=== chunk ${START}-${END} done $(date -u +%H:%M:%S)"
