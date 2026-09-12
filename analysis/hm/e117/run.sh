#!/usr/bin/env bash
# Entry 117 / queue 2 (方針欄 2026-09-11 (3)).  Two measurements:
#
#   1. the BBOB-24 dim2 gate for `commit_place_r010` -- the one adoption
#      candidate of the six that had never been through it -- plus a probe that
#      counts whether the arm fires there at all (entry 29's type (i)/(ii)),
#   2. `comp4`: the four gate-passing arms loaded at the same time, against
#      base, on the CEC2013 functions the adoption procedure names.
#
# The gate writes into results/ (gitignored); the function-level aggregate is
# kept as bbob_gate_commit_place.csv.  Everything else lands next to this file.
#
# pynmmso is not needed here (no NMMSO arm), so the stub of entry 86 is enough
# to get core.optimizers to import:
#   mkdir -p /tmp/stub/pynmmso && printf 'class Nmmso:\n    def __init__(self,*a,**k):\n        raise RuntimeError("stub")\n' > /tmp/stub/pynmmso/__init__.py
#   export PYTHONPATH=/tmp/stub
set -u
cd "$(dirname "$0")/../../.."
OUT=analysis/hm/e117
PY=${PY:-python3}

# -- 1. the gate (8.3 min, 960 run) ------------------------------------------
./run.sh quick --all --n-runs 20 --max-evals 5000 \
    --methods MC-ESO,commit_place_r010 --label e117gate

# -- 1b. gate power: does the committed restart fire on those cells? (5 min) --
for s in 0 5 10 15; do
  $PY "$OUT/gate_power_commit.py" --seed-start "$s" --seeds 5 --evals 5000 \
      --csv "$OUT/gatepower_commit_5k_s${s}.csv" &
done
wait

# -- 2. the composite arm, paired by seed at the suite's own budget -----------
#    N06: 12 seeds in 3 shards per arm (25 s / run).
for v in base comp4; do
  for s in 0 4 8; do
    $PY scripts/diagnose_niching.py --funcs N06-Shubert2D --evals 200000 \
        --variant "$v" --seed-start "$s" --seeds 4 --eps 1e-5,1e-3 \
        --fast-scoring --csv "$OUT/n06_${v}_s${s}.csv" &
  done
done
wait
#    N08: 8 seeds in 2 shards per arm (4 processes = 4 cores).
for v in base comp4; do
  for s in 0 4; do
    $PY scripts/diagnose_niching.py --funcs N08-Shubert3D --evals 400000 \
        --variant "$v" --seed-start "$s" --seeds 4 --eps 1e-5,1e-3 \
        --fast-scoring --csv "$OUT/n08_${v}_s${s}.csv" &
  done
done
wait
#    class identity check: all four layers off must reproduce base exactly.
$PY scripts/diagnose_niching.py --funcs N06-Shubert2D --evals 200000 \
    --variant comp4_off --seed-start 0 --seeds 4 --eps 1e-5,1e-3 \
    --fast-scoring --csv "$OUT/n06_comp4off.csv"

# -- scoring ------------------------------------------------------------------
$PY "$OUT/analyze.py" --tag n06
$PY "$OUT/analyze.py" --tag n08
# N09-Vincent3D (400k x 2 arms x 8 seeds) is the recorded next step: it did not
# fit the frame this cycle.
