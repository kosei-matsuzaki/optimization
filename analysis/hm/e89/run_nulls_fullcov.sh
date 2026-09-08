#!/bin/sh
# Entry 89, question 2 (ii): does the *descent model* move the class ceiling?
#
# Entry 87/88 established the ceiling of "uniform restart + ISOTROPIC descent"
# (CMA_on = 0, step size only) on all seven F14-F20 functions, and found it
# below the published best PR@1e-5 everywhere.  The queue's next step is to ask
# whether anisotropy -- the same descent with CMA's covariance adaptation on --
# reaches the optima the isotropic descent never reaches (CF3's {2, 3}).  Still
# no optimizer runs: `--full-cov` flips one flag in the same null, and draw k
# starts from the same x0 as in e87/e88 (rng seed 1000000+k), so the comparison
# against the stored isotropic CSVs is paired draw by draw.
#
# Short descents = MC-ESO's own hunt length (1499 evals); long descents let
# CMA-ES stop itself.  The class ceiling is the better of the two allocations.
set -e
cd "$(dirname "$0")/../../.."
FUNCS="N14-CF3-3D N15-CF4-3D N16-CF3-5D N17-CF4-5D N18-CF3-10D N19-CF4-10D N20-CF4-20D"
for F in $FUNCS; do
  echo "########## $F  short"
  python3 scripts/hunt_coverage.py --null --func "$F" --full-cov \
      --descents 2000 --budget 1499 --procs 4 --geo-draws 1000 \
      --csv "analysis/hm/e89/null_cov_b1499_${F}.csv"
done
for F in $FUNCS; do
  echo "########## $F  long"
  python3 scripts/hunt_coverage.py --null --func "$F" --full-cov \
      --descents 300 --budget 14990 --procs 4 --geo-draws 1000 \
      --csv "analysis/hm/e89/null_cov_b14990_${F}.csv"
done
