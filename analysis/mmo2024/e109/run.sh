#!/usr/bin/env bash
# Entry 109 -- NMMSO on the GECCO'2024 suite at D=10, full budget. Turns the
# suite's only comparison (MC-ESO vs our own null) into a ranking.
#
#   analysis/mmo2024/e109/run.sh [seeds] [methods]
#
# One CSV per (problem, seed) shard, seed-major order, so a cycle that is cut
# short still leaves whole runs behind and a complete seed 0 first (entry 85's
# rule + entry 109's prereg). 4-wide on this 4-core container.
#
# NMMSO needs steps 2 and 3 of the environment section (sdist copied into
# site-packages, `random.sample(<set>, k)` patched in 4 places). The guard below
# refuses to measure against the stub, which raises on __init__ by design.
set -u
cd "$(dirname "$0")/../../.."
SEEDS="${1:-3}"
METHODS="${2:-NMMSO}"
OUT="analysis/mmo2024/e109"

python3 - "$METHODS" <<'EOF' || exit 1
import sys
import numpy as np
import scripts.niching_baseline as nb
want = sys.argv[1].split(",")
missing = [m for m in want if m not in nb._METHODS]
assert not missing, f"methods missing from _METHODS: {missing}"
if "NMMSO" in want:
    # entry 109: the stub used on NMMSO-free cycles raises on __init__, and
    # niching_baseline swallows the failure -- catch it here instead.
    from pynmmso import Nmmso
    class _P:
        @staticmethod
        def get_bounds():
            return np.zeros(2), np.ones(2)
        @staticmethod
        def fitness(x):
            return -float(np.sum(x * x))
    Nmmso(_P())
    print("guard: NMMSO importable and instantiable (not the stub)")
EOF

mkdir -p "$OUT/by_run"
for s in $(seq 0 $((SEEDS - 1))); do
  for p in $(seq -w 1 16); do echo "M${p}-D10-PIN01 $s"; done
done | xargs -P 4 -n 2 bash -c '
  name="$0"; seed="$1"
  python3 scripts/niching_baseline.py --funcs "$name" --methods "'"$METHODS"'" \
    --evals-frac 1.0 --seeds 1 --seed-offset "$seed" --report-rule current \
    --csv "'"$OUT"'/by_run/${name}_s${seed}.csv" \
    > "'"$OUT"'/by_run/${name}_s${seed}.log" 2>&1
  echo "done $name seed $seed"
'
# Entry 104's trap, in a new form: niching_baseline.py opens its --csv at
# start, so every shard file exists (0 bytes) seconds after launch and a plain
# file count reads as "all done". Count NON-EMPTY shards.
echo "shards finished: $(find "$OUT/by_run" -name '*.csv' -size +0c | wc -l)" \
     "(expect $((SEEDS * 16)))"
