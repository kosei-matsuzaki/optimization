#!/usr/bin/env bash
# その129 (キュー 1(A)) — 3 本目の seed（200）で RR-CMA-ES と Restart-Lander を対にする。
#
# **2 手法を 1 つの 4 並列プールに問題ごとに交互に入れる**（手法ごとに固めると
# 打ち切り時に対が 1 つも作れない。その125 で実証済み）。
# 駆動はその127（RR）とその115（null）と同一で、変えたのは seed だけ。
set -u
cd "$(dirname "$0")/../../.."
PYSTUB="${PYSTUB:-/tmp/pystub}"
mkdir -p "$PYSTUB/pynmmso"
cat > "$PYSTUB/pynmmso/__init__.py" <<'EOF'
class Nmmso:
    def __init__(self, *a, **k):
        raise RuntimeError("pynmmso stub: NMMSO is not runnable here")
EOF
export PYTHONPATH="${PYTHONPATH:-}:$PYSTUB"

OUT="analysis/mmo2024/e129"
SEED="${SEED:-200}"
SEED_INDEX=$((SEED / 100))
export OUT SEED SEED_INDEX
export RESTART_LANDER_DUMP="$OUT/descents"
mkdir -p "$OUT/dumps" "$OUT/descents" "$OUT/by_problem"
LOG="$OUT/run_seed${SEED}.log"
: > "$LOG"

run_one() {
  kind="$1"; name="$2"
  if [ "$kind" = rr ]; then
    python3 analysis/mmo2024/e127/run_rrcma.py "$name" "$SEED" >> "$OUT/run_seed${SEED}.log" 2>&1
  else
    python3 scripts/niching_baseline.py --funcs "$name" --methods Restart-Lander \
      --evals-frac 1.0 --seeds 1 --seed-offset "$SEED_INDEX" --report-rule current \
      --csv "$OUT/by_problem/${name}.csv" > "$OUT/by_problem/${name}.log" 2>&1
  fi
  echo "done $kind $name $(date -u +%H:%M:%S)" >> "$OUT/run_seed${SEED}.log"
}
export -f run_one

for i in $(seq -w 1 16); do
  echo "rr M${i}-D10-PIN01"
  echo "null M${i}-D10-PIN01"
done | xargs -P 4 -n 2 bash -c 'run_one "$0" "$1"'
echo "=== all done $(date -u +%H:%M:%S)" >> "$LOG"
