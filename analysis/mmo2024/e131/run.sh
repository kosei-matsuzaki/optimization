#!/usr/bin/env bash
# その131 (キュー 2) — seed 200 の欠けたセルだけを埋める（9 run）。
#
# 欠けは保存物の `problem` 列を数えて確認した（その123 の教訓）:
#   RR-CMA-ES      : M13 / M14 / M15 / M16            （4 本）
#   Restart-Lander : M11 / M13 / M14 / M15 / M16      （5 本）
# 駆動はその129 の run.sh と同一で、変えたのは**回すセルの一覧だけ**。
# 出力先もその129 と同じ（RR は e127/dumps、null は e129/descents）。あとで e129 の
# 既存 combined `.csv.gz` に足し直して per-problem は消す（e131/fold_into_e129.py）。
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
LOGDIR="analysis/mmo2024/e131"
SEED="${SEED:-200}"
SEED_INDEX=$((SEED / 100))
export OUT LOGDIR SEED SEED_INDEX
export RESTART_LANDER_DUMP="$OUT/descents"
mkdir -p "analysis/mmo2024/e127/dumps" "$OUT/descents" "$OUT/by_problem"
LOG="$LOGDIR/run_seed${SEED}.log"
: > "$LOG"

run_one() {
  kind="$1"; name="$2"
  if [ "$kind" = rr ]; then
    python3 analysis/mmo2024/e127/run_rrcma.py "$name" "$SEED" >> "$LOGDIR/run_seed${SEED}.log" 2>&1
  else
    python3 scripts/niching_baseline.py --funcs "$name" --methods Restart-Lander \
      --evals-frac 1.0 --seeds 1 --seed-offset "$SEED_INDEX" --report-rule current \
      --csv "$OUT/by_problem/${name}.csv" > "$OUT/by_problem/${name}.log" 2>&1
  fi
  echo "done $kind $name $(date -u +%H:%M:%S)" >> "$LOGDIR/run_seed${SEED}.log"
}
export -f run_one

# 長い側（RR、1 run 124-642 秒）を先に入れ、短い側（null、~70 秒）で埋める。
{
  for i in 13 14 15 16; do echo "rr M${i}-D10-PIN01"; done
  for i in 11 13 14 15 16; do echo "null M${i}-D10-PIN01"; done
} | xargs -P 4 -n 2 bash -c 'run_one "$0" "$1"'
echo "=== all done $(date -u +%H:%M:%S)" >> "$LOG"
