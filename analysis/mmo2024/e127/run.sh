#!/usr/bin/env bash
# その127 (キュー 1) — RR-CMA-ES を 16 問 × seed 0 × 正規予算 50 万で回す。
# 4 並列（このコンテナは 4 コア。acceptance_topology.md の環境節）。
set -u
cd "$(dirname "$0")/../../.."
# pynmmso の空 stub（その86 の形。コンテナが変わると消えるので毎回作り直す）
PYSTUB="${PYSTUB:-/tmp/pystub}"
mkdir -p "$PYSTUB/pynmmso"
cat > "$PYSTUB/pynmmso/__init__.py" <<'EOF'
class Nmmso:
    def __init__(self, *a, **k):
        raise RuntimeError("pynmmso stub: NMMSO is not runnable here")
EOF
export PYTHONPATH="${PYTHONPATH:-}:$PYSTUB"
SEED="${SEED:-0}"
LOG="analysis/mmo2024/e127/run_seed${SEED}.log"
: > "$LOG"
for i in $(seq -w 1 16); do
  echo "M${i}-D10-PIN01"
done | xargs -P 4 -I{} sh -c \
  "python3 analysis/mmo2024/e127/run_rrcma.py {} $SEED >> $LOG 2>&1"
echo "done: $(grep -c 'seed' "$LOG") / 16"
