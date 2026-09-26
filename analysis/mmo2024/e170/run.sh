#!/bin/bash
# e170: CF3 の梯子（D=2/3/5）を RL の実走 ＋ desc null で引く。
# NMMSO は回さない（pynmmso は stub ＝ その86 の近道）。core/ は触らない。
# 使い方: PYNMMSO_STUB=<stub dir> ./analysis/mmo2024/e170/run.sh
set -u
cd "$(dirname "$0")/../../.."
OUT=analysis/mmo2024/e170
export PYTHONPATH=${PYNMMSO_STUB:?set PYNMMSO_STUB to the stub dir}
FUNCS="N13-CF3-2D N14-CF3-3D N16-CF3-5D"

# ── (a) RL の実走。降下ダンプ（land_opt / best_f / evals）と報告集合ダンプを取る ──
for F in $FUNCS; do
  RESTART_LANDER_DUMP=$OUT/rl_descents/$F \
  REPORT_SET_DUMP=$OUT/rl_reports/$F \
  python3 scripts/niching_baseline.py --funcs $F --methods Restart-Lander \
    --seeds 3 --evals-frac 1.0 --csv $OUT/rl_$F.csv > $OUT/rl_$F.log 2>&1 &   # 後で baseline_all.csv / run_all.log に畳む
done
wait
echo "=== (a) done: $(find $OUT/rl_descents -type f | wc -l) descent dumps ==="

# ── (b) desc null（その88・その89 と同じ既定: sigma-ratio 0.2、1499 評価）──
for F in $FUNCS; do
  python3 scripts/hunt_coverage.py --null --func $F --descents 2000 \
    --budget 1499 --geo-draws 200000 --procs 1 \
    --csv $OUT/null/${F}_iso.csv > $OUT/null_$F.log 2>&1 &
done
wait

# ── (d) 事後 —— null の降下を RL の実値に合わせる ─────────────────────────
# RL の既定は sigma_ratio=0.1 / descent_budget=12500
# （`core/optimizers/restart_lander.py:55-56`）だが、`hunt_coverage.py --null` の
# 既定は 0.2 / 1499 ＝ **その88・その89 が上限を引いた降下は RL の降下ではない。**
for F in $FUNCS; do
  python3 scripts/hunt_coverage.py --null --func $F --descents 2000 \
    --budget 12500 --sigma-ratio 0.1 --geo-draws 1000 --procs 1 \
    --csv $OUT/null/${F}_iso_s01.csv > $OUT/null01_$F.log 2>&1 &
done
wait

# `hunt_coverage.py --null --csv` は素の CSV を書く（拡張子は見ない）ので自分で圧縮する
gzip -f $OUT/null/*.csv 2>/dev/null
echo "=== (b)(d) done: $(ls $OUT/null/*.gz | wc -l) null dumps ==="
echo "次: python3 analysis/mmo2024/e170/fold.py && python3 analysis/mmo2024/e170/analyze.py"
