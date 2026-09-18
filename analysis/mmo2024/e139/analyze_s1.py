#!/usr/bin/env python3
"""その139 の確認 — seed 0 で符号が薄かった 5 問を seed 1 で読み直す。

**NMMSO は seed 固定でも run 間で完全再現しない**ので、主判定（null を上回る問題があるか）が
+0.0007 のような薄い差に乗っている場合、1 seed では符号を主張できない。
ここは **同じ規則・同じ採点器で seed 1 の対**を作るだけ。**新しい統計量は定義しない。**

null 側は追加評価ゼロ（`e115/s1/descents/` の保存ダンプ）。
NMMSO 側は `e139/report_sets.csv.gz` の `seed == 1` の行。

使い方: python3 analysis/mmo2024/e139/analyze_s1.py
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MMO, "e115"))

import importlib.util                                               # noqa: E402

from analyze import SPAN, aggregate, read_dump, rule_indices        # noqa: E402

# e139/analyze.py の読み口（`read_report_sets` / `attribute` / `bench`）を再利用する
# ＝ 採点経路を 2 度書かない。ファイル名で読むのは e115 の `analyze` と名前が衝突するため。
_spec = importlib.util.spec_from_file_location(
    "e139_analyze", os.path.join(HERE, "analyze.py"))
_e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e)

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN
PROBS = [f"M{n}-D10-PIN01" for n in ("01", "02", "03", "04", "10")]
# その139 seed 0 の実測（scored.txt §2）
S0 = {"M01-D10-PIN01": (0.7607, 0.7600), "M02-D10-PIN01": (0.6421, 0.6069),
      "M03-D10-PIN01": (0.5594, 0.5717), "M04-D10-PIN01": (0.7618, 0.7189),
      "M10-D10-PIN01": (0.7600, 0.6529)}


def main() -> int:
    arm_fn = lambda r: rule_indices(ARM, r["f"], r["K"], r["x"], ARM_R)  # noqa: E731
    runs = []
    for prob, (f, x) in _e.read_report_sets(seed=1).items():
        if len(f) == 0:
            continue
        K, opts = _e.bench(prob)
        runs.append(dict(problem=prob, method="NMMSO-10D", seed=1, K=K,
                         f=f, x=x, opt=_e.attribute(x, opts)))
    dd = os.path.join(MMO, "e115", "s1", "descents")
    for fn in sorted(os.listdir(dd)) if os.path.isdir(dd) else []:
        prob = fn.split("_")[0]
        if prob not in PROBS:
            continue
        f, _opt, x = read_dump(os.path.join(dd, fn))
        if x is None:
            continue
        K, opts = _e.bench(prob)
        runs.append(dict(problem=prob, method="Restart-Lander", seed=1, K=K,
                         f=f, x=x, opt=_e.attribute(x, opts)))

    agg = {m: aggregate([r for r in runs if r["method"] == m], arm_fn)
           for m in ("NMMSO-10D", "Restart-Lander")}
    probs = [p for p in PROBS
             if p in agg["NMMSO-10D"] and p in agg["Restart-Lander"]]
    print("=" * 84)
    print("その139 確認 — seed 0 で薄かった 5 問を seed 1 で読み直す")
    print("=" * 84)
    print(f"  揃った問題: {len(probs)}/{len(PROBS)}"
          + ("   ** 部分結果 **" if len(probs) < len(PROBS) else ""))
    if not probs:
        return 1
    print(f"\n{'問題':<16}{'NMMSO s1':>10}{'null s1':>9}{'差 s1':>9}"
          f"{'差 s0':>9}{'符号一致':>10}")
    rows = []
    for p in probs:
        n1, r1 = agg["NMMSO-10D"][p]["score"], agg["Restart-Lander"][p]["score"]
        d1, d0 = n1 - r1, S0[p][0] - S0[p][1]
        same = "○" if (d1 > 0) == (d0 > 0) else "×"
        print(f"{p:<16}{n1:>10.4f}{r1:>9.4f}{d1:>+9.4f}{d0:>+9.4f}{same:>10}")
        rows.append([p, f"{n1:.4f}", f"{r1:.4f}", f"{d1:+.4f}", f"{d0:+.4f}", same])
    up = [p for p in probs if agg["NMMSO-10D"][p]["score"]
          > agg["Restart-Lander"][p]["score"]]
    print(f"\n  seed 1 で null を上回った問題: {len(up)}/{len(probs)}"
          + (f"  ({', '.join(up)})" if up else ""))
    print("  ==> 2 seed とも上回る問題があれば、主判定の発火は 1 run の揺れでは説明できない。")
    out = os.path.join(HERE, "seed1_check.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "nmmso_s1", "null_s1", "diff_s1", "diff_s0",
                    "sign_agrees"])
        w.writerows(rows)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
