#!/usr/bin/env python3
"""その131 — 16 問 × 3 seed で RR-CMA-ES 対 `Restart-Lander` の直接対決を閉じる（キュー 2）。

**新しい規則も新しい統計量も 1 つも定義しない。**

  * 採点規則 `rule_indices` / 採点 `score` / 対比較 `paired` … `analysis/mmo2024/e115/analyze.py`
  * ダンプの読み込みと cell の組み立て … `analysis/mmo2024/e129/analyze.py` の
    `SRC` / `COMBINED` / `score_dump` / `read_combined` / `score_arrays` / `K_of` / `find`

その129 との違いは **(a) 16 問がそろうこと**と **(b) (J5) 11 問 対 5 問の部分集合の比較**だけ
（(J5) も平均を取るだけで、新しい統計量ではない）。

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e131/analyze.py
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location(
    "e129_analyze", os.path.join(MMO, "e129", "analyze.py"))
a129 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a129)          # e115 の analyze を内部で import する

paired = sys.modules["analyze"].paired  # e115 の対比較（a129 が読み込んだもの）

PROBS, METHODS, SEEDS = a129.PROBS, a129.METHODS, a129.SEEDS
GROUP_A, GROUP_B = a129.GROUP_A, a129.GROUP_B
PUBLISHED_RR, TOL = a129.PUBLISHED_RR, a129.TOL_PUBLISHED
GATE_NULL, GATE_RR = a129.GATE_NULL, a129.GATE_RR
# その129 が 11 問で出した seed 200 の 11 問平均（関門 3）
GATE_S200_11 = {"RR-CMA-ES": 0.5940, "Restart-Lander": 0.6475}
PROBS11 = [f"M{i:02d}-D10-PIN01" for i in list(range(1, 11)) + [12]]
PROBS5 = [p for p in PROBS if p not in PROBS11]


def build_cell():
    """その129 と同じ経路でダンプを読んで採点する（入出力の写し。統計量は含まない）。"""
    cell = {(m, s): {} for m in METHODS for s in SEEDS}
    for (m, s), (d, pat) in a129.SRC.items():
        for p in PROBS:
            q = a129.find(d, pat.format(p=p))
            if q:
                cell[(m, s)][p] = a129.score_dump(q, a129.K_of(p))
    for path, key in ((a129.COMBINED_RR0, ("RR-CMA-ES", 0)),
                      (a129.COMBINED_RR100, ("RR-CMA-ES", 100))):
        if os.path.exists(path):
            for p, (f, opt, xs) in a129.read_combined(path).items():
                if p in PROBS and p not in cell[key]:
                    cell[key][p] = a129.score_arrays(f, opt, xs, a129.K_of(p))
    for key, path in a129.COMBINED.items():
        if os.path.exists(path):
            for p, (f, opt, xs) in a129.read_combined(path).items():
                if p in PROBS and p not in cell[key]:
                    cell[key][p] = a129.score_arrays(f, opt, xs, a129.K_of(p))
    return cell


def mean_over(cell, m, s, sub, key):
    v = [cell[(m, s)][p][key] for p in sub if p in cell[(m, s)]]
    return float(np.mean(v)) if v else float("nan")


def main():
    cell = build_cell()
    print("=" * 104)
    print("その131 — 16 問 × 3 seed で直接対決を閉じる（キュー 2、新規 run は 9 本）")
    print("=" * 104)
    print(f"\n  規則: {a129.ARM} (r={a129.ARM_R})")
    for key in sorted(cell, key=lambda k: (k[0], k[1])):
        miss = [p for p in PROBS if p not in cell[key]]
        tag = "揃った" if not miss else f"**欠け: {','.join(x[:3] for x in miss)}**"
        print(f"    {key[0]:<16} seed {key[1]:>3}: {len(cell[key])}/16  {tag}")

    probs = [p for p in PROBS if all(p in cell[(m, s)] for m in METHODS for s in SEEDS)]
    if len(probs) < 16:
        print(f"\n  **16 問そろわなかった（n={len(probs)}）: "
              f"{', '.join(p for p in PROBS if p not in probs)}**")

    # --------------------------------------------------------------- 関門
    ok = True
    for m, g in (("Restart-Lander", GATE_NULL), ("RR-CMA-ES", GATE_RR)):
        a = mean_over(cell, m, 0, PROBS, "mpr")
        b = mean_over(cell, m, 0, PROBS, "score")
        good = abs(a - g["mpr"]) < 5e-5 and abs(b - g["score"]) < 5e-5
        ok &= good
        print(f"\n  関門1/2（{m} seed 0 の 16 問平均）: MPR {a:.4f} (記録 {g['mpr']}) / "
              f"Score {b:.4f} (記録 {g['score']}) -> {'一致' if good else '**不一致**'}")
    for m, g in GATE_S200_11.items():
        v = mean_over(cell, m, 200, PROBS11, "score")
        good = abs(v - g) < 5e-5
        ok &= good
        print(f"  関門3（{m} seed 200 の既存 11 問平均 Score）: {v:.4f} (記録 {g}) "
              f"-> {'一致' if good else '**不一致**'}")
    if not ok:
        print("\n  **採点経路が再現しないので判定に進まない。**")
        return

    # --------------------------------------------------------- 問題別 Score
    print(f"\n  [問題別 Score]  n={len(probs)}")
    print(f"{'problem':<16}{'K':>4}{'RR s200':>10}{'RL s200':>10}{'d s200':>10}"
          f"{'RR 3seed':>11}{'RL 3seed':>11}{'d 3seed':>10}")
    print("-" * 104)
    for p in probs:
        rr, rl = cell[("RR-CMA-ES", 200)][p], cell[("Restart-Lander", 200)][p]
        rr3 = np.mean([cell[("RR-CMA-ES", s)][p]["score"] for s in SEEDS])
        rl3 = np.mean([cell[("Restart-Lander", s)][p]["score"] for s in SEEDS])
        mark = "  <- 今回埋めた" if p in PROBS5 else ""
        print(f"{p:<16}{a129.K_of(p):>4}{rr['score']:10.4f}{rl['score']:10.4f}"
              f"{rl['score'] - rr['score']:+10.4f}{rr3:11.4f}{rl3:11.4f}"
              f"{rl3 - rr3:+10.4f}{mark}")

    for label, sub in (("全体 16 問", probs),
                       ("群 A", [p for p in probs if p in GROUP_A]),
                       ("群 B", [p for p in probs if p in GROUP_B])):
        print(f"\n  [{label}] n={len(sub)}")
        for m in METHODS:
            for s in SEEDS:
                print(f"    {m:<16} seed {s:>3}  MPR {mean_over(cell, m, s, sub, 'mpr'):.4f}   "
                      f"mean-F1 {mean_over(cell, m, s, sub, 'f1'):.4f}   "
                      f"Score {mean_over(cell, m, s, sub, 'score'):.4f}   "
                      f"報告点数 {mean_over(cell, m, s, sub, 'n'):.1f}")

    # ------------------------------------------------------ (J4) 公表値の残差
    print(f"\n  (J4) 公表値との差（Score、{len(probs)} 問平均）")
    rrv = [mean_over(cell, "RR-CMA-ES", s, probs, "score") for s in SEEDS]
    rlv = [mean_over(cell, "Restart-Lander", s, probs, "score") for s in SEEDS]
    for lab, v in list(zip([f"seed {s}" for s in SEEDS], rrv)) + \
            [("**3 seed 平均**", float(np.mean(rrv)))]:
        d = v - PUBLISHED_RR
        print(f"    {lab:<16} 実測 {v:.4f} 対 資料 {PUBLISHED_RR:.4f} = {d:+.4f}  "
              f"-> {'±0.05 の内側' if abs(d) <= TOL else '**±0.05 の外**'}")
    print(f"    16 問平均の seed 間の散らばり（n=3 の SD）: "
          f"RR-CMA-ES {np.std(rrv, ddof=1):.4f}（{min(rrv):.4f}-{max(rrv):.4f}） / "
          f"Restart-Lander {np.std(rlv, ddof=1):.4f}（{min(rlv):.4f}-{max(rlv):.4f}）")

    # --------------------------------------------------- (J1)(J2) 直接対決
    print("\n  (J1)(J2) 対比較 Restart-Lander − RR-CMA-ES（問題ごと、Wilcoxon exact ＋ rank-biserial）")
    rows_pair = []
    for key in ("mpr", "score"):
        bases = [(f"seed {s}", (lambda m, p, s=s, k=key: cell[(m, s)][p][k])) for s in SEEDS]
        bases.append(("3 seed 平均",
                      lambda m, p, k=key: float(np.mean([cell[(m, s)][p][k] for s in SEEDS]))))
        for lab, getter in bases:
            a = {p: getter("Restart-Lander", p) for p in probs}
            b = {p: getter("RR-CMA-ES", p) for p in probs}
            r = paired(a, b, probs)
            print(f"    {key:<5} {lab:<12} n={len(probs):>2}  mean {r['mean']:+.4f}  "
                  f"W/T/L {r['w']}/{r['t']}/{r['l']}  p={r['p']:.4g}  rb={r['rb']:+.3f}")
            rows_pair.append([key, lab, len(probs), f"{r['mean']:.6f}", r["w"], r["t"], r["l"],
                              f"{r['p']:.6g}", f"{r['rb']:.4f}"])

    # ------------------------------------ (J5) 11 問部分集合は易しい側だったか
    print("\n  (J5) その129 の 11 問 対 今回埋めた 5 問（3 seed 平均 Score。平均を取るだけ）")
    print(f"{'部分集合':<14}{'n':>3}{'RR':>10}{'RL':>10}{'差':>10}")
    for lab, sub in (("11 問（旧）", PROBS11), ("5 問（新）", PROBS5), ("16 問", probs)):
        rr3 = float(np.mean([np.mean([cell[("RR-CMA-ES", s)][p]["score"] for s in SEEDS])
                             for p in sub]))
        rl3 = float(np.mean([np.mean([cell[("Restart-Lander", s)][p]["score"] for s in SEEDS])
                             for p in sub]))
        print(f"{lab:<14}{len(sub):>3}{rr3:10.4f}{rl3:10.4f}{rl3 - rr3:+10.4f}")

    # --------------------------------------------------------------- 集計 CSV
    out = os.path.join(HERE, "by_problem_d10.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "method", "seed", "arm", "n_reported",
                    "mpr", "mean_f1", "score"])
        for p in PROBS:
            for m in METHODS:
                for s in SEEDS:
                    v = cell[(m, s)].get(p)
                    if v is None:
                        continue
                    w.writerow([p, a129.K_of(p), m, s, a129.ARM, f"{v['n']:.0f}",
                                f"{v['mpr']:.6f}", f"{v['f1']:.6f}", f"{v['score']:.6f}"])
    out2 = os.path.join(HERE, "paired_d10.csv")
    with open(out2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "basis", "n", "mean_diff", "win", "tie", "loss", "p", "rb"])
        w.writerows(rows_pair)
    print(f"\n  -> {out}\n  -> {out2}")


if __name__ == "__main__":
    main()
