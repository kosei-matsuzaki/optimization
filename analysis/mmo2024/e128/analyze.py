#!/usr/bin/env python3
"""その128 — RR-CMA-ES 対 `Restart-Lander` の直接対決を seed 100 に広げる（キュー 1）。

**新しい規則も新しい統計量も 1 つも定義しない。** `rule_indices` / `score` / `paired` は
`analysis/mmo2024/e115/analyze.py` から、問題別の記録値は e116 / e127 から読む。違うのは入力だけ。

  * `RR-CMA-ES` seed 0    -> analysis/mmo2024/e127/dumps/*_rrcma_seed0.csv.gz  （その127 の保存物）
  * `RR-CMA-ES` seed 100  -> analysis/mmo2024/e128/dumps/*_rrcma_seed100.csv.gz（今回の 16 run）
  * `Restart-Lander` s0   -> analysis/mmo2024/e115/descents      （保存物。追加 run ゼロ）
  * `Restart-Lander` s100 -> analysis/mmo2024/e115/s1/descents   （保存物。追加 run ゼロ）

**関門は 2 つとも seed 0 側**（`prereg.md`。seed 100 には突き合わせる記録がまだ無い）:
  1. `Restart-Lander` seed 0 の 16 問平均が MPR 0.5644 / Score 0.6284（その116・その127）。
  2. `RR-CMA-ES` seed 0 の 16 問平均が MPR 0.4594 / Score 0.5377（その127）。

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e128/analyze.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MMO, "e115"))

from analyze import SPAN, paired, read_dump, rule_indices, score   # noqa: E402
from core.benchmarks import niching_by_name                        # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN
GATE_NULL = {"mpr": 0.5644, "score": 0.6284}       # その116・その127 の記録
GATE_RR = {"mpr": 0.4594, "score": 0.5377}         # その127 の記録
PUBLISHED_RR = 0.5730                              # 競技資料（出典 1 本、15 instance 平均）
TOL_PUBLISHED = 0.05                               # 事前登録した許容幅
PROBS = [f"M{i:02d}-D10-PIN01" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]

SRC = {
    ("Restart-Lander", 0):   (os.path.join(MMO, "e115", "descents"), "{p}_seed0.csv"),
    ("Restart-Lander", 100): (os.path.join(MMO, "e115", "s1", "descents"), "{p}_seed100.csv"),
    ("RR-CMA-ES", 0):        (os.path.join(MMO, "e127", "dumps"), "{p}_rrcma_seed0.csv"),
}
# 今回の 16 run は **1 本にまとめた `.csv.gz`**（`problem` 列つき、503 行）。
# 問題ごとに 16 ファイルへ散らすとファイル数だけが増えて中身は同じなので、
# CLAUDE.md の保持規則（行単位は .csv.gz）を満たす形のまま 1 本に畳んである。
COMBINED_RR100 = os.path.join(HERE, "dumps_rrcma_seed100.csv.gz")
METHODS = ["RR-CMA-ES", "Restart-Lander"]
SEEDS = [0, 100]

_K: dict = {}


def K_of(prob):
    if prob not in _K:
        _K[prob] = int(niching_by_name(prob).n_global_optima)
    return _K[prob]


def find(d, name):
    for ext in ("", ".gz"):
        p = os.path.join(d, name + ext)
        if os.path.exists(p):
            return p
    return None


def read_combined(path):
    """`problem` 列つきの 1 本のダンプを problem -> (f, opt, xs) に割る。

    列の意味も採点も per-problem のファイルと同一（`read_dump` と同じ 3 つを返す）。
    """
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    out: dict = {}
    for r in rows:
        out.setdefault(r["problem"], []).append(r)
    res = {}
    for p, rs in out.items():
        dim = sum(1 for k in rs[0] if k.startswith("x") and k[1:].isdigit())
        res[p] = (np.array([float(r["best_f"]) for r in rs]),
                  np.array([int(r["land_opt"]) for r in rs]),
                  np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rs]))
    return res


def score_arrays(f, opt, xs, K):
    idx = rule_indices(ARM, f, K, x=xs, r=ARM_R)
    recall, prec, f1, sc, n = score(idx, f, opt, K)
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()),
                score=float(sc.mean()), n=int(n), ndump=len(f))


def score_dump(path, K):
    f, opt, xs = read_dump(path)
    return score_arrays(f, opt, xs, K)


def mean_over(cell, m, s, sub, key):
    vals = [cell[(m, s)][p][key] for p in sub if p in cell[(m, s)]]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    cell = {(m, s): {} for m in METHODS for s in SEEDS}
    for (m, s), (d, pat) in SRC.items():
        for p in PROBS:
            q = find(d, pat.format(p=p))
            if q:
                cell[(m, s)][p] = score_dump(q, K_of(p))
    if os.path.exists(COMBINED_RR100):
        for p, (f, opt, xs) in read_combined(COMBINED_RR100).items():
            if p in PROBS:
                cell[("RR-CMA-ES", 100)][p] = score_arrays(f, opt, xs, K_of(p))

    print("=" * 100)
    print("その128 — RR-CMA-ES 対 Restart-Lander を seed 100 に広げる（キュー 1、新規 run は RR 側 16 本だけ）")
    print("=" * 100)
    print(f"\n  規則: {ARM} (r={ARM_R})")
    for (m, s) in sorted(cell, key=lambda k: (k[0], k[1])):
        miss = [p for p in PROBS if p not in cell[(m, s)]]
        tag = "揃った" if not miss else f"**欠け: {','.join(x[:3] for x in miss)}**"
        print(f"    {m:<16} seed {s:>3}: {len(cell[(m, s)])}/16  {tag}")

    # 判定に使う問題 = RR-CMA-ES が両 seed で揃っている問題
    probs = [p for p in PROBS
             if all(p in cell[(m, s)] for m in METHODS for s in SEEDS)]
    dropped = [p for p in PROBS if p not in probs]
    if dropped:
        print(f"\n  **判定から外した問題（どれかの seed が欠けている）: {', '.join(dropped)}**")

    # ------------------------------------------------- 関門（どちらも seed 0）
    ok = True
    for m, g in (("Restart-Lander", GATE_NULL), ("RR-CMA-ES", GATE_RR)):
        a = mean_over(cell, m, 0, PROBS, "mpr")
        b = mean_over(cell, m, 0, PROBS, "score")
        good = abs(a - g["mpr"]) < 5e-5 and abs(b - g["score"]) < 5e-5
        ok &= good
        print(f"\n  関門（{m} seed 0 の 16 問平均）: MPR {a:.4f} (記録 {g['mpr']}) / "
              f"Score {b:.4f} (記録 {g['score']})  -> {'一致' if good else '**不一致**'}")
    if not ok:
        print("\n  **採点経路が再現しないので判定に進まない。**")
        return

    # ------------------------------------------------- 問題別（seed 100）
    print(f"\n  [問題別 Score / MPR]  n={len(probs)}")
    print(f"{'problem':<16}{'K':>4}" + f"{'RR s0':>10}{'RR s100':>10}{'RL s0':>10}{'RL s100':>10}"
          + f"{'RR n s100':>11}{'d(RL-RR) s100':>15}")
    print("-" * 100)
    for p in probs:
        rr0, rr1 = cell[("RR-CMA-ES", 0)][p], cell[("RR-CMA-ES", 100)][p]
        rl0, rl1 = cell[("Restart-Lander", 0)][p], cell[("Restart-Lander", 100)][p]
        print(f"{p:<16}{K_of(p):>4}{rr0['score']:10.4f}{rr1['score']:10.4f}"
              f"{rl0['score']:10.4f}{rl1['score']:10.4f}{rr1['n']:11.0f}"
              f"{rl1['score'] - rr1['score']:+15.4f}")

    for label, sub in (("全体", probs),
                       ("群 A", [p for p in probs if p in GROUP_A]),
                       ("群 B", [p for p in probs if p in GROUP_B])):
        if not sub:
            continue
        print(f"\n  [{label}] n={len(sub)}")
        for m in METHODS:
            for s in SEEDS:
                print(f"    {m:<16} seed {s:>3}  MPR {mean_over(cell, m, s, sub, 'mpr'):.4f}   "
                      f"mean-F1 {mean_over(cell, m, s, sub, 'f1'):.4f}   "
                      f"Score {mean_over(cell, m, s, sub, 'score'):.4f}   "
                      f"報告点数 {mean_over(cell, m, s, sub, 'n'):.1f}")

    # ------------------------------------------------- (J1) 公表値の再現
    rr0 = mean_over(cell, "RR-CMA-ES", 0, probs, "score")
    rr1 = mean_over(cell, "RR-CMA-ES", 100, probs, "score")
    rr2 = (rr0 + rr1) / 2.0
    print(f"\n  (J1) 公表値との差（Score、16 問平均）")
    for lab, v in (("seed 0", rr0), ("seed 100", rr1), ("**2 seed 平均**", rr2)):
        d = v - PUBLISHED_RR
        print(f"    {lab:<16} 実測 {v:.4f} 対 資料 {PUBLISHED_RR:.4f} = {d:+.4f}  "
              f"-> {'±0.05 の内側' if abs(d) <= TOL_PUBLISHED else '**±0.05 の外**'}")

    # ------------------------------------------------- (J2) 直接対決
    print(f"\n  (J2) 対比較 Restart-Lander − RR-CMA-ES（問題ごと、Wilcoxon exact ＋ rank-biserial）")
    rows_pair = []
    for key in ("mpr", "score"):
        for lab, getter in (
            ("seed 0", lambda m, p, k=key: cell[(m, 0)][p][k]),
            ("seed 100", lambda m, p, k=key: cell[(m, 100)][p][k]),
            ("2 seed 平均", lambda m, p, k=key: (cell[(m, 0)][p][k] + cell[(m, 100)][p][k]) / 2.0),
        ):
            a = {p: getter("Restart-Lander", p) for p in probs}
            b = {p: getter("RR-CMA-ES", p) for p in probs}
            r = paired(a, b, probs)
            print(f"    {key:<5} {lab:<12} mean {r['mean']:+.4f}  W/T/L {r['w']}/{r['t']}/{r['l']}  "
                  f"p={r['p']:.4g}  rb={r['rb']:+.3f}")
            rows_pair.append([key, lab, f"{r['mean']:.6f}", r["w"], r["t"], r["l"],
                              f"{r['p']:.6g}", f"{r['rb']:.4f}"])

    # ------------------------------------------------- (J4) seed 間のずれ
    print(f"\n  (J4) 16 問平均の seed 0 → seed 100 のずれ（2 点なので SD は書かない）")
    for m in METHODS:
        for key in ("mpr", "score"):
            v0 = mean_over(cell, m, 0, probs, key)
            v1 = mean_over(cell, m, 100, probs, key)
            print(f"    {m:<16} {key:<6} {v0:.4f} -> {v1:.4f}  ({v1 - v0:+.4f})")
    dif = [abs(cell[(m, 0)][p]["score"] - cell[(m, 100)][p]["score"])
           for m in METHODS for p in probs]
    print(f"    問題別 |Δscore| の中央値 {np.median(dif):.4f} / 最大 {np.max(dif):.4f}"
          f"（2 手法 × {len(probs)} 問）")

    # ------------------------------------------------- 集計 CSV
    out = os.path.join(HERE, "by_problem_d10.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "method", "seed", "arm", "n_reported",
                    "mpr", "mean_f1", "score"])
        for p in probs:
            for m in METHODS:
                for s in SEEDS:
                    v = cell[(m, s)][p]
                    w.writerow([p, K_of(p), m, s, ARM, f"{v['n']:.0f}",
                                f"{v['mpr']:.6f}", f"{v['f1']:.6f}", f"{v['score']:.6f}"])
    out2 = os.path.join(HERE, "paired_d10.csv")
    with open(out2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "basis", "mean_diff", "win", "tie", "loss", "p", "rb"])
        w.writerows(rows_pair)
    print(f"\n  -> {out}\n  -> {out2}")


if __name__ == "__main__":
    main()
