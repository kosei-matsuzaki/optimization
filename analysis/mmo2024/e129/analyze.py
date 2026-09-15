#!/usr/bin/env python3
"""その129 — RR-CMA-ES 対 `Restart-Lander` の直接対決を 3 本目の seed（200）で決める（キュー 1(A)）。

**新しい規則も新しい統計量も 1 つも定義しない。** `rule_indices` / `score` / `paired` は
`analysis/mmo2024/e115/analyze.py` から import する（その116 以降の全サイクルと同一）。
`read_combined` は e128 の**入出力**ヘルパの写しで、統計量ではない。

  * `RR-CMA-ES` seed 0    -> analysis/mmo2024/e127/dumps/*_rrcma_seed0.csv.gz   （その127 の保存物）
  * `RR-CMA-ES` seed 100  -> analysis/mmo2024/e128/dumps_rrcma_seed100.csv.gz   （その128 の保存物）
  * `RR-CMA-ES` seed 200  -> analysis/mmo2024/e129/dumps/*_rrcma_seed200.csv.gz （今回）
  * `Restart-Lander` s0   -> analysis/mmo2024/e115/descents      （保存物）
  * `Restart-Lander` s100 -> analysis/mmo2024/e115/s1/descents   （保存物）
  * `Restart-Lander` s200 -> analysis/mmo2024/e129/descents      （今回）

**関門は 2 つとも seed 0 側**（`prereg.md`）。

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e129/analyze.py
"""
from __future__ import annotations

import csv
import gzip
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
GATE_NULL = {"mpr": 0.5644, "score": 0.6284}       # その116・その127・その128 の記録
GATE_RR = {"mpr": 0.4594, "score": 0.5377}         # その127・その128 の記録
PUBLISHED_RR = 0.5730                              # 競技資料（出典 1 本、15 instance 平均）
TOL_PUBLISHED = 0.05                               # 事前登録した許容幅
PROBS = [f"M{i:02d}-D10-PIN01" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]

SRC = {
    ("Restart-Lander", 0):   (os.path.join(MMO, "e115", "descents"), "{p}_seed0.csv"),
    ("Restart-Lander", 100): (os.path.join(MMO, "e115", "s1", "descents"), "{p}_seed100.csv"),
    ("Restart-Lander", 200): (os.path.join(HERE, "descents"), "{p}_seed200.csv"),
    ("RR-CMA-ES", 0):        (os.path.join(MMO, "e127", "dumps"), "{p}_rrcma_seed0.csv"),
    ("RR-CMA-ES", 200):      (os.path.join(HERE, "dumps"), "{p}_rrcma_seed200.csv"),
}
COMBINED_RR100 = os.path.join(MMO, "e128", "dumps_rrcma_seed100.csv.gz")
METHODS = ["RR-CMA-ES", "Restart-Lander"]
SEEDS = [0, 100, 200]

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
    """`problem` 列つきの 1 本のダンプを problem -> (f, opt, xs) に割る（e128 の写し）。"""
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

    print("=" * 104)
    print("その129 — 直接対決を 3 本目の seed（200）で決める（キュー 1(A)、新規 run は 32 本）")
    print("=" * 104)
    print(f"\n  規則: {ARM} (r={ARM_R})")
    for (m, s) in sorted(cell, key=lambda k: (k[0], k[1])):
        miss = [p for p in PROBS if p not in cell[(m, s)]]
        tag = "揃った" if not miss else f"**欠け: {','.join(x[:3] for x in miss)}**"
        print(f"    {m:<16} seed {s:>3}: {len(cell[(m, s)])}/16  {tag}")

    # 判定に使う問題 = 3 seed × 2 手法がすべて揃っている問題（事前登録の打ち切り規則）
    probs = [p for p in PROBS
             if all(p in cell[(m, s)] for m in METHODS for s in SEEDS)]
    # seed 200 単独の判定は seed 200 が両手法そろった問題で取る
    probs200 = [p for p in PROBS if all(p in cell[(m, 200)] for m in METHODS)]
    dropped = [p for p in PROBS if p not in probs]
    if dropped:
        print(f"\n  **3 seed が揃わなかった問題（3 seed 平均の判定から外す）: {', '.join(dropped)}**")
    if [p for p in PROBS if p not in probs200]:
        print(f"  **seed 200 が揃わなかった問題: "
              f"{', '.join(p for p in PROBS if p not in probs200)}**")

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

    # ------------------------------------------------- 問題別（seed 200）
    print(f"\n  [問題別 Score]  seed 200 が揃った n={len(probs200)}")
    print(f"{'problem':<16}{'K':>4}{'RR s200':>10}{'RL s200':>10}{'d s200':>10}"
          f"{'RR 3seed':>11}{'RL 3seed':>11}{'d 3seed':>10}")
    print("-" * 104)
    for p in probs200:
        rr, rl = cell[("RR-CMA-ES", 200)][p], cell[("Restart-Lander", 200)][p]
        line = (f"{p:<16}{K_of(p):>4}{rr['score']:10.4f}{rl['score']:10.4f}"
                f"{rl['score'] - rr['score']:+10.4f}")
        if p in probs:
            rr3 = np.mean([cell[("RR-CMA-ES", s)][p]["score"] for s in SEEDS])
            rl3 = np.mean([cell[("Restart-Lander", s)][p]["score"] for s in SEEDS])
            line += f"{rr3:11.4f}{rl3:11.4f}{rl3 - rr3:+10.4f}"
        print(line)

    for label, sub in (("全体（seed 200 が揃った問題）", probs200),
                       ("群 A", [p for p in probs200 if p in GROUP_A]),
                       ("群 B", [p for p in probs200 if p in GROUP_B])):
        if not sub:
            continue
        print(f"\n  [{label}] n={len(sub)}")
        for m in METHODS:
            for s in SEEDS:
                if not any(p in cell[(m, s)] for p in sub):
                    continue
                print(f"    {m:<16} seed {s:>3}  MPR {mean_over(cell, m, s, sub, 'mpr'):.4f}   "
                      f"mean-F1 {mean_over(cell, m, s, sub, 'f1'):.4f}   "
                      f"Score {mean_over(cell, m, s, sub, 'score'):.4f}   "
                      f"報告点数 {mean_over(cell, m, s, sub, 'n'):.1f}")

    # ------------------------------------------------- (J4) 公表値の再現
    print(f"\n  (J4) 公表値との差（Score、{len(probs)} 問平均・3 seed が揃った問題だけ）")
    rrv = [mean_over(cell, "RR-CMA-ES", s, probs, "score") for s in SEEDS]
    rlv = [mean_over(cell, "Restart-Lander", s, probs, "score") for s in SEEDS]
    for lab, v in list(zip([f"seed {s}" for s in SEEDS], rrv)) + \
            [("**3 seed 平均**", float(np.mean(rrv)))]:
        d = v - PUBLISHED_RR
        print(f"    {lab:<16} 実測 {v:.4f} 対 資料 {PUBLISHED_RR:.4f} = {d:+.4f}  "
              f"-> {'±0.05 の内側' if abs(d) <= TOL_PUBLISHED else '**±0.05 の外**'}")
    print(f"    16 問平均の seed 間の散らばり（n=3 の SD）: "
          f"RR-CMA-ES {np.std(rrv, ddof=1):.4f}（{min(rrv):.4f}-{max(rrv):.4f}） / "
          f"Restart-Lander {np.std(rlv, ddof=1):.4f}（{min(rlv):.4f}-{max(rlv):.4f}）")

    # ------------------------------------------------- (J1)(J2) 直接対決
    print(f"\n  (J1)(J2) 対比較 Restart-Lander − RR-CMA-ES（問題ごと、Wilcoxon exact ＋ rank-biserial）")
    rows_pair = []
    for key in ("mpr", "score"):
        bases = [(f"seed {s}", probs200 if s == 200 else probs,
                  (lambda m, p, s=s, k=key: cell[(m, s)][p][k])) for s in SEEDS]
        bases.append(("3 seed 平均", probs,
                      lambda m, p, k=key: float(np.mean([cell[(m, s)][p][k] for s in SEEDS]))))
        for lab, sub, getter in bases:
            sub = [p for p in sub if all(p in cell[(m, s)] for m in METHODS for s in SEEDS)] \
                if lab == "3 seed 平均" else sub
            a = {p: getter("Restart-Lander", p) for p in sub}
            b = {p: getter("RR-CMA-ES", p) for p in sub}
            r = paired(a, b, sub)
            print(f"    {key:<5} {lab:<12} n={len(sub):>2}  mean {r['mean']:+.4f}  "
                  f"W/T/L {r['w']}/{r['t']}/{r['l']}  p={r['p']:.4g}  rb={r['rb']:+.3f}")
            rows_pair.append([key, lab, len(sub), f"{r['mean']:.6f}", r["w"], r["t"], r["l"],
                              f"{r['p']:.6g}", f"{r['rb']:.4f}"])

    # ------------------------------------------------- 集計 CSV
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
                    w.writerow([p, K_of(p), m, s, ARM, f"{v['n']:.0f}",
                                f"{v['mpr']:.6f}", f"{v['f1']:.6f}", f"{v['score']:.6f}"])
    out2 = os.path.join(HERE, "paired_d10.csv")
    with open(out2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "basis", "n", "mean_diff", "win", "tie", "loss", "p", "rb"])
        w.writerows(rows_pair)
    print(f"\n  -> {out}\n  -> {out2}")


if __name__ == "__main__":
    main()
