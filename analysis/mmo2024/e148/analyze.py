#!/usr/bin/env python3
"""その148 — キュー 1: 崖の位置を 1 段詰める（1540 / **2200** / 3000 / 6000 / 12500）。

**採点・規則・統計量は e115 からそのまま import する**（`rule_indices` / `aggregate`
/ `paired` / `mean_of` / `SPAN` / `LEVEL_NAMES`）。**新しい統計量は 1 つも定義しない。**
被覆は その113・その145・その146 と同じ定義。

入力の出自:

  * **2200（この回の腕）**      -> `e148/descents_2200/`（畳んだ後は `e148/descents.csv.gz`）
  * **1840 / 2570（2 段目）**   -> 同上（置いた場合だけ）
  * **3000 / 6000（その146）**  -> `e146/descents.csv.gz` の `arm` 列（**追加評価ゼロ**）
  * **1540（その145）**         -> `e145/descents.csv.gz`（**追加評価ゼロ**）
  * **12500（現行 null、seed 0）** -> `e115/descents/`（**追加評価ゼロ**、関門にも使う）
  * 比較先の定数 -> `e116/ranking_d10.csv`（関門）、その139 / その116 の 16 問平均

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e148/analyze.py
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

from analyze import (LEVEL_NAMES, SPAN, aggregate, mean_of,        # noqa: E402
                     paired, rule_indices)
from core.benchmarks import niching_by_name                        # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN     # その115 の合法な最良腕
EPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

REF_MCESO_SCORE = 0.1385        # その116（seed 0、16 問平均 Score）
REF_NMMSO_SCORE = 0.4144        # その139（新既定 10·D、seed 0）
REF_NULL_SCORE = 0.6284         # その116（seed 0）
REF_1540_SCORE = 0.3284         # その145（seed 0）
REF_3000_SCORE = 0.5633         # その146（seed 0）
REF_6000_SCORE = 0.6141         # その146（seed 0）

NEW_ARMS = [2200, 1840, 2570]   # この回が置きうる水準（存在するものだけ使う）
OLD = {1540: ("e145", None), 3000: ("e146", "3000"), 6000: ("e146", "6000")}
BY_PROBLEM = os.path.join(HERE, "by_problem.csv")
FOLDED = os.path.join(HERE, "descents.csv.gz")

_BENCH: dict = {}


def bench(prob):
    if prob not in _BENCH:
        b = niching_by_name(prob)
        _BENCH[prob] = (int(b.n_global_optima), np.asarray(b.optima_pos, dtype=float))
    return _BENCH[prob]


def attribute(x, optima):
    """最近傍の最適への帰属（`core.runner.count_goptima_nn` と同順序）。オラクル量。採点専用。"""
    d = np.linalg.norm(x[:, None, :] - optima[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def rows_of(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return list(csv.DictReader(fh))


def coverage(rows, k, fcol="best_f"):
    """その113・その145・その146 と同じ定義: detected(eps) = |{land_opt : best_f <= eps}| / K。"""
    return np.array([len({int(r["land_opt"]) for r in rows
                          if float(r[fcol]) <= e}) / k for e in EPS])


def runs_from_rows(rows, prob, method):
    K, opts = bench(prob)
    f = np.array([float(r["best_f"]) for r in rows])
    x = np.array([[float(r[f"x{i}"]) for i in range(opts.shape[1])] for r in rows])
    return dict(problem=prob, method=method, seed=0, K=K,
                f=f, x=x, opt=attribute(x, opts))


def dir_rows(path):
    """降下ダンプのディレクトリを {問題: rows} で読む。"""
    out: dict = {}
    if not os.path.isdir(path):
        return out
    for fn in sorted(os.listdir(path)):
        if fn.endswith((".csv", ".csv.gz")):
            out[fn.split("_")[0]] = rows_of(os.path.join(path, fn))
    return out


def load_all():
    """水準ごとの {問題: rows}。この回の腕は畳んだ 1 本があればそちらから読む。"""
    per: dict = {}
    if os.path.exists(FOLDED):
        for r in rows_of(FOLDED):
            per.setdefault(int(r["arm"]), {}).setdefault(r["problem"], []).append(r)
    else:
        for b in NEW_ARMS:
            d = dir_rows(os.path.join(HERE, f"descents_{b}"))
            if d:
                per[b] = d
    for r in rows_of(os.path.join(MMO, "e145", "descents.csv.gz")):
        per.setdefault(1540, {}).setdefault(r["problem"], []).append(r)
    for r in rows_of(os.path.join(MMO, "e146", "descents.csv.gz")):
        per.setdefault(int(r["arm"]), {}).setdefault(r["problem"], []).append(r)
    per[12500] = dir_rows(os.path.join(MMO, "e115", "descents"))
    return per


def stored_e116(method="Restart-Lander"):
    out = {}
    with open(os.path.join(MMO, "e116", "ranking_d10.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == "legal" and r["method"] == method:
                out[r["problem"]] = float(r["score"])
    return out


def write_by_problem(budgets, probs, agg, cov, nn):
    """問題別の値を 1 本の集計 CSV に（水準 × 問題。**行単位の生データではない**）。

    **この回が依拠する数値はすべてここと `scored.txt` と docs に載る** ＝
    降下ダンプを畳んでも消しても数値は残る。
    """
    with open(BY_PROBLEM, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["descent_budget", "problem", "K", "n_descents", "score", "mpr",
                    "mean_f1", "n_reported"]
                   + [f"cov_{lv}" for lv in LEVEL_NAMES] + ["cov_mean"])
        for b in budgets:
            for p in probs:
                g = agg[b][p]
                w.writerow([b, p, bench(p)[0], nn[b][p], f"{g['score']:.4f}",
                            f"{g['mpr']:.4f}", f"{g['f1']:.4f}", f"{g['n']:.0f}"]
                           + [f"{v:.4f}" for v in cov[b][p]]
                           + [f"{cov[b][p].mean():.4f}"])


def main() -> int:
    per = load_all()
    budgets = sorted(b for b in per if per[b])
    new = [b for b in NEW_ARMS if b in budgets]
    have = {b: set(per[b]) for b in budgets}
    probs = sorted(set.intersection(*[have[b] for b in budgets]))

    print("=" * 100)
    print("その148 — 崖の位置を 1 段詰める（" + " / ".join(str(b) for b in budgets) + "）")
    print("=" * 100)
    for b in budgets:
        tag = "  <== この回" if b in new else ""
        print(f"  descent_budget={b:>6}: {len(have[b])}/16 問{tag}")
    print(f"  全水準そろった問題: {len(probs)}/16"
          + ("   ** 部分結果（その131: 部分集合は中立ではない）**" if len(probs) < 16 else ""))
    if not probs:
        print("  全水準そろった問題が 1 つも無い。集計しない。")
        return 1
    print(f"  規則: {ARM}  r = {ARM_R / SPAN:g} x span   予算: 50 万、seed 0、D=10、PIN01")

    arm_fn = lambda r: rule_indices(ARM, r["f"], r["K"], r["x"], ARM_R)   # noqa: E731
    agg, cov, nn = {}, {}, {}
    for b in budgets:
        runs = [runs_from_rows(per[b][p], p, f"RL-{b}") for p in probs]
        agg[b] = aggregate(runs, arm_fn)
        cov[b] = {p: coverage(per[b][p], bench(p)[0]) for p in probs}
        nn[b] = {p: len(per[b][p]) for p in probs}

    # ---------------------------------------------------------------- 関門
    print("\n## 0. 関門 — 採点経路が その116 と同一か（現行 null 12500 の再採点）\n")
    s116 = stored_e116()
    bad = [(p, agg[12500][p]["score"], s116[p]) for p in probs
           if abs(agg[12500][p]["score"] - s116[p]) > 5e-5]
    if bad:
        for p, a, b_ in bad:
            print(f"  ** ずれ ** {p}: 再採点 {a:.4f} 対 e116 記録 {b_:.4f}")
        print("\n  採点経路が当時と違う。判定に進まない。")
        return 1
    print(f"  {len(probs)} 問すべてで現行 null の Score が e116/ranking_d10.csv と 4 桁一致"
          " ==> 採点経路は同一。")
    print("  既存 4 水準の 16 問平均が記録値と一致するか（その145・その146 の再現）:")
    for b, ref in ((1540, REF_1540_SCORE), (3000, REF_3000_SCORE),
                   (6000, REF_6000_SCORE), (12500, REF_NULL_SCORE)):
        if b in agg:
            m = mean_of(agg[b], probs, "score")
            ok = "一致" if abs(m - ref) <= 5e-5 else "** ずれ **"
            print(f"    {b:>6}: 再採点 {m:.4f} 対 記録 {ref:.4f}  {ok}")

    # ---------------------------------------------------------- 読み A（境界）
    print("\n## 1. 読み A — 問題別 Score\n")
    print(f"{'問題':<16}{'K':>4}" + "".join(f"{b:>10}" for b in budgets))
    for p in probs:
        print(f"{p:<16}{bench(p)[0]:>4}"
              + "".join(f"{agg[b][p]['score']:>10.4f}" for b in budgets))
    print("\n" + f"{'descent_budget':<16}{'MPR':>9}{'mean-F1':>10}{'Score':>9}"
                 f"{'n_rep':>9}{'記録値':>10}")
    rec = {1540: REF_1540_SCORE, 3000: REF_3000_SCORE,
           6000: REF_6000_SCORE, 12500: REF_NULL_SCORE}
    means = {}
    for b in budgets:
        means[b] = mean_of(agg[b], probs, "score")
        r = f"{rec[b]:.4f}" if b in rec else "新規"
        print(f"{b:<16}{mean_of(agg[b], probs, 'mpr'):>9.4f}"
              f"{mean_of(agg[b], probs, 'f1'):>10.4f}{means[b]:>9.4f}"
              f"{mean_of(agg[b], probs, 'n'):>9.1f}{r:>10}")
    print(f"{'NMMSO-10D':<16}{'':>9}{'':>10}{REF_NMMSO_SCORE:>9.4f}   （その139）")
    print(f"{'MC-ESO':<16}{'':>9}{'':>10}{REF_MCESO_SCORE:>9.4f}   （その116）")

    print("\n  対検定（両側 Wilcoxon exact、隣り合う水準）")
    for lo_, hi_ in zip(budgets, budgets[1:]):
        r = paired({p: agg[hi_][p]["score"] for p in probs},
                   {p: agg[lo_][p]["score"] for p in probs}, probs)
        print(f"    {hi_:>6} − {lo_:<6}: {r['mean']:+.4f}  {r['w']}/{r['t']}/{r['l']}"
              f"  p={r['p']:.4g}  rb={r['rb']:+.3f}")
    for b in new:
        for other in (1540, 3000):
            if other in agg:
                r = paired({p: agg[b][p]["score"] for p in probs},
                           {p: agg[other][p]["score"] for p in probs}, probs)
                print(f"    {b:>6} − {other:<6}: {r['mean']:+.4f}  {r['w']}/{r['t']}/{r['l']}"
                      f"  p={r['p']:.4g}  rb={r['rb']:+.3f}   （副次）")

    # ------------------------------------------------- 事前登録した反証条件
    print("\n## 2. 事前登録した反証条件（読み A）\n")
    for b in new:
        c1 = means[b] > REF_NMMSO_SCORE
        c3 = means[b] > means.get(3000, float("inf"))
        print(f"  [{b}] 平均 Score {means[b]:.4f}")
        print(f"    (C1) > NMMSO {REF_NMMSO_SCORE:.4f} か: "
              f"{'** 発火 **' if c1 else 'いいえ'}")
        print(f"    (C2) < NMMSO {REF_NMMSO_SCORE:.4f} か: "
              f"{'** 発火 ** ==> 内挿より急な段がある ＝「崖」と書いてよい' if not c1 else 'いいえ'}")
        print(f"    (C3) > 3000 の {means.get(3000, float('nan')):.4f} か（単調の破れ）: "
              f"{'** 発火 ** ==> この回で判定を書かず俯瞰に上げる' if c3 else 'いいえ'}")

    lo = max([b for b in budgets if means[b] < REF_NMMSO_SCORE], default=None)
    hi = min([b for b in budgets if means[b] > REF_NMMSO_SCORE], default=None)
    if lo is not None and hi is not None:
        print(f"\n  ** 交差点を挟んだ区間: {lo} < X < {hi} 評価（{hi / lo:.2f} 倍幅）"
              f" ＝ 正規予算 500000 の 1/{500000 / hi:.0f} 〜 1/{500000 / lo:.0f} **")
        print(f"  ** 既定 12500 は下限 {lo} の {12500 / lo:.1f} 倍上 **")
    else:
        print("\n  ** 全水準が NMMSO の同じ側 ＝ この点では境界を挟めなかった **")
    print(f"  参考: log-budget 線形内挿による交差点の見込み（事前登録）= 1963")

    # ---------------------------------------------------------- 読み B（機序）
    print("\n## 3. 読み B — 降下ダンプからの被覆（その113 と同じ定義。追加評価ゼロ）\n")
    print(f"{'水準':<8}{'降下本数':>10}" + "".join(f"{lv:>9}" for lv in LEVEL_NAMES)
          + f"{'5 水準平均':>11}{'深さ保持率':>11}")
    lv_of = {}
    for b in budgets:
        lv = [float(np.mean([cov[b][p][i] for p in probs])) for i in range(5)]
        lv_of[b] = lv
        print(f"{b:<8}{np.mean([nn[b][p] for p in probs]):>10.1f}"
              + "".join(f"{v:>9.4f}" for v in lv)
              + f"{float(np.mean([cov[b][p].mean() for p in probs])):>11.4f}"
              + f"{lv[4] / lv[0] if lv[0] else float('nan'):>11.3f}")
    print("\n  事前登録した予測 1: この回の腕の 1e-1 被覆は 3000 の値の 95% 以上か")
    for b in new:
        v, v3 = lv_of[b][0], lv_of[3000][0]
        print(f"    {b:>6}: {v:.4f} 対 3000 の {v3:.4f} = {v / v3 * 100:.1f}%  "
              f"{'==> 予測どおり' if v / v3 >= 0.95 else '==> ** 外れ ** 「広さは配置が作る」の読みを疑うこと'}")
    print("  事前登録した予測 2: 深さ保持率が 1540（0.186）と 3000 の間に入るか（単調）")
    r1540, r3000 = lv_of[1540][4] / lv_of[1540][0], lv_of[3000][4] / lv_of[3000][0]
    for b in new:
        rb_ = lv_of[b][4] / lv_of[b][0]
        ok = r1540 <= rb_ <= r3000
        print(f"    {b:>6}: {rb_:.3f}  （1540 {r1540:.3f} / 3000 {r3000:.3f}）  "
              f"{'==> 予測どおり' if ok else '==> ** 外れ ** 「深さは hunt 長が作る」の読みを疑うこと'}")

    write_by_problem(budgets, probs, agg, cov, nn)
    print(f"\n  問題別の値を {os.path.relpath(BY_PROBLEM, ROOT)} に書いた。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
