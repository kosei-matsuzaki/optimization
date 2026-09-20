#!/usr/bin/env python3
"""その147 — キュー 2: `sigma_ratio` を振る（2 つ目のつまみの監査）。

**採点・規則・統計量は e115 からそのまま import する**（`rule_indices` / `aggregate`
/ `paired` / `mean_of` / `SPAN` / `LEVEL_NAMES`）。**新しい統計量は 1 つも定義しない。**
被覆は その113・その145・その146 と同じ定義（`coverage` は e146/analyze.py と同一の式）。

入力の出自:

  * **0.05 / 0.2（この回の 2 腕）** -> `e147/descents_s005/` `e147/descents_s020/`
    （畳んだ後は `e147/descents.csv.gz` の `arm` 列）
  * **0.1（現行 null、seed 0）**   -> `e115/descents/`（**追加評価ゼロ**、関門にも使う）
  * 比較先の定数 -> `e116/ranking_d10.csv`（関門）、その139 / その116 の 16 問平均

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e147/analyze.py
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
REF_NULL_SCORE = 0.6284         # その116（seed 0、sigma_ratio=0.1）

SIGMAS = [0.05, 0.1, 0.2]
BASE = 0.1                                      # 現行 null（対照）
DUMPDIR = {0.05: "descents_s005", 0.2: "descents_s020"}
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
    """3 水準ぶんの {問題: rows} を返す。畳んだ 1 本があればこの回の 2 腕はそちらから読む。"""
    per: dict = {}
    if os.path.exists(FOLDED):
        for r in rows_of(FOLDED):
            per.setdefault(float(r["arm"]), {}).setdefault(r["problem"], []).append(r)
    else:
        for s, d in DUMPDIR.items():
            per[s] = dir_rows(os.path.join(HERE, d))
    per[BASE] = dir_rows(os.path.join(MMO, "e115", "descents"))
    return per


def stored_e116(method="Restart-Lander"):
    out = {}
    with open(os.path.join(MMO, "e116", "ranking_d10.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == "legal" and r["method"] == method:
                out[r["problem"]] = float(r["score"])
    return out


def write_by_problem(probs, agg, cov, nn):
    """問題別の値を 1 本の集計 CSV に書き出す（3 水準 × 問題数。**行単位の生データではない**）。

    **この回が依拠する数値はすべてここと `scored.txt` と docs に載る** ＝
    降下ダンプを畳んでも消しても数値は残る。
    """
    with open(BY_PROBLEM, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sigma_ratio", "problem", "K", "n_descents", "score", "mpr",
                    "mean_f1", "n_reported"]
                   + [f"cov_{lv}" for lv in LEVEL_NAMES] + ["cov_mean"])
        for s in SIGMAS:
            for p in probs:
                g = agg[s][p]
                w.writerow([s, p, bench(p)[0], nn[s][p], f"{g['score']:.4f}",
                            f"{g['mpr']:.4f}", f"{g['f1']:.4f}", f"{g['n']:.0f}"]
                           + [f"{v:.4f}" for v in cov[s][p]]
                           + [f"{cov[s][p].mean():.4f}"])


def main() -> int:
    per = load_all()
    have = {s: set(per.get(s, {})) for s in SIGMAS}
    probs = sorted(set.intersection(*[have[s] for s in SIGMAS]))

    print("=" * 100)
    print("その147 — `sigma_ratio` の監査（0.05 / 0.1 = 現行 / 0.2 = MC-ESO の σ_init）")
    print("=" * 100)
    for s in SIGMAS:
        print(f"  sigma_ratio={s:<6}: {len(have[s])}/16 問")
    print(f"  3 水準そろった問題: {len(probs)}/16"
          + ("   ** 部分結果（その131: 部分集合は中立ではない）**" if len(probs) < 16 else ""))
    if not probs:
        print("  3 水準そろった問題が 1 つも無い。集計しない。")
        return 1
    print(f"  規則: {ARM}  r = {ARM_R / SPAN:g} x span   予算: 50 万、seed 0、D=10、PIN01"
          "   descent_budget=12500 据え置き")

    arm_fn = lambda r: rule_indices(ARM, r["f"], r["K"], r["x"], ARM_R)   # noqa: E731
    agg, cov, nn = {}, {}, {}
    for s in SIGMAS:
        runs = [runs_from_rows(per[s][p], p, f"RL-s{s}") for p in probs]
        agg[s] = aggregate(runs, arm_fn)
        cov[s] = {p: coverage(per[s][p], bench(p)[0]) for p in probs}
        nn[s] = {p: len(per[s][p]) for p in probs}

    # ---------------------------------------------------------------- 関門
    print("\n## 0. 関門 — 採点経路が その116 と同一か（現行 null σ=0.1 の再採点）\n")
    s116 = stored_e116()
    bad = [(p, agg[BASE][p]["score"], s116[p]) for p in probs
           if abs(agg[BASE][p]["score"] - s116[p]) > 5e-5]
    if bad:
        for p, a, b_ in bad:
            print(f"  ** ずれ ** {p}: 再採点 {a:.4f} 対 e116 記録 {b_:.4f}")
        print("\n  採点経路が当時と違う。判定に進まない。")
        return 1
    print(f"  {len(probs)} 問すべてで現行 null の Score が e116/ranking_d10.csv と 4 桁一致"
          " ==> 採点経路は同一。")

    # ---------------------------------------------------------- 読み A（判定）
    print("\n## 1. 読み A — 問題別 Score（3 水準）\n")
    print(f"{'問題':<16}{'K':>4}" + "".join(f"{s:>10}" for s in SIGMAS))
    for p in probs:
        print(f"{p:<16}{bench(p)[0]:>4}"
              + "".join(f"{agg[s][p]['score']:>10.4f}" for s in SIGMAS))
    print("\n" + f"{'sigma_ratio':<16}{'MPR':>9}{'mean-F1':>10}{'Score':>9}"
                 f"{'n_rep':>9}{'記録値':>10}")
    rec = {BASE: REF_NULL_SCORE}
    means = {}
    for s in SIGMAS:
        means[s] = mean_of(agg[s], probs, "score")
        r = f"{rec[s]:.4f}" if s in rec else "-"
        print(f"{s:<16}{mean_of(agg[s], probs, 'mpr'):>9.4f}"
              f"{mean_of(agg[s], probs, 'f1'):>10.4f}{means[s]:>9.4f}"
              f"{mean_of(agg[s], probs, 'n'):>9.1f}{r:>10}")
    print(f"{'NMMSO-10D':<16}{'':>9}{'':>10}{REF_NMMSO_SCORE:>9.4f}   （その139）")
    print(f"{'MC-ESO':<16}{'':>9}{'':>10}{REF_MCESO_SCORE:>9.4f}   （その116）")

    print("\n  対検定（両側 Wilcoxon exact、σ=0.1 − 各腕）")
    for s in SIGMAS:
        if s == BASE:
            continue
        for key, lab in (("score", "Score"), ("mpr", "MPR"), ("f1", "mean-F1")):
            r = paired({p: agg[BASE][p][key] for p in probs},
                       {p: agg[s][p][key] for p in probs}, probs)
            print(f"    0.1 − {s:<5} {lab:<8}: {r['mean']:+.4f}  {r['w']}/{r['t']}/{r['l']}"
                  f"  p={r['p']:.4g}  rb={r['rb']:+.3f}")

    # ------------------------------------------------- 事前登録した反証条件
    print("\n## 2. 事前登録した反証条件 (A2)（読み A）\n")
    fired = [s for s in SIGMAS if s != BASE and means[s] < REF_NMMSO_SCORE]
    for s in SIGMAS:
        if s == BASE:
            continue
        f = means[s] < REF_NMMSO_SCORE
        print(f"  σ={s}: 16 問平均 Score {means[s]:.4f} < NMMSO {REF_NMMSO_SCORE:.4f} か: "
              + ("** 発火 ** ==> 頭の結果は 2 個目のつまみにも依存する。主張は書き換えず俯瞰に上げる"
                 if f else "いいえ"))
    if not fired:
        print("\n  ** (A2) 不発 ==> `sigma_ratio` については監査が通る（弱い環 1 の半分が閉じる）**")
        near = [s for s in SIGMAS if s != BASE and abs(means[s] - REF_NULL_SCORE) <= 0.05]
        print(f"  0.6284 の ±0.05 に入る腕: {near if near else 'なし'}"
              + ("  ==> 頭の Score は σ の 4 倍の幅に鈍感、と書ける" if len(near) == 2 else ""))
    print(f"  σ 4 倍（0.05 -> 0.2）で動いた Score の幅: "
          f"{max(means.values()) - min(means.values()):.4f}"
          f"   （参考: その145 の descent_budget 8.1 倍の幅は 0.3000）")

    # ---------------------------------------------------------- 読み B（機序）
    print("\n## 3. 読み B — 降下ダンプからの被覆（その113 と同じ定義。追加評価ゼロ）\n")
    print(f"{'σ':<8}{'降下本数':>10}" + "".join(f"{lv:>9}" for lv in LEVEL_NAMES)
          + f"{'5 水準平均':>11}{'深さ保持率':>11}")
    lv_of = {}
    for s in SIGMAS:
        lv = [float(np.mean([cov[s][p][i] for p in probs])) for i in range(5)]
        lv_of[s] = lv
        print(f"{s:<8}{np.mean([nn[s][p] for p in probs]):>10.1f}"
              + "".join(f"{v:>9.4f}" for v in lv)
              + f"{float(np.mean([cov[s][p].mean() for p in probs])):>11.4f}"
              + f"{lv[4] / lv[0] if lv[0] else float('nan'):>11.3f}")
    c0 = lv_of[BASE][0]
    print(f"\n  事前登録した予測: σ を振っても 1e-1 被覆は現行（{c0:.4f}）の ±10% 内か")
    for s in SIGMAS:
        if s == BASE:
            continue
        v = lv_of[s][0]
        ok = abs(v / c0 - 1.0) <= 0.10
        print(f"    σ={s}: {v:.4f} = {v / c0 * 100:.1f}%  "
              + ("==> 予測どおり（広さは σ ではなく draw の一様性）" if ok
                 else "==> ** 外れ ** その145 の「広さは配置」は「配置と σ」に弱まる。俯瞰に上げる"))

    print("\n  対検定（両側 Wilcoxon exact、σ=0.1 − 各腕）")
    for s in SIGMAS:
        if s == BASE:
            continue
        for i, lab in ((0, "被覆@1e-1"), (4, "被覆@1e-5")):
            r = paired({p: float(cov[BASE][p][i]) for p in probs},
                       {p: float(cov[s][p][i]) for p in probs}, probs)
            print(f"    0.1 − {s:<5} {lab:<10}: {r['mean']:+.4f}  {r['w']}/{r['t']}/{r['l']}"
                  f"  p={r['p']:.4g}  rb={r['rb']:+.3f}")
        r = paired({p: float(cov[BASE][p].mean()) for p in probs},
                   {p: float(cov[s][p].mean()) for p in probs}, probs)
        print(f"    0.1 − {s:<5} 5 水準平均 : {r['mean']:+.4f}  {r['w']}/{r['t']}/{r['l']}"
              f"  p={r['p']:.4g}  rb={r['rb']:+.3f}")

    write_by_problem(probs, agg, cov, nn)
    print(f"\n  問題別の値を {os.path.relpath(BY_PROBLEM, ROOT)} に書き出した"
          f"（{len(probs) * len(SIGMAS)} 行の集計）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
