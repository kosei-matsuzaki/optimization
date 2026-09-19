#!/usr/bin/env python3
"""その145 — キュー 1: 勝っている null 自身のつまみ（`descent_budget`）を振る。

**採点・規則・統計量は e115 からそのまま import する**（`rule_indices` / `score` /
`aggregate` / `paired` / `mean_of` / `SPAN` / `LEVEL_NAMES` / `read_dump`）。
**新しい統計量は 1 つも定義しない。** 読み B の被覆は その113 と同じ定義を使う。

入力の出自:

  * **`Restart-Lander-1540`（この回の腕）** -> `e145/descents/`（畳んだ後は `e145/descents.csv.gz`）
  * **`Restart-Lander`（現行 null、seed 0）** -> `e115/descents/`（座標つき。その116・その142 と同じ経路）
  * **MC-ESO** -> `e113/hunts/*_segments.csv.gz`（その113 の保存ダンプ。**追加評価ゼロ**）
  * 比較先の定数 -> `e116/ranking_d10.csv`（関門）、その139 / その116 の 16 問平均

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e145/analyze.py
"""
from __future__ import annotations

import csv
import glob
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
                     paired, read_dump, rule_indices)
from core.benchmarks import niching_by_name                        # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN     # その115 の合法な最良腕
EPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

REF_MCESO_SCORE = 0.1385        # その116（seed 0、16 問平均 Score）
REF_NMMSO_SCORE = 0.4144        # その139（新既定 10·D、seed 0）
REF_NULL_SCORE = 0.6284         # その116（seed 0）
REF_NULL_COV = 0.5644           # その113（null、5 水準平均被覆）
REF_MCESO_COV = 0.1094          # その113（MC-ESO、5 水準平均被覆）

ARM_DIR = os.path.join(HERE, "descents")
ARM_FOLDED = os.path.join(HERE, "descents.csv.gz")
NULL_DIR = os.path.join(MMO, "e115", "descents")
MC_HUNTS = os.path.join(MMO, "e113", "hunts")


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
    """その113 と同じ定義: detected(eps) = |{land_opt : best_f <= eps}| / K。"""
    return np.array([len({int(r["land_opt"]) for r in rows
                          if float(r[fcol]) <= e}) / k for e in EPS])


def arm_rows():
    """腕の降下ダンプを {問題: rows} で返す。畳んだ 1 本があればそちらを読む。"""
    out: dict = {}
    if os.path.exists(ARM_FOLDED):
        for r in rows_of(ARM_FOLDED):
            out.setdefault(r["problem"], []).append(r)
        return out
    for fn in sorted(os.listdir(ARM_DIR)) if os.path.isdir(ARM_DIR) else []:
        if not fn.endswith((".csv", ".csv.gz")):
            continue
        out[fn.split("_")[0]] = rows_of(os.path.join(ARM_DIR, fn))
    return out


def runs_from_rows(rows, prob, method):
    K, opts = bench(prob)
    f = np.array([float(r["best_f"]) for r in rows])
    x = np.array([[float(r[f"x{i}"]) for i in range(opts.shape[1])] for r in rows])
    return dict(problem=prob, method=method, seed=0, K=K,
                f=f, x=x, opt=attribute(x, opts))


def stored_e116(method="Restart-Lander"):
    out = {}
    with open(os.path.join(MMO, "e116", "ranking_d10.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == "legal" and r["method"] == method:
                out[r["problem"]] = float(r["score"])
    return out


BY_PROBLEM = os.path.join(HERE, "by_problem.csv")


def read_mceso():
    """MC-ESO の問題別 被覆と降下本数を返す。

    **`e113/hunts/` があればそこから計算し、無ければこの回が書き出した
    `e145/by_problem.csv` から読む。** その113 の hunt ダンプ（3.0 MB）は
    CLAUDE.md の保持規則によりこの回が消すので、**消したあとも本 script が
    そのまま走るように控えを持つ**（値はこの回が hunts から計算したもの）。
    """
    cov, n = {}, {}
    paths = sorted(glob.glob(os.path.join(MC_HUNTS, "*_segments.csv.gz")))
    if paths:
        for sp in paths:
            prob = os.path.basename(sp).split("_seed")[0]
            rs = rows_of(sp)
            cov[prob], n[prob] = coverage(rs, bench(prob)[0]), len(rs)
        return cov, n
    if os.path.exists(BY_PROBLEM):
        for r in csv.DictReader(open(BY_PROBLEM)):
            if r["series"] != "MC-ESO":
                continue
            cov[r["problem"]] = np.array([float(r[f"cov_{lv}"]) for lv in LEVEL_NAMES])
            n[r["problem"]] = int(r["n_hunts"])
    return cov, n


def write_by_problem(probs, agg_arm, agg_null, a_cov, n_cov, mc_cov, a_n, n_n, mc_n):
    """問題別の値を 1 本の集計 CSV に書き出す（48 行。**行単位の生データではない**）。

    **この回が依拠する数値はすべてここと `scored.txt` と docs に載る** ＝
    降下ダンプを畳んでも hunt ダンプを消しても、数値は残る。
    """
    with open(BY_PROBLEM, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "problem", "K", "n_hunts", "score", "mpr", "mean_f1",
                    "n_reported"] + [f"cov_{lv}" for lv in LEVEL_NAMES] + ["cov_mean"])
        for lab, agg, cov, nn in (
                ("Restart-Lander-1540", agg_arm, a_cov, a_n),
                ("Restart-Lander", agg_null, n_cov, n_n),
                ("MC-ESO", None, mc_cov, mc_n)):
            for p in probs:
                if p not in cov:
                    continue
                g = agg[p] if agg else {}
                w.writerow([lab, p, bench(p)[0], nn.get(p, ""),
                            f"{g['score']:.4f}" if g else "",
                            f"{g['mpr']:.4f}" if g else "",
                            f"{g['f1']:.4f}" if g else "",
                            f"{g['n']:.0f}" if g else ""]
                           + [f"{v:.4f}" for v in cov[p]] + [f"{cov[p].mean():.4f}"])


def main() -> int:
    arm = arm_rows()
    null_runs, null_rows = [], {}
    for fn in sorted(os.listdir(NULL_DIR)):
        if not fn.endswith((".csv", ".csv.gz")):
            continue
        prob = fn.split("_")[0]
        null_rows[prob] = rows_of(os.path.join(NULL_DIR, fn))
        null_runs.append(runs_from_rows(null_rows[prob], prob, "Restart-Lander"))
    arm_runs = [runs_from_rows(rs, p, "Restart-Lander-1540") for p, rs in arm.items()]

    probs = sorted(set(arm) & set(null_rows))
    print("=" * 100)
    print("その145 — 勝っている null 自身のつまみを振る（`descent_budget` 12500 -> 1540）")
    print("=" * 100)
    print(f"  腕が揃った問題: {len(arm)}/16   対が取れた問題: {len(probs)}/16"
          + ("   ** 部分結果（その131: 部分集合は中立ではない）**" if len(probs) < 16 else ""))
    if not probs:
        print("  対が 1 問も無い。集計しない。")
        return 1
    print(f"  規則: {ARM}  r = {ARM_R / SPAN:g} x span   予算: 50 万、seed 0、D=10、PIN01")

    arm_fn = lambda r: rule_indices(ARM, r["f"], r["K"], r["x"], ARM_R)   # noqa: E731
    agg_arm = aggregate([r for r in arm_runs if r["problem"] in probs], arm_fn)
    agg_null = aggregate([r for r in null_runs if r["problem"] in probs], arm_fn)

    # ---------------------------------------------------------------- 関門
    print("\n## 0. 関門 — 採点経路が その116 と同一か（現行 null の再採点）\n")
    s116 = stored_e116()
    bad = [(p, agg_null[p]["score"], s116[p]) for p in probs
           if abs(agg_null[p]["score"] - s116[p]) > 5e-5]
    if bad:
        for p, a, b in bad:
            print(f"  ** ずれ ** {p}: 再採点 {a:.4f} 対 e116 記録 {b:.4f}")
        print("\n  採点経路が当時と違う。判定に進まない。")
        return 1
    print(f"  {len(probs)} 問すべてで現行 null の Score が e116/ranking_d10.csv と 4 桁一致"
          " ==> 採点経路は同一。")

    # ---------------------------------------------------------- 読み A（頑健性）
    print("\n## 1. 読み A — 16 問平均 Score（頑健性）\n")
    print(f"{'問題':<16}{'K':>4}{'腕 1540':>11}{'null 12500':>12}{'差':>10}"
          f"{'n_rep 腕':>10}{'n_rep null':>12}")
    for p in probs:
        a, n = agg_arm[p], agg_null[p]
        print(f"{p:<16}{bench(p)[0]:>4}{a['score']:>11.4f}{n['score']:>12.4f}"
              f"{a['score'] - n['score']:>+10.4f}{a['n']:>10.1f}{n['n']:>12.1f}")
    m_arm = mean_of(agg_arm, probs, "score")
    m_null = mean_of(agg_null, probs, "score")
    print(f"\n{'系列':<28}{'MPR':>9}{'mean-F1':>10}{'Score':>9}{'n_rep':>9}")
    for lab, g in (("Restart-Lander-1540（腕）", agg_arm),
                   ("Restart-Lander（現行 null）", agg_null)):
        print(f"{lab:<28}{mean_of(g, probs, 'mpr'):>9.4f}{mean_of(g, probs, 'f1'):>10.4f}"
              f"{mean_of(g, probs, 'score'):>9.4f}{mean_of(g, probs, 'n'):>9.1f}")
    print(f"{'MC-ESO（その116）':<28}{'':>9}{'':>10}{REF_MCESO_SCORE:>9.4f}")
    print(f"{'NMMSO-10D（その139）':<28}{'':>9}{'':>10}{REF_NMMSO_SCORE:>9.4f}")

    r = paired({p: agg_null[p]["score"] for p in probs},
               {p: agg_arm[p]["score"] for p in probs}, probs)
    print(f"\n  対検定（両側 Wilcoxon exact、現行 null − 腕）: {r['mean']:+.4f}"
          f"  {r['w']}/{r['t']}/{r['l']}  p={r['p']:.4g}  rb={r['rb']:+.3f}")

    print("\n## 2. 事前登録した反証条件（読み A）\n")
    fired = m_arm < REF_NMMSO_SCORE
    print(f"  (A1) 腕の平均 Score {m_arm:.4f} < NMMSO の {REF_NMMSO_SCORE:.4f} か: "
          f"{'** 発火 ** ==> 頭の結果は descent_budget の選び方に依存する' if fired else 'いいえ'}")
    print(f"  (A2) {m_arm:.4f} >= {REF_NMMSO_SCORE:.4f} かつ >= MC-ESO {REF_MCESO_SCORE:.4f} か: "
          f"{'いいえ' if fired else '** はい ==> 8.1 倍振っても null は両方を上回る。弱い環 2 は descent_budget については閉じる **'}")

    # ---------------------------------------------------------- 読み B（機序）
    print("\n## 3. 読み B — 降下ダンプからの被覆（その113 と同じ定義）\n")
    mc_cov, mc_n = read_mceso()
    a_cov = {p: coverage(arm[p], bench(p)[0]) for p in probs}
    n_cov = {p: coverage(null_rows[p], bench(p)[0]) for p in probs}
    a_n = {p: len(arm[p]) for p in probs}
    n_n = {p: len(null_rows[p]) for p in probs}

    print(f"{'問題':<16}{'K':>4}{'本数 腕':>9}{'MC':>7}{'null':>7}"
          f"{'cov 腕':>10}{'cov MC':>9}{'cov null':>10}")
    for p in probs:
        print(f"{p:<16}{bench(p)[0]:>4}{a_n[p]:>9}{mc_n.get(p, -1):>7}{n_n[p]:>7}"
              f"{a_cov[p].mean():>10.4f}{mc_cov.get(p, np.full(5, np.nan)).mean():>9.4f}"
              f"{n_cov[p].mean():>10.4f}")
    cov_arm = float(np.mean([a_cov[p].mean() for p in probs]))
    cov_mc = float(np.mean([mc_cov[p].mean() for p in probs if p in mc_cov]))
    cov_null = float(np.mean([n_cov[p].mean() for p in probs]))
    print(f"\n  5 水準平均被覆: 腕 {cov_arm:.4f}   MC-ESO {cov_mc:.4f}"
          f"（その113 記録 {REF_MCESO_COV:.4f}）   null {cov_null:.4f}"
          f"（その113 記録 {REF_NULL_COV:.4f}）")
    print(f"  降下本数の平均: 腕 {np.mean([a_n[p] for p in probs]):.1f}"
          f"   MC-ESO {np.mean([mc_n[p] for p in probs if p in mc_n]):.1f}"
          f"   null {np.mean([n_n[p] for p in probs]):.1f}")
    print("\n  水準別（5 水準）")
    print(f"{'水準':<8}{'腕':>10}{'MC-ESO':>10}{'null':>10}")
    for i, lv in enumerate(LEVEL_NAMES):
        print(f"{lv:<8}{np.mean([a_cov[p][i] for p in probs]):>10.4f}"
              f"{np.mean([mc_cov[p][i] for p in probs if p in mc_cov]):>10.4f}"
              f"{np.mean([n_cov[p][i] for p in probs]):>10.4f}")

    rb1 = paired({p: float(n_cov[p].mean()) for p in probs},
                 {p: float(a_cov[p].mean()) for p in probs}, probs)
    rb2 = paired({p: float(a_cov[p].mean()) for p in probs},
                 {p: float(mc_cov[p].mean()) for p in probs if p in mc_cov},
                 [p for p in probs if p in mc_cov])
    print(f"\n  対検定  null − 腕     {rb1['mean']:+.4f}  {rb1['w']}/{rb1['t']}/{rb1['l']}"
          f"  p={rb1['p']:.4g}  rb={rb1['rb']:+.3f}")
    print(f"  対検定  腕 − MC-ESO  {rb2['mean']:+.4f}  {rb2['w']}/{rb2['t']}/{rb2['l']}"
          f"  p={rb2['p']:.4g}  rb={rb2['rb']:+.3f}")

    print("\n## 4. 事前登録した棄却条件（読み B）\n")
    if cov_arm >= 0.40:
        print(f"  腕の被覆 {cov_arm:.4f} >= 0.40 ==> ** hunt 長は原因ではない ＝ 選抜が原因 **")
    elif cov_arm < 0.20:
        print(f"  腕の被覆 {cov_arm:.4f} < 0.20 ==> ** hunt 長だけでほぼ説明がつく **")
    else:
        gap = cov_null - cov_mc
        by_len = (cov_null - cov_arm) / gap if gap else float("nan")
        print(f"  腕の被覆 {cov_arm:.4f} は 0.20-0.40 ==> ** 両方が効いている **")
        print("     腕は MC-ESO と hunt 本数・1 hunt の長さが揃い、配置だけが違う"
              "（一様再起動 対 MC-ESO の選抜）。")
        print("     腕は null と配置が揃い、hunt 本数・長さだけが違う。== 差は 2 つに分かれる。")
        print(f"     hunt 長・本数の寄与 = (null {cov_null:.4f} − 腕 {cov_arm:.4f}) /"
              f" (null {cov_null:.4f} − MC {cov_mc:.4f}) = {by_len:.3f}")
        print(f"     配置・選抜の寄与   = (腕 {cov_arm:.4f} − MC {cov_mc:.4f}) / 同 = {1 - by_len:.3f}")

    print("\n## 5. 【事前登録の外・記述のみ】寄与は水準で入れ替わるか\n")
    print(f"{'水準':<8}{'null':>9}{'腕':>9}{'MC-ESO':>9}{'hunt 長の寄与':>15}{'配置の寄与':>13}")
    for i, lv in enumerate(LEVEL_NAMES):
        n_ = float(np.mean([n_cov[p][i] for p in probs]))
        a_ = float(np.mean([a_cov[p][i] for p in probs]))
        m_ = float(np.mean([mc_cov[p][i] for p in probs if p in mc_cov]))
        g = n_ - m_
        bl = (n_ - a_) / g if g else float("nan")
        print(f"{lv:<8}{n_:>9.4f}{a_:>9.4f}{m_:>9.4f}{bl:>15.3f}{1 - bl:>13.3f}")
    write_by_problem(probs, agg_arm, agg_null, a_cov, n_cov, mc_cov, a_n, n_n, mc_n)
    print(f"\n  問題別の値を {os.path.relpath(BY_PROBLEM, ROOT)} に書き出した"
          f"（{len(probs) * 2 + len(mc_cov)} 行の集計）。")
    print("\n  深さの保持率（@1e-5 / @1e-1）: "
          f"腕 {float(np.mean([a_cov[p][4] for p in probs])) / float(np.mean([a_cov[p][0] for p in probs])):.3f}"
          f"   null {float(np.mean([n_cov[p][4] for p in probs])) / float(np.mean([n_cov[p][0] for p in probs])):.3f}"
          f"   MC-ESO {float(np.mean([mc_cov[p][4] for p in probs if p in mc_cov])) / float(np.mean([mc_cov[p][0] for p in probs if p in mc_cov])):.3f}"
          "   （その113 の記録: MC 0.275 / null 0.850）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
