#!/usr/bin/env python3
"""その142 — キュー 1: 外部 niching 手法 2 本（3 腕）を新 suite で回し、手法軸を n=2 -> n=4 にする。

**採点・規則・統計量は e115 からそのまま import する**（`rule_indices` / `score` /
`aggregate` / `paired` / `mean_of` / `SPAN` / `LEVEL_NAMES` / `read_dump`）。
**新しい統計量は 1 つも定義しない。**

入力の出自:

  * **NCDE / r3pso(30) / r3pso(400)** -> `e142/report_sets.csv.gz`
    （この回の `REPORT_SET_DUMP` を `problem` / `method` / `seed` 列つきで 1 本に畳んだもの。
    畳む前は `e142/dumps/` の per-run ダンプで、畳む前後で本 script の出力が 1 文字も
    変わらないことを確認してから消してある）
  * **`Restart-Lander`** -> `e115/descents/`（seed 0、座標つき）を**再採点する** ＝ その116・その139 と同じ経路。
    **再採点値が `e116/ranking_d10.csv` と 4 桁一致することを関門にする。**
  * **MC-ESO / NMMSO（旧既定）** -> `e116/ranking_d10.csv` の `arm=legal` 行。
  * **NMMSO（新既定 `10·D`）** -> `e139/by_problem.csv` の該当系列（その139 の seed 0）。

使い方: python3 analysis/mmo2024/e142/analyze.py
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
                     paired, read_dump, rule_indices)
from core.benchmarks import niching_by_name                        # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN     # その115 の合法な最良腕
ARMS_PLANNED = ["NCDE", "r3pso", "r3pso-p400"]   # 事前登録した 3 腕
ARMS: list = []                                  # 16 問が揃った腕だけがここに入る
REF_RL_SEED0 = 0.6284       # その116（seed 0、16 問平均 Score）
REF_RL_3SEED = 0.6238       # その131（3 seed 平均）
REF_MCESO = 0.1385          # その116
REF_NMMSO_NEW = 0.4144      # その139（新既定 10·D、seed 0）
NULL_SEED_SD = 0.0066       # その131（null の 16 問平均 Score の n=3 SD）

FOLDED = os.path.join(HERE, "report_sets.csv.gz")
DUMPS = os.path.join(HERE, "dumps")


def attribute(x: np.ndarray, optima: np.ndarray) -> np.ndarray:
    """最近傍の最適への帰属（`core.runner.count_goptima_nn` と同順序）。**オラクル量。採点専用。**"""
    d = np.linalg.norm(x[:, None, :] - optima[None, :, :], axis=2)
    return np.argmin(d, axis=1)


_BENCH: dict = {}


def bench(prob):
    if prob not in _BENCH:
        b = niching_by_name(prob)
        _BENCH[prob] = (int(b.n_global_optima), np.asarray(b.optima_pos, dtype=float))
    return _BENCH[prob]


def read_this_cycle(seed=0):
    """この回の報告集合を [(問題, 手法, f, x)] で返す。

    畳んだ 1 本があればそれを読み、無ければ `dumps/` の per-run ダンプを読む
    （**どちらを読んでも同じ**ことが畳みの検査条件）。"""
    acc: dict = {}
    if os.path.exists(FOLDED):
        with gzip.open(FOLDED, "rt") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return []
        dim = sum(1 for k in rows[0] if k.startswith("x") and k[1:].isdigit())
        for r in rows:
            if int(r["seed"]) != seed:
                continue
            key = (r["problem"], r["method"])
            f, x = acc.setdefault(key, ([], []))
            f.append(float(r["f"]))
            x.append([float(r[f"x{i}"]) for i in range(dim)])
    elif os.path.isdir(DUMPS):
        for fn in sorted(os.listdir(DUMPS)):
            if not fn.endswith(".csv.gz"):
                continue
            stem = fn[: -len(".csv.gz")]
            prob, rest = stem.split("_", 1)
            meth, sd = rest.rsplit("_seed", 1)
            if int(sd) != seed:
                continue
            f, x = read_report_dump(os.path.join(DUMPS, fn))
            if len(f) == 0:
                continue
            acc[(prob, meth)] = (list(f), [list(v) for v in x])
    return [(p, m, np.array(f), np.array(x)) for (p, m), (f, x) in acc.items()]


def read_report_dump(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return np.array([]), np.zeros((0, 0))
    dim = sum(1 for k in rows[0] if k.startswith("x") and k[1:].isdigit())
    f = np.array([float(r["f"]) for r in rows])
    x = np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rows])
    return f, x


def stored_e116():
    out: dict = {}
    with open(os.path.join(MMO, "e116", "ranking_d10.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] != "legal":
                continue
            out.setdefault(r["method"], {})[r["problem"]] = dict(
                mpr=float(r["mpr"]), f1=float(r["mean_f1"]),
                score=float(r["score"]), n=float(r["n_reported"]))
    return out


def stored_e139():
    """その139 の `by_problem.csv` から新既定 NMMSO の seed 0 を拾う。"""
    out: dict = {}
    path = os.path.join(MMO, "e139", "by_problem.csv")
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        if r["series"].startswith("NMMSO-10D"):
            out[r["problem"]] = dict(mpr=float(r["mpr"]), f1=float(r["mean_f1"]),
                                     score=float(r["score"]), n=float(r["n_reported"]))
    return out


def load_runs():
    runs = []
    dd = os.path.join(MMO, "e115", "descents")           # Restart-Lander seed 0
    for fn in sorted(os.listdir(dd)) if os.path.isdir(dd) else []:
        if not fn.endswith((".csv", ".csv.gz")):
            continue
        prob = fn.split("_")[0]
        f, _opt, x = read_dump(os.path.join(dd, fn))
        if x is None:
            continue
        K, opts = bench(prob)
        runs.append(dict(problem=prob, method="Restart-Lander", seed=0, K=K,
                         f=f, x=x, opt=attribute(x, opts)))
    for prob, meth, f, x in read_this_cycle(seed=0):
        if len(f) == 0:
            continue
        K, opts = bench(prob)
        runs.append(dict(problem=prob, method=meth, seed=0, K=K,
                         f=f, x=x, opt=attribute(x, opts)))
    return runs


def main() -> int:
    runs = load_runs()
    have = {m: {r["problem"] for r in runs if r["method"] == m}
            for m in ["Restart-Lander"] + ARMS_PLANNED}
    stored116, stored139 = stored_e116(), stored_e139()
    # 事前登録した予算の逃げ道: 枠が binding したら **腕を落とす。問題は落とさない**
    # （Score の比較可能性）。16 問が揃っていない腕はこの回の判定に入れない。
    global ARMS
    ARMS = [a for a in ARMS_PLANNED if len(have[a]) >= 16]
    dropped = [a for a in ARMS_PLANNED if a not in ARMS]
    if not ARMS:
        print("16 問が揃った腕が 1 本も無い。集計しない。")
        return 1
    probs = sorted(have["Restart-Lander"].intersection(*[have[a] for a in ARMS])
                   & set(stored116.get("MC-ESO", {})) & set(stored139))
    print("=" * 96)
    print("その142 — 外部 niching 手法 2 本（3 腕）を新 suite で回す（キュー 1: 手法軸 n=2 -> n=4）")
    print("=" * 96)
    for a in ARMS_PLANNED:
        print(f"  {a:<12} が揃った問題: {len(have[a])}/16"
              + ("   ** 16 問未満 ＝ この回の判定から外す（事前登録した予算の逃げ道）**"
                 if a in dropped else ""))
    if dropped:
        print(f"  ** 落とした腕: {', '.join(dropped)} —— 問題は 1 つも落としていない **")
    print(f"  全系列が揃った問題: {len(probs)}/16"
          + ("   ** 部分結果 **" if len(probs) < 16 else ""))
    if not probs:
        print("\n  対が 1 問も揃っていない。集計しない。")
        return 1
    print(f"  規則: {ARM}  r = {ARM_R / SPAN:g} x span（その115 の合法な最良腕）")
    print("  予算: --evals-frac 1.0（正規予算 50 万）、seed 0、D=10、PIN01")

    arm_fn = lambda r: rule_indices(ARM, r["f"], r["K"], r["x"], ARM_R)  # noqa: E731
    agg = {m: aggregate([r for r in runs if r["method"] == m and r["problem"] in probs],
                        arm_fn) for m in ["Restart-Lander"] + ARMS}

    # ---------------------------------------------------------------- 関門
    print("\n## 0. 関門 — 採点経路が その116 と同一か（`Restart-Lander` の再採点）\n")
    bad = [(p, agg["Restart-Lander"][p]["score"], stored116["Restart-Lander"][p]["score"])
           for p in probs
           if abs(agg["Restart-Lander"][p]["score"]
                  - stored116["Restart-Lander"][p]["score"]) > 5e-5]
    if bad:
        for p, a, b in bad:
            print(f"  ** ずれ ** {p}: 再採点 {a:.4f} 対 e116 記録 {b:.4f}")
        print("\n  採点経路が当時と違う。判定に進まない。")
        return 1
    print(f"  {len(probs)} 問すべてで `Restart-Lander` の Score が e116/ranking_d10.csv と 4 桁一致"
          " ==> 採点経路は同一。")

    rows = {a: {p: agg[a][p] for p in probs} for a in ARMS}
    rows["MC-ESO（その116）"] = {p: stored116["MC-ESO"][p] for p in probs}
    rows["NMMSO-10D（その139）"] = {p: stored139[p] for p in probs}
    rows["Restart-Lander（null）"] = {p: agg["Restart-Lander"][p] for p in probs}

    # ---------------------------------------------------------------- 集計
    print(f"\n## 1. {len(probs)} 問平均（5 水準等重み、seed 0 の 1 本）\n")
    print(f"{'系列':<26}{'MPR':>9}{'mean-F1':>10}{'Score':>9}{'n_rep':>9}")
    means = {}
    for k, g in rows.items():
        means[k] = mean_of(g, probs, "score")
        print(f"{k:<26}{mean_of(g, probs, 'mpr'):>9.4f}"
              f"{mean_of(g, probs, 'f1'):>10.4f}{means[k]:>9.4f}"
              f"{mean_of(g, probs, 'n'):>9.1f}")
    rl = means["Restart-Lander（null）"]

    # ---------------------------------------------------------------- 主判定
    print("\n## 2. 主判定 — 3 腕は `Restart-Lander` を下回るか\n")
    print(f"{'問題':<16}{'K':>4}" + "".join(f"{a:>13}" for a in ARMS)
          + f"{'null':>9}" + "".join(f"{'d:' + a:>13}" for a in ARMS))
    for p in probs:
        r = rows["Restart-Lander（null）"][p]["score"]
        vals = [rows[a][p]["score"] for a in ARMS]
        print(f"{p:<16}{bench(p)[0]:>4}" + "".join(f"{v:>13.4f}" for v in vals)
              + f"{r:>9.4f}" + "".join(f"{v - r:>+13.4f}" for v in vals))

    print()
    fired_a = []
    for a in ARMS:
        beat = [p for p in probs
                if rows[a][p]["score"] > rows["Restart-Lander（null）"][p]["score"]]
        d = means[a] - rl
        print(f"  {a:<12} 16 問平均 Score {means[a]:.4f}  (null {rl:.4f}, {d:+.4f})"
              f"   null を上回った問題 {len(beat)}/{len(probs)}"
              + (f"  ({', '.join(beat)})" if beat else ""))
        if d > 0:
            fired_a.append(a)

    print("\n  対検定（問題を対に、両側 Wilcoxon exact。null − 腕）")
    for k in ARMS + ["NMMSO-10D（その139）", "MC-ESO（その116）"]:
        r = paired({p: rows["Restart-Lander（null）"][p]["score"] for p in probs},
                   {p: rows[k][p]["score"] for p in probs}, probs)
        print(f"    null − {k:<22} {r['mean']:+.4f}  {r['w']}/{r['t']}/{r['l']}"
              f"  p={r['p']:.4g}  rb={r['rb']:+.3f}")

    # ------------------------------------------------------- 事前登録した反証条件
    print("\n## 3. 事前登録した反証条件\n")
    print(f"  (a) どれかの腕が 16 問平均で null を上回ったか: "
          + ("発火（" + ", ".join(fired_a) + "）** 主張を書き換えず俯瞰に上げる **"
             if fired_a else "不発（3 腕とも下回る ==> 手法軸は n=4）"))
    if "r3pso" in ARMS and "r3pso-p400" in ARMS:
        d30 = means["r3pso"] - rl
        d400 = means["r3pso-p400"] - rl
        split = (d30 > 0) != (d400 > 0)
        print(f"  (b) r3pso の 30 と 400 で符号が割れたか: 30 {d30:+.4f} / 400 {d400:+.4f}"
              f"  ==> {'発火（割れた）' if split else '不発（同符号）'}")
        print("      r3pso 400 − 30（問題を対に）: ", end="")
        r = paired({p: rows["r3pso-p400"][p]["score"] for p in probs},
                   {p: rows["r3pso"][p]["score"] for p in probs}, probs)
        print(f"{r['mean']:+.4f}  {r['w']}/{r['t']}/{r['l']}  p={r['p']:.4g}  rb={r['rb']:+.3f}")
    else:
        print("  (b) r3pso の 30 と 400 の符号比較: **測れていない**"
              " —— 400 の腕を予算の逃げ道で落としたため。**未検証として持ち越す。**")
    print(f"  (限界) null の seed 間 SD は {NULL_SEED_SD:.4f}（その131）。"
          f"この帯に入る腕は符号を主張しない。")
    for a in ARMS:
        print(f"      {a:<12} と null の差 {means[a] - rl:+.4f}"
              f"  ==> {'帯の内側（符号は保留）' if abs(means[a] - rl) <= NULL_SEED_SD else '帯の外側'}")

    # ------------------------------------------------------------ 手法軸の一覧
    print("\n## 4. 手法軸（16 問平均 Score、すべて seed 0・PIN01・同じ規則・同じ採点器）\n")
    order = [("MC-ESO（その116）", REF_MCESO), ("NMMSO-10D（その139）", REF_NMMSO_NEW)]
    print(f"{'手法':<26}{'MPR':>9}{'mean-F1':>10}{'Score':>9}{'記録値':>10}")
    for k, ref in order:
        print(f"{k:<26}{mean_of(rows[k], probs, 'mpr'):>9.4f}"
              f"{mean_of(rows[k], probs, 'f1'):>10.4f}{means[k]:>9.4f}{ref:>10.4f}")
    for a in ARMS:
        print(f"{a + '（今回）':<26}{mean_of(rows[a], probs, 'mpr'):>9.4f}"
              f"{mean_of(rows[a], probs, 'f1'):>10.4f}{means[a]:>9.4f}{'—':>10}")
    print(f"{'Restart-Lander（null）':<26}{mean_of(rows['Restart-Lander（null）'], probs, 'mpr'):>9.4f}"
          f"{mean_of(rows['Restart-Lander（null）'], probs, 'f1'):>10.4f}"
          f"{rl:>9.4f}{REF_RL_SEED0:>10.4f}")

    print("\n## 5. 水準別 PR / F1（問題平均）\n")
    print(f"{'系列':<26}{'量':<5}" + "".join(f"{l:>9}" for l in LEVEL_NAMES))
    for k in ARMS + ["Restart-Lander（null）"]:
        g = rows[k]
        if "pr_lv" not in next(iter(g.values())):
            continue
        pr = np.mean([g[p]["pr_lv"] for p in probs], axis=0)
        f1 = np.mean([g[p]["f1_lv"] for p in probs], axis=0)
        print(f"{k:<26}{'PR':<5}" + "".join(f"{v:>9.4f}" for v in pr))
        print(f"{'':<26}{'F1':<5}" + "".join(f"{v:>9.4f}" for v in f1))

    out = os.path.join(HERE, "by_problem.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "series", "n_reported", "mpr", "mean_f1", "score"])
        for k, g in rows.items():
            for p in probs:
                w.writerow([p, bench(p)[0], k, f"{g[p]['n']:.1f}", f"{g[p]['mpr']:.4f}",
                            f"{g[p]['f1']:.4f}", f"{g[p]['score']:.4f}"])
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
