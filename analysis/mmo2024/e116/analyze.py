#!/usr/bin/env python3
"""その116 — 同じ合法な報告規則を 3 手法に与えたときに順位はどう動くか（キュー 1 の次の一手）。

その115 は `Restart-Lander` だけに規則を与えて Score 0.3629 -> 0.6224 を出した。
**報告規則は手法非依存に当てられる後処理**なので、その数字を順位に使う前に
**MC-ESO と NMMSO にも同じ規則を与える**（測定の鉄則: 他手法にも与えられる介入は
他手法にも与えてから読む）。

採点・規則の定義は e115 からそのまま import する（`rule_indices` / `score` / `paired`）。
**追加の規則は 1 つも定義しない。** 違うのは入力だけ:

  * `Restart-Lander` -> e115 の降下ダンプ（seed 0、座標つき）
  * `MC-ESO` / `NMMSO` -> e116 の報告集合ダンプ（seed 0、座標つき、上限なし）

**最近傍最適への帰属（`opt`）はここで計算する。** ダンプにオラクル列は入れていない
（その115 の教訓）。`opt` は**採点にだけ**使い、規則の定義には使わない。

使い方: python3 analysis/mmo2024/e116/analyze.py
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
                     oracle_indices, paired, read_dump, rule_indices)
from core.benchmarks import niching_by_name                        # noqa: E402

# その115 が選んだ合法な最良腕（oracle 上限と 16/16 厳密一致）
ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN
METHODS = ["Restart-Lander", "MC-ESO", "NMMSO"]
# その112 の現行規則での Score（対照。ここでは再計算するので参照値としてだけ書く）
E112_CUR = {"Restart-Lander": 0.3639, "MC-ESO": 0.0638, "NMMSO": 0.1211}


def attribute(x: np.ndarray, optima: np.ndarray) -> np.ndarray:
    """各点を最近傍の最適に帰属させる（`core.runner.count_goptima_nn` と同順序）。
    **オラクル量。採点にだけ使う。**"""
    d = np.linalg.norm(x[:, None, :] - optima[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def read_report_dump(path):
    """e116 の報告集合ダンプ（f と座標のみ）。"""
    with gzip.open(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return np.zeros(0), np.zeros((0, 0))
    dim = sum(1 for k in rows[0] if k.startswith("x") and k[1:].isdigit())
    f = np.array([float(r["f"]) for r in rows])
    x = np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rows])
    return f, x


_BENCH: dict = {}


def bench(prob):
    if prob not in _BENCH:
        b = niching_by_name(prob)
        _BENCH[prob] = (int(b.n_global_optima),
                        np.asarray(b.optima_pos, dtype=float))
    return _BENCH[prob]


def load_all():
    """1 run 1 要素。手法ごとに出自が違うので読み口だけ分ける。"""
    runs = []
    # Restart-Lander: e115 の座標つき降下ダンプ（seed 0）。opt 列はあるが使わず、
    # 他の 2 手法と同じ手続きで座標から計算し直す（採点の経路を 1 本に揃える）。
    dd = os.path.join(MMO, "e115", "descents")
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
    # MC-ESO / NMMSO: e116 の報告集合ダンプ
    dd = os.path.join(HERE, "dumps")
    for fn in sorted(os.listdir(dd)) if os.path.isdir(dd) else []:
        if not fn.endswith(".csv.gz"):
            continue
        prob, meth, _seed = fn[:-len(".csv.gz")].split("_", 2)
        f, x = read_report_dump(os.path.join(dd, fn))
        if len(f) == 0:
            continue
        K, opts = bench(prob)
        runs.append(dict(problem=prob, method=meth, seed=0, K=K,
                         f=f, x=x, opt=attribute(x, opts)))
    return runs


def main() -> int:
    runs = load_all()
    have = {m: {r["problem"] for r in runs if r["method"] == m} for m in METHODS}
    probs = sorted(set.intersection(*have.values())) if all(have.values()) else []
    print("=" * 88)
    print("その116 — 同じ合法な報告規則を 3 手法に与える（キュー 1 の次の一手）")
    print("=" * 88)
    for m in METHODS:
        print(f"  {m:<16} {len(have[m]):>2} 問")
    print(f"\n  3 手法が揃った問題: {len(probs)}/16" +
          ("  ** 部分結果 **" if len(probs) < 16 else ""))
    if not probs:
        print("\n  対が 1 問も揃っていない。集計しない。")
        return 1
    print(f"  規則: {ARM}  r = {ARM_R / SPAN:g} x span  "
          f"(その115 の合法な最良腕、oracle 上限と 16/16 一致)")

    arms = {
        "cur": lambda r: rule_indices("cur", r["f"], r["K"]),
        "legal": lambda r: rule_indices(ARM, r["f"], r["K"], r["x"], ARM_R),
        "oracle_e112": lambda r: oracle_indices(r["f"], r["opt"],
                                                detected_only=True),
    }
    agg = {}
    for m in METHODS:
        sub = [r for r in runs if r["method"] == m and r["problem"] in probs]
        for a, fn in arms.items():
            agg[(m, a)] = aggregate(sub, fn)

    print("\n## 1. 16 問平均（5 水準等重み、seed 0 の 1 本）\n")
    print(f"{'手法':<16}{'腕':<14}{'MPR':>9}{'mean-F1':>10}{'Score':>9}{'n_rep':>9}")
    for m in METHODS:
        for a in arms:
            g = agg[(m, a)]
            print(f"{m:<16}{a:<14}{mean_of(g, probs, 'mpr'):>9.4f}"
                  f"{mean_of(g, probs, 'f1'):>10.4f}"
                  f"{mean_of(g, probs, 'score'):>9.4f}"
                  f"{mean_of(g, probs, 'n'):>9.1f}")
        print()

    print("## 2. 順位（Score、規則を与えた後）と対検定\n")
    order = sorted(METHODS, key=lambda m: -mean_of(agg[(m, "legal")], probs, "score"))
    for i, m in enumerate(order, 1):
        print(f"  {i}. {m:<16} Score {mean_of(agg[(m, 'legal')], probs, 'score'):.4f}"
              f"   (現行規則 {mean_of(agg[(m, 'cur')], probs, 'score'):.4f}"
              f" / その112 記録 {E112_CUR.get(m, float('nan')):.4f})")
    print()
    for a in ("cur", "legal"):
        print(f"  [{a}] 対 `Restart-Lander`（問題を対に、両側 Wilcoxon exact）")
        ref = {p: agg[("Restart-Lander", a)][p]["score"] for p in probs}
        for m in METHODS[1:]:
            x = {p: agg[(m, a)][p]["score"] for p in probs}
            r = paired(ref, x, probs)
            print(f"    RL - {m:<12} Score {r['mean']:+.4f}  "
                  f"{r['w']}/{r['l']}/{r['t']}  p={r['p']:.4g}  rb={r['rb']:+.3f}")
        print()

    print("## 3. 規則の利得は手法によって違うか（legal - cur、問題を対に）\n")
    for m in METHODS:
        x = {p: agg[(m, "legal")][p]["score"] for p in probs}
        c = {p: agg[(m, "cur")][p]["score"] for p in probs}
        r = paired(x, c, probs)
        dn = (mean_of(agg[(m, "legal")], probs, "n")
              - mean_of(agg[(m, "cur")], probs, "n"))
        print(f"  {m:<16} Score {r['mean']:+.4f}  {r['w']}/{r['l']}/{r['t']}  "
              f"p={r['p']:.4g}  rb={r['rb']:+.3f}   n_rep {dn:+.1f}")
    print("\n  利得どうしの対（手法間で利得が違うか）")
    gains = {m: {p: agg[(m, "legal")][p]["score"] - agg[(m, "cur")][p]["score"]
                 for p in probs} for m in METHODS}
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            r = paired(gains[a], gains[b], probs)
            print(f"    {a:<16}- {b:<16}{r['mean']:+.4f}  "
                  f"{r['w']}/{r['l']}/{r['t']}  p={r['p']:.4g}")

    print("\n## 4. 水準別 PR / F1（legal、問題平均）\n")
    print(f"{'手法':<16}{'量':<5}" + "".join(f"{l:>9}" for l in LEVEL_NAMES))
    for m in METHODS:
        g = agg[(m, "legal")]
        pr = np.mean([g[p]["pr_lv"] for p in probs], axis=0)
        f1 = np.mean([g[p]["f1_lv"] for p in probs], axis=0)
        print(f"{m:<16}{'PR':<5}" + "".join(f"{v:>9.4f}" for v in pr))
        print(f"{'':<16}{'F1':<5}" + "".join(f"{v:>9.4f}" for v in f1))

    print("\n## 5. 事前登録した棄却条件\n")
    rl = mean_of(agg[("Restart-Lander", "legal")], probs, "score")
    beaten = [m for m in METHODS[1:]
              if mean_of(agg[(m, "legal")], probs, "score") > rl]
    if not beaten:
        print("  `Restart-Lander` が規則を与えた後も Score で 2 手法に勝つ")
        print("  ==> その112 の「null が公式指標でも 2 手法に勝つ」は報告規則に依らない")
    else:
        print(f"  順位が入れ替わった（`Restart-Lander` を上回る: {', '.join(beaten)}）")
        print("  ==> その112 §3 の「MC-ESO が単独最下位」は報告規則のアーティファクト。俯瞰へ上げる")

    out = os.path.join(HERE, "ranking_d10.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "method", "arm", "n_reported",
                    "mpr", "mean_f1", "score"])
        for m in METHODS:
            for a in arms:
                for p in probs:
                    g = agg[(m, a)][p]
                    w.writerow([p, bench(p)[0], m, a, f"{g['n']:.1f}",
                                f"{g['mpr']:.4f}", f"{g['f1']:.4f}",
                                f"{g['score']:.4f}"])
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
