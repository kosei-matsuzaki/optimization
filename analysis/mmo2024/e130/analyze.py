#!/usr/bin/env python3
"""その130 — 報告半径 `r` の掃引を両手法・3 seed で取り、順位の半径依存を見る（キュー 1）。

**追加評価ゼロ。最適化 run ゼロ。** 保存済みの報告集合ダンプを読み直すだけ。
**新しい規則も新しい統計量も 1 つも定義しない** —— `rule_indices` / `score` / `paired` は
`analysis/mmo2024/e115/analyze.py` から import する（その116 以降の全サイクルと同一）。
`read_combined` は e128/e129 の**入出力**ヘルパの写しで、統計量ではない。

判定は `prereg.md` の J1 / J2 / J3。

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e130/analyze.py
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

from analyze import LEVELS, SPAN, paired, read_dump, rule_indices, score  # noqa: E402
from core.benchmarks import niching_by_name                              # noqa: E402

ARM = "eps_loose+dedup"
ARM_R_REF = 0.05 * SPAN                       # 現行の既定（関門の列）
# 事前登録した掃引軸（後から点を足さない）
FRACS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5]
PUBLISHED_BEST = 0.6080                       # TRDE-LR（競技優勝）公表値
PUBLISHED_RR = 0.5730                         # RR-CMA-ES（競技次点）公表値
GATE = {("Restart-Lander", "mpr"): 0.5644, ("Restart-Lander", "score"): 0.6284,
        ("RR-CMA-ES", "mpr"): 0.4594, ("RR-CMA-ES", "score"): 0.5377}
PROBS = [f"M{i:02d}-D10-PIN01" for i in range(1, 17)]
METHODS = ["Restart-Lander", "RR-CMA-ES"]
SEEDS = [0, 100, 200]

# (ディレクトリ, ファイル名パターン) か、`problem` 列つきの 1 本
SRC_DIR = {
    ("Restart-Lander", 0):   (os.path.join(MMO, "e115", "descents"), "{p}_seed0.csv"),
    ("Restart-Lander", 100): (os.path.join(MMO, "e115", "s1", "descents"), "{p}_seed100.csv"),
}
SRC_COMBINED = {
    ("Restart-Lander", 200): os.path.join(MMO, "e129", "descents_seed200.csv.gz"),
    ("RR-CMA-ES", 0):        os.path.join(MMO, "e127", "dumps_rrcma_seed0.csv.gz"),
    ("RR-CMA-ES", 100):      os.path.join(MMO, "e128", "dumps_rrcma_seed100.csv.gz"),
    ("RR-CMA-ES", 200):      os.path.join(MMO, "e129", "dumps_rrcma_seed200.csv.gz"),
}

_K: dict = {}


def K_of(prob):
    if prob not in _K:
        _K[prob] = int(niching_by_name(prob).n_global_optima)
    return _K[prob]


def find(d, name):
    for ext in ("", ".gz"):
        q = os.path.join(d, name + ext)
        if os.path.exists(q):
            return q
    return None


def read_combined(path):
    """`problem` 列つきの 1 本を problem -> (f, opt, xs) に割る（e128/e129 の写し）。"""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    grp: dict = {}
    for r in rows:
        grp.setdefault(r["problem"], []).append(r)
    res = {}
    for p, rs in grp.items():
        dim = sum(1 for k in rs[0] if k.startswith("x") and k[1:].isdigit())
        res[p] = (np.array([float(r["best_f"]) for r in rs]),
                  np.array([int(r["land_opt"]) for r in rs]),
                  np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rs]))
    return res


def load_all():
    """(method, seed) -> problem -> (f, opt, xs)。"""
    raw = {(m, s): {} for m in METHODS for s in SEEDS}
    for (m, s), (d, pat) in SRC_DIR.items():
        for p in PROBS:
            q = find(d, pat.format(p=p))
            if q:
                raw[(m, s)][p] = read_dump(q)
    for (m, s), path in SRC_COMBINED.items():
        if os.path.exists(path):
            for p, v in read_combined(path).items():
                if p in PROBS:
                    raw[(m, s)][p] = v
    return raw


def score_at(f, opt, xs, K, r_abs):
    """半径 `r_abs` で報告集合を作って採点する（r=0 は間引きなし ＝ 規則 `eps_loose`）。"""
    if r_abs <= 0.0:
        idx = rule_indices("eps_loose", f, K)
    else:
        idx = rule_indices(ARM, f, K, x=xs, r=r_abs)
    recall, prec, f1, sc, n = score(idx, f, opt, K)
    return dict(mpr=float(recall.mean()), prec=float(prec.mean()),
                f1=float(f1.mean()), score=float(sc.mean()), n=int(n),
                pr_tight=float(recall[LEVELS.index(1e-5)]))


def mean_over(cell, key, probs):
    return float(np.mean([cell[p][key] for p in probs]))


def main():
    raw = load_all()
    print("=" * 108)
    print("その130 — 報告半径 `r` の掃引（両手法 × 3 seed、追加評価ゼロ・最適化 run ゼロ）")
    print("=" * 108)
    for m in METHODS:
        for s in SEEDS:
            print(f"    {m:<16} seed {s:>3}: {len(raw[(m, s)])}/16 問")

    # 事前登録した 2 つの基礎
    base_a = [p for p in PROBS if all(p in raw[(m, 0)] for m in METHODS)]
    base_b = [p for p in PROBS
              if all(p in raw[(m, s)] for m in METHODS for s in SEEDS)]
    print(f"\n  基礎 A（seed 0）: n={len(base_a)}   "
          f"基礎 B（3 seed 平均）: n={len(base_b)}"
          f"  外れた: {', '.join(p[:3] for p in PROBS if p not in base_b)}")

    # cell[(m, s, frac)][problem] = dict
    cell = {}
    for (m, s), d in raw.items():
        for frac in FRACS:
            cell[(m, s, frac)] = {p: score_at(*v, K_of(p), frac * SPAN)
                                  for p, v in d.items()}

    # ------------------------------------------------------------- 関門
    print("\n## 0. 関門（r = 0.05*span の列が その116・その127 の記録と 4 桁一致するか）\n")
    ok = True
    for m in METHODS:
        c = cell[(m, 0, 0.05)]
        a, b = mean_over(c, "mpr", PROBS), mean_over(c, "score", PROBS)
        good = (abs(a - GATE[(m, "mpr")]) < 5e-5
                and abs(b - GATE[(m, "score")]) < 5e-5)
        ok &= good
        print(f"  {m:<16} MPR {a:.4f} (記録 {GATE[(m, 'mpr')]}) / "
              f"Score {b:.4f} (記録 {GATE[(m, 'score')]})  -> "
              f"{'一致' if good else '**不一致**'}")
    if not ok:
        print("\n  **採点経路が再現しないので判定に進まない。**")
        return 1

    def curve(m, frac, basis):
        """基礎 A は seed 0、基礎 B は 3 seed 平均。problem -> value の dict を返す。"""
        if basis == "A":
            return {p: cell[(m, 0, frac)][p] for p in base_a}
        out = {}
        for p in base_b:
            vs = [cell[(m, s, frac)][p] for s in SEEDS]
            out[p] = {k: float(np.mean([v[k] for v in vs])) for k in vs[0]}
        return out

    rows = []
    for basis, probs in (("A", base_a), ("B", base_b)):
        lab = ("基礎 A: seed 0・16 問" if basis == "A"
               else "基礎 B: 3 seed 平均・11 問")
        print(f"\n## 1{basis}. 半径ごとの 16 問平均（{lab}）\n")
        print(f"{'r/span':>9}{'r':>7} | " + " | ".join(
            f"{m[:13]:^40}" for m in METHODS))
        print(f"{'':>9}{'':>7} | " + " | ".join(
            f"{'MPR':>8}{'F1':>9}{'Score':>9}{'n_rep':>7} " for _ in METHODS))
        print("-" * 108)
        for frac in FRACS:
            cs = {m: curve(m, frac, basis) for m in METHODS}
            line = f"{frac:>9g}{frac * SPAN:>7.3g} | "
            line += " | ".join(
                f"{mean_over(cs[m], 'mpr', probs):>8.4f}"
                f"{mean_over(cs[m], 'f1', probs):>9.4f}"
                f"{mean_over(cs[m], 'score', probs):>9.4f}"
                f"{mean_over(cs[m], 'n', probs):>7.1f} " for m in METHODS)
            print(line)

        # --------------------------------------------------- J1 / J2
        print(f"\n## 2{basis}. J1（対 公表最良 {PUBLISHED_BEST}）と "
              f"J2（対比較 Restart-Lander − RR-CMA-ES）\n")
        print(f"{'r/span':>9}{'RL Score':>10}{'J1 差':>9}{'J1':>6}"
              f"{'  |  ':>5}{'d Score':>9}{'W/T/L':>9}{'p':>10}{'rb':>8}"
              f"{'  |  ':>5}{'d MPR':>9}{'p(MPR)':>10}")
        print("-" * 108)
        for frac in FRACS:
            cs = {m: curve(m, frac, basis) for m in METHODS}
            rl = mean_over(cs["Restart-Lander"], "score", probs)
            j1 = rl - PUBLISHED_BEST
            res = {}
            for key in ("score", "mpr"):
                a = {p: cs["Restart-Lander"][p][key] for p in probs}
                b = {p: cs["RR-CMA-ES"][p][key] for p in probs}
                res[key] = paired(a, b, probs)
            r = res["score"]
            wtl = "{}/{}/{}".format(r['w'], r['t'], r['l'])
            print(f"{frac:>9g}{rl:>10.4f}{j1:>+9.4f}"
                  f"{('勝ち' if j1 > 0 else '**負け**'):>6}{'  |  ':>5}"
                  f"{r['mean']:>+9.4f}{wtl:>9}"
                  f"{r['p']:>10.4g}{r['rb']:>+8.3f}{'  |  ':>5}"
                  f"{res['mpr']['mean']:>+9.4f}{res['mpr']['p']:>10.4g}")
            rows.append([basis, f"{frac:g}", f"{frac * SPAN:g}", len(probs),
                         f"{rl:.6f}",
                         f"{mean_over(cs['RR-CMA-ES'], 'score', probs):.6f}",
                         f"{mean_over(cs['Restart-Lander'], 'mpr', probs):.6f}",
                         f"{mean_over(cs['RR-CMA-ES'], 'mpr', probs):.6f}",
                         f"{mean_over(cs['Restart-Lander'], 'n', probs):.3f}",
                         f"{mean_over(cs['RR-CMA-ES'], 'n', probs):.3f}",
                         f"{j1:.6f}", f"{r['mean']:.6f}", r["w"], r["t"], r["l"],
                         f"{r['p']:.6g}", f"{r['rb']:.4f}",
                         f"{res['mpr']['mean']:.6f}", f"{res['mpr']['p']:.6g}"])

        sgn = [np.sign(paired({p: curve("Restart-Lander", f, basis)[p]["score"] for p in probs},
                              {p: curve("RR-CMA-ES", f, basis)[p]["score"] for p in probs},
                              probs)["mean"]) for f in FRACS]
        j1v = [mean_over(curve("Restart-Lander", f, basis), "score", probs) - PUBLISHED_BEST
               for f in FRACS]
        print(f"\n  **J2 判定（基礎 {basis}）**: 符号が正の半径 {int(sum(s > 0 for s in sgn))}/{len(FRACS)} 点"
              f"  -> {'全域で正（順位は半径に依らない）' if all(s > 0 for s in sgn) else '**反転あり**'}")
        print(f"  **J1 判定（基礎 {basis}）**: 公表最良を上回る半径 "
              f"{int(sum(v > 0 for v in j1v))}/{len(FRACS)} 点"
              f"  -> {'全域で上回る' if all(v > 0 for v in j1v) else '**但し書きが要る**'}"
              f"（負けるのは r/span = "
              f"{', '.join(f'{f:g}' for f, v in zip(FRACS, j1v) if v <= 0) or 'なし'}）")

    # ------------------------------------------------------------- J3
    print("\n## 3. J3（機序）—— 半径を下げたとき落ちるのは precision だけか（基礎 A、5 水準平均）\n")
    print(f"{'r/span':>9} | " + " | ".join(
        f"{m[:13]:^34}" for m in METHODS))
    print(f"{'':>9} | " + " | ".join(
        f"{'recall(MPR)':>12}{'precision':>11}{'n_rep':>8} " for _ in METHODS))
    print("-" * 108)
    for frac in FRACS:
        line = f"{frac:>9g} | "
        line += " | ".join(
            f"{mean_over(cell[(m, 0, frac)], 'mpr', base_a):>12.4f}"
            f"{mean_over(cell[(m, 0, frac)], 'prec', base_a):>11.4f}"
            f"{mean_over(cell[(m, 0, frac)], 'n', base_a):>8.1f} " for m in METHODS)
        print(line)
    print("\n  recall の r=0 からの損失（基礎 A、MPR 5 水準平均 / PR@1e-5）:")
    for m in METHODS:
        b0 = mean_over(cell[(m, 0, 0.0)], "mpr", base_a)
        t0 = mean_over(cell[(m, 0, 0.0)], "pr_tight", base_a)
        s = "    ".join(
            f"{f:g}: {mean_over(cell[(m, 0, f)], 'mpr', base_a) - b0:+.4f}"
            f"/{mean_over(cell[(m, 0, f)], 'pr_tight', base_a) - t0:+.4f}"
            for f in FRACS[1:])
        print(f"    {m:<16} {s}")

    out = os.path.join(HERE, "radius_sweep_d10.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["basis", "r_frac", "r_abs", "n_problems", "rl_score", "rr_score",
                    "rl_mpr", "rr_mpr", "rl_n_rep", "rr_n_rep", "j1_vs_published",
                    "d_score", "win", "tie", "loss", "p_score", "rb_score",
                    "d_mpr", "p_mpr"])
        w.writerows(rows)
    # 871 行あるので `.csv.gz`（保持規約: 生の .csv は数百行の集計まで）
    out2 = os.path.join(HERE, "by_problem_radius.csv.gz")
    with gzip.open(out2, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "method", "seed", "r_frac", "n_reported",
                    "mpr", "precision", "mean_f1", "score"])
        for p in PROBS:
            for m in METHODS:
                for s in SEEDS:
                    for frac in FRACS:
                        v = cell[(m, s, frac)].get(p)
                        if v is None:
                            continue
                        w.writerow([p, K_of(p), m, s, f"{frac:g}", v["n"],
                                    f"{v['mpr']:.6f}", f"{v['prec']:.6f}",
                                    f"{v['f1']:.6f}", f"{v['score']:.6f}"])
    print(f"\n  -> {out}\n  -> {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
