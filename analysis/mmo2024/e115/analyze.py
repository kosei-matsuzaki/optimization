#!/usr/bin/env python3
"""その115 — 合法な報告規則で oracle 余地 0.2517 のどれだけが実際に取れるか（キュー 1）。

保存済みの降下ダンプから報告集合だけを作り直して、競技の公式指標
（PR / static F1 / Score、`competition_setup_TR2024001.txt` §4）で採点する。
**探索は 1 ビットも変えない。追加評価ゼロ。**

合法性（§5）: 使ってよいのは D・探索域・評価予算・eps_tight=1e-5・eps_loose=1e-1 だけ。
K（NGM）も f* も使えないので、現行の `max(100,2K)` cap は**それ自体が規則違反**。
ダンプの `land_opt` / `dist` はオラクル列で、**採点にだけ使い規則の定義には使わない**。

使い方: python3 analysis/mmo2024/e115/analyze.py
"""
from __future__ import annotations

import csv
import gzip
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
LEVELS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
LEVEL_NAMES = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]
EPS_LOOSE, EPS_TIGHT = 1e-1, 1e-5
SPAN = 10.0  # 探索域は [-5,5]^D（§3）。合法に使ってよい情報。

# その112 の比較先 3 本
CUR_SCORE, ORACLE_SCORE, PUBLISHED_BEST = 0.3639, 0.6156, 0.6080
PREREG_GATE = 0.45


# ------------------------------------------------------------------ 読み込み
def read_dump(path):
    """降下ダンプを読む。座標列 x* があれば一緒に返す（無ければ None）。"""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    f = np.array([float(r["best_f"]) for r in rows])
    opt = np.array([int(r["land_opt"]) for r in rows])
    xs = None
    if rows and "x0" in rows[0]:
        dim = sum(1 for k in rows[0] if k.startswith("x") and k[1:].isdigit())
        xs = np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rows])
    return f, opt, xs


def problem_K(bp_dir, prob):
    """by_problem の集計 CSV から K を読む（採点のためだけに使う）。"""
    with open(os.path.join(bp_dir, f"{prob}.csv")) as fh:
        row = next(csv.DictReader(fh))
    return int(row["n_optima"])


# (降下ダンプのディレクトリ, by_problem のディレクトリ, seed 番号)
SRC_PLAIN = [(os.path.join(MMO, "e110", "descents"), os.path.join(MMO, "e110", "by_problem"), 0),
             (os.path.join(MMO, "e111", "descents"), os.path.join(MMO, "e111", "by_problem"), 1)]
SRC_COORD = [(os.path.join(HERE, "descents"), os.path.join(HERE, "by_problem"), 0),
             (os.path.join(HERE, "s1", "descents"), os.path.join(HERE, "s1", "by_problem"), 1)]


def load_runs(with_coords=False):
    """1 run 1 要素で読む。`with_coords` は座標つきの再 run（その115）を読む。"""
    runs = []
    sources = SRC_COORD if with_coords else SRC_PLAIN
    for dd, bp, seed in sources:
        exp = os.path.basename(os.path.dirname(dd))
        if not os.path.isdir(dd):
            continue
        for fn in sorted(os.listdir(dd)):
            if not fn.endswith((".csv", ".csv.gz")):
                continue
            prob = fn.split("_")[0]
            f, opt, xs = read_dump(os.path.join(dd, fn))
            if with_coords and xs is None:
                continue
            runs.append(dict(problem=prob, exp=exp, seed=seed, K=problem_K(bp, prob),
                             f=f, opt=opt, x=xs))
    return runs


# ------------------------------------------------------------------ 報告規則
def rule_indices(name, f, K, x=None, r=None):
    """報告する降下の index 集合を返す。K を使う腕は `cur` だけ（規則違反の対照）。"""
    order = np.argsort(f, kind="stable")          # f 昇順
    if name == "cur":                             # 現行（違反・対照）
        return order[: max(100, 2 * K)]
    if name == "cap100":
        return order[:100]
    if name == "eps_loose":
        return order[f[order] <= f.min() + EPS_LOOSE]
    if name == "eps_tight":
        return order[f[order] <= f.min() + EPS_TIGHT]
    if name == "eps_loose_cap100":
        return order[f[order] <= f.min() + EPS_LOOSE][:100]
    if name == "eps_tight_cap100":
        return order[f[order] <= f.min() + EPS_TIGHT][:100]
    if name == "fdedup":
        # 座標を使わない間引き（負の対照）: f の値が eps_tight 以内なら同じ最適とみなす。
        # **大域最適はどれも f = f* なので、この規則は全部を 1 点に潰す。**
        keep, seen = [], []
        for i in order:
            if all(abs(f[i] - f[j]) > EPS_TIGHT for j in seen):
                keep.append(int(i))
                seen.append(int(i))
        return np.array(keep, dtype=int)
    if name.startswith("dedup") or "+dedup" in name:   # 貪欲な最小間隔の間引き
        base = order
        if name.startswith("eps_loose+"):
            base = order[f[order] <= f.min() + EPS_LOOSE]
        keep = []
        for i in base:
            if all(np.linalg.norm(x[i] - x[j]) > r for j in keep):
                keep.append(int(i))
        return np.array(keep, dtype=int)
    raise KeyError(name)


def score(idx, f, opt, K):
    """報告集合 idx を 5 水準で採点して (PR, precision, F1, Score, n) を返す。"""
    n = len(idx)
    det = np.array([len({int(o) for o, v in zip(opt[idx], f[idx]) if v <= e})
                    for e in LEVELS], dtype=float)
    recall = det / K
    prec = det / n if n > 0 else np.zeros_like(det)
    dn = prec + recall
    f1 = np.where(dn > 0, 2 * prec * recall / np.where(dn > 0, dn, 1.0), 0.0)
    return recall, prec, f1, (recall + f1) / 2.0, n


def oracle_indices(f, opt, detected_only=False):
    """各最適につき f 最小の 1 点（上限。`land_opt` を使うので実装不可能）。

    `detected_only=True` は その112 §5 の定義（`n = detected(1e-1)` ＝ 1e-1 に
    届かなかった着地は報告しない ＝ **完全な間引き ＋ 完全な足切り**）。
    False は**完全な間引きだけ**（触れた盆地の数だけ報告する）で、
    距離による間引きが届きうる天井はこちら。
    """
    best = {}
    for i, (o, v) in enumerate(zip(opt, f)):
        if o not in best or v < f[best[o]]:
            best[o] = i
    idx = sorted(best.values())
    if detected_only:
        idx = [i for i in idx if f[i] <= EPS_LOOSE]
    return np.array(idx, dtype=int)


# ------------------------------------------------------------------ 集約・検定
def aggregate(runs, arm_fn):
    """問題 -> seed 平均 の dict を返す。"""
    by = defaultdict(list)
    for r in runs:
        idx = arm_fn(r)
        recall, prec, f1, sc, n = score(idx, r["f"], r["opt"], r["K"])
        by[r["problem"]].append(dict(mpr=recall.mean(), f1=f1.mean(),
                                     score=sc.mean(), n=n,
                                     pr_lv=recall, f1_lv=f1))
    out = {}
    for p, rs in by.items():
        out[p] = dict(mpr=float(np.mean([v["mpr"] for v in rs])),
                      f1=float(np.mean([v["f1"] for v in rs])),
                      score=float(np.mean([v["score"] for v in rs])),
                      n=float(np.mean([v["n"] for v in rs])),
                      pr_lv=np.mean([v["pr_lv"] for v in rs], axis=0),
                      f1_lv=np.mean([v["f1_lv"] for v in rs], axis=0))
    return out


def paired(a, b, probs):
    d = np.array([a[p] - b[p] for p in probs])
    w, t, l = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(mean=0.0, w=w, t=t, l=l, p=1.0, rb=0.0)
    try:
        p = float(stats.wilcoxon(nz, method="exact").pvalue)
    except Exception:
        p = float(stats.wilcoxon(nz).pvalue)
    rk = stats.rankdata(np.abs(nz))
    rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
    return dict(mean=float(d.mean()), w=w, t=t, l=l, p=p, rb=float(rb))


def mean_of(agg, probs, key):
    return float(np.mean([agg[p][key] for p in probs]))


# ------------------------------------------------------------------ main
def main():
    runs = load_runs()
    probs = sorted({r["problem"] for r in runs})
    print("=" * 82)
    print("その115 — 合法な報告規則で余地のどれだけが取れるか（キュー 1、追加評価ゼロ）")
    print("=" * 82)
    print(f"\n  run: {len(runs)} 本（16 問 × 2 seed）  比較先: 現行 {CUR_SCORE:.4f} / "
          f"oracle {ORACLE_SCORE:.4f} / 公表最良 {PUBLISHED_BEST:.4f}  ゲート {PREREG_GATE}")

    arms = ["cur", "cap100", "eps_loose", "eps_loose_cap100", "eps_tight",
            "eps_tight_cap100", "fdedup"]
    agg = {a: aggregate(runs, lambda r, a=a: rule_indices(a, r["f"], r["K"]))
           for a in arms}
    agg["oracle_dedup"] = aggregate(runs, lambda r: oracle_indices(r["f"], r["opt"]))
    agg["oracle_e112"] = aggregate(
        runs, lambda r: oracle_indices(r["f"], r["opt"], detected_only=True))

    print("\n## 1. 16 問平均（seed 平均のあと問題平均、5 水準等重み）\n")
    print(f"{'腕':<20}{'合法':>6}{'MPR':>9}{'mean-F1':>10}{'Score':>9}{'n_rep':>8}"
          f"{'対現行':>9}")
    legal = {"cur": "×", "oracle_dedup": "上限", "oracle_e112": "上限"}
    for a in arms + ["oracle_dedup", "oracle_e112"]:
        m = {k: mean_of(agg[a], probs, k) for k in ("mpr", "f1", "score", "n")}
        print(f"{a:<20}{legal.get(a, '○'):>6}{m['mpr']:>9.4f}{m['f1']:>10.4f}"
              f"{m['score']:>9.4f}{m['n']:>8.1f}"
              f"{m['score'] - mean_of(agg['cur'], probs, 'score'):>+9.4f}")

    print("\n## 2. 対検定（16 問を対にした両側 Wilcoxon exact、対 `cur`）\n")
    cur = {p: agg["cur"][p]["score"] for p in probs}
    for a in arms[1:] + ["oracle_dedup", "oracle_e112"]:
        x = {p: agg[a][p]["score"] for p in probs}
        r = paired(x, cur, probs)
        print(f"  {a:<20} Score {r['mean']:+.4f}  {r['w']}/{r['l']}/{r['t']}  "
              f"p={r['p']:.4g}  rb={r['rb']:+.3f}")

    print("\n## 3. 水準別（16 問平均）\n")
    print(f"{'腕':<20}{'量':<5}" + "".join(f"{l:>9}" for l in LEVEL_NAMES))
    for a in arms + ["oracle_dedup", "oracle_e112"]:
        pr = np.mean([agg[a][p]["pr_lv"] for p in probs], axis=0)
        f1 = np.mean([agg[a][p]["f1_lv"] for p in probs], axis=0)
        print(f"{a:<20}{'PR':<5}" + "".join(f"{v:>9.4f}" for v in pr))
        print(f"{'':<20}{'F1':<5}" + "".join(f"{v:>9.4f}" for v in f1))

    # ---- 閾値の掃引（1e-1 と 1e-5 だけが厳密に合法。中間は診断値）
    print("\n## 4. 閾値の掃引 `best_f <= min + t`（**1e-1 と 1e-5 だけが合法**、中間は診断）\n")
    print(f"{'t':<10}{'MPR':>9}{'mean-F1':>10}{'Score':>9}{'n_rep':>8}")
    for t in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-8):
        a = aggregate(runs, lambda r, t=t: np.argsort(r["f"], kind="stable")[
            np.sort(r["f"]) <= r["f"].min() + t])
        tag = f"{t:g}" + ("" if t in (1e-1, 1e-5) else " (診断)")
        print(f"{tag:<10}{mean_of(a, probs, 'mpr'):>9.4f}"
              f"{mean_of(a, probs, 'f1'):>10.4f}{mean_of(a, probs, 'score'):>9.4f}"
              f"{mean_of(a, probs, 'n'):>8.1f}")

    # ---- 座標つきの腕（e115 の再 run）
    cruns = load_runs(with_coords=True)
    best_overall = max((mean_of(agg[a], probs, "score"), a) for a in arms[1:])
    if cruns:
        cprobs = sorted({r["problem"] for r in cruns})
        print(f"\n## 5. 最小間隔の間引き（座標つき、その115 の再 run {len(cruns)} 本 / "
              f"{len(cprobs)} 問）\n")
        # 恒等検査: e110 と best_f が一致するか
        ref = {(r["seed"], r["problem"]): r for r in load_runs()}
        same = sum(1 for r in cruns
                   if (r["seed"], r["problem"]) in ref
                   and len(r["f"]) == len(ref[(r["seed"], r["problem"])]["f"])
                   and np.array_equal(r["f"], ref[(r["seed"], r["problem"])]["f"]))
        print(f"  恒等検査（e110/e111 の同 seed と `best_f` 列が完全一致）: {same}/{len(cruns)} run")
        print(f"\n{'腕':<24}{'MPR':>9}{'mean-F1':>10}{'Score':>9}{'n_rep':>8}")
        base = aggregate(cruns, lambda r: rule_indices("cur", r["f"], r["K"]))
        print(f"{'cur (対照)':<24}{mean_of(base, cprobs, 'mpr'):>9.4f}"
              f"{mean_of(base, cprobs, 'f1'):>10.4f}"
              f"{mean_of(base, cprobs, 'score'):>9.4f}"
              f"{mean_of(base, cprobs, 'n'):>8.1f}")
        orc = aggregate(cruns, lambda r: oracle_indices(r["f"], r["opt"]))
        orc112 = aggregate(cruns, lambda r: oracle_indices(r["f"], r["opt"],
                                                           detected_only=True))
        cbest = (0.0, None, None)
        for nm in ("dedup", "eps_loose+dedup"):
            for frac in (1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1):
                r_abs = frac * SPAN
                a = aggregate(cruns, lambda r, nm=nm, rr=r_abs:
                              rule_indices(nm, r["f"], r["K"], r["x"], rr))
                sc = mean_of(a, cprobs, "score")
                print(f"{nm + f' r={frac:g}*span':<24}{mean_of(a, cprobs, 'mpr'):>9.4f}"
                      f"{mean_of(a, cprobs, 'f1'):>10.4f}{sc:>9.4f}"
                      f"{mean_of(a, cprobs, 'n'):>8.1f}")
                if sc > cbest[0]:
                    cbest = (sc, f"{nm} r={frac:g}*span", a)
        for nm, a in (("oracle_dedup (間引きだけ)", orc),
                      ("oracle_e112 (間引き+足切り)", orc112)):
            print(f"{nm:<24}{mean_of(a, cprobs, 'mpr'):>9.4f}"
                  f"{mean_of(a, cprobs, 'f1'):>10.4f}"
                  f"{mean_of(a, cprobs, 'score'):>9.4f}"
                  f"{mean_of(a, cprobs, 'n'):>8.1f}")
        if cbest[2] is not None:
            r = paired({p: cbest[2][p]["score"] for p in cprobs},
                       {p: base[p]["score"] for p in cprobs}, cprobs)
            print(f"\n  最良 {cbest[1]} 対 cur（問題を対に）: "
                  f"{r['mean']:+.4f}  {r['w']}/{r['l']}/{r['t']}  p={r['p']:.4g}  rb={r['rb']:+.3f}")
            for nm, a in (("oracle_dedup", orc), ("oracle_e112", orc112)):
                r2 = paired({p: cbest[2][p]["score"] for p in cprobs},
                            {p: a[p]["score"] for p in cprobs}, cprobs)
                print(f"  最良 {cbest[1]} 対 {nm:<12}: "
                      f"{r2['mean']:+.4f}  {r2['w']}/{r2['l']}/{r2['t']}  p={r2['p']:.4g}")
            best_overall = max(best_overall, (cbest[0], cbest[1]))

    if cruns:
        print("\n## 5b. 間引き半径が効く理由（オラクル診断。規則の定義には使わない）\n")
        print(f"{'問題':<16}{'降下':>6}{'触れた最適':>10}{'同一最適の広がり(max)':>22}"
              f"{'異なる最適の最小距離':>22}{'PR損失 r=0.1':>13}")
        for r in sorted(cruns, key=lambda r: r["problem"]):
            within, centres = [], {}
            for o in set(r["opt"].tolist()):
                pts = r["x"][r["opt"] == o]
                within.append(float(np.max(np.linalg.norm(pts - pts[np.argmin(r["f"][r["opt"] == o])], axis=1))))
                centres[o] = pts[np.argmin(r["f"][r["opt"] == o])]
            ks = sorted(centres)
            between = min((float(np.linalg.norm(centres[a] - centres[b]))
                           for i, a in enumerate(ks) for b in ks[i + 1:]), default=float("nan"))
            idx = rule_indices("dedup", r["f"], r["K"], r["x"], 0.1 * SPAN)
            det_d = len({int(o) for o, v in zip(r["opt"][idx], r["f"][idx]) if v <= EPS_LOOSE})
            det_o = len({int(o) for o, v in zip(r["opt"], r["f"]) if v <= EPS_LOOSE})
            print(f"{r['problem']:<16}{len(r['f']):>6}{len(centres):>10}"
                  f"{max(within):>22.3g}{between:>22.3g}{det_o - det_d:>13}")

    print("\n## 6. 事前登録した棄却条件\n")
    print(f"  合法な腕の最良: {best_overall[1]} = Score {best_overall[0]:.4f}")
    print(f"  ゲート {PREREG_GATE}: {'到達' if best_overall[0] >= PREREG_GATE else '未達'}")
    print(f"  ==> {'枝: 3 手法に広げる' if best_overall[0] >= PREREG_GATE else '枝: 報告軸は oracle でしか開かない、と書いて閉じる'}")

    # CSV（問題 × 腕の集計。数百行なので生のまま置いてよい）
    out = os.path.join(HERE, "rules_d10.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "arm", "n_reported", "mpr", "mean_f1", "score"])
        for a in arms + ["oracle_dedup", "oracle_e112"]:
            for p in probs:
                w.writerow([p, next(r["K"] for r in runs if r["problem"] == p), a,
                            f"{agg[a][p]['n']:.1f}", f"{agg[a][p]['mpr']:.4f}",
                            f"{agg[a][p]['f1']:.4f}", f"{agg[a][p]['score']:.4f}"])
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
