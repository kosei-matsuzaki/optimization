#!/usr/bin/env python3
"""その112 — 保存済み run を競技の公式指標（PR / static F1 / Score）で採点し直す。

追加評価ゼロ。`competition_setup_TR2024001.txt` §4 の定義:

    recall(eps)    = PR(eps) = detected(eps) / K
    precision(eps) = detected(eps) / n_reported     <- 分子は PR の分子と同じ
    F1(eps)        = 2 P R / (P + R)
    Score(eps)     = (PR + F1) / 2

集約は run -> (seed 平均) 問題 -> 16 問平均。水準 1e-1..1e-5 は等重み。

使い方: python3 analyze.py   （カレントは analysis/mmo2024/e112/ でも repo root でもよい）
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
LEVELS = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]

# 公表値（gecco2024_results_deck.txt 68-80 行、D=10）。Score は (MPR + mean-F1)/2。
PUBLISHED = {
    "RR-CMA-ES":       (0.651, 0.495),
    "TRDE-LR":         (0.538, 0.678),
    "Niching CMA-ES":  (0.078, 0.020),
}


def read_runs(path, method=None, rule="current"):
    """1 run 1 行の CSV を読む。返すのは dict のリスト。"""
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if rule is not None and row["rule"] != rule:
                continue
            if method is not None and row["method"] != method:
                continue
            k = int(row["n_optima"])
            rec = {
                "problem": row["function"],
                "method": row["method"],
                "seed": int(row["seed"]),
                "K": k,
                "n_reported": int(row["n_reported"]),
                "pr": np.array([float(row[f"pr_{l}"]) for l in LEVELS]),
            }
            rec["detected"] = np.rint(rec["pr"] * k).astype(int)
            out.append(rec)
    return out


def score_run(rec):
    """run 1 本を 5 水準で採点して PR / F1 / Score のベクトルを返す。"""
    k, n = rec["K"], rec["n_reported"]
    det = rec["detected"].astype(float)
    recall = det / k
    precision = det / n if n > 0 else np.zeros_like(det)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
    return recall, precision, f1, (recall + f1) / 2.0


def aggregate(runs):
    """問題ごとに seed 平均した (MPR, meanF1, Score, n_reported) を返す。"""
    by = defaultdict(list)
    for r in runs:
        by[r["problem"]].append(r)
    per_problem = {}
    for prob, rs in sorted(by.items()):
        mpr, mf1, msc, nrep, prl, f1l = [], [], [], [], [], []
        for r in rs:
            recall, prec, f1, sc = score_run(r)
            mpr.append(recall.mean())
            mf1.append(f1.mean())
            msc.append(sc.mean())
            nrep.append(r["n_reported"])
            prl.append(recall)
            f1l.append(f1)
        per_problem[prob] = {
            "K": rs[0]["K"],
            "seeds": len(rs),
            "mpr": float(np.mean(mpr)),
            "f1": float(np.mean(mf1)),
            "score": float(np.mean(msc)),
            "n_reported": float(np.mean(nrep)),
            "pr_levels": np.mean(prl, axis=0),
            "f1_levels": np.mean(f1l, axis=0),
        }
    return per_problem


def paired(a, b, probs):
    """問題を対にした両側 Wilcoxon。既定と exact の p を両方返す（その111 の教訓）。"""
    d = np.array([a[p] - b[p] for p in probs])
    wins = int((d > 0).sum())
    ties = int((d == 0).sum())
    loss = int((d < 0).sum())
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(mean=0.0, w=wins, t=ties, l=loss, stat=float("nan"),
                    p_def=1.0, p_exact=1.0, rb=0.0)
    res_def = stats.wilcoxon(nz)
    try:
        res_ex = stats.wilcoxon(nz, method="exact")
        p_ex = float(res_ex.pvalue)
    except Exception:
        p_ex = float("nan")
    r = stats.rankdata(np.abs(nz))
    rp = r[nz > 0].sum()
    rn = r[nz < 0].sum()
    rb = (rp - rn) / (rp + rn)
    return dict(mean=float(d.mean()), w=wins, t=ties, l=loss,
                stat=float(res_def.statistic), p_def=float(res_def.pvalue),
                p_exact=p_ex, rb=float(rb))


# ---------------------------------------------------------------- 恒等検査
def identity_check():
    """降下ダンプから detected(eps) を作り直し、by_problem の pr*K と突き合わせる。

    報告集合は by_problem と同じく f で昇順 trim（cap = max(100, 2K)）。
    """
    rows = []
    # その111 のダンプは optimiser seed（100）で名前が付いている（by_problem の
    # seed 列は index 1）。名前の規則が exp ごとに違うだけで中身は同じ形。
    for exp, seed, tag in (("e110", 0, "seed0"), ("e111", 1, "seed100")):
        bp = os.path.join(MMO, exp, "by_problem")
        for fn in sorted(os.listdir(bp)):
            rec = read_runs(os.path.join(bp, fn))[0]
            dump = os.path.join(MMO, exp, "descents",
                                f"{rec['problem']}_{tag}.csv.gz")
            if not os.path.exists(dump):
                rows.append((rec["problem"], seed, "missing", None, None))
                continue
            with gzip.open(dump, "rt") as fh:
                d = [(float(x["best_f"]), int(x["land_opt"])) for x in csv.DictReader(fh)]
            cap = max(100, 2 * rec["K"])
            d.sort(key=lambda t: t[0])
            kept = d[:cap]
            ok = True
            for j, lv in enumerate(LEVELS):
                eps = float(lv)
                got = len({o for f, o in kept if f <= eps})
                if got != rec["detected"][j]:
                    ok = False
            rows.append((rec["problem"], seed, "ok" if ok else "MISMATCH",
                         len(d), len(kept)))
    return rows


def main():
    mceso = read_runs(os.path.join(MMO, "e106", "baseline_d10_runs.csv"), "MC-ESO")
    nmmso = read_runs(os.path.join(MMO, "e109", "nmmso_runs_d10.csv"), "NMMSO")
    lander = []
    for exp in ("e110", "e111"):
        bp = os.path.join(MMO, exp, "by_problem")
        for fn in sorted(os.listdir(bp)):
            lander += read_runs(os.path.join(bp, fn))

    methods = {"MC-ESO": mceso, "NMMSO": nmmso, "Restart-Lander": lander}
    agg = {m: aggregate(rs) for m, rs in methods.items()}
    probs = sorted(agg["MC-ESO"])
    for m in agg:
        assert sorted(agg[m]) == probs, f"{m}: 問題集合が違う"

    print("=" * 78)
    print("その112 — 公式指標（PR / static F1 / Score）での再採点。追加評価ゼロ")
    print("=" * 78)

    print("\n## 1. 16 問平均（seed 平均のあと問題平均。水準 1e-1..1e-5 は等重み）\n")
    print(f"{'手法':<16}{'run':>5}{'MPR':>9}{'mean-F1':>10}{'Score':>9}"
          f"{'n_rep':>8}{'K':>6}")
    means = {}
    for m in ("Restart-Lander", "NMMSO", "MC-ESO"):
        a = agg[m]
        mpr = np.mean([a[p]["mpr"] for p in probs])
        f1 = np.mean([a[p]["f1"] for p in probs])
        sc = np.mean([a[p]["score"] for p in probs])
        nr = np.mean([a[p]["n_reported"] for p in probs])
        kk = np.mean([a[p]["K"] for p in probs])
        nrun = sum(len(v) for v in [[r for r in methods[m] if r["problem"] == p] for p in probs])
        means[m] = dict(mpr=mpr, f1=f1, score=sc, n_reported=nr)
        print(f"{m:<16}{nrun:>5}{mpr:>9.4f}{f1:>10.4f}{sc:>9.4f}{nr:>8.1f}{kk:>6.1f}")
    print()
    for name, (p_mpr, p_f1) in PUBLISHED.items():
        print(f"{name+' (公表)':<16}{'-':>5}{p_mpr:>9.3f}{p_f1:>10.3f}"
              f"{(p_mpr+p_f1)/2:>9.4f}{'-':>8}{'-':>6}")
    print("  ※ 公表値は 15 instance 平均、手元は PIN01 のみ。同一視しない。")

    print("\n## 2. 事前登録した 2 枝の判定\n")
    lan = means["Restart-Lander"]
    d_mpr = 0.651 - lan["mpr"]
    d_sc = 0.6080 - lan["score"]
    order_mpr = sorted(means, key=lambda m: -means[m]["mpr"])
    order_sc = sorted(means, key=lambda m: -means[m]["score"])
    cond1 = order_mpr == order_sc
    cond2 = d_sc <= 0.1532
    print(f"  MPR 順位   : {' > '.join(order_mpr)}")
    print(f"  Score 順位 : {' > '.join(order_sc)}")
    print(f"  条件 1（順位が同じ）           : {cond1}")
    print(f"  Delta_MPR   = 0.651  - {lan['mpr']:.4f} = {d_mpr:.4f}")
    print(f"  Delta_Score = 0.6080 - {lan['score']:.4f} = {d_sc:.4f}")
    print(f"  条件 2（Delta_Score <= 0.1532）: {cond2}")
    print(f"\n  ==> 発火したのは枝 {'A' if (cond1 and cond2) else 'B'}")

    print("\n## 3. 水準別（16 問平均。precision の分母は水準に依らない）\n")
    print(f"{'手法':<16}{'量':<5}" + "".join(f"{l:>9}" for l in LEVELS))
    for m in ("Restart-Lander", "NMMSO", "MC-ESO"):
        a = agg[m]
        pr = np.mean([a[p]["pr_levels"] for p in probs], axis=0)
        f1 = np.mean([a[p]["f1_levels"] for p in probs], axis=0)
        print(f"{m:<16}{'PR':<5}" + "".join(f"{v:>9.4f}" for v in pr))
        print(f"{'':<16}{'F1':<5}" + "".join(f"{v:>9.4f}" for v in f1))

    print("\n## 4. 対検定（16 問を対にした両側 Wilcoxon、Score の上で）\n")
    pairs = [("Restart-Lander", "MC-ESO"), ("Restart-Lander", "NMMSO"),
             ("NMMSO", "MC-ESO")]
    for key in ("score", "mpr", "f1"):
        print(f"  [{key}]")
        for x, y in pairs:
            ax = {p: agg[x][p][key] for p in probs}
            ay = {p: agg[y][p][key] for p in probs}
            r = paired(ax, ay, probs)
            print(f"    {x:<15} - {y:<15} mean {r['mean']:+.4f}  "
                  f"{r['w']}/{r['l']}/{r['t']}  W={r['stat']:.1f}  "
                  f"p={r['p_def']:.4g} (exact {r['p_exact']:.4g})  rb={r['rb']:+.3f}")

    print("\n## 5. 問題別（seed 平均）\n")
    hdr = f"{'問題':<16}{'K':>4}"
    for m in ("Restart-Lander", "NMMSO", "MC-ESO"):
        hdr += f"{m[:9]+' PR':>14}{'F1':>8}{'Sc':>8}{'nrep':>7}"
    print(hdr)
    for p in probs:
        line = f"{p:<16}{agg['MC-ESO'][p]['K']:>4}"
        for m in ("Restart-Lander", "NMMSO", "MC-ESO"):
            a = agg[m][p]
            line += f"{a['mpr']:>14.4f}{a['f1']:>8.4f}{a['score']:>8.4f}{a['n_reported']:>7.1f}"
        print(line)

    # CSV（数百行の集計なので生のまま置いてよい）
    out = os.path.join(HERE, "score_d10.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "method", "seeds", "n_reported",
                    "mpr", "mean_f1", "score"]
                   + [f"pr_{l}" for l in LEVELS] + [f"f1_{l}" for l in LEVELS])
        for p in probs:
            for m in ("Restart-Lander", "NMMSO", "MC-ESO"):
                a = agg[m][p]
                w.writerow([p, a["K"], m, a["seeds"], f"{a['n_reported']:.1f}",
                            f"{a['mpr']:.4f}", f"{a['f1']:.4f}", f"{a['score']:.4f}"]
                           + [f"{v:.4f}" for v in a["pr_levels"]]
                           + [f"{v:.4f}" for v in a["f1_levels"]])
    print(f"\n  -> {out}")

    print("\n## 6. 報告軸の上限（oracle trim。枝 B が開け直した軸に、安いスクリーンを当てる）\n")
    print("  規則: 各最適につき f 最小の 1 点だけを報告する（n = detected(1e-1)）。")
    print("  PR は 5 水準とも不変（1e-5 で当たる点は 1e-1 でも当たる ＝ 入れ子）、")
    print("  precision(eps) = detected(eps) / detected(1e-1) が上限。")
    print("  **最適の位置を使うので実装可能な規則ではない。`mpr_sup` と同じ身分の上限。**\n")
    print(f"{'手法':<16}{'MPR':>9}{'F1_sup':>9}{'Score_sup':>11}{'現 Score':>10}{'余地':>8}")
    sup = {}
    for m in ("Restart-Lander", "NMMSO", "MC-ESO"):
        per = []
        for p in probs:
            rs = [r for r in methods[m] if r["problem"] == p]
            vals = []
            for r in rs:
                det = r["detected"].astype(float)
                n = det[0]  # detected(1e-1)
                recall = det / r["K"]
                prec = det / n if n > 0 else np.zeros_like(det)
                dn = prec + recall
                f1 = np.where(dn > 0, 2 * prec * recall / np.where(dn > 0, dn, 1), 0.0)
                vals.append((recall.mean(), f1.mean(), ((recall + f1) / 2).mean()))
            per.append(np.mean(vals, axis=0))
        per = np.mean(per, axis=0)
        sup[m] = per
        print(f"{m:<16}{per[0]:>9.4f}{per[1]:>9.4f}{per[2]:>11.4f}"
              f"{means[m]['score']:>10.4f}{per[2]-means[m]['score']:>8.4f}")
    print(f"\n  公表最良 Score（D=10、TRDE-LR）= 0.6080 / RR-CMA-ES = 0.5730")
    ls = sup["Restart-Lander"][2]
    print(f"  `Restart-Lander` の上限 {ls:.4f} は公表最良を "
          f"{'上回る' if ls > 0.6080 else '下回る'}（差 {ls-0.6080:+.4f}）")
    print("  ==> 報告軸は F1 の上では空ではない。ただし上限であって到達値ではない。")

    print("\n## 7. 恒等検査（降下ダンプ -> detected を作り直して by_problem と突き合わせ）\n")
    rows = identity_check()
    bad = [r for r in rows if r[2] != "ok"]
    print(f"  32 run × 5 水準: 一致 {len(rows)-len(bad)}/{len(rows)} run")
    for r in bad:
        print(f"    {r[0]} seed{r[1]}: {r[2]} (dump {r[3]}, kept {r[4]})")
    if not bad:
        print("  ==> ダンプは報告集合の忠実な表現。報告点数を切る腕は追加評価ゼロで計算できる。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
