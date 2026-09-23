#!/usr/bin/env python3
"""その161 (ii) — その113 の結論を、いま生き残っている入力だけで引き直す。

その113 が読んだ入力は 2 本とも消えている:
  - null 側 `e110/descents/`  -> その118 が削除（`e115/descents/` が厳密な上位集合）
  - MC-ESO 側 `e113/hunts/`   -> その145 が削除（集計は e148 に移設された）

この script は生き残った入力だけから同じ量を組み直し、その113 の記録値
（`e113/scored.txt`、＝ acceptance_topology.md の その113 の節の出典）と機械で照合する。
追加評価ゼロ。使い方: python3 rederive_e113.py
"""
from __future__ import annotations

import csv
import gzip
import os
import re
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
LEVELS = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]
EPS = [float(x) for x in LEVELS]

NULL_DIR = os.path.join(MMO, "e115", "descents")        # e110 の上位集合
MC_AGG = os.path.join(MMO, "e148", "by_problem_e145_arms.csv")   # e145 からの移設先
SCORED = os.path.join(MMO, "e113", "scored.txt")        # その113 の記録出力


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_null():
    """e115/descents から null 側を組み直す。K は e110/by_problem から引く。"""
    if not os.path.isdir(NULL_DIR):
        die(f"null 降下ダンプが無い: {NULL_DIR}")
    out = {}
    for fn in sorted(os.listdir(NULL_DIR)):
        if not fn.endswith("_seed0.csv.gz"):
            continue
        problem = fn.split("_seed")[0]
        kp = os.path.join(MMO, "e110", "by_problem", f"{problem}.csv")
        if not os.path.exists(kp):
            die(f"K を引く集計が無い: {kp}")
        with open(kp) as fh:
            k = int(next(csv.DictReader(fh))["n_optima"])
        with gzip.open(os.path.join(NULL_DIR, fn), "rt") as fh:
            rows = list(csv.DictReader(fh))
        cov = np.array([len({int(r["land_opt"]) for r in rows
                             if float(r["best_f"]) <= e}) / k for e in EPS])
        out[problem] = dict(
            K=k, n=len(rows), cov=cov,
            ev_med=float(np.median([int(r["evals"]) for r in rows])),
            uniq=len({int(r["land_opt"]) for r in rows}),
        )
    if len(out) != 16:
        die(f"null 側が 16 問そろわない（{len(out)} 問）")
    return out


def load_mceso():
    """MC-ESO 側の被覆と hunt 本数は e148 に移設された集計から引く。"""
    if not os.path.exists(MC_AGG):
        die(f"MC-ESO の集計が無い: {MC_AGG}\n"
            f"（e113/HUNTS_REMOVED.md は e145/by_problem.csv を指しているが "
            f"e145 は その150 が削除済み。移設先は e148）")
    out = {}
    with open(MC_AGG) as fh:
        for r in csv.DictReader(fh):
            if r["series"] != "MC-ESO":
                continue
            out[r["problem"]] = dict(
                K=int(r["K"]), n=int(r["n_hunts"]),
                cov=np.array([float(r[f"cov_{lv}"]) for lv in LEVELS]),
            )
    if len(out) != 16:
        die(f"MC-ESO 側が 16 問そろわない（{len(out)} 問）")
    return out


def load_scored():
    """その113 の記録出力から問題別の行を読む（照合の基準）。"""
    if not os.path.exists(SCORED):
        die(f"その113 の記録出力が無い: {SCORED}")
    ref = {}
    pat = re.compile(r"^(M\d\d-D10-PIN01)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+"
                     r"(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    for line in open(SCORED):
        m = pat.match(line.strip())
        if m:
            ref[m.group(1)] = dict(
                K=int(m.group(2)), n_mc=int(m.group(3)), n_nl=int(m.group(4)),
                ev_mc=float(m.group(6)), ev_nl=float(m.group(7)),
                cov_mc_1=float(m.group(9)), cov_nl_1=float(m.group(11)),
            )
    if len(ref) != 16:
        die(f"scored.txt から 16 行読めない（{len(ref)} 行）")
    return ref


def paired(a, b):
    d = np.asarray(b) - np.asarray(a)
    w, t, l = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(mean=0.0, w=w, t=t, l=l, p=1.0, rb=0.0)
    res = stats.wilcoxon(nz)
    rk = stats.rankdata(np.abs(nz))
    rp, rn = float(rk[nz > 0].sum()), float(rk[nz < 0].sum())
    return dict(mean=float(d.mean()), w=w, t=t, l=l,
                p=float(res.pvalue), rb=(rp - rn) / (rp + rn))


def main():
    nul, mc, ref = load_null(), load_mceso(), load_scored()
    pids = sorted(ref)

    print("=== その113 の引き直し（生き残った入力のみ・追加評価ゼロ）===")
    print(f"null 側  : {os.path.relpath(NULL_DIR, MMO)}  （e110/descents の上位集合）")
    print(f"MC-ESO 側: {os.path.relpath(MC_AGG, MMO)}  （e113/hunts の集計の移設先）")
    print(f"照合基準 : {os.path.relpath(SCORED, MMO)}\n")

    print(f"{'problem':<16}{'K':>3}{'n_MC':>6}{'n_null':>7}{'ev/h null':>11}"
          f"{'cov@1e-1 MC':>13}{'null':>8}   照合")
    print("-" * 80)
    bad = []
    for p in pids:
        r, a, b = ref[p], mc[p], nul[p]
        chk = []
        if a["K"] != r["K"] or b["K"] != r["K"]:
            chk.append("K")
        if a["n"] != r["n_mc"]:
            chk.append(f"n_MC {a['n']}!={r['n_mc']}")
        if b["n"] != r["n_nl"]:
            chk.append(f"n_null {b['n']}!={r['n_nl']}")
        if abs(b["ev_med"] - r["ev_nl"]) > 0.5:
            chk.append(f"ev_null {b['ev_med']:.0f}!={r['ev_nl']:.0f}")
        if abs(a["cov"][0] - r["cov_mc_1"]) > 5e-4:
            chk.append(f"covMC {a['cov'][0]:.3f}!={r['cov_mc_1']:.3f}")
        if abs(b["cov"][0] - r["cov_nl_1"]) > 5e-4:
            chk.append(f"covNull {b['cov'][0]:.3f}!={r['cov_nl_1']:.3f}")
        if chk:
            bad.append((p, chk))
        print(f"{p:<16}{r['K']:>3}{a['n']:>6}{b['n']:>7}{b['ev_med']:>11.0f}"
              f"{a['cov'][0]:>13.3f}{b['cov'][0]:>8.3f}   "
              f"{'OK' if not chk else '; '.join(chk)}")

    cov_mc = np.array([mc[p]["cov"] for p in pids])
    cov_nl = np.array([nul[p]["cov"] for p in pids])
    n_mc = np.array([mc[p]["n"] for p in pids], float)
    n_nl = np.array([nul[p]["n"] for p in pids], float)

    print("\n--- 本数（その113 の記録: MC 317.9 / null 95.2 / 16-16）---")
    print(f"n_MC 平均 {n_mc.mean():.1f}  レンジ {n_mc.min():.0f}-{n_mc.max():.0f}")
    print(f"n_null 平均 {n_nl.mean():.1f}  レンジ {n_nl.min():.0f}-{n_nl.max():.0f}")
    print(f"n_MC >= n_null の問題数 {int((n_mc >= n_nl).sum())}/16")

    print("\n--- 被覆（その113 の記録: MC 0.1094 / null 0.5644 / 比 5.16）---")
    print(f"{'水準':<8}{'MC':>10}{'null':>10}{'比':>8}")
    for j, lv in enumerate(LEVELS):
        a, c = cov_mc[:, j].mean(), cov_nl[:, j].mean()
        print(f"{lv:<8}{a:>10.4f}{c:>10.4f}{(c/a if a else float('inf')):>8.2f}")
    a5, c5 = cov_mc.mean(1), cov_nl.mean(1)
    print(f"{'平均':<8}{a5.mean():>10.4f}{c5.mean():>10.4f}{c5.mean()/a5.mean():>8.2f}")

    print("\n--- 深さの保持率（@1e-5 / @1e-1。記録: MC 0.275 / null 0.850）---")
    print(f"MC-ESO {cov_mc[:, 4].mean()/cov_mc[:, 0].mean():.3f}   "
          f"null {cov_nl[:, 4].mean()/cov_nl[:, 0].mean():.3f}")

    s = paired(a5, c5)
    print("\n--- 対検定（16 問、両側 Wilcoxon。記録: +0.4550 / 16-0-0 / "
          "p=3.052e-5 / rb=+1.000）---")
    print(f"全本 null − MC 5 水準平均  平均差 {s['mean']:+.4f}  "
          f"{s['w']}/{s['t']}/{s['l']}  p={s['p']:.4g}  rb={s['rb']:+.3f}")
    s1 = paired(cov_mc[:, 0], cov_nl[:, 0])
    print(f"全本 null − MC @1e-1       平均差 {s1['mean']:+.4f}  "
          f"{s1['w']}/{s1['t']}/{s1['l']}  p={s1['p']:.4g}  rb={s1['rb']:+.3f}")

    print("\n--- 判定 ---")
    if bad:
        print(f"問題別の照合で {len(bad)} 問がずれた:")
        for p, c in bad:
            print(f"  {p}: {'; '.join(c)}")
        print("→ その113 の記録値は生き残った入力から再現しない。")
    else:
        print("問題別の照合 16/16 一致（K・hunt 本数・null の 1 本あたり評価・"
              "@1e-1 被覆の 4 量）。")
        print("→ その113 の結論は生き残った入力だけで引き直せる。")

    with open(os.path.join(HERE, "rederived_e113.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "n_mc", "n_null", "ev_med_null", "uniq_null"]
                   + [f"cov_mc_{lv}" for lv in LEVELS]
                   + [f"cov_null_{lv}" for lv in LEVELS])
        for p in pids:
            w.writerow([p, ref[p]["K"], mc[p]["n"], nul[p]["n"],
                        f"{nul[p]['ev_med']:.0f}", nul[p]["uniq"]]
                       + [f"{v:.4f}" for v in mc[p]["cov"]]
                       + [f"{v:.4f}" for v in nul[p]["cov"]])
    print(f"\n書き出し: {os.path.relpath(os.path.join(HERE, 'rederived_e113.csv'), MMO)}")


if __name__ == "__main__":
    main()
