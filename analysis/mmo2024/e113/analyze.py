#!/usr/bin/env python3
"""その113 — 4.5 倍差の内訳: hunt 本数か、選抜か、降下か。

追加評価ゼロの側（null）は e110 の保存ダンプを読み直すだけ。MC-ESO 側は
e113 の計器つき run（MC-ESO-traced、base とビット一致）の spillover 単位ダンプ。

被覆の定義は e112 の恒等検査が採点器と突き合わせたものと同じ:
    detected(eps) = |{ land_opt : best_f <= eps }|,  PR = detected / K

使い方: python3 analyze.py   （カレントはどこでもよい）
"""
from __future__ import annotations

import csv
import glob
import gzip
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
LEVELS = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]
EPS = [float(x) for x in LEVELS]


def read_dump(path, fcol="best_f"):
    """降下 / segment ダンプを (best_f, land_opt) の列として読む。順序は保存順。"""
    with gzip.open(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def coverage(rows, k, n=None, fcol="best_f"):
    """先頭 n 本（None なら全部）で 5 水準の PR を返す。"""
    use = rows if n is None else rows[:n]
    out = []
    for e in EPS:
        out.append(len({int(r["land_opt"]) for r in use
                        if float(r[fcol]) <= e}) / k)
    return np.array(out)


def k_of(problem):
    """by_problem の集計 CSV から K を引く（e113 が無ければ e110）。"""
    for exp in ("e113", "e110"):
        p = os.path.join(MMO, exp, "by_problem", f"{problem}.csv")
        if os.path.exists(p):
            with open(p) as fh:
                for row in csv.DictReader(fh):
                    return int(row["n_optima"])
    raise FileNotFoundError(problem)


def paired(a, b, labels):
    """b - a の対検定（両側 Wilcoxon）。a, b は同順の 1 次元配列。"""
    d = np.asarray(b) - np.asarray(a)
    wins = int((d > 0).sum()); ties = int((d == 0).sum()); loss = int((d < 0).sum())
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(mean=0.0, w=wins, t=ties, l=loss, p=1.0, rb=0.0)
    res = stats.wilcoxon(nz)
    rp = float(stats.rankdata(np.abs(nz))[nz > 0].sum())
    rn = float(stats.rankdata(np.abs(nz))[nz < 0].sum())
    return dict(mean=float(d.mean()), w=wins, t=ties, l=loss,
                p=float(res.pvalue), rb=(rp - rn) / (rp + rn))


def main():
    seg_paths = sorted(glob.glob(os.path.join(HERE, "hunts", "*_segments.csv.gz")))
    if not seg_paths:
        raise SystemExit("no segment dumps in e113/hunts -- run run.sh first")

    rows = []
    for sp in seg_paths:
        problem = os.path.basename(sp).split("_seed")[0]
        nullp = os.path.join(MMO, "e110", "descents", f"{problem}_seed0.csv.gz")
        if not os.path.exists(nullp):
            print(f"  (skip {problem}: no null dump)")
            continue
        k = k_of(problem)
        seg = read_dump(sp)
        nul = read_dump(nullp)
        n_mc, n_nl = len(seg), len(nul)
        dr_path = sp.replace("_segments.", "_draws.")
        draws = read_dump(dr_path) if os.path.exists(dr_path) else []

        cov_mc = coverage(seg, k)
        cov_nl = coverage(nul, k)
        cov_nl_m = coverage(nul, k, n=n_mc)          # 同本数に切った null
        # 事前登録は「MC のほうが本数が少ない」を想定していた。実測は逆なので、
        # 同本数の比較は MC を n_null 本に切る向きでも出す（切るのは先頭 n 本）。
        cov_mc_m = coverage(seg, k, n=n_nl)

        ev_mc = np.array([int(r["evals"]) for r in seg], dtype=float)
        ev_nl = np.array([int(r["evals"]) for r in nul], dtype=float)
        bs = sum(1 for r in seg if r["basin_switch"] == "1")

        # 選抜: segment の着地が、その segment を開いた draw 群のうち f 最良の
        # 点の最近傍最適と一致するか。
        # 注意: 通常 spillover は f 最良の 1 スロットを温存する（kind=retained）
        # ので、「f 最良の draw」はしばしばその温存点そのもの ＝ 一致は自明になる。
        # 新規 draw だけに絞った版を併記する。
        hit = tot = hit_f = tot_f = 0
        if draws:
            by_seg = defaultdict(list)
            for r in draws:
                by_seg[int(r["segment"])].append(r)
            for r in seg:
                s = int(r["segment"])
                if s not in by_seg:
                    continue
                best = min(by_seg[s], key=lambda q: float(q["f"]))
                tot += 1
                hit += int(best["land_opt"] == r["land_opt"])
                fresh = [q for q in by_seg[s] if q["kind"] == "reseed"]
                if fresh:
                    bf = min(fresh, key=lambda q: float(q["f"]))
                    tot_f += 1
                    hit_f += int(bf["land_opt"] == r["land_opt"])

        # 冗長性: 1 hunt あたり何個の「新しい盆地」を買っているか。分子は
        # 着地した相異なる最適の数（f 閾値なし = 触れた盆地）。
        uniq_mc = len({r["land_opt"] for r in seg})
        uniq_nl = len({r["land_opt"] for r in nul})
        uniq_nl_m = len({r["land_opt"] for r in nul[:n_mc]})

        rows.append(dict(uniq_mc=uniq_mc, uniq_nl=uniq_nl, uniq_nl_m=uniq_nl_m,
                         problem=problem, K=k, n_mc=n_mc, n_nl=n_nl,
                         ev_mc=float(np.median(ev_mc)),
                         ev_nl=float(np.median(ev_nl)),
                         bs=bs, cov_mc=cov_mc, cov_nl=cov_nl,
                         cov_nl_m=cov_nl_m, cov_mc_m=cov_mc_m,
                         sel_hit=hit, sel_tot=tot,
                         self_hit=hit_f, self_tot=tot_f,
                         n_draw=len(draws)))

    if not rows:
        raise SystemExit("nothing paired")

    print(f"\n=== {len(rows)} 問 / D=10 / 50 万評価 / seed 0（MC-ESO は計器つき、"
          f"base とビット一致） ===\n")
    print(f"{'problem':<16}{'K':>4}{'n_MC':>6}{'n_null':>7}{'n_MC/n_null':>12}"
          f"{'ev/hunt MC':>12}{'null':>8}{'bswitch':>8}"
          f"{'cov@1e-1 MC':>13}{'null(n_MC)':>11}{'null(all)':>10}{'sel%':>7}")
    print("-" * 116)
    for r in rows:
        sel = f"{100*r['sel_hit']/r['sel_tot']:.0f}" if r["sel_tot"] else "-"
        print(f"{r['problem']:<16}{r['K']:>4}{r['n_mc']:>6}{r['n_nl']:>7}"
              f"{r['n_mc']/r['n_nl']:>12.2f}{r['ev_mc']:>12.0f}{r['ev_nl']:>8.0f}"
              f"{r['bs']:>8}{r['cov_mc'][0]:>13.3f}{r['cov_nl_m'][0]:>11.3f}"
              f"{r['cov_nl'][0]:>10.3f}{sel:>7}")

    n_mc = np.array([r["n_mc"] for r in rows], float)
    n_nl = np.array([r["n_nl"] for r in rows], float)
    cov_mc = np.array([r["cov_mc"] for r in rows])
    cov_nl = np.array([r["cov_nl"] for r in rows])
    cov_nl_m = np.array([r["cov_nl_m"] for r in rows])
    cov_mc_m = np.array([r["cov_mc_m"] for r in rows])

    print("\n--- 本数 ---")
    print(f"n_MC   平均 {n_mc.mean():.1f}  中央 {np.median(n_mc):.1f}  "
          f"レンジ {n_mc.min():.0f}-{n_mc.max():.0f}")
    print(f"n_null 平均 {n_nl.mean():.1f}  中央 {np.median(n_nl):.1f}  "
          f"レンジ {n_nl.min():.0f}-{n_nl.max():.0f}")
    ratio = n_mc / n_nl
    ge3 = int((ratio >= 1/3).sum())
    print(f"n_MC / n_null: 平均 {ratio.mean():.3f}  中央 {np.median(ratio):.3f}  "
          f"1/3 以上の問題数 {ge3}/{len(rows)}")
    print(f"n_MC >= n_null の問題数 {int((n_mc >= n_nl).sum())}/{len(rows)}")
    ev_mc = np.array([r["ev_mc"] for r in rows]); ev_nl = np.array([r["ev_nl"] for r in rows])
    print(f"1 hunt あたり評価回数（問題ごとの中央値の平均）: MC {ev_mc.mean():.0f}  "
          f"null {ev_nl.mean():.0f}  比 {ev_mc.mean()/ev_nl.mean():.2f}")

    u_mc = np.array([r["uniq_mc"] for r in rows], float)
    u_nl = np.array([r["uniq_nl"] for r in rows], float)
    u_nlm = np.array([r["uniq_nl_m"] for r in rows], float)
    kk = np.array([r["K"] for r in rows], float)
    print("\n--- 冗長性（触れた盆地 ＝ f 閾値なしの相異なる着地最適）---")
    print(f"{'':<22}{'相異なる着地':>12}{'/K':>8}{'1 hunt あたり':>14}")
    for lbl, u, n in (("MC-ESO", u_mc, n_mc), ("null（全本）", u_nl, n_nl),
                      ("null（同本数 n_MC）", u_nlm, n_mc)):
        print(f"{lbl:<22}{u.mean():>12.2f}{(u/kk).mean():>8.3f}{(u/n).mean():>14.4f}")
    print(f"＝ MC-ESO は null の {(n_mc/n_nl).mean():.1f} 倍 hunt を回して "
          f"相異なる盆地は {(u_mc/u_nl).mean():.2f} 倍")

    print("\n--- 被覆（5 水準）---")
    print(f"{'水準':<8}{'MC 全本':>10}{'null 同本数':>13}{'null 全本':>11}"
          f"{'比 同本数':>11}{'比 全本':>10}")
    for j, lv in enumerate(LEVELS):
        a, b, c = cov_mc[:, j].mean(), cov_nl_m[:, j].mean(), cov_nl[:, j].mean()
        print(f"{lv:<8}{a:>10.4f}{b:>13.4f}{c:>11.4f}"
              f"{(b/a if a else float('inf')):>11.2f}{(c/a if a else float('inf')):>10.2f}")
    a5, b5, c5 = cov_mc.mean(1), cov_nl_m.mean(1), cov_nl.mean(1)
    print(f"{'平均':<8}{a5.mean():>10.4f}{b5.mean():>13.4f}{c5.mean():>11.4f}"
          f"{(b5.mean()/a5.mean()):>11.2f}{(c5.mean()/a5.mean()):>10.2f}")

    d5, e5 = cov_mc_m.mean(1), cov_nl.mean(1)
    print("\n--- 同本数（MC を n_null 本に切る向き。実測は n_MC > n_null なので"
          "こちらが効く比較）---")
    print(f"{'水準':<8}{'MC 先頭 n_null 本':>18}{'null 全本':>11}{'比':>8}")
    for j, lv in enumerate(LEVELS):
        a, c = cov_mc_m[:, j].mean(), cov_nl[:, j].mean()
        print(f"{lv:<8}{a:>18.4f}{c:>11.4f}{(c/a if a else float('inf')):>8.2f}")
    print(f"{'平均':<8}{d5.mean():>18.4f}{e5.mean():>11.4f}"
          f"{(e5.mean()/d5.mean() if d5.mean() else float('inf')):>8.2f}")

    print("\n--- 深さの保持率（@1e-5 / @1e-1）---")
    print(f"MC-ESO {cov_mc[:, 4].mean()/cov_mc[:, 0].mean():.3f}   "
          f"null {cov_nl[:, 4].mean()/cov_nl[:, 0].mean():.3f}")

    print("\n--- 対検定（16 問、両側 Wilcoxon）---")
    for lbl, x, y in (("同本数 null − MC @1e-1", cov_mc[:, 0], cov_nl_m[:, 0]),
                      ("同本数 null − MC 5 水準平均", a5, b5),
                      ("全本 null − MC 5 水準平均", a5, c5)):
        s = paired(x, y, None)
        print(f"{lbl:<30} 平均差 {s['mean']:+.4f}  {s['w']}/{s['t']}/{s['l']}  "
              f"p={s['p']:.4g}  rb={s['rb']:+.3f}")

    print("\n--- 事前登録の枝 ---")
    maj = len(rows) / 2
    cov_ratio = b5.mean() / a5.mean() if a5.mean() else float("inf")
    cov_ratio1 = cov_nl_m[:, 0].mean() / cov_mc[:, 0].mean() if cov_mc[:, 0].mean() else float("inf")
    cond_count = ge3 > maj
    print(f"条件 1（n_MC >= n_null/3 が過半）: {cond_count}  ({ge3}/{len(rows)})")
    print(f"条件 2（同本数 null の被覆 < 1.5x MC）: "
          f"@1e-1 比 {cov_ratio1:.2f} → {cov_ratio1 < 1.5}, "
          f"5 水準平均 比 {cov_ratio:.2f} → {cov_ratio < 1.5}")
    if not cond_count:
        print("→ 枝 B（本数）が発火")
    elif cov_ratio1 < 1.5:
        print("→ 枝 A（降下そのもの）が発火")
    else:
        print("→ 枝 C（撒き方・選抜）が発火")

    sel_hit = sum(r["sel_hit"] for r in rows); sel_tot = sum(r["sel_tot"] for r in rows)
    if sel_tot:
        hf = sum(r["self_hit"] for r in rows); tf = sum(r["self_tot"] for r in rows)
        print(f"\n選抜: segment の着地が「その segment の f 最良 draw の最近傍最適」と"
              f"一致した割合 {100*sel_hit/sel_tot:.1f}%  ({sel_hit}/{sel_tot})")
        print(f"      温存スロット（kind=retained）を除いた新規 draw だけでは "
              f"{100*hf/tf:.1f}%  ({hf}/{tf})")
        kinds = defaultdict(int)
        for r in rows:
            kinds["seg"] += r["n_mc"]
        print(f"      draw 行 {sum(r['n_draw'] for r in rows)} / segment "
              f"{int(n_mc.sum())} = 1 segment あたり "
              f"{sum(r['n_draw'] for r in rows)/n_mc.sum():.1f} draw"
              f"（1 問あたり {sum(r['n_draw'] for r in rows)/len(rows):.0f} 評価 = "
              f"予算 50 万の {100*sum(r['n_draw'] for r in rows)/len(rows)/5e5:.1f}%）")
    print(f"basin_switch の総数 {sum(r['bs'] for r in rows)} / segment 総数 "
          f"{int(n_mc.sum())}、draw 行 {sum(r['n_draw'] for r in rows)}")


if __name__ == "__main__":
    main()
