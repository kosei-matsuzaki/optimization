#!/usr/bin/env python3
"""その164 — キュー 1: D=20 で `descent_budget` を 25000 -> 50000 に上げた 1 水準（飽和点を取る）。

**採点器・報告規則・統計量は既存からそのまま import する。新しい統計量は 1 つも定義しない。**

  * 採点（`rule_indices` / `score` / `read_dump` / `SPAN`） -> `e115/analyze.py`
  * 3 分類（`classify`）と対検定（`paired`）                -> `e158/analyze.py`

入力の出自:

  * **50000（この回、新規 16 run）** -> `e164/descents/M??-D20-PIN01_seed0.csv[.gz]`
                                        （畳んだ後は `e164/descents.csv.gz`）
  * **25000（主対照、追加評価ゼロ）** -> `e163/descents.csv.gz` の保存物
  * **12500（既定、追加評価ゼロ）**   -> `e151/descents.csv.gz` の保存物
  * **その158 の D=20 の 3 分類・`maxfevals` 比重** -> `prereg.md` の転記表（測らない）

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e164/analyze.py
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)


def _load(entry):
    path = os.path.join(MMO, entry, "analyze.py")
    if not os.path.isfile(path):
        raise SystemExit(f"採点器が無い: {path}\n"
                         "  -> しきい値を自前で置かない規則なので続行しない。")
    spec = importlib.util.spec_from_file_location(f"{entry}_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E115, E158 = _load("e115"), _load("e158")
SPAN, rule_indices, score = E115.SPAN, E115.rule_indices, E115.score
classify, paired = E158.classify, E158.paired
EPS_TIGHT, EPS_LOOSE = E115.EPS_TIGHT, E115.EPS_LOOSE

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN
PROBS = [f"M{i:02d}" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]

# --- 転記（prereg.md §2）。この回では 1 つも測らない ---
E163_SCORE, E163_MPR, E163_F1 = 0.4922, 0.4188, 0.5656      # その163（関門の照合先）
E151_SCORE, E151_MPR, E151_F1 = 0.4504, 0.3831, 0.5176      # その151
SEED_SD = 0.0066                                             # その123
E158_D20 = dict(new=0.0378, dup=0.4700, miss=0.4922)         # その158（ε=1e-5、12500 側）
E158_MAXFEV = dict(all=0.243, miss=0.423)                    # その158（D=20、12500 側）
E163_MAXFEV = dict(all=0.127, miss=0.195)                    # その163 §4（25000 側）
PUB_D20_SCORE = 0.4445


def _need(path):
    if not os.path.exists(path):
        raise SystemExit(f"入力が無い: {path}\n"
                         "  -> その150 §3 の経路（欠けた入力を黙って飛ばして nan の表を刷る）は踏まない。")
    return path


def _rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(_need(path), "rt") as fh:
        return list(csv.DictReader(fh))


def _pack(rs):
    rs = sorted(rs, key=lambda r: int(r["descent"]))
    dim = sum(1 for k in rs[0] if k.startswith("x") and k[1:].isdigit())
    return dict(f=np.array([float(r["best_f"]) for r in rs]),
                opt=np.array([int(r["land_opt"]) for r in rs]),
                xs=np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rs]),
                stop=[r["stop"] for r in rs],
                evals=np.array([int(r["evals"]) for r in rs]))


def load_folded(entry):
    """保存物（畳んだ 1 本、`problem` 列で割る）。追加評価ゼロ。"""
    by: dict = {}
    for r in _rows(os.path.join(MMO, entry, "descents.csv.gz")):
        by.setdefault(r["problem"].split("-")[0], []).append(r)
    return {p: _pack(rs) for p, rs in by.items()}


def load_this():
    """この回の新規 run。per-problem と畳んだ 1 本の両方を見る（畳む前後で同じ出力になる）。"""
    out: dict = {}
    folded = os.path.join(HERE, "descents.csv.gz")
    if os.path.exists(folded):
        by: dict = {}
        for r in _rows(folded):
            by.setdefault(r["problem"].split("-")[0], []).append(r)
        out.update({p: _pack(rs) for p, rs in by.items()})
    dd = os.path.join(HERE, "descents")
    if os.path.isdir(dd):
        for fn in sorted(os.listdir(dd)):
            if not (fn.endswith(".csv") or fn.endswith(".csv.gz")):
                continue
            p = fn.split("_")[0].split("-")[0]
            out.setdefault(p, _pack(_rows(os.path.join(dd, fn))))
    return out


def K_of(prob):
    from core.benchmarks import niching_by_name
    return int(niching_by_name(f"{prob}-D20-PIN01").n_global_optima)


def scored(run, K):
    idx = rule_indices(ARM, run["f"], K, x=run["xs"], r=ARM_R)
    recall, prec, f1, sc, n = score(idx, run["f"], run["opt"], K)
    cov = float(recall[0])
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()), score=float(sc.mean()),
                n=int(n), ndump=len(run["f"]), cov=cov,
                keep=float(recall[4] / cov) if cov > 0 else float("nan"),
                hits5=int((run["f"] <= EPS_TIGHT).sum()),
                distinct5=len({int(o) for v, o in zip(run["f"], run["opt"]) if v <= EPS_TIGHT}))


def cls(run, eps):
    new, dup, miss, seen, miss_idx = classify(run["f"], run["opt"], eps)
    n = len(run["f"])
    return dict(new=new / n, dup=dup / n, miss=miss / n, seen=seen, n=n,
                dupshare=dup / (new + dup) if (new + dup) else float("nan"),
                miss_idx=miss_idx)


def fmt(d):
    m, p, rb, (w, t, l) = d
    return f"mean {m:+.4f}  {w}/{t}/{l}  p={p:.5g}  rb={rb:+.3f}"


def main():
    out = []
    P = out.append
    C12, C25, B50 = load_folded("e151"), load_folded("e163"), load_this()
    K = {p: K_of(p) for p in PROBS}

    # ------------------------------------------------- 関門（主対照を採点し直す）
    P("## 関門 —— 同じ採点器で その163（25000）の保存物を採点し直し、記録値と一致するか")
    miss_ctrl = [p for p in PROBS if p not in C25]
    if miss_ctrl:
        P(f"  **主対照が欠けている: {' '.join(miss_ctrl)}。関門を通せない。**")
        print("\n".join(out))
        return
    s25 = {p: scored(C25[p], K[p]) for p in PROBS}
    s12 = {p: scored(C12[p], K[p]) for p in PROBS} if all(p in C12 for p in PROBS) else None
    m_sc = float(np.mean([s25[p]["score"] for p in PROBS]))
    m_mpr = float(np.mean([s25[p]["mpr"] for p in PROBS]))
    ok = abs(m_sc - E163_SCORE) < 5e-5 and abs(m_mpr - E163_MPR) < 5e-5
    P(f"  再計算: Score {m_sc:.4f} / MPR {m_mpr:.4f}   記録: {E163_SCORE:.4f} / {E163_MPR:.4f}"
      f"  -> {'通過' if ok else '不一致'}")
    if s12 is not None:
        a_sc = float(np.mean([s12[p]["score"] for p in PROBS]))
        a_mpr = float(np.mean([s12[p]["mpr"] for p in PROBS]))
        P(f"  （副）その151（12500）再計算: Score {a_sc:.4f} / MPR {a_mpr:.4f}   記録: "
          f"{E151_SCORE:.4f} / {E151_MPR:.4f}  -> "
          f"{'一致' if abs(a_sc - E151_SCORE) < 5e-5 and abs(a_mpr - E151_MPR) < 5e-5 else '不一致'}")
    if not ok:
        P("")
        P("**関門が外れた。事前登録どおり判定は出さない。**")
        print("\n".join(out))
        return

    done = [p for p in PROBS if p in B50]
    missing = [p for p in PROBS if p not in B50]
    k = len(done)
    na = sum(1 for p in done if p in GROUP_A)
    P("")
    P("## 50000（この回の新規 run）")
    P(f"  完走: k = {k} / 16  （群 A {na} / 群 B {k - na}）")
    P(f"  **落ちた問題: {' '.join(missing) if missing else '（なし）'}**")
    if k == 0:
        P("  **run がゼロ。判定は出せない。**")
        print("\n".join(out))
        return
    s50 = {p: scored(B50[p], K[p]) for p in done}

    # ------------------------------------------------- 問題別（キュー 1 の指定）
    P("")
    P("  | 問題 | K | Score 12500 | Score 25000 | Score 50000 | Δ(50k−25k) | MPR 25000 | MPR 50000 | "
      "F1 25000 | F1 50000 | 降下 12500 | 25000 | 50000 | 相異なり(1e-5) 12500/25000/50000 |")
    P("  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in done:
        a, b, c = s12[p] if s12 else None, s25[p], s50[p]
        P(f"  | {p} | {K[p]} | {a['score']:.4f} | {b['score']:.4f} | {c['score']:.4f} | "
          f"{c['score'] - b['score']:+.4f} | {b['mpr']:.4f} | {c['mpr']:.4f} | "
          f"{b['f1']:.4f} | {c['f1']:.4f} | {a['ndump']} | {b['ndump']} | {c['ndump']} | "
          f"{a['distinct5']}/{b['distinct5']}/{c['distinct5']} |")

    def mm(s, key):
        return float(np.mean([s[p][key] for p in done]))

    for lab, s in (("12500", s12), ("25000", s25), ("50000", s50)):
        P(f"  {k} 問平均 {lab}: Score {mm(s, 'score'):.4f} / MPR {mm(s, 'mpr'):.4f} / "
          f"mean-F1 {mm(s, 'f1'):.4f} / 報告点数 {mm(s, 'n'):.2f} / 1e-1 被覆 {mm(s, 'cov'):.4f} / "
          f"深さ保持率 {mm(s, 'keep'):.4f} / 降下本数 {mm(s, 'ndump'):.1f}")

    # ------------------------------------------------- 群 A / 群 B（prereg §4-6）
    P("")
    P("## 群 A（K=20, M01-M08）/ 群 B（K=10, M09-M16）の分離 —— 本数が K を下回る代償を読む")
    for gname, G in (("A", GROUP_A), ("B", GROUP_B)):
        g = [p for p in done if p in G]
        if not g:
            P(f"  群 {gname}: run なし")
            continue
        row = []
        for lab, s in (("12500", s12), ("25000", s25), ("50000", s50)):
            row.append(f"{lab} Score {np.mean([s[p]['score'] for p in g]):.4f} "
                       f"(降下 {np.mean([s[p]['ndump'] for p in g]):.1f})")
        P(f"  群 {gname}（{len(g)} 問、K={K[g[0]]}）: " + " / ".join(row))
    P("")
    P("  **降下本数が K を下回る問題（50000 側）**: " + (", ".join(
        f"{p} ({s50[p]['ndump']} 本 < K={K[p]})" for p in done if s50[p]["ndump"] < K[p]) or "（なし）"))
    P("  **同（25000 側、対照）**: " + (", ".join(
        f"{p} ({s25[p]['ndump']} 本 < K={K[p]})" for p in done if s25[p]["ndump"] < K[p]) or "（なし）"))

    # ------------------------------------------------- 群別の対検定（prereg §4-6）
    P("")
    P("## 群別の対検定（50000 − 25000）—— 16 問平均が相殺かどうかを見る")
    for gname, G in (("A(K=20)", GROUP_A), ("B(K=10)", GROUP_B)):
        g = [p for p in done if p in G]
        if len(g) < 2:
            P(f"  群 {gname}: 有効対 {len(g)} 本。検定しない")
            continue
        for key, lab in (("score", "Score"), ("mpr", "MPR"), ("f1", "mean-F1"),
                         ("ndump", "降下本数"), ("distinct5", "相異なり")):
            P(f"  群 {gname} {lab:>8}: "
              f"{fmt(paired([s25[p][key] for p in g], [s50[p][key] for p in g]))}")
        P(f"  群 {gname} 平均 Score: 12500 {np.mean([s12[p]['score'] for p in g]):.4f} -> "
          f"25000 {np.mean([s25[p]['score'] for p in g]):.4f} -> "
          f"50000 {np.mean([s50[p]['score'] for p in g]):.4f}")
        P(f"  群 {gname} 平均 相異なり(1e-5): {np.mean([s12[p]['distinct5'] for p in g]):.2f} -> "
          f"{np.mean([s25[p]['distinct5'] for p in g]):.2f} -> "
          f"{np.mean([s50[p]['distinct5'] for p in g]):.2f}  (K={K[g[0]]})")
    P("")
    P("  降下本数 / K の比（run が 3 水準で異なる問題だけ。1 を切ると被覆が構造的に頭打ち）:")
    for p in done:
        if s50[p]["ndump"] == s25[p]["ndump"] and s25[p]["ndump"] == s12[p]["ndump"]:
            continue
        P(f"    {p} K={K[p]:2d}  降下/K  12500 {s12[p]['ndump'] / K[p]:.2f} / "
          f"25000 {s25[p]['ndump'] / K[p]:.2f} / 50000 {s50[p]['ndump'] / K[p]:.2f}   "
          f"Δscore(50k-25k) {s50[p]['score'] - s25[p]['score']:+.4f}")
    moved = [p for p in done if abs(s50[p]["score"] - s25[p]["score"]) > 1e-12]
    P("")
    P(f"  動いた問題だけの対検定（n={len(moved)}: {' '.join(moved)}）:")
    if len(moved) >= 2:
        for key, lab in (("score", "Score"), ("mpr", "MPR"), ("f1", "mean-F1")):
            P(f"    {lab:>8}: "
              f"{fmt(paired([s25[p][key] for p in moved], [s50[p][key] for p in moved]))}")

    # ------------------------------------------------- 主判定
    P("")
    P("## 判定（prereg.md §4）")
    P("  主判定の対検定は 50000 − 25000 の向き（正なら 50000 が上）。両側 Wilcoxon exact、α=0.05")
    stats_row = {}
    for key, lab in (("score", "Score"), ("mpr", "MPR"), ("f1", "mean-F1"),
                     ("cov", "1e-1 被覆"), ("keep", "深さ保持率"),
                     ("n", "報告点数"), ("ndump", "降下本数"), ("distinct5", "相異なり(1e-5)")):
        d = paired([s25[p][key] for p in done], [s50[p][key] for p in done])
        stats_row[key] = d
        P(f"    {lab:>14}: {fmt(d)}")
    P("")
    P("  （参考）50000 − 12500 の向き（3 点の端どうし）:")
    for key, lab in (("score", "Score"), ("mpr", "MPR"), ("f1", "mean-F1"), ("ndump", "降下本数")):
        P(f"    {lab:>14}: {fmt(paired([s12[p][key] for p in done], [s50[p][key] for p in done]))}")
    dsc, psc = stats_row["score"][0], stats_row["score"][1]
    P("")
    P(f"  **主判定**: Δ Score = {dsc:+.4f}（対照 25000 {mm(s25, 'score'):.4f} -> "
      f"50000 {mm(s50, 'score'):.4f}）、p={psc:.5g}、rb={stats_row['score'][2]:+.3f}")
    P(f"    その123 の seed SD = {SEED_SD:.4f}。|Δ| は SD の {abs(dsc) / SEED_SD:.2f} 倍")
    fire_a = abs(dsc) < SEED_SD and psc >= 0.05
    fire_b = dsc > SEED_SD and psc < 0.05
    fire_c = dsc < -SEED_SD and psc < 0.05
    P(f"  **反証条件 (a)**（|Δ| < {SEED_SD} かつ非有意 ＝ 飽和点は 25000 と 50000 の間。この軸を閉じる）: "
      f"**{'発火' if fire_a else '不発'}**")
    P(f"  **反証条件 (b)**（Δ > +{SEED_SD} かつ有意 ＝ 深さで決まるという強い主張。俯瞰に上げる）: "
      f"**{'発火' if fire_b else '不発'}**")
    P(f"  **反証条件 (c)**（Δ < −{SEED_SD} かつ有意 ＝ 山の頂は 12500 と 50000 の間、単調ではない）: "
      f"**{'発火' if fire_c else '不発'}**")
    if not (fire_a or fire_b or fire_c):
        P("  **どれも発火しない** ＝ 符号は出たが seed SD か有意性のどちらかを満たさない"
          "（「動いたとは書けない」。範囲として報告する）")
    P(f"  （参考）50000 の {k} 問平均 Score {mm(s50, 'score'):.4f} 対 公表最良 D=20 "
      f"{PUB_D20_SCORE:.4f} ＝ {mm(s50, 'score') - PUB_D20_SCORE:+.4f}")

    # ------------------------------------------------- 機序 1: 3 分類
    P("")
    P("## 機序 (1) —— その158 の 3 分類（初出 i / 重複 ii / 未到達 iii）")
    for eps, nm in ((EPS_TIGHT, "1e-5（主水準）"), (EPS_LOOSE, "1e-1（感度）")):
        cc = {lab: {p: cls(R[p], eps) for p in done}
              for lab, R in (("12500", C12), ("25000", C25), ("50000", B50))}
        P(f"  ε={nm}")
        P(f"    {'':>10} {'初出(i)':>10} {'重複(ii)':>10} {'未到達(iii)':>12} {'重複/ヒット':>12} {'降下':>8}")
        for lab in ("12500", "25000", "50000"):
            c = cc[lab]
            P(f"    {lab:>10} {np.mean([c[p]['new'] for p in done]):>10.4f} "
              f"{np.mean([c[p]['dup'] for p in done]):>10.4f} "
              f"{np.mean([c[p]['miss'] for p in done]):>12.4f} "
              f"{np.nanmean([c[p]['dupshare'] for p in done]):>12.4f} "
              f"{np.mean([c[p]['n'] for p in done]):>8.1f}")
        for key, lab in (("new", "初出(i)"), ("dup", "重複(ii)"), ("miss", "未到達(iii)"),
                         ("dupshare", "重複/ヒット")):
            va = [cc["25000"][p][key] for p in done]
            vb = [cc["50000"][p][key] for p in done]
            good = [i for i in range(len(done)) if np.isfinite(va[i]) and np.isfinite(vb[i])]
            if len(good) < 2:
                P(f"      {lab:>12}: 有効対 {len(good)} 本。検定しない")
                continue
            P(f"      {lab:>12} (50k−25k): "
              f"{fmt(paired([va[i] for i in good], [vb[i] for i in good]))}  （有効対 {len(good)}）")
        if eps == EPS_TIGHT:
            c = cc["12500"]
            P(f"      その158 の D=20 既定値との照合（12500 側）: 初出 "
              f"{np.mean([c[p]['new'] for p in done]):.4f} 対 {E158_D20['new']:.4f} / 重複 "
              f"{np.mean([c[p]['dup'] for p in done]):.4f} 対 {E158_D20['dup']:.4f} / 未到達 "
              f"{np.mean([c[p]['miss'] for p in done]):.4f} 対 {E158_D20['miss']:.4f}")

    # ------------------------------------------------- 機序 2: stop の内訳
    P("")
    P("## 機序 (2) —— `stop` の内訳（`maxfevals` ＝ 降下上限を使い切った降下）")
    P(f"  {'':>10} {'全降下':>8} {'maxfevals':>10} {'全体比':>9} {'未到達に占める比':>16} {'1 降下の平均評価':>16}")
    for lab, R in (("12500", C12), ("25000", C25), ("50000", B50)):
        tot = mf = mf_miss = miss_tot = 0
        ev = []
        for p in done:
            st = R[p]["stop"]
            tot += len(st)
            mi = set(cls(R[p], EPS_TIGHT)["miss_idx"])
            miss_tot += len(mi)
            for i, s in enumerate(st):
                if "maxfevals" in s:
                    mf += 1
                    if i in mi:
                        mf_miss += 1
            ev.extend(R[p]["evals"].tolist())
        P(f"  {lab:>10} {tot:>8} {mf:>10} {mf / tot * 100:>8.1f}% "
          f"{(mf_miss / miss_tot * 100 if miss_tot else float('nan')):>15.1f}% {np.mean(ev):>16.1f}")
    P(f"  転記（その158 / その163）: 12500 は 全体比 {E158_MAXFEV['all'] * 100:.1f}% / 未到達比 "
      f"{E158_MAXFEV['miss'] * 100:.1f}%、25000 は {E163_MAXFEV['all'] * 100:.1f}% / "
      f"{E163_MAXFEV['miss'] * 100:.1f}%")
    P("")
    P("  停止理由の全内訳（16 問合算、上位 6 件）:")
    for lab, R in (("12500", C12), ("25000", C25), ("50000", B50)):
        c = Counter(s for p in done for s in R[p]["stop"])
        tot = sum(c.values())
        P(f"    {lab}: " + ", ".join(f"{s} {n} ({n / tot * 100:.1f}%)" for s, n in c.most_common(6)))

    # ------------------------------------------------- 機序 3: 当たり率 x 本数
    P("")
    P("## 機序 (3) —— 当たり率の上昇は本数の減少に食われたか（相異なり数 = 初出率 x 降下本数）")
    P(f"  {'':>10} {'降下本数':>10} {'初出率(1e-5)':>14} {'相異なり数':>12} {'ヒット率':>10}")
    for lab, R in (("12500", C12), ("25000", C25), ("50000", B50)):
        nd = float(np.mean([len(R[p]['f']) for p in done]))
        cc = {p: cls(R[p], EPS_TIGHT) for p in done}
        newr = float(np.mean([cc[p]["new"] for p in done]))
        dist = float(np.mean([cc[p]["seen"] for p in done]))
        hit = float(np.mean([(cc[p]["new"] + cc[p]["dup"]) for p in done]))
        P(f"  {lab:>10} {nd:>10.1f} {newr:>14.4f} {dist:>12.2f} {hit:>10.4f}")

    # ------------------------------------------------- 機序 4: 名指しの問題
    P("")
    P("## 機序 (4) —— 名指しの問題（M03 / M11 = その162、M10 = その163 で唯一下がった、"
      "M07 / M08 / M15 = 12500 で maxfevals ゼロの対照群）")
    for p in ("M03", "M11", "M10", "M07", "M08", "M15"):
        if p not in s50:
            P(f"  {p}: この回の run が無い")
            continue
        P(f"  {p} (K={K[p]}): 相異なり(1e-5) {s12[p]['distinct5']}/{s25[p]['distinct5']}/"
          f"{s50[p]['distinct5']} / ヒット本数 {s12[p]['hits5']}/{s25[p]['hits5']}/{s50[p]['hits5']} / "
          f"Score {s12[p]['score']:.4f}/{s25[p]['score']:.4f}/{s50[p]['score']:.4f} / "
          f"降下 {s12[p]['ndump']}/{s25[p]['ndump']}/{s50[p]['ndump']}")

    print("\n".join(out))


if __name__ == "__main__":
    main()
