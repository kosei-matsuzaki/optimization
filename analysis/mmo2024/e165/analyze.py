#!/usr/bin/env python3
"""その165 — キュー 1: 降下長の最適水準は K に依存するか（`descent_budget` 50000 -> 100000、D=20）。

**採点器・報告規則・統計量は既存からそのまま import する。新しい統計量は 1 つも定義しない。**

  * 採点（`rule_indices` / `score` / `read_dump` / `SPAN`） -> `e115/analyze.py`
  * 3 分類（`classify`）と対検定（`paired`）                -> `e158/analyze.py`

入力の出自:

  * **100000（この回、新規 run）**  -> `e165/descents/M??-D20-PIN01_seed0.csv[.gz]`
                                       （畳んだ後は `e165/descents.csv.gz`）
  * **50000（主対照、追加評価ゼロ）** -> `e164/descents.csv.gz` の保存物
  * **25000（追加評価ゼロ）**         -> `e163/descents.csv.gz` の保存物
  * **12500（既定、追加評価ゼロ）**   -> `e151/descents.csv.gz` の保存物

**主判定は群 B（M09-M16、K=10）の Δ Score（100000 − 50000）**（prereg.md §4）。
群 A（M01-M08、K=20）は任意分で、run が無ければその旨を印字して群 B だけで判定する。

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e165/analyze.py
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
E164_SCORE, E164_MPR, E164_F1 = 0.4973, 0.4244, 0.5702      # その164（関門の主照合先）
E163_SCORE, E163_MPR = 0.4922, 0.4188                        # その163（副の関門）
E151_SCORE, E151_MPR = 0.4504, 0.3831                        # その151（副の関門）
E164_GA, E164_GB = 0.4705, 0.5241                            # その164 §4 の群別 Score
SEED_SD = 0.0066                                             # その123
E164_MAXFEV = dict(all=0.056, miss=0.087)                    # その164 §2（50000 側）
E163_MAXFEV = dict(all=0.127, miss=0.195)                    # その163 §4（25000 側）
E158_MAXFEV = dict(all=0.243, miss=0.423)                    # その158（12500 側）
PUB_D20_SCORE = 0.4445

LEVELS = ("12500", "25000", "50000", "100000")


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


def verdict(dsc, psc, label):
    """prereg.md §4 の (a)(b)(c)。群 B について判定し、群 A は文脈として添える。"""
    fa = abs(dsc) < SEED_SD and psc >= 0.05
    fb = dsc > SEED_SD and psc < 0.05
    fc = dsc < -SEED_SD and psc < 0.05
    return fa, fb, fc


def main():
    out = []
    P = out.append
    R = {"12500": load_folded("e151"), "25000": load_folded("e163"),
         "50000": load_folded("e164"), "100000": load_this()}
    K = {p: K_of(p) for p in PROBS}

    # ------------------------------------------------- 関門（対照を採点し直す）
    P("## 関門 —— 同じ採点器で保存物を採点し直し、記録値と一致するか")
    for lab, (rs, rm) in (("50000（その164、主対照）", (E164_SCORE, E164_MPR)),
                          ("25000（その163）", (E163_SCORE, E163_MPR)),
                          ("12500（その151）", (E151_SCORE, E151_MPR))):
        key = lab.split("（")[0]
        miss_ctrl = [p for p in PROBS if p not in R[key]]
        if miss_ctrl:
            P(f"  **{lab} が欠けている: {' '.join(miss_ctrl)}。関門を通せない。**")
            print("\n".join(out))
            return
        s = {p: scored(R[key][p], K[p]) for p in PROBS}
        m_sc = float(np.mean([s[p]["score"] for p in PROBS]))
        m_mpr = float(np.mean([s[p]["mpr"] for p in PROBS]))
        ok = abs(m_sc - rs) < 5e-5 and abs(m_mpr - rm) < 5e-5
        P(f"  {lab}: 再計算 Score {m_sc:.4f} / MPR {m_mpr:.4f}   記録 {rs:.4f} / {rm:.4f}"
          f"  -> {'通過' if ok else '**不一致**'}")
        if key == "50000":
            ga = float(np.mean([s[p]["score"] for p in GROUP_A]))
            gb = float(np.mean([s[p]["score"] for p in GROUP_B]))
            P(f"    群別の照合: A {ga:.4f} 対 記録 {E164_GA:.4f} / B {gb:.4f} 対 記録 {E164_GB:.4f}"
              f"  -> {'一致' if abs(ga - E164_GA) < 5e-5 and abs(gb - E164_GB) < 5e-5 else '**不一致**'}")
            gate_ok = ok
    if not gate_ok:
        P("")
        P("**主対照の関門が外れた。事前登録どおり判定は出さない。**")
        print("\n".join(out))
        return

    done = [p for p in PROBS if p in R["100000"]]
    missing = [p for p in PROBS if p not in R["100000"]]
    doneB = [p for p in done if p in GROUP_B]
    doneA = [p for p in done if p in GROUP_A]
    P("")
    P("## 100000（この回の新規 run）")
    P(f"  完走: {len(done)} / 16  （**群 B {len(doneB)} / 8（主判定）**、群 A {len(doneA)} / 8（任意分））")
    P(f"  **落ちた問題: {' '.join(missing) if missing else '（なし）'}**")
    if not doneB:
        P("  **群 B の run がゼロ。主判定は出せない。**")
        print("\n".join(out))
        return
    S = {lab: {p: scored(R[lab][p], K[p]) for p in done} for lab in LEVELS}

    # ------------------------------------------------- 問題別
    P("")
    P("  | 問題 | K | Score 12500 | 25000 | 50000 | **100000** | Δ(100k−50k) | MPR 50000 | 100000 | "
      "F1 50000 | 100000 | 降下 50000 | 100000 | 降下/K 100000 | 相異なり(1e-5) 12.5k/25k/50k/100k |")
    P("  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in done:
        a, b, c, d = (S[l][p] for l in LEVELS)
        P(f"  | {p} | {K[p]} | {a['score']:.4f} | {b['score']:.4f} | {c['score']:.4f} | "
          f"**{d['score']:.4f}** | {d['score'] - c['score']:+.4f} | {c['mpr']:.4f} | {d['mpr']:.4f} | "
          f"{c['f1']:.4f} | {d['f1']:.4f} | {c['ndump']} | {d['ndump']} | {d['ndump'] / K[p]:.2f} | "
          f"{a['distinct5']}/{b['distinct5']}/{c['distinct5']}/{d['distinct5']} |")

    def mm(lab, key, g=None):
        g = g if g is not None else done
        return float(np.mean([S[lab][p][key] for p in g]))

    P("")
    for lab in LEVELS:
        P(f"  {len(done)} 問平均 {lab:>6}: Score {mm(lab, 'score'):.4f} / MPR {mm(lab, 'mpr'):.4f} / "
          f"mean-F1 {mm(lab, 'f1'):.4f} / 報告点数 {mm(lab, 'n'):.2f} / 1e-1 被覆 {mm(lab, 'cov'):.4f} / "
          f"深さ保持率 {mm(lab, 'keep'):.4f} / 降下本数 {mm(lab, 'ndump'):.1f}")

    # ------------------------------------------------- 主判定（群 B）
    P("")
    P("## 主判定（prereg.md §4）—— **群 B（K=10）の Δ Score（100000 − 50000）**")
    dB = paired([S["50000"][p]["score"] for p in doneB], [S["100000"][p]["score"] for p in doneB])
    dsc, psc = dB[0], dB[1]
    P(f"  群 B（{len(doneB)} 問）: 50000 {mm('50000', 'score', doneB):.4f} -> "
      f"100000 {mm('100000', 'score', doneB):.4f}   {fmt(dB)}")
    P(f"    その123 の seed SD = {SEED_SD:.4f}。|Δ| は SD の {abs(dsc) / SEED_SD:.2f} 倍")
    fa, fb, fc = verdict(dsc, psc, "B")
    if doneA:
        dA = paired([S["50000"][p]["score"] for p in doneA], [S["100000"][p]["score"] for p in doneA])
        P(f"  群 A（{len(doneA)} 問、任意分）: 50000 {mm('50000', 'score', doneA):.4f} -> "
          f"100000 {mm('100000', 'score', doneA):.4f}   {fmt(dA)}")
        a_up = dA[0] > 0
    else:
        dA, a_up = None, None
        P("  群 A: run なし（この回は削った）")
    P("")
    P(f"  **反証条件 (a)**（群 B の |Δ| < {SEED_SD} かつ非有意 ＝ K=10 でも 50000 で飽和。この枝も閉じる）: "
      f"**{'発火' if fa else '不発'}**")
    P(f"  **反証条件 (b)**（群 B が有意に +{SEED_SD} 超 かつ 群 A が上がらない ＝ 最適降下長は K に依存。俯瞰に上げる）: "
      f"**{'発火' if (fb and (a_up is False)) else '不発'}**"
      + ("" if a_up is None else f"（群 B 側 {'○' if fb else '×'} / 群 A が上がらない {'○' if a_up is False else '×'}）"))
    P(f"  **反証条件 (c)**（群 B も下がる ＝ 群 B の頂は 50000）: **{'発火' if fc else '不発'}**")
    if not (fa or fb or fc):
        P("  **どれも発火しない** ＝ 符号は出たが seed SD か有意性のどちらかを満たさない"
          "（「動いたとは書けない」。範囲として報告する）")

    P("")
    P("  群別の内訳（Score / MPR / mean-F1 / 相異なり、100000 − 50000 の向き）:")
    for gname, G in (("A(K=20)", doneA), ("B(K=10)", doneB)):
        if len(G) < 2:
            P(f"    群 {gname}: 有効対 {len(G)} 本。検定しない")
            continue
        for key, lab in (("score", "Score"), ("mpr", "MPR"), ("f1", "mean-F1"),
                         ("ndump", "降下本数"), ("distinct5", "相異なり"), ("cov", "1e-1 被覆"),
                         ("keep", "深さ保持率")):
            P(f"    群 {gname} {lab:>10}: "
              f"{fmt(paired([S['50000'][p][key] for p in G], [S['100000'][p][key] for p in G]))}")
        P(f"    群 {gname} 平均 Score の 4 点: " + " -> ".join(f"{mm(l, 'score', G):.4f}" for l in LEVELS))
        P(f"    群 {gname} 平均 相異なり(1e-5) の 4 点: "
          + " -> ".join(f"{mm(l, 'distinct5', G):.2f}" for l in LEVELS) + f"  (K={K[G[0]]})")
        P(f"    群 {gname} 平均 降下本数の 4 点: " + " -> ".join(f"{mm(l, 'ndump', G):.1f}" for l in LEVELS))

    # ------------------------------------------------- 全 16 問（参考）
    if len(done) >= 2:
        P("")
        P("## 参考 —— 完走した全問の対検定（100000 − 50000）")
        for key, lab in (("score", "Score"), ("mpr", "MPR"), ("f1", "mean-F1"),
                         ("cov", "1e-1 被覆"), ("keep", "深さ保持率"),
                         ("n", "報告点数"), ("ndump", "降下本数"), ("distinct5", "相異なり(1e-5)")):
            P(f"    {lab:>14}: "
              f"{fmt(paired([S['50000'][p][key] for p in done], [S['100000'][p][key] for p in done]))}")
        P("")
        P("  （参考）100000 − 12500 の向き（4 点の端どうし）:")
        for key, lab in (("score", "Score"), ("mpr", "MPR"), ("f1", "mean-F1"), ("ndump", "降下本数")):
            P(f"    {lab:>14}: "
              f"{fmt(paired([S['12500'][p][key] for p in done], [S['100000'][p][key] for p in done]))}")
        P(f"  （参考）100000 の {len(done)} 問平均 Score {mm('100000', 'score'):.4f} 対 公表最良 D=20 "
          f"{PUB_D20_SCORE:.4f} ＝ {mm('100000', 'score') - PUB_D20_SCORE:+.4f}"
          "（1 instance / 1 seed なので参考）")

    # ------------------------------------------------- 降下/K とキューの代償
    P("")
    P("## 代償 —— 降下本数 / K の比（キューが名指しした量。1 を切ると被覆が構造的に頭打ち）")
    P("  **降下本数が K を下回る問題（100000 側）**: " + (", ".join(
        f"{p} ({S['100000'][p]['ndump']} 本 < K={K[p]})"
        for p in done if S["100000"][p]["ndump"] < K[p]) or "（なし）"))
    for p in done:
        if len({S[l][p]["ndump"] for l in LEVELS}) == 1:
            continue
        P(f"    {p} K={K[p]:2d}  降下/K  " + " / ".join(
            f"{l} {S[l][p]['ndump'] / K[p]:.2f}" for l in LEVELS)
          + f"   Δscore(100k-50k) {S['100000'][p]['score'] - S['50000'][p]['score']:+.4f}")
    moved = [p for p in done if abs(S["100000"][p]["score"] - S["50000"][p]["score"]) > 1e-12]
    P("")
    P(f"  動いた問題だけの対検定（n={len(moved)}: {' '.join(moved)}）:")
    if len(moved) >= 2:
        for key, lab in (("score", "Score"), ("mpr", "MPR"), ("f1", "mean-F1")):
            P(f"    {lab:>8}: "
              f"{fmt(paired([S['50000'][p][key] for p in moved], [S['100000'][p][key] for p in moved]))}")
        P("  動いた問題の K と符号: " + ", ".join(
            f"{p}(K={K[p]}, {S['100000'][p]['score'] - S['50000'][p]['score']:+.4f})" for p in moved))

    # ------------------------------------------------- 機序 1: 3 分類
    P("")
    P("## 機序 (1) —— その158 の 3 分類（初出 i / 重複 ii / 未到達 iii）")
    for eps, nm in ((EPS_TIGHT, "1e-5（主水準）"), (EPS_LOOSE, "1e-1（感度）")):
        cc = {lab: {p: cls(R[lab][p], eps) for p in done} for lab in LEVELS}
        P(f"  ε={nm}（完走した {len(done)} 問の平均）")
        P(f"    {'':>10} {'初出(i)':>10} {'重複(ii)':>10} {'未到達(iii)':>12} {'重複/ヒット':>12} {'降下':>8}")
        for lab in LEVELS:
            c = cc[lab]
            P(f"    {lab:>10} {np.mean([c[p]['new'] for p in done]):>10.4f} "
              f"{np.mean([c[p]['dup'] for p in done]):>10.4f} "
              f"{np.mean([c[p]['miss'] for p in done]):>12.4f} "
              f"{np.nanmean([c[p]['dupshare'] for p in done]):>12.4f} "
              f"{np.mean([c[p]['n'] for p in done]):>8.1f}")
        for key, lab in (("new", "初出(i)"), ("dup", "重複(ii)"), ("miss", "未到達(iii)"),
                         ("dupshare", "重複/ヒット")):
            va = [cc["50000"][p][key] for p in done]
            vb = [cc["100000"][p][key] for p in done]
            good = [i for i in range(len(done)) if np.isfinite(va[i]) and np.isfinite(vb[i])]
            if len(good) < 2:
                P(f"      {lab:>12}: 有効対 {len(good)} 本。検定しない")
                continue
            P(f"      {lab:>12} (100k−50k): "
              f"{fmt(paired([va[i] for i in good], [vb[i] for i in good]))}  （有効対 {len(good)}）")
        if eps == EPS_TIGHT and doneB:
            P("      群 B だけの 4 点（初出率 / 重複÷ヒット）: " + " / ".join(
                f"{lab} {np.mean([cc[lab][p]['new'] for p in doneB]):.4f}・"
                f"{np.nanmean([cc[lab][p]['dupshare'] for p in doneB]):.4f}" for lab in LEVELS))
            if doneA:
                P("      群 A だけの 4 点（初出率 / 重複÷ヒット）: " + " / ".join(
                    f"{lab} {np.mean([cc[lab][p]['new'] for p in doneA]):.4f}・"
                    f"{np.nanmean([cc[lab][p]['dupshare'] for p in doneA]):.4f}" for lab in LEVELS))

    # ------------------------------------------------- 機序 2: stop の内訳
    P("")
    P("## 機序 (2) —— `stop` の内訳（`maxfevals` ＝ 降下上限を使い切った降下）")
    P(f"  {'':>10} {'全降下':>8} {'maxfevals':>10} {'全体比':>9} {'未到達に占める比':>16} {'1 降下の平均評価':>16}")
    for lab in LEVELS:
        tot = mf = mf_miss = miss_tot = 0
        ev = []
        for p in done:
            st = R[lab][p]["stop"]
            tot += len(st)
            mi = set(cls(R[lab][p], EPS_TIGHT)["miss_idx"])
            miss_tot += len(mi)
            for i, s in enumerate(st):
                if "maxfevals" in s:
                    mf += 1
                    if i in mi:
                        mf_miss += 1
            ev.extend(R[lab][p]["evals"].tolist())
        P(f"  {lab:>10} {tot:>8} {mf:>10} {mf / tot * 100:>8.1f}% "
          f"{(mf_miss / miss_tot * 100 if miss_tot else float('nan')):>15.1f}% {np.mean(ev):>16.1f}")
    P(f"  転記（その158 / その163 / その164、いずれも 16 問）: 12500 は 全体比 "
      f"{E158_MAXFEV['all'] * 100:.1f}% / 未到達比 {E158_MAXFEV['miss'] * 100:.1f}%、25000 は "
      f"{E163_MAXFEV['all'] * 100:.1f}% / {E163_MAXFEV['miss'] * 100:.1f}%、50000 は "
      f"{E164_MAXFEV['all'] * 100:.1f}% / {E164_MAXFEV['miss'] * 100:.1f}%")
    P("")
    P("  停止理由の全内訳（完走した問だけ合算、上位 6 件）:")
    for lab in LEVELS:
        c = Counter(s for p in done for s in R[lab][p]["stop"])
        tot = sum(c.values())
        P(f"    {lab:>6}: " + ", ".join(f"{s} {n} ({n / tot * 100:.1f}%)" for s, n in c.most_common(6)))

    # ------------------------------------------------- 機序 3: 当たり率 x 本数
    P("")
    P("## 機序 (3) —— 当たり率の上昇は本数の減少に食われたか（相異なり数 = 初出率 x 降下本数）")
    for gname, G in (("完走した全問", done), ("群 A(K=20)", doneA), ("群 B(K=10)", doneB)):
        if not G:
            continue
        P(f"  【{gname}、{len(G)} 問】")
        P(f"    {'':>10} {'降下本数':>10} {'初出率(1e-5)':>14} {'相異なり数':>12} {'ヒット率':>10}")
        for lab in LEVELS:
            nd = float(np.mean([len(R[lab][p]['f']) for p in G]))
            cc = {p: cls(R[lab][p], EPS_TIGHT) for p in G}
            newr = float(np.mean([cc[p]["new"] for p in G]))
            dist = float(np.mean([cc[p]["seen"] for p in G]))
            hit = float(np.mean([(cc[p]["new"] + cc[p]["dup"]) for p in G]))
            P(f"    {lab:>10} {nd:>10.1f} {newr:>14.4f} {dist:>12.2f} {hit:>10.4f}")

    # ------------------------------------------------- 機序 4: 名指しの問題
    P("")
    P("## 機序 (4) —— 名指しの問題（M03 / M11 = その162、M10 = その163、"
      "M07 / M08 / M15 = 12500 で maxfevals ゼロの対照群）")
    for p in ("M03", "M11", "M10", "M09", "M12", "M07", "M08", "M15"):
        if p not in done:
            P(f"  {p}: この回の run が無い")
            continue
        P(f"  {p} (K={K[p]}): 相異なり(1e-5) " + "/".join(str(S[l][p]['distinct5']) for l in LEVELS)
          + " / ヒット本数 " + "/".join(str(S[l][p]['hits5']) for l in LEVELS)
          + " / Score " + "/".join(f"{S[l][p]['score']:.4f}" for l in LEVELS)
          + " / 降下 " + "/".join(str(S[l][p]['ndump']) for l in LEVELS))

    print("\n".join(out))


if __name__ == "__main__":
    main()
