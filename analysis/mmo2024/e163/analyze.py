#!/usr/bin/env python3
"""その163 — キュー 1: D=20 で `descent_budget` を 12500 -> 25000 に上げた 1 水準。

**採点器・報告規則・統計量は既存からそのまま import する。新しい統計量は 1 つも定義しない。**

  * 採点（`rule_indices` / `score` / `read_dump` / `SPAN`） -> `e115/analyze.py`
  * 3 分類（`classify`）と対検定（`paired`）                -> `e158/analyze.py`

入力の出自:

  * **25000（この回、新規 16 run）** -> `e163/descents/M??-D20-PIN01_seed0.csv[.gz]`
                                        （畳んだ後は `e163/descents.csv.gz`）
  * **12500（対照、追加評価ゼロ）**  -> `e151/descents.csv.gz` の保存物
  * **その158 の D=20 の 3 分類・`maxfevals` 比重** -> `prereg.md` の転記表（測らない）

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e163/analyze.py
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
SPAN, read_dump, rule_indices, score = E115.SPAN, E115.read_dump, E115.rule_indices, E115.score
classify, paired = E158.classify, E158.paired
EPS_TIGHT, EPS_LOOSE = E115.EPS_TIGHT, E115.EPS_LOOSE

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN
PROBS = [f"M{i:02d}" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]

# --- 転記（prereg.md §2）。この回では 1 つも測らない ---
E151_SCORE, E151_MPR, E151_F1 = 0.4504, 0.3831, 0.5176      # その151（関門の照合先）
SEED_SD = 0.0066                                             # その123
E158_D20 = dict(new=0.0378, dup=0.4700, miss=0.4922)         # その158（ε=1e-5）
E158_MAXFEV = dict(all=0.243, miss=0.423)                    # その158（D=20）
PUB_D20_SCORE = 0.4445


def _need(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"入力が無い: {path}\n"
            "  -> その150 §3 の経路（欠けた入力を黙って飛ばして nan の表を刷る）は踏まない。\n"
            "  -> `e163` / `e164` の `descents.csv.gz` は **その166（2026-09-25）が意図的に削除した**。\n"
            "     降下長の軸は その163〜その165 の 4 水準で閉じており、数値は\n"
            "     docs/acceptance_topology.md の その165 の節「この軸の数値を消す前に移した表」に全部ある。\n"
            "     経緯は analysis/mmo2024/e163/DUMPS_REMOVED.md ／ e164/DUMPS_REMOVED.md。\n"
            "     ＝ これはバグではない。この script はもう再走できない。")
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


def load_12500():
    """対照 = その151 の保存物（畳んだ 1 本。`problem` 列で割る）。追加評価ゼロ。"""
    by: dict = {}
    for r in _rows(os.path.join(MMO, "e151", "descents.csv.gz")):
        by.setdefault(r["problem"].split("-")[0], []).append(r)
    return {p: _pack(rs) for p, rs in by.items()}


def load_25000():
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
    if not out:
        # その161 が見つけた走査の穴 (2)（肯定形の `os.path.exists` ガードに `else` が無い）
        # をこの 2 本のガードが踏んでいた ＝ 入力ゼロでも「完走 0/16」の表を刷って exit 0 に
        # なる。その166 が塞いだ。
        raise SystemExit(
            f"入力が無い: {folded} も {dd} も無い\n"
            "  -> 入力ゼロで「完走 0 / 16」の表を刷る経路（その150 §3 / その161 の走査の穴 (2)）は踏まない。\n"
            "  -> この回の降下ダンプは **その166（2026-09-25）が意図的に削除した**。\n"
            "     降下長の軸は その163〜その165 の 4 水準で閉じており、数値は\n"
            "     docs/acceptance_topology.md の その165 の節「この軸の数値を消す前に移した表」に全部ある。\n"
            "     経緯は analysis/mmo2024/e163/DUMPS_REMOVED.md ／ e164/DUMPS_REMOVED.md。\n"
            "     ＝ これはバグではない。この script はもう再走できない。")
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
    A, B = load_12500(), load_25000()          # A = 12500（対照）, B = 25000（この回）
    K = {p: K_of(p) for p in PROBS}

    # ------------------------------------------------- 関門（対照を採点し直す）
    P("## 関門 —— 同じ採点器で その151（12500）の保存物を採点し直し、記録値と一致するか")
    miss_ctrl = [p for p in PROBS if p not in A]
    if miss_ctrl:
        P(f"  **対照が欠けている: {' '.join(miss_ctrl)}。関門を通せない。**")
        print("\n".join(out))
        return
    sa = {p: scored(A[p], K[p]) for p in PROBS}
    m_sc = float(np.mean([sa[p]["score"] for p in PROBS]))
    m_mpr = float(np.mean([sa[p]["mpr"] for p in PROBS]))
    ok = abs(m_sc - E151_SCORE) < 5e-5 and abs(m_mpr - E151_MPR) < 5e-5
    P(f"  再計算: Score {m_sc:.4f} / MPR {m_mpr:.4f}   記録: {E151_SCORE:.4f} / {E151_MPR:.4f}"
      f"  -> {'通過' if ok else '不一致'}")
    if not ok:
        P("")
        P("**関門が外れた。事前登録どおり判定は出さない。**")
        print("\n".join(out))
        return

    done = [p for p in PROBS if p in B]
    missing = [p for p in PROBS if p not in B]
    k = len(done)
    na = sum(1 for p in done if p in GROUP_A)
    P("")
    P("## 25000（この回の新規 run）")
    P(f"  完走: k = {k} / 16  （群 A {na} / 群 B {k - na}）")
    P(f"  **落ちた問題: {' '.join(missing) if missing else '（なし）'}**")
    if k == 0:
        P("  **run がゼロ。判定は出せない。**")
        print("\n".join(out))
        return
    sb = {p: scored(B[p], K[p]) for p in done}

    # ------------------------------------------------- 問題別（キュー 1 の指定）
    P("")
    P("  | 問題 | K | Score 12500 | Score 25000 | Δ | MPR 12500 | MPR 25000 | F1 12500 | F1 25000 | "
      "降下 12500 | 降下 25000 | 相異なり(1e-5) 12500 | 25000 |")
    P("  |---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in done:
        a, b = sa[p], sb[p]
        P(f"  | {p} | {K[p]} | {a['score']:.4f} | {b['score']:.4f} | {b['score'] - a['score']:+.4f} | "
          f"{a['mpr']:.4f} | {b['mpr']:.4f} | {a['f1']:.4f} | {b['f1']:.4f} | "
          f"{a['ndump']} | {b['ndump']} | {a['distinct5']} | {b['distinct5']} |")

    def mA(key):
        return float(np.mean([sa[p][key] for p in done]))

    def mB(key):
        return float(np.mean([sb[p][key] for p in done]))

    P("")
    P(f"  {k} 問平均 12500: Score {mA('score'):.4f} / MPR {mA('mpr'):.4f} / mean-F1 {mA('f1'):.4f} / "
      f"報告点数 {mA('n'):.2f} / 1e-1 被覆 {mA('cov'):.4f} / 深さ保持率 {mA('keep'):.4f} / "
      f"降下本数 {mA('ndump'):.1f}")
    P(f"  {k} 問平均 25000: Score {mB('score'):.4f} / MPR {mB('mpr'):.4f} / mean-F1 {mB('f1'):.4f} / "
      f"報告点数 {mB('n'):.2f} / 1e-1 被覆 {mB('cov'):.4f} / 深さ保持率 {mB('keep'):.4f} / "
      f"降下本数 {mB('ndump'):.1f}")

    # ------------------------------------------------- 主判定
    P("")
    P("## 判定（prereg.md §4）")
    P("  対検定はすべて 25000 − 12500 の向き（正なら 25000 が上）。両側 Wilcoxon exact、α=0.05")
    stats_row = {}
    for key, lab in (("score", "Score"), ("mpr", "MPR"), ("f1", "mean-F1"),
                     ("cov", "1e-1 被覆"), ("keep", "深さ保持率"),
                     ("n", "報告点数"), ("ndump", "降下本数"), ("distinct5", "相異なり(1e-5)")):
        d = paired([sa[p][key] for p in done], [sb[p][key] for p in done])
        stats_row[key] = d
        P(f"    {lab:>14}: {fmt(d)}")
    dsc, psc = stats_row["score"][0], stats_row["score"][1]
    P("")
    P(f"  **主判定**: Δ Score = {dsc:+.4f}（対照 {mA('score'):.4f} -> {mB('score'):.4f}）、"
      f"p={psc:.5g}、rb={stats_row['score'][2]:+.3f}")
    P(f"    その123 の seed SD = {SEED_SD:.4f}。|Δ| は SD の {abs(dsc) / SEED_SD:.2f} 倍")
    fire_a = abs(dsc) < SEED_SD and psc >= 0.05
    fire_b = dsc > SEED_SD and psc < 0.05
    fire_c = dsc < -SEED_SD and psc < 0.05
    P(f"  **反証条件 (a)**（|Δ| < {SEED_SD} かつ非有意 ＝ 長さは binding ではない）: "
      f"**{'発火' if fire_a else '不発'}**")
    P(f"  **反証条件 (b)**（Δ > +{SEED_SD} かつ有意 ＝ その152 に範囲の但し書きが要る。俯瞰に上げる）: "
      f"**{'発火' if fire_b else '不発'}**")
    P(f"  **反証条件 (c)**（Δ < −{SEED_SD} かつ有意 ＝ 12500 は D=20 でも最適の側）: "
      f"**{'発火' if fire_c else '不発'}**")
    if not (fire_a or fire_b or fire_c):
        P("  **どれも発火しない** ＝ 符号は出たが seed SD か有意性のどちらかを満たさない"
          "（「動いたとは書けない」。範囲として報告する）")
    P(f"  （参考）25000 の {k} 問平均 Score {mB('score'):.4f} 対 公表最良 D=20 {PUB_D20_SCORE:.4f} "
      f"＝ {mB('score') - PUB_D20_SCORE:+.4f}")

    # ------------------------------------------------- 機序 1: 3 分類
    P("")
    P("## 機序 (1) —— その158 の 3 分類（初出 i / 重複 ii / 未到達 iii）")
    for eps, nm in ((EPS_TIGHT, "1e-5（主水準）"), (EPS_LOOSE, "1e-1（感度）")):
        ca = {p: cls(A[p], eps) for p in done}
        cb = {p: cls(B[p], eps) for p in done}
        P(f"  ε={nm}")
        P(f"    {'':>10} {'初出(i)':>10} {'重複(ii)':>10} {'未到達(iii)':>12} {'重複/ヒット':>12} {'降下':>8}")
        for lab, c in (("12500", ca), ("25000", cb)):
            P(f"    {lab:>10} {np.mean([c[p]['new'] for p in done]):>10.4f} "
              f"{np.mean([c[p]['dup'] for p in done]):>10.4f} "
              f"{np.mean([c[p]['miss'] for p in done]):>12.4f} "
              f"{np.nanmean([c[p]['dupshare'] for p in done]):>12.4f} "
              f"{np.mean([c[p]['n'] for p in done]):>8.1f}")
        for key, lab in (("new", "初出(i)"), ("dup", "重複(ii)"), ("miss", "未到達(iii)"),
                         ("dupshare", "重複/ヒット")):
            va = [ca[p][key] for p in done]
            vb = [cb[p][key] for p in done]
            good = [i for i in range(len(done)) if np.isfinite(va[i]) and np.isfinite(vb[i])]
            if len(good) < 2:
                P(f"      {lab:>12}: 有効対 {len(good)} 本。検定しない")
                continue
            P(f"      {lab:>12}: {fmt(paired([va[i] for i in good], [vb[i] for i in good]))}"
              f"  （有効対 {len(good)}）")
        if eps == EPS_TIGHT:
            P(f"      その158 の D=20 既定値との照合（12500 側）: 初出 "
              f"{np.mean([ca[p]['new'] for p in done]):.4f} 対 {E158_D20['new']:.4f} / 重複 "
              f"{np.mean([ca[p]['dup'] for p in done]):.4f} 対 {E158_D20['dup']:.4f} / 未到達 "
              f"{np.mean([ca[p]['miss'] for p in done]):.4f} 対 {E158_D20['miss']:.4f}")

    # ------------------------------------------------- 機序 2: stop の内訳
    P("")
    P("## 機序 (2) —— `stop` の内訳（`maxfevals` ＝ 降下上限を使い切った降下）")
    P(f"  {'':>10} {'全降下':>8} {'maxfevals':>10} {'全体比':>9} {'未到達に占める比':>16} {'1 降下の平均評価':>16}")
    for lab, R in (("12500", A), ("25000", B)):
        tot = mf = 0
        mf_miss = miss_tot = 0
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
    P(f"  その158 の D=20 既定値（転記）: 全体比 {E158_MAXFEV['all'] * 100:.1f}% / "
      f"未到達に占める比 {E158_MAXFEV['miss'] * 100:.1f}%")
    P("")
    P("  停止理由の全内訳（16 問合算、上位 6 件）:")
    for lab, R in (("12500", A), ("25000", B)):
        c = Counter(s for p in done for s in R[p]["stop"])
        tot = sum(c.values())
        P(f"    {lab}: " + ", ".join(f"{s} {n} ({n / tot * 100:.1f}%)"
                                     for s, n in c.most_common(6)))

    # ------------------------------------------------- 機序 3: 当たり率 x 本数
    P("")
    P("## 機序 (3) —— 当たり率の上昇は本数の減少に食われたか（相異なり数 = 初出率 x 降下本数）")
    P(f"  {'':>10} {'降下本数':>10} {'初出率(1e-5)':>14} {'相異なり数':>12} {'ヒット率':>10}")
    for lab, R in (("12500", A), ("25000", B)):
        nd = float(np.mean([len(R[p]['f']) for p in done]))
        cc = {p: cls(R[p], EPS_TIGHT) for p in done}
        newr = float(np.mean([cc[p]["new"] for p in done]))
        dist = float(np.mean([cc[p]["seen"] for p in done]))
        hit = float(np.mean([(cc[p]["new"] + cc[p]["dup"]) for p in done]))
        P(f"  {lab:>10} {nd:>10.1f} {newr:>14.4f} {dist:>12.2f} {hit:>10.4f}")

    # ------------------------------------------------- 機序 4: M03 / M11
    P("")
    P("## 機序 (4) —— その162 が名指しした M03 / M11（既定 12500 の実機は 1e-5 到達 0 個）")
    for p in ("M03", "M11"):
        if p not in sb:
            P(f"  {p}: この回の run が無い")
            continue
        P(f"  {p} (K={K[p]}): 相異なり(1e-5) {sa[p]['distinct5']} -> {sb[p]['distinct5']} / "
          f"ヒット本数 {sa[p]['hits5']} -> {sb[p]['hits5']} / "
          f"Score {sa[p]['score']:.4f} -> {sb[p]['score']:.4f} / "
          f"降下 {sa[p]['ndump']} -> {sb[p]['ndump']}")

    print("\n".join(out))


if __name__ == "__main__":
    main()
