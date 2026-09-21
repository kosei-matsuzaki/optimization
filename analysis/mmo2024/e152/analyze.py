#!/usr/bin/env python3
"""その152 — キュー 1: RR-CMA-ES を D=20 で 16 問（直接対決の 2 次元目）。

**採点・規則・統計量は `e115/analyze.py` からそのまま import する**
（`rule_indices` / `score` / `paired` / `read_dump` / `SPAN`）。**新しい統計量は 1 つも定義しない。**

入力の出自:

  * **D=20 RR-CMA-ES**     -> `e152/dumps/M??-D20-PIN01_rrcma_seed0.csv`（この回、新規 16 run）
  * **D=20 Restart-Lander** -> `e151/descents.csv.gz`（その151 の保存物、**追加評価ゼロ**）
  * **D=10 の 2 手法**      -> `prereg.md` §2 が指す その131 §7 の表を下に転記（**測らない**）
  * **公表最良 D=20**       -> 転記（測らない）

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e152/analyze.py
"""
from __future__ import annotations

import csv
import gzip
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MMO, "e115"))

from analyze import SPAN, paired, read_dump, rule_indices, score   # noqa: E402
from core.benchmarks import niching_by_name                        # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN

PROBS = [f"M{i:02d}" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]

# --- 転記 1: その151 の 16 問平均（関門の照合先） ---
GATE_RL_D20 = {"score": 0.4504, "mpr": 0.3831}

# --- 転記 2: その131 §7 の表（D=10 / PIN01 / seed 0）。この回では 1 つも測らない ---
D10_SCORE_RR = dict(zip(PROBS, [0.7189, 0.3808, 0.5833, 0.4857, 0.5833, 0.5353, 0.3808, 0.7189,
                                0.6750, 0.4857, 0.6750, 0.4857, 0.7618, 0.3808, 0.2667, 0.4857]))
D10_SCORE_RL = dict(zip(PROBS, [0.7600, 0.6069, 0.5717, 0.7189, 0.7189, 0.4857, 0.5234, 0.7189,
                                0.5625, 0.6529, 0.5175, 0.8416, 0.8444, 0.3808, 0.3886, 0.7618]))
D10_MPR_RR = dict(zip(PROBS, [0.65, 0.30, 0.50, 0.40, 0.50, 0.45, 0.30, 0.65,
                              0.60, 0.40, 0.60, 0.40, 0.70, 0.30, 0.20, 0.40]))
D10_MPR_RL = dict(zip(PROBS, [0.72, 0.53, 0.49, 0.65, 0.65, 0.40, 0.44, 0.65,
                              0.50, 0.60, 0.46, 0.82, 0.80, 0.30, 0.32, 0.70]))
GATE_D10 = {"rr_score": 0.5377, "rl_score": 0.6284, "rr_mpr": 0.4594, "rl_mpr": 0.5644}

PUB_D20_SCORE, PUB_D20_MPR, PUB_D20_F1 = 0.4445, 0.476, 0.413
R_INSTANCE = 0.4899   # その149 の instance 間相関（反証条件 (b) の閾値）

_K: dict = {}


def K_of(prob):
    if prob not in _K:
        _K[prob] = int(niching_by_name(prob).n_global_optima)
    return _K[prob]


def find(*cands):
    for c in cands:
        for ext in ("", ".gz"):
            if os.path.exists(c + ext):
                return c + ext
    return None


def score_arrays(f, opt, xs, K):
    idx = rule_indices(ARM, f, K, x=xs, r=ARM_R)
    recall, prec, f1, sc, n = score(idx, f, opt, K)
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()), score=float(sc.mean()),
                n=int(n), ndump=len(f))


def score_dump(path, K):
    return score_arrays(*read_dump(path), K)


# --- その151 の畳んだダンプ（`problem` 列つき 1 本）を problem -> (f, opt, xs) に割る ---
FOLDED_RL_D20 = os.path.join(MMO, "e151", "descents.csv.gz")
_FOLD: dict = {}


def folded(prob):
    if not _FOLD:
        if not os.path.exists(FOLDED_RL_D20):
            _FOLD["_"] = None
            return None
        with gzip.open(FOLDED_RL_D20, "rt") as fh:
            rows = list(csv.DictReader(fh))
        by: dict = {}
        for r in rows:
            by.setdefault(r["problem"], []).append(r)
        for pr, rs in by.items():
            dim = sum(1 for kk in rs[0] if kk.startswith("x") and kk[1:].isdigit())
            _FOLD[pr] = (np.array([float(r["best_f"]) for r in rs]),
                         np.array([int(r["land_opt"]) for r in rs]),
                         np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rs]))
    return _FOLD.get(prob)


def fmt(d):
    return (f"mean {d['mean']:+.4f}  {d['w']}/{d['t']}/{d['l']}  "
            f"p={d['p']:.5g}  rb={d['rb']:+.3f}")


def main():
    out = []
    P = out.append

    # ---------------------------------------------- 関門 1: 転記表（その131 §7）
    P("## 関門 1 —— 転記した その131 §7 の表が記録値を再現するか（追加評価ゼロ）")
    g1 = {"rr_score": float(np.mean(list(D10_SCORE_RR.values()))),
          "rl_score": float(np.mean(list(D10_SCORE_RL.values()))),
          "rr_mpr": float(np.mean(list(D10_MPR_RR.values()))),
          "rl_mpr": float(np.mean(list(D10_MPR_RL.values())))}
    ok1 = all(abs(g1[k] - GATE_D10[k]) < 5e-5 for k in GATE_D10)
    for k in ("rr_score", "rl_score", "rr_mpr", "rl_mpr"):
        P(f"  {k}: 転記から {g1[k]:.4f}   記録 {GATE_D10[k]:.4f}   差 {g1[k] - GATE_D10[k]:+.5f}")
    P(f"  -> {'通過' if ok1 else '**不一致**'}")

    # ---------------------------------------------- 関門 2: その151 の保存物を採点し直す
    P("")
    P("## 関門 2 —— その151 の D=20 保存物を<u>この回の採点コード</u>で読み直す（追加評価ゼロ）")
    rl20 = {}
    for p in PROBS:
        fa = folded(f"{p}-D20-PIN01")
        if fa is None:
            P(f"  **{p} の D=20 保存物が無い。関門を通せない。**")
            print("\n".join(out))
            return
        rl20[p] = score_arrays(*fa, K_of(f"{p}-D20-PIN01"))
    m_sc = float(np.mean([rl20[p]["score"] for p in PROBS]))
    m_mpr = float(np.mean([rl20[p]["mpr"] for p in PROBS]))
    ok2 = (abs(m_sc - GATE_RL_D20["score"]) < 5e-5 and abs(m_mpr - GATE_RL_D20["mpr"]) < 5e-5)
    P(f"  再計算: Score {m_sc:.4f} / MPR {m_mpr:.4f}   その151 の記録: "
      f"{GATE_RL_D20['score']:.4f} / {GATE_RL_D20['mpr']:.4f}  -> {'通過' if ok2 else '**不一致**'}")
    if not (ok1 and ok2):
        P("")
        P("**関門が外れた。事前登録（§4）どおり判定は出さない。**")
        print("\n".join(out))
        return

    # ---------------------------------------------- D=20 RR の採点
    P("")
    P("## D=20 RR-CMA-ES（この回の新規 run）")
    rr20, done, missing = {}, [], []
    for p in PROBS:
        prob = f"{p}-D20-PIN01"
        path = find(os.path.join(HERE, "dumps", f"{prob}_rrcma_seed0.csv"))
        if path is None:
            missing.append(p)
            continue
        rr20[p] = score_dump(path, K_of(prob))
        done.append(p)
    k = len(done)
    na = sum(1 for p in done if p in GROUP_A)
    P(f"  完走: k = {k} / 16  （群 A {na} / 群 B {k - na}）")
    P(f"  完走した問題: {' '.join(done) if done else '（なし）'}")
    P(f"  **落ちた問題: {' '.join(missing) if missing else '（なし）'}**")
    if k == 0:
        P("  **run がゼロ。判定は出せない。**")
        print("\n".join(out))
        return

    P("")
    P("  | 問題 | K | RR MPR | RR F1 | RR Score | RR 報告点数 | RL Score (その151) | 差 RL−RR | "
      "(参考) D10 差 |")
    P("  |---|---|---|---|---|---|---|---|---|")
    for p in done:
        a, b = rr20[p], rl20[p]
        d20 = b["score"] - a["score"]
        d10 = D10_SCORE_RL[p] - D10_SCORE_RR[p]
        P(f"  | {p} | {K_of(f'{p}-D20-PIN01')} | {a['mpr']:.4f} | {a['f1']:.4f} | {a['score']:.4f} | "
          f"{a['n']} | {b['score']:.4f} | {d20:+.4f} | {d10:+.4f} |")

    def mrr(key):
        return float(np.mean([rr20[p][key] for p in done]))

    def mrl(key):
        return float(np.mean([rl20[p][key] for p in done]))

    P("")
    P(f"  {k} 問平均 RR-CMA-ES（D=20）: Score {mrr('score'):.4f} / MPR {mrr('mpr'):.4f} / "
      f"mean-F1 {mrr('f1'):.4f} / 報告点数 {mrr('n'):.2f}")
    P(f"  {k} 問平均 Restart-Lander（D=20、同じ問題に制限）: Score {mrl('score'):.4f} / "
      f"MPR {mrl('mpr'):.4f} / mean-F1 {mrl('f1'):.4f} / 報告点数 {mrl('n'):.2f}")
    P(f"  公表最良 D=20 は Score {PUB_D20_SCORE:.4f}（MPR {PUB_D20_MPR:.3f} / mean-F1 {PUB_D20_F1:.3f}）"
      f" —— RR は {mrr('score') - PUB_D20_SCORE:+.4f}、RL は {mrl('score') - PUB_D20_SCORE:+.4f}")

    # ---------------------------------------------- (i) D=20 の対検定
    P("")
    P("## 判定 (i) —— D=20 の 16 問対検定（両側 Wilcoxon exact、RL − RR）")
    res20 = {}
    for key, lab in (("score", "Score（主）"), ("mpr", "MPR"), ("f1", "mean-F1")):
        a = {p: rl20[p][key] for p in done}
        b = {p: rr20[p][key] for p in done}
        res20[key] = paired(a, b, done)
        P(f"  {lab:<12}: {fmt(res20[key])}")
    sign20 = res20["score"]["mean"]
    P(f"  **反証条件 (a)**: D=20 の Score 差の符号 = {'正（RL > RR、D=10 と同じ）' if sign20 > 0 else '**負（RR > RL ＝ 反転）**'}"
      f" ＝ **{'不発' if sign20 > 0 else '発火（頭の主張は D=10 に限定される。俯瞰に上げる）'}**")

    # ---------------------------------------------- (iii) D 間の相関（(ii) の前に出す）
    P("")
    P("## 判定 (iii) —— D 間の差の相関（反証条件 (b) の判定。(ii) の採否を決めるので先に出す）")
    d10v = np.array([D10_SCORE_RL[p] - D10_SCORE_RR[p] for p in done])
    d20v = np.array([rl20[p]["score"] - rr20[p]["score"] for p in done])
    r, p_r = stats.pearsonr(d10v, d20v)
    rho, p_rho = stats.spearmanr(d10v, d20v)
    agree = int(np.sum(np.sign(d10v) == np.sign(d20v)))
    P(f"  Pearson r = {r:+.4f}（p={p_r:.4g}、n={k}） / Spearman rho = {rho:+.4f}（p={p_rho:.4g}）")
    P(f"  符号一致は {agree}/{k}。D=10 の差 mean {d10v.mean():+.4f} / D=20 の差 mean {d20v.mean():+.4f}")
    P(f"  その149 の instance 間 r = {R_INSTANCE:+.4f} が閾値")
    fired_b = r >= R_INSTANCE
    P(f"  **反証条件 (b)**: r {r:+.4f} {'≥' if fired_b else '<'} {R_INSTANCE:.4f} ＝ "
      f"**{'発火（次元も疑似反復。n=32 は主張に使わない）' if fired_b else '不発（次元は疑似反復ではない ＝ n=32 が使える）'}**")

    # ---------------------------------------------- (ii) n=32
    P("")
    P("## 判定 (ii) —— D=10 の 16 対と合わせた n=32 の対検定（Score）")
    keys32 = [f"{p}-D10" for p in done] + [f"{p}-D20" for p in done]
    a32 = {f"{p}-D10": D10_SCORE_RL[p] for p in done} | {f"{p}-D20": rl20[p]["score"] for p in done}
    b32 = {f"{p}-D10": D10_SCORE_RR[p] for p in done} | {f"{p}-D20": rr20[p]["score"] for p in done}
    res32 = paired(a32, b32, keys32)
    P(f"  n=32: {fmt(res32)}")
    d10only = paired({p: D10_SCORE_RL[p] for p in done}, {p: D10_SCORE_RR[p] for p in done}, done)
    P(f"  （内訳）D=10 の 16 対: {fmt(d10only)}")
    P(f"  （内訳）D=20 の 16 対: {fmt(res20['score'])}")
    P(f"  **採否**: n=32 の p={res32['p']:.5g} は "
      f"**{'主張に使わない（(b) が発火）' if fired_b else '主張に使える（(b) は不発）'}**")

    # ---------------------------------------------- 群別（参考）
    P("")
    P("## 群別（参考。K=20 の群 A と K=10 の群 B）")
    for lab, grp in (("群 A (K=20)", [p for p in done if p in GROUP_A]),
                     ("群 B (K=10)", [p for p in done if p in GROUP_B])):
        if not grp:
            continue
        srr = float(np.mean([rr20[p]["score"] for p in grp]))
        srl = float(np.mean([rl20[p]["score"] for p in grp]))
        P(f"  {lab} n={len(grp)}: RR {srr:.4f} / RL {srl:.4f} / 差 {srl - srr:+.4f}")

    # ---------------------------------------------- 次元間の内訳（参考。対検定はしない）
    P("")
    P("## RR-CMA-ES は次元で何を失ったか（参考。予算も K も違うので対検定はしない）")
    P(f"  RR Score:  D=10 {float(np.mean([D10_SCORE_RR[p] for p in done])):.4f} -> D=20 {mrr('score'):.4f}"
      f"  （比 {mrr('score') / float(np.mean([D10_SCORE_RR[p] for p in done])):.3f}）")
    P(f"  RR MPR:    D=10 {float(np.mean([D10_MPR_RR[p] for p in done])):.4f} -> D=20 {mrr('mpr'):.4f}"
      f"  （比 {mrr('mpr') / float(np.mean([D10_MPR_RR[p] for p in done])):.3f}）")
    P(f"  RL Score:  D=10 {float(np.mean([D10_SCORE_RL[p] for p in done])):.4f} -> D=20 {mrl('score'):.4f}"
      f"  （比 {mrl('score') / float(np.mean([D10_SCORE_RL[p] for p in done])):.3f}）")
    P(f"  RL MPR:    D=10 {float(np.mean([D10_MPR_RL[p] for p in done])):.4f} -> D=20 {mrl('mpr'):.4f}"
      f"  （比 {mrl('mpr') / float(np.mean([D10_MPR_RL[p] for p in done])):.3f}）")
    P(f"  報告点数:  RR D=20 {mrr('n'):.2f} / RL D=20 {mrl('n'):.2f}")

    print("\n".join(out))
    with open(os.path.join(HERE, "scored.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")

    # 問題別の集計 CSV（数百行未満の集計なので生 CSV のまま。行単位は dumps 側）
    with open(os.path.join(HERE, "by_problem_d20.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "rr_d20_mpr", "rr_d20_f1", "rr_d20_score", "rr_d20_n",
                    "rl_d20_mpr", "rl_d20_f1", "rl_d20_score", "rl_d20_n",
                    "rr_d10_score", "rl_d10_score"])
        for p in done:
            a, b = rr20[p], rl20[p]
            w.writerow([p, K_of(f"{p}-D20-PIN01"),
                        f"{a['mpr']:.4f}", f"{a['f1']:.4f}", f"{a['score']:.4f}", a["n"],
                        f"{b['mpr']:.4f}", f"{b['f1']:.4f}", f"{b['score']:.4f}", b["n"],
                        f"{D10_SCORE_RR[p]:.4f}", f"{D10_SCORE_RL[p]:.4f}"])


if __name__ == "__main__":
    main()
