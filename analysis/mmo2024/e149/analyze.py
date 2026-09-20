#!/usr/bin/env python3
"""その149 — キュー 1: instance 軸で直接対決の n を増やす（PIN01 + PIN02）。

**採点・規則・統計量は `e115/analyze.py` からそのまま import する**
（`rule_indices` / `score` / `paired` / `read_dump` / `SPAN`）。**新しい統計量は 1 つも定義しない。**

入力の出自:

  * **PIN02 RR-CMA-ES**    -> `e149/dumps/{prob}_rrcma_seed0.csv[.gz]`（この回、新規 16 run）
  * **PIN02 Restart-Lander** -> `e149/descents/{prob}_seed0.csv[.gz]`（この回、新規 16 run）
  * **PIN01 両手法（seed 0）** -> 下の `PIN01_*` 表 ＝ **`acceptance_topology.md` の その131 §7 が正本**
    （`e131/by_problem_d10.csv` は その132 の統合で削除済み。**追加評価ゼロ**）
  * **関門 1 の再計算** -> `e115/descents/`（保存物、`Restart-Lander` PIN01 seed 0）

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e149/analyze.py
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

from analyze import SPAN, paired, read_dump, rule_indices, score   # noqa: E402
from core.benchmarks import niching_by_name                        # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN

# --- PIN01 seed 0（acceptance_topology.md その131 §7 の表。転記であって再計算ではない）---
#     問題:      (RR MPR, RL MPR, RR Score, RL Score)
PIN01 = {
    "M01": (0.6500, 0.7200, 0.7189, 0.7600), "M02": (0.3000, 0.5300, 0.3808, 0.6069),
    "M03": (0.5000, 0.4900, 0.5833, 0.5717), "M04": (0.4000, 0.6500, 0.4857, 0.7189),
    "M05": (0.5000, 0.6500, 0.5833, 0.7189), "M06": (0.4500, 0.4000, 0.5353, 0.4857),
    "M07": (0.3000, 0.4400, 0.3808, 0.5234), "M08": (0.6500, 0.6500, 0.7189, 0.7189),
    "M09": (0.6000, 0.5000, 0.6750, 0.5625), "M10": (0.4000, 0.6000, 0.4857, 0.6529),
    "M11": (0.6000, 0.4600, 0.6750, 0.5175), "M12": (0.4000, 0.8200, 0.4857, 0.8416),
    "M13": (0.7000, 0.8000, 0.7618, 0.8444), "M14": (0.3000, 0.3000, 0.3808, 0.3808),
    "M15": (0.2000, 0.3200, 0.2667, 0.3886), "M16": (0.4000, 0.7000, 0.4857, 0.7618),
}
GATE_RL = {"mpr": 0.5644, "score": 0.6284}   # その116・その127・その131 の記録（PIN01 seed 0）
GATE_RR = {"mpr": 0.4594, "score": 0.5377}   # その127・その131 の記録（PIN01 seed 0）
PIN01_SCORE_DIFF_16 = 0.0907                 # その131 §2（seed 0、16 問）

PROBS = [f"M{i:02d}" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]

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


def read_combined(path):
    """`problem` 列つきの 1 本のダンプを problem -> (f, opt, xs) に割る（e128/e129 の写し）。"""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    by: dict = {}
    for r in rows:
        by.setdefault(r["problem"], []).append(r)
    res = {}
    for prob, rs in by.items():
        dim = sum(1 for kk in rs[0] if kk.startswith("x") and kk[1:].isdigit())
        res[prob] = (np.array([float(r["best_f"]) for r in rs]),
                     np.array([int(r["land_opt"]) for r in rs]),
                     np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rs]))
    return res


# 畳んだ後の退避路（その148 と同じ形。畳む前後で出力が 1 文字も変わらないことを確認する）
FOLDED = {"rr": os.path.join(HERE, "dumps_rrcma.csv.gz"),
          "rl": os.path.join(HERE, "descents.csv.gz")}
_FOLD: dict = {}


def folded(kind, prob):
    if kind not in _FOLD:
        path = FOLDED[kind]
        _FOLD[kind] = read_combined(path) if os.path.exists(path) else {}
    return _FOLD[kind].get(prob)


def score_arrays(f, opt, xs, K):
    idx = rule_indices(ARM, f, K, x=xs, r=ARM_R)
    recall, prec, f1, sc, n = score(idx, f, opt, K)
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()),
                score=float(sc.mean()), n=int(n), ndump=len(f))


def score_dump(path, K):
    return score_arrays(*read_dump(path), K)


def fmt(d):
    return (f"mean {d['mean']:+.4f}  {d['w']}/{d['t']}/{d['l']}  "
            f"p={d['p']:.5g}  rb={d['rb']:+.3f}")


def main():
    out = []
    P = out.append

    # ---------------------------------------------------------- 関門 1（再計算）
    P("## 関門 1 —— `Restart-Lander` PIN01 seed 0 を保存ダンプから採点し直す")
    rl01 = {}
    for p in PROBS:
        path = find(os.path.join(MMO, "e115", "descents", f"{p}-D10-PIN01_seed0.csv"))
        rl01[p] = score_dump(path, K_of(f"{p}-D10-PIN01"))
    m_mpr = float(np.mean([rl01[p]["mpr"] for p in PROBS]))
    m_sc = float(np.mean([rl01[p]["score"] for p in PROBS]))
    ok1 = abs(m_mpr - GATE_RL["mpr"]) < 5e-5 and abs(m_sc - GATE_RL["score"]) < 5e-5
    P(f"  再計算: MPR {m_mpr:.4f} / Score {m_sc:.4f}   記録: "
      f"{GATE_RL['mpr']:.4f} / {GATE_RL['score']:.4f}  -> {'通過' if ok1 else '不一致'}")
    # 転記表そのものも照合する（表が正本なので、再計算と表がずれたら転記ミス）
    t_mpr = float(np.mean([PIN01[p][1] for p in PROBS]))
    t_sc = float(np.mean([PIN01[p][3] for p in PROBS]))
    P(f"  その131 §7 の表（RL 列）: MPR {t_mpr:.4f} / Score {t_sc:.4f}")

    # ---------------------------------------------------------- 関門 2（算術）
    P("")
    P("## 関門 2 —— RR-CMA-ES PIN01 seed 0（**再計算できない**。表の算術照合のみ）")
    r_mpr = float(np.mean([PIN01[p][0] for p in PROBS]))
    r_sc = float(np.mean([PIN01[p][2] for p in PROBS]))
    ok2 = abs(r_mpr - GATE_RR["mpr"]) < 5e-5 and abs(r_sc - GATE_RR["score"]) < 5e-5
    P(f"  表の平均: MPR {r_mpr:.4f} / Score {r_sc:.4f}   記録: "
      f"{GATE_RR['mpr']:.4f} / {GATE_RR['score']:.4f}  -> {'通過' if ok2 else '不一致'}")

    if not (ok1 and ok2):
        P("")
        P("**関門が外れた。事前登録どおり判定は出さない。**")
        print("\n".join(out))
        return

    # ---------------------------------------------------------- PIN02 の採点
    P("")
    P("## PIN02（この回の新規 run）—— 両手法が揃った問題だけを対にする")
    rr02, rl02, done = {}, {}, []
    for p in PROBS:
        prob = f"{p}-D10-PIN02"
        a = find(os.path.join(HERE, "dumps", f"{prob}_rrcma_seed0.csv"))
        b = find(os.path.join(HERE, "descents", f"{prob}_seed0.csv"))
        K = K_of(prob)
        ra = score_dump(a, K) if a else (
            score_arrays(*folded("rr", prob), K) if folded("rr", prob) else None)
        rb = score_dump(b, K) if b else (
            score_arrays(*folded("rl", prob), K) if folded("rl", prob) else None)
        if ra and rb:
            rr02[p], rl02[p] = ra, rb
            done.append(p)
    k = len(done)
    na = sum(1 for p in done if p in GROUP_A)
    P(f"  完走した対: k = {k} / 16  （群 A {na} / 群 B {k - na}）  {' '.join(done)}")
    if k == 0:
        P("  **対がゼロ。判定は出せない。**")
        print("\n".join(out))
        return

    P("")
    P("  | 問題 | K | RR MPR | RL MPR | RR Score | RL Score | 差 | RR 報告点数 | RL 報告点数 |")
    P("  |---|---|---|---|---|---|---|---|---|")
    for p in done:
        P(f"  | {p} | {K_of(f'{p}-D10-PIN02')} | {rr02[p]['mpr']:.4f} | {rl02[p]['mpr']:.4f} | "
          f"{rr02[p]['score']:.4f} | {rl02[p]['score']:.4f} | "
          f"{rl02[p]['score'] - rr02[p]['score']:+.4f} | {rr02[p]['n']} | {rl02[p]['n']} |")

    for key in ("mpr", "f1", "score", "n"):
        a = float(np.mean([rr02[p][key] for p in done]))
        b = float(np.mean([rl02[p][key] for p in done]))
        P(f"  {k} 問平均 {key}: RR {a:.4f} / RL {b:.4f}  （差 {b - a:+.4f}）")

    # ---------------------------------------------------------- 判定
    P("")
    P("## 判定")
    # 対の単位は (問題, インスタンス)。PIN01 の 16 対 ＋ PIN02 の k 対。
    def pairs(keys):
        a, b = {}, {}
        for tag, p in keys:
            if tag == "P1":
                a[f"P1:{p}"] = PIN01[p][2]      # RR Score
                b[f"P1:{p}"] = PIN01[p][3]      # RL Score
            else:
                a[f"P2:{p}"] = rr02[p]["score"]
                b[f"P2:{p}"] = rl02[p]["score"]
        return b, a, list(a.keys())            # (RL, RR, 順序)

    def mpr_pairs(keys):
        a, b = {}, {}
        for tag, p in keys:
            if tag == "P1":
                a[f"P1:{p}"], b[f"P1:{p}"] = PIN01[p][0], PIN01[p][1]
            else:
                a[f"P2:{p}"], b[f"P2:{p}"] = rr02[p]["mpr"], rl02[p]["mpr"]
        return b, a, list(a.keys())

    K16 = [("P1", p) for p in PROBS]
    K02 = [("P2", p) for p in done]
    KALL = K16 + K02
    KMATCH = [("P1", p) for p in done] + K02

    for label, keys in (("**J1（主判定）** PIN01 16 対 ＋ PIN02 {} 対（n={}）"
                         .format(k, 16 + k), KALL),
                        ("PIN02 だけ（n={}）".format(k), K02),
                        ("PIN01 だけ（n=16、その131 の再掲）", K16),
                        ("釣り合い型（同じ {} 問の両インスタンス、n={}）".format(k, 2 * k), KMATCH)):
        b, a, order = pairs(keys)
        d = paired(b, a, order)
        bm, am, om = mpr_pairs(keys)
        dm = paired(bm, am, om)
        P(f"  {label}")
        P(f"    Score: {fmt(d)}")
        P(f"    MPR  : {fmt(dm)}")

    # 偏りの検査（k < 16 のとき）
    P("")
    P("## 部分集合の偏り（その131 J5 の検査。追加評価ゼロ）")
    b, a, order = pairs([("P1", p) for p in done])
    dsub = paired(b, a, order)
    P(f"  PIN01 を完走した {k} 問だけに制限した Score 差: {dsub['mean']:+.4f}"
      f"  （全 16 問は {PIN01_SCORE_DIFF_16:+.4f}）")
    P(f"    -> ずれ {dsub['mean'] - PIN01_SCORE_DIFF_16:+.4f}")

    print("\n".join(out))




def addendum():
    """副次（追加評価ゼロ）: 対の独立性と、問題単位に平均した保守的な検定。

    **新しい統計量は定義しない**（`paired` と numpy の相関だけ）。
    `(問題, インスタンス)` の 32 対は**同じ 16 問から 2 つずつ**取っているので独立ではない。
    そこで (i) 2 インスタンスの差の相関、(ii) 問題ごとに 2 インスタンスを平均した n=16 の検定
    （＝ 疑似反復を潰した保守側）を出す。
    """
    out = []
    P = out.append
    rr02, rl02 = {}, {}
    for p in PROBS:
        prob = f"{p}-D10-PIN02"
        K = K_of(prob)
        a = find(os.path.join(HERE, "dumps", f"{prob}_rrcma_seed0.csv"))
        b = find(os.path.join(HERE, "descents", f"{prob}_seed0.csv"))
        rr02[p] = score_dump(a, K) if a else score_arrays(*folded("rr", prob), K)
        rl02[p] = score_dump(b, K) if b else score_arrays(*folded("rl", prob), K)

    d1 = np.array([PIN01[p][3] - PIN01[p][2] for p in PROBS])
    d2 = np.array([rl02[p]["score"] - rr02[p]["score"] for p in PROBS])
    P("")
    P("## 副次（追加評価ゼロ）—— 32 対は独立ではない")
    P(f"  2 インスタンスの Score 差の相関: r = {float(np.corrcoef(d1, d2)[0, 1]):+.4f}"
      f"  （符号が一致した問題 {int(((d1 > 0) == (d2 > 0)).sum())}/16）")
    a = {p: (PIN01[p][2] + rr02[p]["score"]) / 2 for p in PROBS}
    b = {p: (PIN01[p][3] + rl02[p]["score"]) / 2 for p in PROBS}
    P(f"  問題ごとに 2 インスタンスを平均した保守側の検定（n=16）: {fmt(paired(b, a, PROBS))}")
    am = {p: (PIN01[p][0] + rr02[p]["mpr"]) / 2 for p in PROBS}
    bm = {p: (PIN01[p][1] + rl02[p]["mpr"]) / 2 for p in PROBS}
    P(f"    MPR 側: {fmt(paired(bm, am, PROBS))}")
    P("")
    P("## 外部妥当性（PIN02 は 2 本目のインスタンス）")
    rrm = float(np.mean([rr02[p]["score"] for p in PROBS]))
    rlm = float(np.mean([rl02[p]["score"] for p in PROBS]))
    P(f"  RR-CMA-ES PIN02 16 問平均 Score {rrm:.4f}  対 公表値 0.5730 ＝ {rrm - 0.5730:+.4f}"
      f"  （事前登録の許容幅 ±0.05 の{'内側' if abs(rrm - 0.5730) < 0.05 else '外側'}）")
    P(f"  `Restart-Lander` PIN02 16 問平均 Score {rlm:.4f}  対 公表最良 0.6080 ＝ {rlm - 0.6080:+.4f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
    addendum()
