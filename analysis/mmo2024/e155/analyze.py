#!/usr/bin/env python3
"""その155 — キュー 1: RR-CMA-ES の被覆係数 c を振って D=5 の MPR 赤字を切り分ける。

**採点・規則・統計量は `e115/analyze.py` からそのまま import する**（`rule_indices` / `score` / `paired` / `read_dump` / `SPAN`）。
**新しい統計量は 1 つも定義しない。**

入力:
  * `e155/dumps/`（この回の新規 48 run: c ∈ {2,20,200} × 16 問、D=5・PIN01・seed 0）
  * `e155/restarts/`（同じ run の再起動単位の記録。**キュー 1 (ii) が要求した列**）
  * `e153/dumps_rrcma.csv.gz`（その153 の保存物。**関門 G1 の照合先**、追加評価ゼロ）

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e155/analyze.py
"""
from __future__ import annotations

import csv
import glob
import gzip
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MMO, "e115"))

from analyze import SPAN, paired, read_dump, rule_indices, score   # noqa: E402

ARM_RULE, ARM_R = "eps_loose+dedup", 0.05 * SPAN
PROBS = [f"M{i:02d}-D05-PIN01" for i in range(1, 17)]
K_OF = {p: (20 if int(p[1:3]) <= 8 else 10) for p in PROBS}
COVS = [2, 20, 200]
BUDGET = 250_000

# --- 転記: その153 の D=5 / RR-CMA-ES の 16 問平均（関門 G1 の照合先。測らない） ---
E153 = dict(mpr=0.6619, f1=0.7822, score=0.7220)
PUB5 = dict(mpr=0.844, f1=0.585, score=0.7145)     # 競技資料（D=5 の最良は RR-CMA-ES 自身）
SD_RR = 0.0238                                     # その123 の 1 seed ばらつき
GATE_FLAT = 0.05                                   # 反証条件 (a) の閾（≈ 2·SD_RR）


def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


def load_arm(cov):
    """c = cov の 16 問を読む。**1 問でも欠けたら exit 1**（その150 §3 の直し）。"""
    out = {}
    for p in PROBS:
        path = os.path.join(HERE, "dumps", f"{p}_cov{cov:g}_seed0.csv")
        if not os.path.exists(path):
            die(f"入力が無い: {path}  （黙って飛ばさない）")
        f, opt, xs = read_dump(path)
        out[p] = dict(f=f, opt=opt, x=xs, K=K_OF[p])
    return out


def load_restarts(cov):
    out = {}
    for p in PROBS:
        path = os.path.join(HERE, "restarts", f"{p}_cov{cov:g}_seed0.csv")
        if not os.path.exists(path):
            die(f"入力が無い: {path}")
        with open(path) as fh:
            out[p] = list(csv.DictReader(fh))
    return out


def scored(arm):
    """報告規則を当てて問題ごとの (MPR, mean-F1, Score, n) を返す。"""
    res = {}
    for p, r in arm.items():
        idx = rule_indices(ARM_RULE, r["f"], r["K"], x=r["x"], r=ARM_R)
        rec, prec, f1, sc, n = score(idx, r["f"], r["opt"], r["K"])
        res[p] = dict(mpr=float(rec.mean()), f1=float(f1.mean()),
                      score=float(sc.mean()), n=int(n))
    return res


def decompose(arm):
    """archive を deep(f≤1e-5) / distinct / redundant / shallow に割る（oracle 帰属）。"""
    res = {}
    for p, r in arm.items():
        f, o = r["f"], r["opt"]
        deep = f <= 1e-5
        nd, dist = int(deep.sum()), len(set(o[deep]))
        res[p] = dict(arch=len(f) - 1, deep=nd, distinct=dist,
                      redundant=nd - dist - 1, shallow=int((~deep).sum()))
    return res


def main():
    print("=" * 88)
    print("その155 — キュー 1: RR-CMA-ES の被覆係数 c（de Nobel+ 2024 §4.3）を 3 水準で振る")
    print("=" * 88)

    arms = {c: load_arm(c) for c in COVS}
    rst = {c: load_restarts(c) for c in COVS}
    sc = {c: scored(arms[c]) for c in COVS}
    dec = {c: decompose(arms[c]) for c in COVS}

    # ---------------- 関門 ----------------
    print("\n## 関門\n")
    # G1: c=20 が その153 を再現するか
    m20 = {k: float(np.mean([sc[20][p][k] for p in PROBS])) for k in ("mpr", "f1", "score")}
    g1 = abs(m20["mpr"] - E153["mpr"])
    print(f"  G1  c=20（既定 ＝ その153 の配線）の 16 問平均: "
          f"MPR {m20['mpr']:.4f}（その153 {E153['mpr']:.4f}、差 {m20['mpr']-E153['mpr']:+.4f}）／ "
          f"mean-F1 {m20['f1']:.4f}（{E153['f1']:.4f}）／ Score {m20['score']:.4f}（{E153['score']:.4f}）")
    print(f"      -> {'通過' if g1 < 0.01 else '**不通過**'}（閾 0.01）")

    # archive サイズの 1 問ずつの照合
    ref = {}
    with gzip.open(os.path.join(MMO, "e153", "dumps_rrcma.csv.gz"), "rt") as fh:
        for row in csv.DictReader(fh):
            ref[row["problem"]] = ref.get(row["problem"], 0) + (row["stop"] == "taboo")
    diffs = [(p, dec[20][p]["arch"], ref.get(p)) for p in PROBS if dec[20][p]["arch"] != ref.get(p)]
    print(f"  G1b archive サイズの 1 問ずつの一致: {16-len(diffs)}/16"
          + ("" if not diffs else f"  ずれ: {diffs}"))

    # G2: 予算消化
    bad = []
    for c in COVS:
        for p in PROBS:
            ev = int(rst[c][p][-1]["evals_cum"]) if rst[c][p] else 0
            if ev < 0.99 * BUDGET:
                bad.append((c, p, ev))
    print(f"  G2  予算消化 ≥99%: {'通過' if not bad else '**不通過** ' + str(bad[:5])}"
          f"（最終再起動時点の累積評価で見る。予算切れ時の端数は含まない）")

    # ---------------- 容疑「再起動の回数」 ----------------
    print("\n## 1. 容疑「再起動の回数」の実測\n")
    print("  | c | 再起動本数(平均) | 1 本あたり評価(平均) | archive(平均) | 本数→archive の歩留まり |")
    print("  |---|---|---|---|---|")
    for c in COVS:
        nr = np.mean([len(rst[c][p]) for p in PROBS])
        ev = np.mean([np.mean([int(r["evals_this"]) for r in rst[c][p]]) for p in PROBS])
        ar = np.mean([dec[c][p]["arch"] for p in PROBS])
        print(f"  | {c} | {nr:.1f} | {ev:.0f} | {ar:.2f} | {ar/nr:.3f} |")
    print("\n  キュー 1 が書いた「D=5 は 26.2 本（1 本 9,500 評価）」は **archive のサイズ**であって再起動の本数ではない。")

    print("\n  打ち切り理由の内訳（16 問合算、c 別）:")
    for c in COVS:
        cnt = Counter()
        for p in PROBS:
            for r in rst[c][p]:
                cnt[r["reason"]] += 1
        tot = sum(cnt.values())
        top = ", ".join(f"{k}={v} ({v/tot:.1%})" for k, v in cnt.most_common(6))
        print(f"    c={c:<4} n={tot:<5} {top}")

    # ---------------- 主判定 ----------------
    print("\n## 2. 主判定 — MPR は c で動くか\n")
    print("  | c | MPR | mean-F1 | Score | 報告点数 | 公表 MPR 0.844 との差 |")
    print("  |---|---|---|---|---|---|")
    mprs = {}
    for c in COVS:
        m = {k: float(np.mean([sc[c][p][k] for p in PROBS])) for k in ("mpr", "f1", "score")}
        n = float(np.mean([sc[c][p]["n"] for p in PROBS]))
        mprs[c] = m["mpr"]
        print(f"  | {c} | {m['mpr']:.4f} | {m['f1']:.4f} | {m['score']:.4f} | {n:.2f} | {m['mpr']-PUB5['mpr']:+.4f} |")
    rng = max(mprs.values()) - min(mprs.values())
    print(f"\n  3 水準の幅 = {rng:.4f}（反証条件 (a) の閾 {GATE_FLAT}）"
          f" -> (a) は {'**発火**（c では説明できない）' if rng < GATE_FLAT else '不発（c で動く）'}")
    hit = [c for c in COVS if mprs[c] >= PUB5["mpr"] - 0.10]
    print(f"  判定（±0.10 で公表 0.844 に届く c）: {hit if hit else 'なし'}")
    best, worst = max(mprs, key=mprs.get), min(mprs, key=mprs.get)
    print(f"  向き: c={best} が最大 {mprs[best]:.4f}、c={worst} が最小 {mprs[worst]:.4f}"
          f" -> 反証条件 (b)（c を下げると MPR が下がる）は "
          f"{'**発火**' if mprs[2] < mprs[20] else '不発'}")

    print("\n  対比較（c=20 を基準、16 問対、Wilcoxon exact）:")
    for c in (2, 200):
        for k in ("mpr", "score"):
            a = {p: sc[c][p][k] for p in PROBS}
            b = {p: sc[20][p][k] for p in PROBS}
            r = paired(a, b, PROBS)
            print(f"    c={c:<4} {k:<6} 差 {r['mean']:+.4f}  {r['w']}/{r['t']}/{r['l']}  "
                  f"p={r['p']:.4g}  rb={r['rb']:+.3f}")

    # ---------------- archive の分解 ----------------
    print("\n## 3. archive の分解（oracle 帰属、16 問合算）\n")
    print("  | c | archive | deep(f≤1e-5) | うち相異なる | うち冗長 | shallow(局所最適等) | PR@1e-5 |")
    print("  |---|---|---|---|---|---|---|")
    for c in COVS:
        A = sum(dec[c][p]["arch"] for p in PROBS)
        D = sum(dec[c][p]["deep"] for p in PROBS)
        Di = sum(dec[c][p]["distinct"] for p in PROBS)
        R = sum(dec[c][p]["redundant"] for p in PROBS)
        S = sum(dec[c][p]["shallow"] for p in PROBS)
        print(f"  | {c} | {A} | {D} | {Di} | {R} | {S} | {Di/240:.4f} |")
    print("\n  （K の合計は 8×20 + 8×10 = 240。`redundant` は 'final' 行の重複 1 点を各問で引いてある）")

    # ---------------- 問題別 ----------------
    print("\n## 4. 問題別 MPR（関数別の増減を必ず列挙する規則）\n")
    print("  | 問題 | K | c=2 | c=20 | c=200 | c2−c20 | 再起動本数 c=20 |")
    print("  |---|---|---|---|---|---|---|")
    for p in PROBS:
        print(f"  | {p[:3]} | {K_OF[p]} | {sc[2][p]['mpr']:.3f} | {sc[20][p]['mpr']:.3f} | "
              f"{sc[200][p]['mpr']:.3f} | {sc[2][p]['mpr']-sc[20][p]['mpr']:+.3f} | {len(rst[20][p])} |")


if __name__ == "__main__":
    main()
