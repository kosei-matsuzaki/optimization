#!/usr/bin/env python3
"""その127 — RR-CMA-ES を、既存 3 手法とまったく同じ採点器・同じ報告規則で採点する（キュー 1）。

**新しい規則も新しい統計量も 1 つも定義しない。** `rule_indices` / `score` / `paired` は
`analysis/mmo2024/e115/analyze.py` からそのまま import する。違うのは入力だけ。

  * `RR-CMA-ES`       -> analysis/mmo2024/e127/dumps/*_rrcma_seed0.csv(.gz)（今回の run）
  * `Restart-Lander`  -> analysis/mmo2024/e115/descents（seed 0、保存物。追加 run ゼロ）
  * `MC-ESO` / `NMMSO` -> analysis/mmo2024/e116/ranking_d10.csv の `legal` 行（記録値。
                          ダンプはその120 で削除済み）

**関門は 2 つ**（`prereg.md`）:
  1. `Restart-Lander` の 16 問平均が その116 の記録（MPR 0.5644 / Score 0.6284）と 4 桁一致。
  2. `Restart-Lander` の問題別が `ranking_d10.csv` の `legal` 行と 16/16 一致。

使い方: python3 analysis/mmo2024/e127/analyze.py
"""
from __future__ import annotations

import csv
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
GATE = {"mpr": 0.5644, "score": 0.6284}          # その116 の記録
PUBLISHED = {"TRDE-LR": 0.6080, "RR-CMA-ES": 0.5730}   # 競技資料（出典 1 本）
TOL_PUBLISHED = 0.05                             # 事前登録した許容幅
PROBS = [f"M{i:02d}-D10-PIN01" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]
NULL_DIR = os.path.join(MMO, "e115", "descents")
RR_DIR = os.path.join(HERE, "dumps")
RANKING = os.path.join(MMO, "e116", "ranking_d10.csv")

_K: dict = {}


def K_of(prob):
    if prob not in _K:
        _K[prob] = int(niching_by_name(prob).n_global_optima)
    return _K[prob]


def find(d, name):
    for ext in ("", ".gz"):
        p = os.path.join(d, name + ext)
        if os.path.exists(p):
            return p
    return None


def score_dump(path, K):
    f, opt, xs = read_dump(path)
    idx = rule_indices(ARM, f, K, x=xs, r=ARM_R)
    recall, prec, f1, sc, n = score(idx, f, opt, K)
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()),
                score=float(sc.mean()), n=int(n), ndump=len(f))


def load_recorded():
    """e116 の記録値（`legal` 行）を method -> problem -> dict で読む。"""
    out: dict = {}
    with open(RANKING) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] != "legal":
                continue
            out.setdefault(r["method"], {})[r["problem"]] = dict(
                mpr=float(r["mpr"]), f1=float(r["mean_f1"]),
                score=float(r["score"]), n=float(r["n_reported"]))
    return out


def main():
    rec = load_recorded()

    live = {"Restart-Lander": {}, "RR-CMA-ES": {}}
    for p in PROBS:
        q = find(NULL_DIR, f"{p}_seed0.csv")
        if q:
            live["Restart-Lander"][p] = score_dump(q, K_of(p))
        q = find(RR_DIR, f"{p}_rrcma_seed0.csv")
        if q:
            live["RR-CMA-ES"][p] = score_dump(q, K_of(p))

    print("=" * 96)
    print("その127 — RR-CMA-ES（de Nobel+ 2024, modcma の repelling）を同一採点器に載せる（キュー 1）")
    print("=" * 96)
    print(f"\n  規則: {ARM} (r={ARM_R})   RR-CMA-ES が揃った問題: "
          f"{len(live['RR-CMA-ES'])}/16")
    missing = [p for p in PROBS if p not in live["RR-CMA-ES"]]
    if missing:
        print(f"  **欠けた問題: {', '.join(missing)}**")

    # ------------------------------------------------- 関門 1（16 問平均の再現）
    nl = live["Restart-Lander"]
    g_mpr = float(np.mean([nl[p]["mpr"] for p in PROBS if p in nl]))
    g_sc = float(np.mean([nl[p]["score"] for p in PROBS if p in nl]))
    ok1 = abs(g_mpr - GATE["mpr"]) < 5e-5 and abs(g_sc - GATE["score"]) < 5e-5
    print(f"\n  関門 1（null 16 問平均）: MPR {g_mpr:.4f} (記録 {GATE['mpr']}) / "
          f"Score {g_sc:.4f} (記録 {GATE['score']})  -> {'一致' if ok1 else '**不一致**'}")

    # ------------------------------------------------- 関門 2（問題別の一致）
    bad = []
    for p in PROBS:
        if p not in nl or p not in rec.get("Restart-Lander", {}):
            continue
        a, b = nl[p], rec["Restart-Lander"][p]
        if max(abs(a["mpr"] - b["mpr"]), abs(a["f1"] - b["f1"]),
               abs(a["score"] - b["score"])) > 5e-5:
            bad.append(p)
    ok2 = not bad
    print(f"  関門 2（null 問題別 対 ranking_d10.csv の legal 行）: "
          f"{16 - len(bad)}/16 一致  -> {'一致' if ok2 else '**不一致: ' + ','.join(bad) + '**'}")
    if not (ok1 and ok2):
        print("\n  **採点経路が再現しないので判定に進まない。**")
        return

    # ------------------------------------------------- 問題別の表
    methods = ["RR-CMA-ES", "Restart-Lander", "MC-ESO", "NMMSO"]

    def get(m, p):
        if m in live and p in live[m]:
            return live[m][p]
        return rec.get(m, {}).get(p)

    probs = [p for p in PROBS if get("RR-CMA-ES", p) is not None]
    print(f"\n{'problem':<16}{'K':>4}" + "".join(f"{m[:9]+' sc':>14}" for m in methods)
          + f"{'RR n':>7}{'RR mpr':>9}{'RL mpr':>9}")
    print("-" * 96)
    for p in probs:
        vs = [get(m, p) for m in methods]
        print(f"{p:<16}{K_of(p):>4}"
              + "".join(f"{v['score']:14.4f}" if v else f"{'-':>14}" for v in vs)
              + f"{vs[0]['n']:7.0f}{vs[0]['mpr']:9.4f}{vs[1]['mpr']:9.4f}")

    def means(sub, m, key):
        vals = [get(m, p)[key] for p in sub if get(m, p) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    for label, sub in (("全体", probs), ("群 A", [p for p in probs if p in GROUP_A]),
                       ("群 B", [p for p in probs if p in GROUP_B])):
        if not sub:
            continue
        print(f"\n  [{label}] n={len(sub)}")
        for m in methods:
            print(f"    {m:<16} MPR {means(sub, m, 'mpr'):.4f}   "
                  f"mean-F1 {means(sub, m, 'f1'):.4f}   "
                  f"Score {means(sub, m, 'score'):.4f}   "
                  f"報告点数 {means(sub, m, 'n'):.1f}")

    # ------------------------------------------------- 公表値との突き合わせ
    rr = means(probs, "RR-CMA-ES", "score")
    d = rr - PUBLISHED["RR-CMA-ES"]
    print(f"\n  公表値との差: 実測 {rr:.4f} 対 資料 {PUBLISHED['RR-CMA-ES']:.4f}  "
          f"= {d:+.4f}  -> {'±0.05 の内側' if abs(d) <= TOL_PUBLISHED else '**±0.05 の外**'}")

    # ------------------------------------------------- 対比較
    print(f"\n  対比較（問題ごと、Wilcoxon exact ＋ rank-biserial）")
    for key in ("mpr", "score"):
        for x, y in (("Restart-Lander", "RR-CMA-ES"), ("RR-CMA-ES", "MC-ESO"),
                     ("RR-CMA-ES", "NMMSO")):
            sub = [p for p in probs
                   if get(x, p) is not None and get(y, p) is not None]
            a = {p: get(x, p)[key] for p in sub}
            b = {p: get(y, p)[key] for p in sub}
            r = paired(a, b, sub)
            print(f"    {key:<5} {x:<15}-{y:<15} mean {r['mean']:+.4f}  "
                  f"W/T/L {r['w']}/{r['t']}/{r['l']}  p={r['p']:.4g}  rb={r['rb']:+.3f}")

    # ------------------------------------------------- 集計 CSV（数百行未満）
    out = os.path.join(HERE, "by_problem_d10.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "method", "arm", "n_reported", "mpr",
                    "mean_f1", "score", "source"])
        for p in probs:
            for m in methods:
                v = get(m, p)
                if v is None:
                    continue
                src = "e127 run" if (m in live and p in live[m]) else "e116 record"
                w.writerow([p, K_of(p), m, ARM, f"{v['n']:.0f}", f"{v['mpr']:.6f}",
                            f"{v['f1']:.6f}", f"{v['score']:.6f}", src])
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
