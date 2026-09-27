#!/usr/bin/env python3
"""その175 — キュー 1: 論文の背骨を D=5 / D=20 に広げる（NMMSO を新 suite で測る）。

**採点・規則・統計量は `e115/analyze.py` からそのまま import する**
（`SPAN` / `rule_indices` / `score` / `paired` / `read_dump`）。**新しい統計量は 1 つも定義しない。**

入力の出自:

  * **NMMSO D=5** -> `e175/report_sets_D05.csv.gz`（この回の新規 run を畳んだもの）
  * **NMMSO D=20** -> **`e176/report_sets_D20.csv.gz`**（**その176 が 16 問に埋めたので一本化した。
    この回が回した M09 / M10 の 12 行はそちらに含まれる**）
    （この回の新規 run。報告集合ダンプ ＝ `f` ＋ 座標、上限なし。`land_opt` は持たないので
    その139 と同じ最近傍帰属 `attribute()` で付ける）
  * **`Restart-Lander` D=5**  -> `e153/descents.csv.gz`（保存物、PIN01 seed 0。**追加評価ゼロ**）
  * **`Restart-Lander` D=20** -> `e151/descents.csv.gz`（保存物、PIN01 seed 0。**追加評価ゼロ**）
  * **D=10 の 1 点** -> `prereg.md` の転記（その139。測らない）

使い方: PYTHONPATH=<patched pynmmso> python3 analysis/mmo2024/e175/analyze.py [D05 D20 ...]
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

from analyze import LEVEL_NAMES, SPAN, paired, rule_indices, score   # noqa: E402
from core.benchmarks import niching_by_name                          # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN     # その115 の合法な最良腕
PROBS = [f"M{i:02d}" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]

# 転記（prereg.md §2。この回では測らない）— その139 の D=10 の 1 点
REF_D10 = dict(diff=0.2140, p=0.004181, rl=0.6284, nmmso=0.3950)

RL_FOLDED = {"D05": os.path.join(MMO, "e153", "descents.csv.gz"),
             "D20": os.path.join(MMO, "e151", "descents.csv.gz")}

_BENCH: dict = {}


def bench(prob_full):
    if prob_full not in _BENCH:
        b = niching_by_name(prob_full)
        _BENCH[prob_full] = (int(b.n_global_optima),
                             np.asarray(b.optima_pos, dtype=float))
    return _BENCH[prob_full]


def attribute(x, optima):
    """最近傍の最適への帰属（`core.runner.count_goptima_nn` と同順序）。オラクル量。"""
    d = np.linalg.norm(x[:, None, :] - optima[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def read_rl(dim):
    """畳んだ降下ダンプから {問題短名: (f, opt, x)} を返す。"""
    path = RL_FOLDED[dim]
    out: dict = {}
    with gzip.open(path, "rt") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["problem"], []).append(r)
    res = {}
    for full, rows in out.items():
        ncol = sum(1 for k in rows[0] if k.startswith("x") and k[1:].isdigit())
        f = np.array([float(r["best_f"]) for r in rows])
        opt = np.array([int(r["land_opt"]) for r in rows])
        x = np.array([[float(r[f"x{i}"]) for i in range(ncol)] for r in rows])
        res[full.split("-")[0]] = (f, opt, x)
    return res


def read_nmmso(dim):
    """報告集合ダンプから {問題短名: (f, opt, x)} を返す（opt は最近傍帰属）。

    **`fold.py` が per-problem を 1 本に畳んだ後は `report_sets_<DIM>.csv.gz` を読む**
    （`problem` / `seed` 列が付いただけで、採点に渡る中身は per-problem 時代と同一。
    畳む前後でこの script の出力が 1 文字も変わらないことを確認してある ＝ `scored.txt`）。
    畳む前の `dumps/<DIM>/` が残っていればそちらを読む（畳む前後の照合用の退避路）。
    """
    res = {}
    d = os.path.join(HERE, "dumps", dim)
    per_problem = sorted(os.listdir(d)) if os.path.isdir(d) else []
    if per_problem:
        for fn in per_problem:
            if not fn.endswith((".csv", ".csv.gz")):
                continue
            full = fn.split("_")[0]
            op = gzip.open if fn.endswith(".gz") else open
            with op(os.path.join(d, fn), "rt") as fh:
                rows = list(csv.DictReader(fh))
            if rows:
                res[full.split("-")[0]] = _as_run(rows, full)
        if res:
            return res
    folded = os.path.join(HERE, f"report_sets_{dim}.csv.gz")
    if not os.path.exists(folded) and dim == "D20":
        # **その176 が D=20 を 16 問に埋めたので、この回の 2 問ぶんは消して e176 の 1 本に一本化した**
        # （消す前に M09 / M10 の 12 行が e176 側と完全一致することを確認済み ＝ 値は 1 つも消していない）。
        folded = os.path.join(MMO, "e176", "report_sets_D20.csv.gz")
    if not os.path.exists(folded):
        return res
    by: dict = {}
    with gzip.open(folded, "rt") as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["problem"], []).append(r)
    for full, rows in by.items():
        res[full.split("-")[0]] = _as_run(rows, full)
    return res


def _as_run(rows, full):
    ncol = sum(1 for k in rows[0] if k.startswith("x") and k[1:].isdigit())
    f = np.array([float(r["f"]) for r in rows])
    x = np.array([[float(r[f"x{i}"]) for i in range(ncol)] for r in rows])
    return f, attribute(x, bench(full)[1]), x


def score_one(f, opt, x, K):
    idx = rule_indices(ARM, f, K, x=x, r=ARM_R)
    recall, prec, f1, sc, n = score(idx, f, opt, K)
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()),
                score=float(sc.mean()), n=int(n), ndump=len(f),
                pr_lv=recall, f1_lv=f1)


def run_dim(dim, out_rows):
    rl, nm = read_rl(dim), read_nmmso(dim)
    probs = [p for p in PROBS if p in rl and p in nm]
    missing = [p for p in PROBS if p not in nm]
    print(f"\n{'='*78}\n{dim}: NMMSO {len(nm)}/16 問、対照 RL {len(rl)}/16 問 "
          f"-> 対になるのは {len(probs)} 問")
    if missing:
        print(f"  NMMSO 側が欠けている問題: {' '.join(missing)}")
    if not probs:
        print("  -> 対がゼロ。この次元は判定しない。")
        return None
    A, B = {}, {}
    print(f"\n  {'問題':<6}{'K':>4}{'RL_MPR':>9}{'RL_F1':>8}{'RL_Sc':>8}"
          f"{'NM_MPR':>9}{'NM_F1':>8}{'NM_Sc':>8}{'ΔSc':>9}{'RL_n':>6}{'NM_n':>6}")
    for p in probs:
        full = f"{p}-{dim}-PIN01"
        K = bench(full)[0]
        a, b = score_one(*rl[p], K), score_one(*nm[p], K)
        A[p], B[p] = a, b
        print(f"  {p:<6}{K:>4}{a['mpr']:>9.4f}{a['f1']:>8.4f}{a['score']:>8.4f}"
              f"{b['mpr']:>9.4f}{b['f1']:>8.4f}{b['score']:>8.4f}"
              f"{a['score']-b['score']:>+9.4f}{a['n']:>6}{b['n']:>6}")
        out_rows.append(dict(dim=dim, problem=p, K=K,
                             rl_mpr=round(a['mpr'], 4), rl_f1=round(a['f1'], 4),
                             rl_score=round(a['score'], 4), rl_n=a['n'],
                             nm_mpr=round(b['mpr'], 4), nm_f1=round(b['f1'], 4),
                             nm_score=round(b['score'], 4), nm_n=b['n'],
                             d_score=round(a['score'] - b['score'], 4)))

    def mean(d, k):
        return float(np.mean([d[p][k] for p in probs]))

    print(f"\n  {len(probs)} 問平均: RL Score {mean(A,'score'):.4f} "
          f"(MPR {mean(A,'mpr'):.4f} / F1 {mean(A,'f1'):.4f})、"
          f"NMMSO Score {mean(B,'score'):.4f} "
          f"(MPR {mean(B,'mpr'):.4f} / F1 {mean(B,'f1'):.4f})")
    res = {}
    for key in ("score", "mpr", "f1"):
        st = paired({p: A[p][key] for p in probs}, {p: B[p][key] for p in probs}, probs)
        res[key] = st
        print(f"  {key:<6} RL − NMMSO = {st['mean']:+.4f}  "
              f"{st['w']}/{st['t']}/{st['l']}  p={st['p']:.6g}  rb={st['rb']:+.3f}")
    # 群別（K=20 / K=10）と水準別は内訳として出すだけ（検定はしない）
    for name, g in (("群 A K=20", GROUP_A), ("群 B K=10", GROUP_B)):
        gg = [p for p in g if p in probs]
        if gg:
            d = float(np.mean([A[p]['score'] - B[p]['score'] for p in gg]))
            print(f"    {name} ({len(gg)} 問): ΔScore {d:+.4f}")
    print("    水準別 ΔPR: " + "  ".join(
        f"{LEVEL_NAMES[i]} {float(np.mean([A[p]['pr_lv'][i]-B[p]['pr_lv'][i] for p in probs])):+.4f}"
        for i in range(5)))
    res["n_probs"] = len(probs)
    res["mean"] = {k: {s: mean(d, s) for s in ("score", "mpr", "f1", "n")}
                   for k, d in (("rl", A), ("nmmso", B))}
    return res


def main():
    dims = sys.argv[1:] or ["D05", "D20"]
    rows, summary = [], {}
    for dim in dims:
        r = run_dim(dim, rows)
        if r:
            summary[dim] = r
    out = os.path.join(HERE, "by_problem.csv")
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")

    print(f"\n{'='*78}\n次元ごとに別々の検定を並べる（**プールしない** ＝ その152 の疑似反復）")
    print(f"  {'次元':<6}{'n':>4}{'RL_Sc':>8}{'NM_Sc':>8}{'ΔScore':>9}{'w/t/l':>9}{'p':>11}{'rb':>8}")
    print(f"  {'D=10':<6}{16:>4}{REF_D10['rl']:>8.4f}{REF_D10['nmmso']:>8.4f}"
          f"{REF_D10['diff']:>+9.4f}{'12/0/4':>9}{REF_D10['p']:>11.6g}{'—':>8}"
          "   <- その139 の転記（測っていない）")
    for dim, r in summary.items():
        s = r["score"]
        wtl = "%d/%d/%d" % (s["w"], s["t"], s["l"])
        print(f"  {dim:<6}{r['n_probs']:>4}{r['mean']['rl']['score']:>8.4f}"
              f"{r['mean']['nmmso']['score']:>8.4f}{s['mean']:>+9.4f}"
              f"{wtl:>9}{s['p']:>11.6g}{s['rb']:>+8.3f}")

    print(f"\n{'='*78}\n反証条件（prereg.md §4）の判定")
    for dim, r in summary.items():
        s = r["score"]
        neg, sig = s["mean"] < 0, s["p"] < 0.05
        print(f"  {dim}: 符号 {'負' if neg else '正'} / p={s['p']:.6g} "
              f"({'有意' if sig else '非有意'}) -> "
              + ("(a) 発火 ＝ 主張 1 は D=10 限定" if neg
                 else "(b) 側（正かつ有意）" if sig
                 else "(c) 側（正だが非有意。『差が無い』と書かない）"))


if __name__ == "__main__":
    main()
