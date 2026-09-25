#!/usr/bin/env python3
"""その166 — キュー 1: いちばん堅い主張は CEC2013 でも成り立つか（別 suite での直接対決）。

入力（この回の新規 42 run、seed 0/1/2、suite 既定予算 400,000）:

  * `e166/baseline_{a,b,c}.csv`      — seed 0（事前登録した本体）
  * `e166/baseline_s12_{a,b,c}.csv`  — seed 1,2（枠に余裕が出たので足した分。事前登録 §5 の逆順）
  * `e166/descents.csv.gz`          — `Restart-Lander` の降下ダンプ（畳んだ後。`problem`/`seed` 列つき）

**統計量は `e115/analyze.py` の `paired` をそのまま import する。新しい統計量は 1 つも定義しない。**
Score / MPR / mean-F1 の定義も `e115/analyze.py:130-139` と同じ式で、det だけ CEC2013 公式の
rho ベース（`core/runner._niching_counts` が出した `pr_*` × K）から取る。

使い方: python3 analysis/mmo2024/e166/analyze.py
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

from analyze import paired                                        # noqa: E402

LEVELS = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]
PROBS = ["N14-CF3-3D", "N15-CF4-3D", "N16-CF3-5D", "N17-CF4-5D",
         "N18-CF3-10D", "N19-CF4-10D", "N20-CF4-20D"]
RL, NM = "Restart-Lander", "NMMSO"
FOLDED = os.path.join(HERE, "descents.csv.gz")
KEYS = ("mpr", "f1", "score")


def load():
    """(problem, method, seed) -> dict(mpr, f1, score, n, K, pr, evals)。"""
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "baseline_*.csv"))):
        with open(p, newline="") as fh:
            rows += [r for r in csv.DictReader(fh) if r["rule"] == "current"]
    if not rows:
        sys.exit("baseline_*.csv が無い（run.sh を先に回すこと）")
    out = {}
    for r in rows:
        K, n = int(r["n_optima"]), int(r["n_reported"])
        recall = np.array([float(r[f"pr_{L}"]) for L in LEVELS])
        det = recall * K
        prec = det / n if n > 0 else np.zeros_like(det)
        dn = prec + recall
        f1 = np.where(dn > 0, 2 * prec * recall / np.where(dn > 0, dn, 1.0), 0.0)
        out[(r["function"], r["method"], int(r["seed"]))] = dict(
            mpr=float(recall.mean()), f1=float(f1.mean()),
            score=float(((recall + f1) / 2.0).mean()), n=n, K=K,
            pr={L: float(recall[i]) for i, L in enumerate(LEVELS)},
            evals=int(r["evals"]))
    return out


def cell(agg, prob, method, seeds, key):
    """seed 平均。`key` が 'pr:<level>' なら その水準の PR。"""
    vals = []
    for s in seeds:
        r = agg.get((prob, method, s))
        if r is None:
            continue
        vals.append(r["pr"][key[3:]] if key.startswith("pr:") else r[key])
    return float(np.mean(vals)) if vals else float("nan")


def read_descents():
    """降下ダンプ -> (problem, seed) -> (本数, 評価回数の平均, stop の Counter)。"""
    rows = []
    if os.path.exists(FOLDED):
        with gzip.open(FOLDED, "rt") as fh:
            rows = list(csv.DictReader(fh))
    else:
        for p in sorted(glob.glob(os.path.join(HERE, "descents", "*.csv"))):
            base = os.path.basename(p)[: -len(".csv")]
            name, _, sd = base.rpartition("_seed")
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    r["problem"], r["seed"] = name, str(int(sd) // 100)
                    rows.append(r)
    out: dict = {}
    for r in rows:
        k = (r["problem"], int(r["seed"]))
        d = out.setdefault(k, dict(n=0, ev=[], stop=Counter()))
        d["n"] += 1
        d["ev"].append(int(r["evals"]))
        d["stop"][r["stop"]] += 1
    return out


def block(L, agg, probs, seeds, label):
    """1 つの seed 集合についての §2/§3 と反証条件の判定。戻り値は Score の paired。"""
    P = L.append
    P(f"### {label}")
    P("")
    for key, lab in (("score", "Score（主）"), ("mpr", "MPR"), ("f1", "mean-F1")):
        a = {p: cell(agg, p, RL, seeds, key) for p in probs}
        b = {p: cell(agg, p, NM, seeds, key) for p in probs}
        r = paired(a, b, probs)
        P(f"* **{lab}**: RL {np.mean(list(a.values())):.4f} 対 NM {np.mean(list(b.values())):.4f} "
          f"＝ 対差 {r['mean']:+.4f}（{r['w']}/{r['t']}/{r['l']}、**有効 n={r['w'] + r['l']}**、"
          f"p={r['p']:.6g}、rb={r['rb']:+.3f}）")
        if key == "score":
            out = r
    P("")
    P("| 水準 | RL 平均 PR | NM 平均 PR | 対差 | w/t/l | 有効 n | p | rb |")
    P("|---|---|---|---|---|---|---|---|")
    for Lv in LEVELS:
        a = {p: cell(agg, p, RL, seeds, "pr:" + Lv) for p in probs}
        b = {p: cell(agg, p, NM, seeds, "pr:" + Lv) for p in probs}
        r = paired(a, b, probs)
        P(f"| {Lv} | {np.mean(list(a.values())):.4f} | {np.mean(list(b.values())):.4f} | "
          f"{r['mean']:+.4f} | {r['w']}/{r['t']}/{r['l']} | {r['w'] + r['l']} | "
          f"{r['p']:.6g} | {r['rb']:+.3f} |")
    P("")
    return out


def main():
    agg = load()
    seeds_all = sorted({k[2] for k in agg})
    have = [p for p in PROBS if (p, RL, 0) in agg and (p, NM, 0) in agg]
    miss = [p for p in PROBS if p not in have]
    de = read_descents()

    L = []
    P = L.append
    P("=" * 96)
    P("その166 — キュー 1: いちばん堅い主張は CEC2013 でも成り立つか（別 suite での直接対決）")
    P("=" * 96)
    P(f"CEC2013 F14-F20（{len(have)}/7 問）、seed {seeds_all}、suite 既定予算 400,000、")
    P("報告規則 current、採点は CEC2013 公式 rho ベース（rho=0.01）。")
    P("Score = mean over 5 levels of (recall+F1)/2（`e115/analyze.py:130-139`）。")
    P("**seed 0 が事前登録した本体。seed 1,2 は枠に余裕が出たので足した**（事前登録 §5 の落とす順の逆）。")
    if miss:
        P(f"**枠に入らなかった問題: {', '.join(miss)}**")
    P("")

    # ---- §1 関数別（その162 の教訓: 平均だけ見ない） ----
    P("## §1 関数別（seed 平均。Score / MPR / mean-F1 / 報告点数 / 降下）")
    P("")
    P("| 問題 | D | K | RL Score | NM Score | 差 | RL MPR | NM MPR | RL F1 | NM F1 "
      "| RL rep | NM rep | RL 降下 | 降下/K | 1 降下の評価 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in have:
        rs, ns = cell(agg, p, RL, seeds_all, "score"), cell(agg, p, NM, seeds_all, "score")
        K = agg[(p, RL, 0)]["K"]
        ds = [de[(p, s)] for s in seeds_all if (p, s) in de]
        nd = float(np.mean([d["n"] for d in ds])) if ds else float("nan")
        ev = float(np.mean([np.mean(d["ev"]) for d in ds])) if ds else float("nan")
        P(f"| {p} | {p.split('-')[-1].rstrip('D')} | {K} | {rs:.4f} | {ns:.4f} | {rs - ns:+.4f} "
          f"| {cell(agg, p, RL, seeds_all, 'mpr'):.4f} | {cell(agg, p, NM, seeds_all, 'mpr'):.4f} "
          f"| {cell(agg, p, RL, seeds_all, 'f1'):.4f} | {cell(agg, p, NM, seeds_all, 'f1'):.4f} "
          f"| {cell(agg, p, RL, seeds_all, 'n'):.0f} | {cell(agg, p, NM, seeds_all, 'n'):.0f} "
          f"| {nd:.0f} | {nd / K:.1f} | {ev:.0f} |")
    P("")

    # ---- §2 対検定 ----
    P("## §2 対検定（主指標 Score。反証条件 (a)/(b)）")
    P("")
    r0 = block(L, agg, have, [0], "seed 0（事前登録した本体。n=7 の対比較）")
    rall = None
    if len(seeds_all) > 1:
        rall = block(L, agg, have, seeds_all,
                     f"seed {seeds_all} 平均（追加。同点が減るぶん有効 n が増える）")

    # ---- §3 反証条件の判定 ----
    P("## §3 事前登録した反証条件の判定（判定は seed 0 の本体で行う）")
    P("")
    for tag, r in (("seed 0（本体）", r0),) + ((("seed 平均（参考）", rall),) if rall else ()):
        fired = (r["mean"] <= 0) or (r["w"] < 4)
        P(f"* **{tag}** —— (a)〔符号が正でない、または勝ち数 4 未満 ＝ 主張 1 はこの suite 固有〕: "
          f"**{'発火' if fired else '不発'}**（対差 {r['mean']:+.4f}、勝ち {r['w']}/{len(have)}、"
          f"負け {r['l']}、同点 {r['t']}）。(b) は{'不発' if fired else '発火'}。")
    P("")

    # ---- §4 降下の打ち切り理由（机上予測の検算） ----
    P("## §4 降下の検算 —— 机上予測 32 本は 1 桁外れた。`descent_budget` はこの suite で binding しない")
    P("")
    P("事前登録 §3 の机上計算は「400,000 ÷ 12,500 = 32 本」だった（＝ どの降下も予算を使い切る前提）。")
    stop = Counter()
    for d in de.values():
        stop += d["stop"]
    tot = sum(stop.values())
    P("")
    P("| 問題 | 降下本数 | 降下/K | 1 降下の評価（平均） | 上限 12500 に対する比 |")
    P("|---|---|---|---|---|")
    for p in have:
        ds = [de[(p, s)] for s in seeds_all if (p, s) in de]
        if not ds:
            continue
        nd = float(np.mean([d["n"] for d in ds]))
        ev = float(np.mean([np.mean(d["ev"]) for d in ds]))
        P(f"| {p} | {nd:.0f} | {nd / agg[(p, RL, 0)]['K']:.1f} | {ev:.0f} | {ev / 12500:.3f} |")
    P("")
    P(f"**打ち切り理由の内訳（全 {tot} 降下）**: "
      + " ／ ".join(f"`{k}` {v} 本（{v / tot:.1%}）" for k, v in stop.most_common()))
    P("")
    P("**＝ 降下/K は 6〜60 で、その165 が崩れを見た壁（0.90〜1.00）から 1 桁以上上にある。**")
    P("**その理由は予算配分ではなく停止規則で、CMA-ES が `tolflatfitness` で自分から降りるため。**")

    out = "\n".join(L)
    print(out)
    with open(os.path.join(HERE, "scored.txt"), "w") as fh:
        fh.write(out + "\n")

    with open(os.path.join(HERE, "by_problem.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "method", "seed", "K", "evals", "n_reported",
                    "mpr", "f1", "score"] + [f"pr_{x}" for x in LEVELS]
                   + ["descents", "descent_evals_mean"])
        for p in have:
            for m in (RL, NM):
                for s in seeds_all:
                    x = agg.get((p, m, s))
                    if x is None:
                        continue
                    d = de.get((p, s)) if m == RL else None
                    w.writerow([p, m, s, x["K"], x["evals"], x["n"],
                                f"{x['mpr']:.4f}", f"{x['f1']:.4f}", f"{x['score']:.4f}"]
                               + [f"{x['pr'][v]:.4f}" for v in LEVELS]
                               + ([d["n"], f"{np.mean(d['ev']):.0f}"] if d else ["", ""]))


if __name__ == "__main__":
    main()
