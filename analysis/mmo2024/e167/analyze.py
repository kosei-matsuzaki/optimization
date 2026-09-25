#!/usr/bin/env python3
"""その167 — キュー 1 の追加 3 問（CEC2013 F11-F13、2D）の `Restart-Lander` 側。

NMMSO はこのコンテナで回せない（`prereg.md` §5）ので、**対検定はこの回では引けない。**
この script が出すのは、次のサイクルが NMMSO 側 9 run を足すだけで対検定に入れるよう
**RL 側の確定値**と、**事前登録 (c)(d) の判定に要る 2 つの診断**である:

  1. **seed 方向のばらつき** — RL の MPR が seed で動かない問題は、対差の符号が
     NMMSO 側の 1 点だけで決まる ＝ 同点の出やすさを直接規定する。
  2. **降下本数 ÷ K** — その165 が崩れを見た壁（0.90〜1.00）の上か下か。
     その166 §4 は F14-F20 で 5.9〜59.8 ＝ 壁の 1 桁上だったが、2D は予算が半分なので
     机上では 200,000 / 12,500 = 16 本、K=6 で 2.67 ＝ 壁に近づく側。

出力は `scored.txt`（この script の stdout をそのまま保存したもの）。
"""
from __future__ import annotations

import csv
import glob
import gzip
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]
K = {"N11-CF1-2D": 6, "N12-CF2-2D": 8, "N13-CF3-2D": 6}


def _rows() -> list[dict]:
    out: list[dict] = []
    for tag in ("a", "b", "c"):
        p = os.path.join(HERE, f"baseline_{tag}.csv")
        if not os.path.exists(p):                      # 畳んだあと
            p = os.path.join(HERE, "baseline_all.csv")
            if not os.path.exists(p):
                sys.exit("baseline_*.csv が無い。この回の入力は削除された "
                         "—— 数値は docs/acceptance_topology.md の その167 の節にある。")
            with open(p) as fh:
                return list(csv.DictReader(fh))
        with open(p) as fh:
            out.extend(csv.DictReader(fh))
    return out


def _mpr(r: dict) -> float:
    return sum(float(r[k]) for k in LEVELS) / len(LEVELS)


def _f1(r: dict) -> float:
    """mean-F1 over the 5 levels (precision = det/|rep|, recall = det/K)."""
    k, rep = int(r["n_optima"]), int(r["n_reported"])
    vals = []
    for lv in LEVELS:
        recall = float(r[lv])
        det = recall * k
        prec = det / rep if rep else 0.0
        vals.append(2 * prec * recall / (prec + recall) if prec + recall else 0.0)
    return sum(vals) / len(vals)


def _score(r: dict) -> float:
    """`e115/analyze.py:130-139` の定義（水準ごとの (recall + F1)/2 の平均）。"""
    k, rep = int(r["n_optima"]), int(r["n_reported"])
    vals = []
    for lv in LEVELS:
        recall = float(r[lv])
        det = recall * k
        prec = det / rep if rep else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
        vals.append((recall + f1) / 2)
    return sum(vals) / len(vals)


def _descents() -> dict[tuple[str, int], tuple[int, dict[str, int], float]]:
    """(func, seed) -> (本数, 打ち切り理由の内訳, 1 降下あたり平均評価)."""
    out = {}
    pats = [os.path.join(HERE, "descents_*", "*.csv"),
            os.path.join(HERE, "descents.csv.gz")]
    files = sorted(glob.glob(pats[0]))
    if files:
        for p in files:
            base = os.path.basename(p)[:-4]
            func, _, s = base.rpartition("_seed")
            with open(p) as fh:
                rows = list(csv.DictReader(fh))
            stops: dict[str, int] = {}
            for r in rows:
                stops[r["stop"]] = stops.get(r["stop"], 0) + 1
            ev = statistics.mean(int(r["evals"]) for r in rows) if rows else 0.0
            out[(func, int(s) // 100)] = (len(rows), stops, ev)
        return out
    if os.path.exists(pats[1]):                        # 畳んだあと
        with gzip.open(pats[1], "rt") as fh:
            rows = list(csv.DictReader(fh))
        grp: dict[tuple[str, int], list[dict]] = {}
        for r in rows:
            grp.setdefault((r["function"], int(r["seed"])), []).append(r)
        for kk, rs in grp.items():
            stops: dict[str, int] = {}
            for r in rs:
                stops[r["stop"]] = stops.get(r["stop"], 0) + 1
            out[kk] = (len(rs), stops, statistics.mean(int(r["evals"]) for r in rs))
        return out
    return {}          # その169 が畳んだ（下の §3 が理由を印字する）


def main() -> None:
    rows = _rows()
    print("## その167 §1 `Restart-Lander` の確定値（CEC2013 F11-F13、2D、予算 200,000、報告規則 current）\n")
    print(f"{'問題':<13}{'K':>3}{'seed':>6}{'|rep|':>7}"
          + "".join(f"{lv[3:]:>8}" for lv in LEVELS) + f"{'MPR':>9}{'F1':>9}{'Score':>9}")
    per_func: dict[str, list[float]] = {}
    per_func_score: dict[str, list[float]] = {}
    for r in sorted(rows, key=lambda r: (r["function"], int(r["seed"]))):
        f = r["function"]
        print(f"{f:<13}{r['n_optima']:>3}{r['seed']:>6}{r['n_reported']:>7}"
              + "".join(f"{float(r[lv]):>8.4f}" for lv in LEVELS)
              + f"{_mpr(r):>9.4f}{_f1(r):>9.4f}{_score(r):>9.4f}")
        per_func.setdefault(f, []).append(_mpr(r))
        per_func_score.setdefault(f, []).append(_score(r))

    print("\n## その167 §2 診断 (1) —— seed 方向のばらつき（同点の出やすさを規定する）\n")
    print(f"{'問題':<13}{'K':>3}{'MPR 平均':>10}{'MPR SD':>9}{'刻み 1/K':>10}"
          f"{'Score 平均':>12}{'Score SD':>10}{'相異なり数':>12}")
    for f in sorted(per_func):
        v = per_func[f]
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        sv = per_func_score[f]
        ssd = statistics.stdev(sv) if len(sv) > 1 else 0.0
        mean = sum(v) / len(v)
        print(f"{f:<13}{K[f]:>3}{mean:>10.4f}{sd:>9.4f}{1 / K[f]:>10.4f}"
              f"{sum(sv) / len(sv):>12.4f}{ssd:>10.4f}{mean * K[f]:>12.2f}")

    print("\n## その167 §3 診断 (2) —— 降下本数 ÷ K（その165 の壁 0.90〜1.00 の上か下か）\n")
    d = _descents()
    if not d:
        global DEGRADED
        DEGRADED = True
        print("**§3 は引けない —— `descents.csv.gz` は その169 が畳んだ**"
              "（キュー 1 の軸が閉じたため。判断の根拠は その169 の節）。\n"
              "この節の数値は全部 docs/acceptance_topology.md の "
              "「### §3 `descent_budget` は 2D でも binding しない」の表にある:\n"
              "  降下本数 N11 204/207/207・N12 177/179/179・N13 243/246/239、"
              "降下/K 22〜41、1 降下 820〜1120 評価（上限 12500 の 0.065〜0.090）、\n"
              "  打ち切りは全 1881 降下で tolxstagnation 1042（55.4%）／"
              "tolflatfitness 830（44.1%）／budget 9（0.5%）。")
        return
    print(f"{'問題':<13}{'seed':>6}{'降下本数':>10}{'降下/K':>9}"
          f"{'1 降下の評価':>14}{'上限 12500 比':>14}  打ち切り内訳")
    allstops: dict[str, int] = {}
    tot = 0
    for (f, s) in sorted(d):
        n, stops, ev = d[(f, s)]
        tot += n
        for kk, vv in stops.items():
            allstops[kk] = allstops.get(kk, 0) + vv
        inner = " ".join(f"{kk}={vv}" for kk, vv in sorted(stops.items()))
        print(f"{f:<13}{s:>6}{n:>10}{n / K[f]:>9.2f}{ev:>14.0f}"
              f"{ev / 12500:>14.3f}  {inner}")
    print(f"\n全 {tot} 降下の打ち切り内訳: "
          + " / ".join(f"{kk} {vv} 本（{100 * vv / tot:.1f}%）"
                       for kk, vv in sorted(allstops.items(), key=lambda t: -t[1])))


# ---------------------------------------------------------------------------
# §4 決定境界（追加評価ゼロ）。次のサイクルが NMMSO 側 9 run を足したとき、
# 10 問の対検定が α=0.05 に届きうるかを、**NMMSO の取りうる値を全部数えて**先に決める。
#
# 根拠: CEC2013 公式 PR は `det/K` なので **NMMSO の値は 1/K 刻みの有限集合**である
# （K=6 なら 7 通り、K=8 なら 9 通り）。RL 側は この回で確定したので、
# 残る自由度は追加 3 問の NMMSO の値だけ ＝ 7 x 9 x 7 = 441 通りを全部引ける。
# その166 の 7 問（seed 0）は `e166/baseline_all.csv` から読む。
# ---------------------------------------------------------------------------
E166 = os.path.join(os.path.dirname(HERE), "e166", "baseline_all.csv")


def _e166_seed0_mpr() -> list[float]:
    """その166 の 7 問の seed 0 対差（RL - NM、MPR）。

    **入力が無ければ黙って空を返さず、理由を印字して落ちる**（その169）——
    以前は `return []` で、`scan_silent_null.py` の走査 B に「黙って抜ける loader」として
    掛かっていた。この節は 7 対差が無ければ引けないので、空は帰無ではなく**欠損**である。
    """
    if not os.path.exists(E166):
        sys.exit(f"その166 の 42 run が無い（{E166}）。§4 の決定境界は "
                 "7 対差なしには引けない —— 数値は docs/acceptance_topology.md の "
                 "その167 §4 の節にある（勝ち 0/3 → 最小 p 0.250、3/3 → 0.03125）。")
    with open(E166) as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["seed"]) == 0]
    by: dict[str, dict[str, float]] = {}
    for r in rows:
        by.setdefault(r["function"], {})[r["method"]] = _mpr(r)
    return [by[f]["Restart-Lander"] - by[f]["NMMSO"]
            for f in sorted(by) if len(by[f]) == 2]


def power() -> None:
    from itertools import product

    from scipy.stats import wilcoxon

    base = _e166_seed0_mpr()
    print("\n## その167 §4 決定境界 —— 10 問で α=0.05 に届きうるか"
          "（NMMSO の取りうる値を全部数える。追加評価ゼロ）\n")
    if len(base) != 7:
        print("その166 の 7 問が読めない。§4 は引けない。")
        return
    print(f"その166（seed 0、MPR）の 7 対差: "
          + ", ".join(f"{d:+.4f}" for d in sorted(base, reverse=True)))

    rl = {"N11-CF1-2D": 4 / 6, "N12-CF2-2D": 5 / 8, "N13-CF3-2D": 4 / 6}
    grids = [[rl[f] - m / K[f] for m in range(K[f] + 1)] for f in sorted(rl)]

    best_p, best_combo, n_ok, n_tot = 1.0, None, 0, 0
    by_wins: dict[int, float] = {}
    for combo in product(*grids):
        d = base + list(combo)
        nz = [x for x in d if x != 0.0]
        n_tot += 1
        if len(nz) < 1:
            continue
        p = float(wilcoxon(nz, alternative="two-sided", mode="exact").pvalue) \
            if len(nz) <= 25 else float(wilcoxon(nz).pvalue)
        wins = sum(1 for x in combo if x > 0)
        by_wins[wins] = min(by_wins.get(wins, 1.0), p)
        if p < 0.05:
            n_ok += 1
        if p < best_p:
            best_p, best_combo = p, combo

    print(f"\n全 {n_tot} 通りのうち p < 0.05 になるのは **{n_ok} 通り**"
          f"（{100 * n_ok / n_tot:.1f}%）。**達成可能な最小 p = {best_p:.6f}**")
    if best_combo is not None:
        print("  その最小を与える追加 3 問の対差: "
              + ", ".join(f"{f} {d:+.4f}"
                          for f, d in zip(sorted(rl), best_combo)))
    print("\n追加 3 問での RL の勝ち数ごとの、達成可能な最小 p:")
    for w in sorted(by_wins):
        print(f"  勝ち {w}/3 -> 最小 p = {by_wins[w]:.6f}"
              + ("  ← α=0.05 に届きうる" if by_wins[w] < 0.05 else ""))


DEGRADED = False

if __name__ == "__main__":
    main()
    power()
    if DEGRADED:
        print("\n[exit 1] §3 の入力（`descents.csv.gz`）は畳まれている。"
              "§1 / §2 / §4 は保存物から引けており、上の出力は有効。")
        sys.exit(1)
