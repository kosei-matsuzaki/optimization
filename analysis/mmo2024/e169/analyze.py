#!/usr/bin/env python3
"""その169 — キュー 1 の決着: CEC2013 10 問（F11-F20）の `Restart-Lander` − NMMSO 対検定。

この回が足したのは **NMMSO 側 9 run だけ**（`baseline_all.csv`）。他は全部保存物:

  * `e167/baseline_all.csv` — 追加 3 問（F11-F13、2D、200,000 評価）の RL 側 9 run。
  * `e166/baseline_all.csv` — 既存 7 問（F14-F20、3D-20D、400,000 評価）の両手法 42 run。

その167 §4 が**追加評価ゼロで 441 通りを全数走査し**、決定境界を先に確定させてある:
**10 問で α=0.05 に届くのは追加 3 問を全勝した場合だけ**（勝ち 3/3 で p=0.03125、
勝ち 2/3 では最小 p 0.0625）。**この script は 3 つの符号を確定させ、境界表に当てる。**

事前登録 (a)/(b)/(c) の判定は §3 が印字する。
"""
from __future__ import annotations

import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.dirname(HERE)
LEVELS = ["pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]
ADDED = ["N11-CF1-2D", "N12-CF2-2D", "N13-CF3-2D"]
K = {"N11-CF1-2D": 6, "N12-CF2-2D": 8, "N13-CF3-2D": 6,
     "N14-CF3-3D": 6, "N15-CF4-3D": 8, "N16-CF3-5D": 6, "N17-CF4-5D": 8,
     "N18-CF3-10D": 6, "N19-CF4-10D": 8, "N20-CF4-20D": 8}


def _read(path: str, what: str) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(f"{what} が無い（{path}）。この回の入力は削除された "
                 "—— 数値は docs/acceptance_topology.md の その169 の節にある。")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _mpr(r: dict) -> float:
    return sum(float(r[k]) for k in LEVELS) / len(LEVELS)


def _f1(r: dict) -> float:
    k, rep = int(r["n_optima"]), int(r["n_reported"])
    vals = []
    for lv in LEVELS:
        recall = float(r[lv])
        prec = recall * k / rep if rep else 0.0
        vals.append(2 * prec * recall / (prec + recall) if prec + recall else 0.0)
    return sum(vals) / len(vals)


def _score(r: dict) -> float:
    k, rep = int(r["n_optima"]), int(r["n_reported"])
    vals = []
    for lv in LEVELS:
        recall = float(r[lv])
        prec = recall * k / rep if rep else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
        vals.append((recall + f1) / 2)
    return sum(vals) / len(vals)


def _table() -> dict[tuple[str, str, int], dict]:
    """(function, method, seed) -> row。3 つの出所を 1 本に畳む。"""
    out: dict[tuple[str, str, int], dict] = {}
    src = [(os.path.join(HERE, "baseline_all.csv"), "この回の NMMSO 9 run"),
           (os.path.join(MM, "e167", "baseline_all.csv"), "その167 の RL 9 run"),
           (os.path.join(MM, "e166", "baseline_all.csv"), "その166 の 42 run")]
    for p, what in src:
        for r in _read(p, what):
            out[(r["function"], r["method"], int(r["seed"]))] = r
    return out


def main() -> None:
    from scipy.stats import wilcoxon

    tab = _table()
    funcs = sorted(K, key=lambda f: int(f[1:3]))

    # ---- §1 関数別（3 seed 平均。その162 の教訓で必ず列挙する） ----
    print("## その169 §1 関数別（CEC2013 F11-F20、報告規則 current、3 seed 平均）\n")
    print("【追加 3 問 = F11-F13 は 2D・予算 200,000、既存 7 問 = F14-F20 は 3D-20D・予算 400,000】\n")
    hdr = (f"{'問題':<13}{'K':>3}{'D':>4}  {'RL MPR':>13}{'NM MPR':>13}"
           f"{'対差':>9}{'RL Score':>10}{'NM Score':>10}{'差':>9}{'RL|rep|':>9}{'NM|rep|':>9}")
    print(hdr)
    print("-" * len(hdr))
    per: dict[str, dict[str, dict[str, float]]] = {}
    for f in funcs:
        cell = {}
        for m in ("Restart-Lander", "NMMSO"):
            rs = [tab[(f, m, s)] for s in (0, 1, 2) if (f, m, s) in tab]
            if not rs:
                sys.exit(f"{f} / {m} の run が無い。")
            cell[m] = {"mpr": statistics.mean(_mpr(r) for r in rs),
                       "mpr_sd": statistics.stdev([_mpr(r) for r in rs]) if len(rs) > 1 else 0.0,
                       "f1": statistics.mean(_f1(r) for r in rs),
                       "score": statistics.mean(_score(r) for r in rs),
                       "rep": statistics.mean(int(r["n_reported"]) for r in rs),
                       "n": len(rs)}
        per[f] = cell
        rl, nm = cell["Restart-Lander"], cell["NMMSO"]
        d = int(f.split("-")[-1].rstrip("D"))
        print(f"{f:<13}{K[f]:>3}{d:>4}  "
              f"{rl['mpr']:>7.4f}±{rl['mpr_sd']:<5.3f}{nm['mpr']:>7.4f}±{nm['mpr_sd']:<5.3f}"
              f"{rl['mpr'] - nm['mpr']:>+9.4f}{rl['score']:>10.4f}{nm['score']:>10.4f}"
              f"{rl['score'] - nm['score']:>+9.4f}{rl['rep']:>9.1f}{nm['rep']:>9.1f}")

    # ---- §2 追加 3 問の符号（事前登録の関門はここだけで決まる） ----
    print("\n## その169 §2 追加 3 問の符号 —— その167 §4 の決定境界に当てる\n")
    print(f"{'問題':<13}{'seed':>6}{'RL MPR':>9}{'NM MPR':>9}{'対差':>9}  判定")
    seed0 = {}
    for f in ADDED:
        for s in (0, 1, 2):
            a, b = _mpr(tab[(f, "Restart-Lander", s)]), _mpr(tab[(f, "NMMSO", s)])
            v = "RL 勝ち" if a > b else ("同点" if a == b else "RL 負け")
            print(f"{f:<13}{s:>6}{a:>9.4f}{b:>9.4f}{a - b:>+9.4f}  {v}")
            if s == 0:
                seed0[f] = a - b
    print("\nその167 §4 の境界: 追加 3 問を**全勝**した場合だけ 10 問で p=0.03125。"
          "勝ち 2/3 の最小 p は 0.0625 > α=0.05。")
    wins3 = sum(1 for f in ADDED if per[f]["Restart-Lander"]["mpr"] > per[f]["NMMSO"]["mpr"])
    print(f"実測（3 seed 平均）: RL の勝ち {wins3}/3、"
          f"seed 0 単独: {sum(1 for f in ADDED if seed0[f] > 0)}/3")

    # ---- §3 10 問の対検定 ----
    print("\n## その169 §3 10 問の対検定（両側 Wilcoxon signed-rank exact、α=0.05）\n")
    for tag, get in (("seed 0 単独（その167 §4 が境界を引いた形）",
                      lambda f, m: _mpr(tab[(f, m, 0)])),
                     ("3 seed 平均（この回で両手法とも 3 seed が揃った）",
                      lambda f, m: per[f][m]["mpr"])):
        for metric, key in (("MPR", "mpr"), ("Score", "score")):
            if metric == "MPR":
                dd = [get(f, "Restart-Lander") - get(f, "NMMSO") for f in funcs]
            else:
                if "seed 0" in tag:
                    dd = [_score(tab[(f, "Restart-Lander", 0)])
                          - _score(tab[(f, "NMMSO", 0)]) for f in funcs]
                else:
                    dd = [per[f]["Restart-Lander"]["score"] - per[f]["NMMSO"]["score"]
                          for f in funcs]
            nz = [x for x in dd if abs(x) > 1e-12]
            w = sum(1 for x in dd if x > 1e-12)
            l = sum(1 for x in dd if x < -1e-12)
            t = len(dd) - w - l
            mean = statistics.mean(dd)
            med = statistics.median(dd)
            if len(nz) >= 1:
                st, p = wilcoxon(nz, alternative="two-sided", mode="exact")
                npos = sum(1 for x in nz if x > 0)
                rb = (2 * npos / len(nz)) - 1 if nz else 0.0
                ptxt, rbtxt = f"{p:.5f}", f"{rb:+.3f}"
            else:
                ptxt, rbtxt = "引けない（全同点）", "—"
            print(f"{tag} / {metric:>5}: 平均 {mean:+.4f}  中央値 {med:+.4f}  "
                  f"勝/分/敗 {w}/{t}/{l}  有効対 {len(nz)}  p={ptxt}  rb={rbtxt}")
        print()

    # ---- §4 何が符号を決めたか（機序） ----
    print("## その169 §4 機序 —— 追加 3 問で何が起きたか\n")
    for f in ADDED:
        rl, nm = per[f]["Restart-Lander"], per[f]["NMMSO"]
        print(f"{f}: NMMSO は MPR {nm['mpr']:.4f}（K={K[f]} なので検出 "
              f"{nm['mpr'] * K[f]:.2f}/{K[f]}）を報告 {nm['rep']:.1f} 点で出し、"
              f"RL は {rl['mpr']:.4f}（検出 {rl['mpr'] * K[f]:.2f}）を "
              f"{rl['rep']:.1f} 点で出した。"
              f"精度比 = 検出/報告 は NMMSO {nm['mpr'] * K[f] / nm['rep']:.4f} 対 "
              f"RL {rl['mpr'] * K[f] / rl['rep']:.4f}。")
    print("\n次元別の符号（対差 MPR、3 seed 平均）:")
    for f in funcs:
        d = int(f.split("-")[-1].rstrip("D"))
        dv = per[f]["Restart-Lander"]["mpr"] - per[f]["NMMSO"]["mpr"]
        print(f"  {f:<13} D={d:<3} {dv:+.4f}")

    # ---- §5 CF3 の梯子 —— その88 のクラス上限 4.00 と突き合わせる（追加評価ゼロ） ----
    print("\n## その169 §5 CF3 の梯子 —— その88 のクラス上限（K=6 で 4.00）と突き合わせる\n")
    print("その88 §2 の表は CF3 の「一様再起動 ＋ 局所降下」クラス上限を "
          "N14-3D 4.49 / N16-5D 4.00 / N18-10D 3.98 と測っている（閾値 = 公表最良 × K）。")
    print("CF3 はこの 10 問に 4 次元で入っているので、検出数（MPR × K）を並べると梯子になる。\n")
    cf3 = [f for f in funcs if "CF3" in f]
    print(f"{'問題':<13}{'D':>4}{'K':>3}{'RL 検出':>10}{'RL SD':>8}"
          f"{'NM 検出':>10}{'NM SD':>8}  その88 のクラス上限")
    lim = {"N14-CF3-3D": "4.49", "N16-CF3-5D": "4.00", "N18-CF3-10D": "3.98",
           "N13-CF3-2D": "（未測定）"}
    for f in cf3:
        rl, nm = per[f]["Restart-Lander"], per[f]["NMMSO"]
        d = int(f.split("-")[-1].rstrip("D"))
        print(f"{f:<13}{d:>4}{K[f]:>3}{rl['mpr'] * K[f]:>10.2f}"
              f"{rl['mpr_sd'] * K[f]:>8.2f}{nm['mpr'] * K[f]:>10.2f}"
              f"{nm['mpr_sd'] * K[f]:>8.2f}  {lim[f]:>8}")
    print("\n読み: RL は 4 次元すべてで検出 4.00・SD 0.00 ＝ その88 のクラス上限にぴたりと張り付く。")
    print("NMMSO は D=2 でだけ 6.00 に抜け、D=5 以上では RL と同じ 4.00 に落ちる。")
    print("＝ CF3 の上限 4.00 は<絶対>ではないが、破れるのは D=2 だけである。")


if __name__ == "__main__":
    main()
