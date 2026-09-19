#!/usr/bin/env python3
"""その140 — キュー 1 の残り: M01 / M02 / M04 / M10 の符号を 5 seed で確定させる。

その139 は seed 0 で 4 問、seed 1 で 3 問において **直した NMMSO（`swarm_size` 既定 `10·D`）が
記憶なし多スタート `Restart-Lander` を上回る**と出したが、**差はどれも ±0.05 以内**で、
**NMMSO は seed を固定しても run 間で完全再現しない**（環境節）。**1-2 run では符号を主張できない。**
ここは seed を 5 本に増やし、**事前登録した「正の seed が 4/5 以上」の規則で数え直すだけ。**

**採点・規則・統計量は `e115/analyze.py` からそのまま import する**（`rule_indices` / `score` /
`aggregate` / `paired` / `SPAN`）。**新しい統計量は 1 つも定義しない。**

入力の出自:

  * NMMSO seed 0 / 1  -> `e139/report_sets.csv.gz`（その139 が畳んだ報告集合ダンプ）
  * NMMSO seed 2/3/4  -> `e140/dumps/`（この回の `REPORT_SET_DUMP`。**新規 run 12 本**）
  * null  seed 0      -> `e115/descents/`（再採点。追加評価ゼロ）
  * null  seed 100    -> `e115/s1/descents/`（同上。その139 が「seed 1」と呼んだ列）
  * null  seed 2/3/4  -> `e140/descents/`（この回の `RESTART_LANDER_DUMP`。**新規 run 12 本**）

**【その143 で保持整理】この回の行単位の入力（`e140/report_sets.csv.gz` /
`e140/descents.csv.gz` と per-run の `dumps/` `descents/`）は路線を畳んだ際に削除した。
集計は `by_seed.csv` / `means_5seed.csv` に残っており、結論は
`docs/acceptance_topology.md` の その140 の節にある（bytes は git 履歴）。
＝ **この script はそのままでは再実行できない。**

使い方: python3 analysis/mmo2024/e140/analyze.py
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

import importlib.util                                               # noqa: E402

from analyze import SPAN, aggregate, read_dump, rule_indices        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "e139_analyze", os.path.join(MMO, "e139", "analyze.py"))
_e139 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e139)

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN
PROBS = [f"M{n}-D10-PIN01" for n in ("01", "02", "04", "10")]
NEW_SEEDS = (2, 3, 4)

# その139 が記録した seed 0 の Score（関門 (b) の照合先。scored.txt §2）
GATE_S0 = {"M01-D10-PIN01": (0.7607, 0.7600), "M02-D10-PIN01": (0.6421, 0.6069),
           "M04-D10-PIN01": (0.7618, 0.7189), "M10-D10-PIN01": (0.7600, 0.6529)}
# その139 が記録した seed 1 の NMMSO / null（seed1_check.csv）
GATE_S1_NULL = {"M01-D10-PIN01": None}  # 値は seed1_check.csv から読む


def read_report_dump(path):
    """`REPORT_SET_DUMP` の per-problem ダンプを (f, x) で読む（上限なし、座標つき）。"""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return np.array([]), np.zeros((0, 0))
    dim = sum(1 for k in rows[0] if k.startswith("x") and k[1:].isdigit())
    f = np.array([float(r["f"]) for r in rows])
    x = np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rows])
    return f, x


def read_folded(path, fcol):
    """畳んだ 1 本（`problem` / `seed` 列つき）を [(問題, seed, f, x)] で返す。"""
    with gzip.open(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    dim = sum(1 for k in rows[0] if k.startswith("x") and k[1:].isdigit())
    acc: dict = {}
    for r in rows:
        key = (r["problem"], int(r["seed"]))
        f, x = acc.setdefault(key, ([], []))
        f.append(float(r[fcol]))
        x.append([float(r[f"x{i}"]) for i in range(dim)])
    return [(p, s, np.array(f), np.array(x)) for (p, s), (f, x) in acc.items()]


def read_new(folded, dirname, reader):
    """この回の run を読む。**畳んだ 1 本があればそちらを使う**（per-problem は畳んだら消す）。

    畳む前後で採点に渡る中身は同一で、`problem` / `seed` 列が付くだけ。
    `analyze.py` の出力が 1 文字も変わらないことを確認してから per-problem を消すこと。"""
    fp = os.path.join(HERE, folded)
    if os.path.exists(fp):
        fcol = "best_f" if "descent" in folded else "f"
        return [(p, s, f, x) for (p, s, f, x) in read_folded(fp, fcol)
                if p in PROBS and s in NEW_SEEDS]
    out = []
    dd = os.path.join(HERE, dirname)
    for fn in sorted(os.listdir(dd)) if os.path.isdir(dd) else []:
        prob = fn.split("_")[0]
        sd = int(fn.rsplit("seed", 1)[1].split(".")[0])
        # **降下ダンプのファイル名の seed は `seed_index x 100`**（`niching_baseline.py:281`、
        # その123 の罠）。報告集合ダンプは index そのもの。ここで index に揃える。
        if sd >= 100:
            sd //= 100
        if prob not in PROBS or sd not in NEW_SEEDS:
            continue
        f, x = reader(os.path.join(dd, fn))
        if x is not None and len(f):
            out.append((prob, sd, f, x))
    return out


def mk(prob, method, seed, f, x):
    K, opts = _e139.bench(prob)
    return dict(problem=prob, method=method, seed=seed, K=K,
                f=f, x=x, opt=_e139.attribute(x, opts))


def load_runs():
    runs = []
    # --- NMMSO seed 0 / 1（その139 の畳んだダンプ）
    for sd in (0, 1):
        for prob, (f, x) in _e139.read_report_sets(seed=sd).items():
            if prob in PROBS and len(f):
                runs.append(mk(prob, "NMMSO-10D", sd, f, x))
    # --- NMMSO seed 2/3/4（この回。畳んだ 1 本があればそちら、無ければ per-problem）
    for prob, sd, f, x in read_new("report_sets.csv.gz", "dumps",
                                   read_report_dump):
        runs.append(mk(prob, "NMMSO-10D", sd, f, x))
    # --- null seed 0 / 100（保存ダンプ。追加評価ゼロ）
    for dd, sd in ((os.path.join(MMO, "e115", "descents"), 0),
                   (os.path.join(MMO, "e115", "s1", "descents"), 1)):
        for fn in sorted(os.listdir(dd)) if os.path.isdir(dd) else []:
            prob = fn.split("_")[0]
            if prob not in PROBS:
                continue
            f, _o, x = read_dump(os.path.join(dd, fn))
            if x is not None:
                runs.append(mk(prob, "Restart-Lander", sd, f, x))
    # --- null seed 2/3/4（この回。同上）
    for prob, sd, f, x in read_new("descents.csv.gz", "descents",
                                   lambda pth: read_dump(pth)[::2]):
        runs.append(mk(prob, "Restart-Lander", sd, f, x))
    return runs


def per_seed(runs, arm_fn):
    """{(手法, 問題, seed): 集計} —— aggregate を 1 run ずつに当てるだけ。"""
    out = {}
    for r in runs:
        a = aggregate([r], arm_fn)[r["problem"]]
        out[(r["method"], r["problem"], r["seed"])] = a
    return out


def main() -> int:
    arm_fn = lambda r: rule_indices(ARM, r["f"], r["K"], r["x"], ARM_R)  # noqa: E731
    runs = load_runs()
    ps = per_seed(runs, arm_fn)

    print("=" * 92)
    print("その140 — M01/M02/M04/M10 で NMMSO が null を上回るか、5 seed で符号を確定させる")
    print("=" * 92)

    # ---- 関門 (b): null seed 0 の再採点が その139 の記録と 4 桁一致するか
    print("\n[関門 (b)] `Restart-Lander` seed 0 の再採点 対 その139 の記録")
    ok = True
    for p in PROBS:
        k = ("Restart-Lander", p, 0)
        if k not in ps:
            print(f"  {p:<16} 欠測"); ok = False; continue
        got, want = ps[k]["score"], GATE_S0[p][1]
        hit = abs(got - want) < 5e-5
        ok &= hit
        print(f"  {p:<16} 再採点 {got:.4f}  記録 {want:.4f}  "
              + ("一致" if hit else "** 不一致 **"))
    print(f"  ==> 関門 {'合格' if ok else '不合格（判定に進まない）'}")
    if not ok:
        return 1

    # ---- seed ごとの Score 差
    # **反復の順序は固定して書く**（sorted に任せると null の 100 が末尾に来て対がずれる）。
    # 2 手法の seed 値は揃わないので、対にしているのは「k 本目の独立な反復」であって
    # 同じ乱数列ではない ＝ **どの反復とどの反復を並べるかは任意**。
    # 事前登録の「正の seed が 4/5 以上」はこの任意の並べ方に依存するので、
    # 下で 5 本の min/max も出して重なりが見えるようにする。
    # seed 番号は **index**（`niching_baseline.py` の「seed index i ＝ optimiser seed i*100」）。
    ORDER_N, ORDER_R = [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]
    have = lambda m, o: [s for s in o                                 # noqa: E731
                         if any((m, p, s) in ps for p in PROBS)]
    seeds_n, seeds_r = have("NMMSO-10D", ORDER_N), have("Restart-Lander", ORDER_R)
    print(f"\n  NMMSO の反復: {seeds_n}    null の反復: {seeds_r}")
    print("  （2 手法の seed 値は揃わない ＝ 対にするのは問題であって seed 値ではない。"
          "k 本目どうしを並べる）")
    pairs = list(zip(seeds_n, seeds_r))
    print(f"  対にする k 本目: {pairs}")

    rows, verdict = [], {}
    print(f"\n{'問題':<16}" + "".join(f"{'差 s%d' % k:>10}" for k in range(len(pairs)))
          + f"{'平均差':>10}{'正の数':>8}{'判定':>8}")
    for p in PROBS:
        diffs = []
        for sn, sr in pairs:
            a, b = ps.get(("NMMSO-10D", p, sn)), ps.get(("Restart-Lander", p, sr))
            diffs.append(None if (a is None or b is None) else a["score"] - b["score"])
        got = [d for d in diffs if d is not None]
        npos = sum(1 for d in got if d > 0)
        mean = float(np.mean(got)) if got else float("nan")
        up = len(got) >= 5 and npos >= 4
        verdict[p] = dict(n=len(got), npos=npos, mean=mean, up=up, diffs=diffs)
        cells = "".join(("%+10.4f" % d) if d is not None else f"{'-':>10}"
                        for d in diffs)
        print(f"{p:<16}{cells}{mean:>+10.4f}{npos:>4}/{len(got):<3}"
              + f"{('上回る' if up else ('—' if len(got) >= 5 else '未了')):>8}")
        rows.append([p, len(got), npos, f"{mean:+.4f}", "up" if up else "no"]
                    + [("" if d is None else f"{d:+.6f}") for d in diffs])

    # ---- 主判定と反証条件 (a)
    done = [p for p in PROBS if verdict[p]["n"] >= 5]
    ups = [p for p in done if verdict[p]["up"]]
    print(f"\n  揃った問題: {len(done)}/{len(PROBS)}"
          + ("   ** 部分結果 **" if len(done) < len(PROBS) else ""))
    print(f"  主判定（正の seed が 4/5 以上）: {len(ups)} 問" 
          + (f"  ({', '.join(ups)})" if ups else ""))
    core3 = [p for p in ("M01-D10-PIN01", "M04-D10-PIN01", "M10-D10-PIN01")
             if p in done]
    strad = [p for p in core3 if not (verdict[p]["mean"] > 0
                                      and verdict[p]["npos"] >= 4)]
    print(f"  反証条件 (a)（その139 の 3 問とも 5 seed 平均が 0 をまたぐ）: "
          + ("発火" if len(core3) == 3 and all(verdict[p]["mean"] <= 0 for p in core3)
             else "不発")
          + f"   （4/5 に届かない 3 問中: {len(strad)}）")

    # ---- 5 seed 平均の Score（内訳つき）
    print(f"\n{'問題':<16}{'NMMSO Score':>13}{'null Score':>12}{'NMMSO MPR':>11}"
          f"{'null MPR':>10}{'NMMSO F1':>10}{'null F1':>9}{'NMMSO 報告':>11}{'null 報告':>10}")
    detail = []
    for p in PROBS:
        def m(meth, seeds, key):
            v = [ps[(meth, p, s)][key] for s in seeds if (meth, p, s) in ps]
            return float(np.mean(v)) if v else float("nan")
        r = [m("NMMSO-10D", seeds_n, "score"), m("Restart-Lander", seeds_r, "score"),
             m("NMMSO-10D", seeds_n, "mpr"), m("Restart-Lander", seeds_r, "mpr"),
             m("NMMSO-10D", seeds_n, "f1"), m("Restart-Lander", seeds_r, "f1"),
             m("NMMSO-10D", seeds_n, "n"), m("Restart-Lander", seeds_r, "n")]
        print(f"{p:<16}{r[0]:>13.4f}{r[1]:>12.4f}{r[2]:>11.4f}{r[3]:>10.4f}"
              f"{r[4]:>10.4f}{r[5]:>9.4f}{r[6]:>11.1f}{r[7]:>10.1f}")
        detail.append([p] + [f"{v:.4f}" for v in r])
    print("\n  5 本の Score の範囲（対の並べ方に依らない読み。重なっていれば「勝つ」とは書けない）")
    print(f"{'問題':<16}{'NMMSO min..max':>24}{'null min..max':>24}{'重なり':>8}")
    for p in PROBS:
        a = [ps[("NMMSO-10D", p, s)]["score"] for s in seeds_n
             if ("NMMSO-10D", p, s) in ps]
        b = [ps[("Restart-Lander", p, s)]["score"] for s in seeds_r
             if ("Restart-Lander", p, s) in ps]
        if not a or not b:
            continue
        lap = "有" if (min(a) <= max(b) and min(b) <= max(a)) else "無"
        print(f"{p:<16}{f'{min(a):.4f} .. {max(a):.4f}':>24}"
              f"{f'{min(b):.4f} .. {max(b):.4f}':>24}{lap:>8}")

    out = os.path.join(HERE, "by_seed.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "n_pairs", "n_positive", "mean_diff", "verdict"]
                   + [f"diff_k{k}" for k in range(len(pairs))])
        w.writerows(rows)
    out2 = os.path.join(HERE, "means_5seed.csv")
    with open(out2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "nmmso_score", "null_score", "nmmso_mpr",
                    "null_mpr", "nmmso_f1", "null_f1", "nmmso_nrep", "null_nrep"])
        w.writerows(detail)
    print(f"\n  -> {out}\n  -> {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
