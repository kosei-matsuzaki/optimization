#!/usr/bin/env python3
"""その139 — 直した NMMSO（`swarm_size` 既定 = `10·D`）で新 suite を測り直す（キュー 1 の残り (c)）。

その116 の NMMSO Score 0.1548 は **旧既定 `swarm_size=10`（公表設定 `10·D` の 1/D 倍、その136）**
で取った数値だった。ここは**その 1 行だけを直して同じ条件で測り直した run** を、同じ採点器・
同じ報告規則で読む。

**採点・規則の定義は e115 からそのまま import する**（`rule_indices` / `score` / `paired` /
`aggregate` / `mean_of` / `SPAN` / `LEVEL_NAMES`）。**新しい統計量は 1 つも定義しない。**

入力の出自:

  * **NMMSO（新既定）** -> `e139/report_sets.csv.gz`（この回の報告集合ダンプ、座標つき、上限なし。
    **21 本の per-problem ダンプを `problem` / `seed` 列つきで 1 本に畳んだもの**。畳む前後で
    この script と `analyze_s1.py` の出力が 1 文字も変わらないことを確認してある）
  * **`Restart-Lander`** -> `e115/descents/`（seed 0、座標つき）を**再採点する**
    ＝ その116 と同じ経路。**再採点値が `e116/ranking_d10.csv` と 4 桁一致することを関門にする**
      （採点経路が当時と同一であることの確認。一致しなければ判定に進まない）。
  * **NMMSO（旧既定）/ MC-ESO** -> `e116/ranking_d10.csv` の `arm=legal` 行
    （ダンプは その120 が削除済み。**集計値は残っている**）。

使い方: python3 analysis/mmo2024/e139/analyze.py
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

from analyze import (LEVEL_NAMES, SPAN, aggregate, mean_of,        # noqa: E402
                     paired, read_dump, rule_indices)
from core.benchmarks import niching_by_name                        # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN     # その115 の合法な最良腕
# 事前登録した対照（prereg.md。16 問平均 Score、seed 0・PIN01・同じ規則）
REF_OLD_NMMSO = 0.1548          # その116、旧既定 swarm_size=10
REF_MCESO = 0.1385              # その116
REF_RL_SEED0 = 0.6284           # その116（seed 0）
REF_RL_3SEED = 0.6238           # その131（3 seed 平均）
BAND = 0.02                     # 反証条件 (b) の帯


def attribute(x: np.ndarray, optima: np.ndarray) -> np.ndarray:
    """最近傍の最適への帰属（`core.runner.count_goptima_nn` と同順序）。
    **オラクル量。採点にだけ使う。**（e116/analyze.py と同じ手続き）"""
    d = np.linalg.norm(x[:, None, :] - optima[None, :, :], axis=2)
    return np.argmin(d, axis=1)


REPORT_SETS = os.path.join(HERE, "report_sets.csv.gz")


def read_report_sets(seed=0):
    """畳んだ報告集合ダンプを `{問題: (f, x)}` で返す（`f` と座標のみ、上限なし）。

    採点に渡る中身は per-problem ダンプ時代と同一で、`problem` / `seed` 列が付いただけ
    （e116/analyze.py の `read_report_dump` と同じ読み口を 1 本にまとめた形）。"""
    out: dict = {}
    with gzip.open(REPORT_SETS, "rt") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return out
    dim = sum(1 for k in rows[0] if k.startswith("x") and k[1:].isdigit())
    for r in rows:
        if int(r["seed"]) != seed:
            continue
        f, x = out.setdefault(r["problem"], ([], []))
        f.append(float(r["f"]))
        x.append([float(r[f"x{i}"]) for i in range(dim)])
    return {p: (np.array(f), np.array(x)) for p, (f, x) in out.items()}


_BENCH: dict = {}


def bench(prob):
    if prob not in _BENCH:
        b = niching_by_name(prob)
        _BENCH[prob] = (int(b.n_global_optima),
                        np.asarray(b.optima_pos, dtype=float))
    return _BENCH[prob]


def load_runs():
    runs = []
    dd = os.path.join(MMO, "e115", "descents")           # Restart-Lander
    for fn in sorted(os.listdir(dd)) if os.path.isdir(dd) else []:
        if not fn.endswith((".csv", ".csv.gz")):
            continue
        prob = fn.split("_")[0]
        f, _opt, x = read_dump(os.path.join(dd, fn))
        if x is None:
            continue
        K, opts = bench(prob)
        runs.append(dict(problem=prob, method="Restart-Lander", seed=0, K=K,
                         f=f, x=x, opt=attribute(x, opts)))
    for prob, (f, x) in read_report_sets(seed=0).items():   # NMMSO（新既定）
        if len(f) == 0:
            continue
        K, opts = bench(prob)
        runs.append(dict(problem=prob, method="NMMSO-10D", seed=0, K=K,
                         f=f, x=x, opt=attribute(x, opts)))
    return runs


def stored_e116():
    """e116/ranking_d10.csv の `arm=legal` を {手法: {問題: 行}} で返す。"""
    out: dict = {}
    with open(os.path.join(MMO, "e116", "ranking_d10.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] != "legal":
                continue
            out.setdefault(r["method"], {})[r["problem"]] = dict(
                mpr=float(r["mpr"]), f1=float(r["mean_f1"]),
                score=float(r["score"]), n=float(r["n_reported"]))
    return out


def reported_counts(seed=0):
    """`runs.csv`（driver の 1 run 1 行、21 本を畳んだもの）から `n_reported`
    （＝ 手法の報告点数、上限なし）と予算を拾う。反証条件 (a) の (i)(ii) 用。"""
    out = {}
    with open(os.path.join(HERE, "runs.csv")) as fh:
        for r in csv.DictReader(fh):
            if int(r["seed"]) != seed:
                continue
            out[r["function"]] = dict(n_reported=float(r["n_reported"]),
                                      evals=int(r["evals"]),
                                      seed=int(r["seed"]))
    return out


def main() -> int:
    runs = load_runs()
    have = {m: {r["problem"] for r in runs if r["method"] == m}
            for m in ("Restart-Lander", "NMMSO-10D")}
    stored = stored_e116()
    probs = sorted(have["Restart-Lander"] & have["NMMSO-10D"]
                   & set(stored.get("NMMSO", {})) & set(stored.get("MC-ESO", {})))
    print("=" * 92)
    print("その139 — 直した NMMSO（swarm_size 既定 = 10·D）で新 suite を測り直す")
    print("=" * 92)
    print(f"  NMMSO（新既定）が揃った問題: {len(have['NMMSO-10D'])}/16")
    print(f"  4 系列が揃った問題: {len(probs)}/16"
          + ("   ** 部分結果 **" if len(probs) < 16 else ""))
    if not probs:
        print("\n  対が 1 問も揃っていない。集計しない。")
        return 1
    print(f"  規則: {ARM}  r = {ARM_R / SPAN:g} x span（その115 の合法な最良腕）")
    print(f"  予算: --evals-frac 1.0（正規予算）、seed 0、D=10、PIN01")

    arm_fn = lambda r: rule_indices(ARM, r["f"], r["K"], r["x"], ARM_R)  # noqa: E731
    agg = {}
    for m in ("Restart-Lander", "NMMSO-10D"):
        agg[m] = aggregate([r for r in runs if r["method"] == m
                            and r["problem"] in probs], arm_fn)

    # ---------------------------------------------------------------- 関門
    print("\n## 0. 関門 — 採点経路が その116 と同一か（`Restart-Lander` の再採点）\n")
    bad = []
    for p in probs:
        a, b = agg["Restart-Lander"][p]["score"], stored["Restart-Lander"][p]["score"]
        if abs(a - b) > 5e-5:
            bad.append((p, a, b))
    if bad:
        for p, a, b in bad:
            print(f"  ** ずれ ** {p}: 再採点 {a:.4f} 対 e116 記録 {b:.4f}")
        print("\n  採点経路が当時と違う。判定に進まない。")
        return 1
    print(f"  {len(probs)} 問すべてで `Restart-Lander` の Score が "
          f"e116/ranking_d10.csv と 4 桁一致 ==> 採点経路は同一。")

    # ---------------------------------------------------------------- 集計
    rows = {
        "NMMSO-10D（今回・新既定）": {p: agg["NMMSO-10D"][p] for p in probs},
        "NMMSO（その116・旧既定）": {p: stored["NMMSO"][p] for p in probs},
        "MC-ESO（その116）": {p: stored["MC-ESO"][p] for p in probs},
        "Restart-Lander（null）": {p: agg["Restart-Lander"][p] for p in probs},
    }
    print(f"\n## 1. {len(probs)} 問平均（5 水準等重み、seed 0 の 1 本）\n")
    print(f"{'系列':<28}{'MPR':>9}{'mean-F1':>10}{'Score':>9}{'n_rep':>9}")
    for k, g in rows.items():
        print(f"{k:<28}{mean_of(g, probs, 'mpr'):>9.4f}"
              f"{mean_of(g, probs, 'f1'):>10.4f}"
              f"{mean_of(g, probs, 'score'):>9.4f}{mean_of(g, probs, 'n'):>9.1f}")
    new = mean_of(rows["NMMSO-10D（今回・新既定）"], probs, "score")
    old = mean_of(rows["NMMSO（その116・旧既定）"], probs, "score")
    rl = mean_of(rows["Restart-Lander（null）"], probs, "score")

    # ---------------------------------------------------------------- 主判定
    print("\n## 2. 主判定 — NMMSO が `Restart-Lander` を上回る問題はあるか\n")
    beat = [p for p in probs
            if rows["NMMSO-10D（今回・新既定）"][p]["score"]
            > rows["Restart-Lander（null）"][p]["score"]]
    print(f"{'問題':<16}{'K':>4}{'NMMSO新':>10}{'NMMSO旧':>10}{'MC-ESO':>9}"
          f"{'null':>9}{'新-null':>10}{'新-旧':>9}")
    for p in probs:
        n, o = (rows["NMMSO-10D（今回・新既定）"][p]["score"],
                rows["NMMSO（その116・旧既定）"][p]["score"])
        m, r = (rows["MC-ESO（その116）"][p]["score"],
                rows["Restart-Lander（null）"][p]["score"])
        print(f"{p:<16}{bench(p)[0]:>4}{n:>10.4f}{o:>10.4f}{m:>9.4f}{r:>9.4f}"
              f"{n - r:>+10.4f}{n - o:>+9.4f}")
    print(f"\n  `Restart-Lander` を上回った問題: {len(beat)}/{len(probs)}"
          + (f"  ({', '.join(beat)})" if beat else ""))
    if beat:
        print("  ==> 「16/16 全敗」は崩れた。**実行役は主張を書き換えない。俯瞰に上げる。**")
    else:
        print("  ==> 直した NMMSO も全問で null に負ける ＝ 16/16 は崩れない。")
    print(f"\n  {len(probs)} 問平均 Score: NMMSO 新 {new:.4f}  対 null seed0 "
          f"{rl:.4f} / 3seed {REF_RL_3SEED:.4f}")

    print("\n  対検定（問題を対に、両側 Wilcoxon exact）")
    for k in ("NMMSO-10D（今回・新既定）", "NMMSO（その116・旧既定）", "MC-ESO（その116）"):
        r = paired({p: rows["Restart-Lander（null）"][p]["score"] for p in probs},
                   {p: rows[k][p]["score"] for p in probs}, probs)
        print(f"    null - {k:<28} {r['mean']:+.4f}  {r['w']}/{r['t']}/{r['l']}"
              f"  p={r['p']:.4g}  rb={r['rb']:+.3f}")

    # ------------------------------------------------------- 反証条件 (a)(b)
    print("\n## 3. 事前登録した反証条件\n")
    d_old = new - old
    print(f"  (a) NMMSO が下がったか: 新 {new:.4f} − 旧 {old:.4f} = {d_old:+.4f}"
          f"  ==> {'発火（下がった）' if d_old < 0 else '不発（下がっていない）'}")
    r = paired({p: rows["NMMSO-10D（今回・新既定）"][p]["score"] for p in probs},
               {p: rows["NMMSO（その116・旧既定）"][p]["score"] for p in probs}, probs)
    print(f"      新 − 旧（問題を対に）: {r['mean']:+.4f}  {r['w']}/{r['t']}/{r['l']}"
          f"  p={r['p']:.4g}  rb={r['rb']:+.3f}")
    inband = abs(new - REF_OLD_NMMSO) <= BAND
    print(f"  (b) 帯 [{REF_OLD_NMMSO - BAND:.4f}, {REF_OLD_NMMSO + BAND:.4f}] の内側か: "
          f"{new:.4f} ==> {'発火（内側 ＝ この路線を閉じる）' if inband else '不発（外側）'}")

    # --------------------------------------------- 予算・報告の計器（(a) 用）
    print("\n## 4. 予算と報告点数の計器（反証条件 (a) の (i)(ii)）\n")
    cnt = reported_counts()
    print(f"{'問題':<16}{'予算':>9}{'seed':>6}{'報告点数(上限なし)':>20}"
          f"{'規則適用後 n':>14}{'旧既定 n':>11}")
    for p in probs:
        c = cnt.get(p, {})
        print(f"{p:<16}{c.get('evals', 0):>9}{c.get('seed', -1):>6}"
              f"{c.get('n_reported', float('nan')):>20.0f}"
              f"{rows['NMMSO-10D（今回・新既定）'][p]['n']:>14.1f}"
              f"{rows['NMMSO（その116・旧既定）'][p]['n']:>11.1f}")

    print("\n## 5. 水準別 PR / F1（問題平均）\n")
    print(f"{'系列':<28}{'量':<5}" + "".join(f"{l:>9}" for l in LEVEL_NAMES))
    for k in ("NMMSO-10D（今回・新既定）", "Restart-Lander（null）"):
        g = rows[k]
        if "pr_lv" not in next(iter(g.values())):
            continue
        pr = np.mean([g[p]["pr_lv"] for p in probs], axis=0)
        f1 = np.mean([g[p]["f1_lv"] for p in probs], axis=0)
        print(f"{k:<28}{'PR':<5}" + "".join(f"{v:>9.4f}" for v in pr))
        print(f"{'':<28}{'F1':<5}" + "".join(f"{v:>9.4f}" for v in f1))

    out = os.path.join(HERE, "by_problem.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "series", "n_reported", "mpr", "mean_f1",
                    "score"])
        for k, g in rows.items():
            for p in probs:
                w.writerow([p, bench(p)[0], k, f"{g[p]['n']:.1f}",
                            f"{g[p]['mpr']:.4f}", f"{g[p]['f1']:.4f}",
                            f"{g[p]['score']:.4f}"])
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
