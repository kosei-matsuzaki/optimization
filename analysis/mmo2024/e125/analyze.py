#!/usr/bin/env python3
"""その125 — 再投入腕 / 素の basin-hopping / 一様 null を、その116 と同じ合法規則で採点する。

**新しい規則も新しい統計量も 1 つも定義しない。** `rule_indices` / `score` / `paired` は
e115 の `analyze.py` からそのまま import する。違うのは入力（どの降下ダンプか）だけ。

  * `null`   -> analysis/mmo2024/e115/descents（seed 0、座標つき、保存物。追加 run ゼロ）
  * `reseed` -> analysis/mmo2024/e125/descents_reseed_seed0.csv.gz（その131 が 16 本を畳んだ）
  * `bhop`   -> analysis/mmo2024/e125/descents_bhop_seed0.csv.gz（同上）

**関門**: `null` をこのコードで採点し直した 16 問平均が、その116 の記録
（MPR 0.5644 / Score 0.6284）と 4 桁一致すること。外れたら採点経路が壊れているので判定に進まない。

使い方: python3 analysis/mmo2024/e125/analyze.py
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
GATE = {"mpr": 0.5644, "score": 0.6284}          # その116 の記録（再現の関門）
PROBS = [f"M{i:02d}-D10-PIN01" for i in range(1, 17)]
GROUP_A = PROBS[:8]                              # 塊のある側（その118）
GROUP_B = PROBS[8:]
NULL_DIR = os.path.join(MMO, "e115", "descents")
ARM_DIR = os.path.join(HERE, "descents")
# **その131 の片付けで、腕のダンプ 32 本は腕ごとに 1 本の combined へ畳んだ**
# （`descents_reseed_seed0.csv.gz` / `descents_bhop_seed0.csv.gz`、`problem` 列つき）。
# `ARM_DIR` の per-problem が残っていればそちらを使い、無ければ combined から読む。
# **畳む前後で本 script の出力が 1 文字も変わらないことは その131 が確認済み。**
COMBINED_ARM = {arm: os.path.join(HERE, f"descents_{arm}_seed0.csv.gz")
                for arm in ("reseed", "bhop")}

_K: dict = {}
_COMB: dict = {}


def K_of(prob):
    if prob not in _K:
        _K[prob] = int(niching_by_name(prob).n_global_optima)
    return _K[prob]


def find(d, *cands):
    for c in cands:
        for ext in ("", ".gz"):
            p = os.path.join(d, c + ext)
            if os.path.exists(p):
                return p
    return None


def arm_rows(prob, arm):
    """畳んだ combined から 1 問ぶんの生の行を返す（入出力の写し。統計量は含まない）。"""
    if arm not in _COMB:
        path = COMBINED_ARM.get(arm)
        out: dict = {}
        if path and os.path.exists(path):
            with gzip.open(path, "rt") as fh:
                for r in csv.DictReader(fh):
                    out.setdefault(r.pop("problem"), []).append(r)
        _COMB[arm] = out
    return _COMB[arm].get(prob)


def load(prob, arm):
    if arm == "null":
        p = find(NULL_DIR, f"{prob}_seed0.csv")
    else:
        p = find(ARM_DIR, f"{prob}_{arm}_seed0.csv")
    if p is None and arm != "null":
        rows = arm_rows(prob, arm)
        if not rows:
            return None
        dim = sum(1 for k in rows[0] if k.startswith("x") and k[1:].isdigit())
        f = np.array([float(r["best_f"]) for r in rows])
        opt = np.array([int(r["land_opt"]) for r in rows])
        xs = np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rows])
        idx = rule_indices(ARM, f, K_of(prob), x=xs, r=ARM_R)
        recall, prec, f1, sc, n = score(idx, f, opt, K_of(prob))
        return dict(mpr=float(recall.mean()), f1=float(f1.mean()),
                    score=float(sc.mean()), n=int(n), ndesc=len(f),
                    pr_lv=recall, path=COMBINED_ARM[arm])
    if p is None:
        return None
    f, opt, xs = read_dump(p)
    idx = rule_indices(ARM, f, K_of(prob), x=xs, r=ARM_R)
    recall, prec, f1, sc, n = score(idx, f, opt, K_of(prob))
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()),
                score=float(sc.mean()), n=int(n), ndesc=len(f),
                pr_lv=recall, path=p)


def rho_trace(prob, arm):
    """腕の計器（`mode` / `rho` 列）。腕の中身が想定どおり動いたかの確認だけに使う。"""
    p = find(ARM_DIR, f"{prob}_{arm}_seed0.csv")
    if p is None:
        rows = arm_rows(prob, arm)          # 畳んだ combined からの退避路（その131）
    else:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as fh:
            rows = list(csv.DictReader(fh))
    if not rows or "mode" not in rows[0]:
        return None
    modes = [r["mode"] for r in rows]
    rho = [float(r["rho"]) for r in rows if r["rho"] not in ("nan", "")]
    return dict(n=len(rows), uniform=modes.count("uniform"),
                perturbed=len(modes) - modes.count("uniform"),
                rho_med=float(np.median(rho)) if rho else float("nan"),
                rho_last=rho[-1] if rho else float("nan"))


def main():
    arms = ["null", "reseed", "bhop"]
    data = {a: {} for a in arms}
    for p in PROBS:
        for a in arms:
            v = load(p, a)
            if v is not None:
                data[a][p] = v
    # 3 腕すべてが揃った問題だけで対にする（事前登録の打ち切り規則）
    probs = [p for p in PROBS if all(p in data[a] for a in arms)]
    missing = [p for p in PROBS if p not in probs]

    print("=" * 90)
    print("その125 — 着地点近傍への再投入は採点者の情報なしでも利得が残るか（キュー 2）")
    print("=" * 90)
    print(f"\n  規則: {ARM} (r={ARM_R})   3 腕が揃った問題: {len(probs)}/16")
    if missing:
        print(f"  **欠けた問題: {', '.join(missing)}**  "
              f"(揃っている本数: " +
              ", ".join(f"{a}={len(data[a])}" for a in arms) + ")")

    # --------------------------------------------------- 関門（null の再現）
    g_mpr = float(np.mean([data["null"][p]["mpr"] for p in PROBS if p in data["null"]]))
    g_sc = float(np.mean([data["null"][p]["score"] for p in PROBS if p in data["null"]]))
    ok = abs(g_mpr - GATE["mpr"]) < 5e-5 and abs(g_sc - GATE["score"]) < 5e-5
    print(f"\n  関門（null 16 問平均の再現）: MPR {g_mpr:.4f} (記録 {GATE['mpr']}) / "
          f"Score {g_sc:.4f} (記録 {GATE['score']})  -> {'一致' if ok else '**不一致**'}")
    if not ok:
        print("  採点経路が再現しないので判定に進まない。")

    # --------------------------------------------------- 問題別
    print(f"\n{'problem':<16}" + "".join(f"{a+' mpr':>12}" for a in arms)
          + f"{'d(rs-nl)':>10}{'d(bh-nl)':>10}" + "".join(f"{a+' sc':>11}" for a in arms))
    print("-" * 90)
    for p in probs:
        r = [data[a][p] for a in arms]
        print(f"{p:<16}" + "".join(f"{v['mpr']:12.4f}" for v in r)
              + f"{r[1]['mpr']-r[0]['mpr']:10.4f}{r[2]['mpr']-r[0]['mpr']:10.4f}"
              + "".join(f"{v['score']:11.4f}" for v in r))

    def means(sub, key):
        return {a: float(np.mean([data[a][p][key] for p in sub])) for a in arms}

    for label, sub in (("全体", probs),
                       ("群 A", [p for p in probs if p in GROUP_A]),
                       ("群 B", [p for p in probs if p in GROUP_B])):
        if not sub:
            continue
        m, f1, sc = means(sub, "mpr"), means(sub, "f1"), means(sub, "score")
        nn = means(sub, "n")
        print(f"\n  [{label}] n={len(sub)}")
        for a in arms:
            print(f"    {a:<7} MPR {m[a]:.4f}   mean-F1 {f1[a]:.4f}   "
                  f"Score {sc[a]:.4f}   報告点数 {nn[a]:.1f}")
        print(f"    差(MPR): reseed-null {m['reseed']-m['null']:+.4f}   "
              f"bhop-null {m['bhop']-m['null']:+.4f}   "
              f"reseed-bhop {m['reseed']-m['bhop']:+.4f}")
        print(f"    差(Score): reseed-null {sc['reseed']-sc['null']:+.4f}   "
              f"bhop-null {sc['bhop']-sc['null']:+.4f}   "
              f"reseed-bhop {sc['reseed']-sc['bhop']:+.4f}")
        for key in ("mpr", "score"):
            for x, y in (("reseed", "null"), ("bhop", "null"), ("reseed", "bhop")):
                d = paired({p: data[x][p][key] for p in sub},
                           {p: data[y][p][key] for p in sub}, sub)
                print(f"    Wilcoxon {key:<5} {x}-{y:<7} mean {d['mean']:+.4f}  "
                      f"W/T/L {d['w']}/{d['t']}/{d['l']}  p={d['p']:.4g}  rb={d['rb']:+.3f}")

    # --------------------------------------------------- 腕の計器
    print(f"\n  腕の計器（`mode` / `rho` 列。腕が想定どおり動いたかの確認）")
    print(f"{'problem':<16}" + f"{'rs desc':>9}{'rs unif':>9}{'rs pert':>9}{'rs rho~':>9}"
          + f"{'bh desc':>9}{'bh unif':>9}{'bh pert':>9}{'bh rho~':>9}{'bh rho_end':>11}")
    for p in probs:
        a, b = rho_trace(p, "reseed"), rho_trace(p, "bhop")
        if not a or not b:
            continue
        print(f"{p:<16}{a['n']:9d}{a['uniform']:9d}{a['perturbed']:9d}{a['rho_med']:9.3f}"
              f"{b['n']:9d}{b['uniform']:9d}{b['perturbed']:9d}{b['rho_med']:9.3f}"
              f"{b['rho_last']:11.4f}")

    # --------------------------------------------------- 集計 CSV（数百行未満）
    out = os.path.join(HERE, "by_problem_current.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "arm", "K", "mpr", "f1", "score", "n_reported",
                    "n_descents"])
        for p in probs:
            for a in arms:
                v = data[a][p]
                w.writerow([p, a, K_of(p), f"{v['mpr']:.6f}", f"{v['f1']:.6f}",
                            f"{v['score']:.6f}", v["n"], v["ndesc"]])
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
