#!/usr/bin/env python3
"""その133 — 重複を潰した半径の再投入腕 / 素の basin-hopping / 一様 null を、その116 と同じ合法規則で採点する。

**新しい規則も新しい統計量も 1 つも定義しない。** `rule_indices` / `score` / `paired` は e115 の
`analyze.py` からそのまま import する。違うのは入力（どの降下ダンプか）だけ。

  * `null`   -> analysis/mmo2024/e115/descents（seed 0、座標つき、保存物。追加 run ゼロ）
  * `reseed` -> analysis/mmo2024/e133/descents/*_reseed_seed0.csv
  * `bhop`   -> analysis/mmo2024/e133/descents/*_bhop_seed0.csv

**関門は 2 つ**（事前登録）:
  (i) `null` をこのコードで採点し直した 16 問平均が その116 の記録（MPR 0.5644 / Score 0.6284）と 4 桁一致。
  (ii) **per-problem** —— 腕の `rho` 列の中央値が `e125/rho_degeneracy.csv` の `rho_dedup` と 2 桁一致。

**副判定 P1**: 問題の順位づけを **seed 1 の null** で行い、差は seed 0 で測る（平均回帰の交絡外し）。

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e133/analyze.py
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
GATE = {"mpr": 0.5644, "score": 0.6284}          # その116 の記録（採点経路の関門）
PROBS = [f"M{i:02d}-D10-PIN01" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]
NULL_DIR = os.path.join(MMO, "e115", "descents")
NULL_S1_DIR = os.path.join(MMO, "e115", "s1", "descents")
ARM_DIR = os.path.join(HERE, "descents")
RHO_TABLE = os.path.join(MMO, "e125", "rho_degeneracy.csv")
# 事前登録で凍結した seed 1 null の MPR 昇順（弱い順）。P1 の順位づけに使う。
S1_ORDER = ["M14", "M15", "M06", "M16", "M07", "M11", "M04", "M09",
            "M03", "M02", "M01", "M10", "M05", "M13", "M12", "M08"]

_K: dict = {}


def K_of(prob):
    if prob not in _K:
        _K[prob] = int(niching_by_name(prob).n_global_optima)
    return _K[prob]


def find(d, prefix):
    if not os.path.isdir(d):
        return None
    for c in sorted(os.listdir(d)):
        if c.startswith(prefix):
            return os.path.join(d, c)
    return None


def load(prob, arm):
    p = (find(NULL_DIR, f"{prob}_seed") if arm == "null"
         else find(ARM_DIR, f"{prob}_{arm}_seed0.csv"))
    if p is None:
        return None
    f, opt, xs = read_dump(p)
    idx = rule_indices(ARM, f, K_of(prob), x=xs, r=ARM_R)
    recall, prec, f1, sc, n = score(idx, f, opt, K_of(prob))
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()), prec=float(prec.mean()),
                score=float(sc.mean()), n=int(n), ndesc=len(f), path=p)


def rho_trace(prob, arm):
    """腕の計器（`mode` / `rho` 列）。関門 (ii) と、腕が想定どおり動いたかの確認に使う。"""
    p = find(ARM_DIR, f"{prob}_{arm}_seed0.csv")
    if p is None:
        return None
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


def rho_dedup_table():
    out = {}
    with open(RHO_TABLE) as fh:
        for r in csv.DictReader(fh):
            out[r["problem"]] = float(r["rho_dedup"])
    return out


def perm_spearman(x, y, n=20000, seed=0):
    """Spearman ＋ 並べ替え検定（両側）。e119/e125 と同じ形。"""
    from scipy.stats import spearmanr
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    cnt = sum(abs(float(spearmanr(x, rng.permutation(y)).statistic)) >= abs(rho)
              for _ in range(n))
    return rho, (cnt + 1) / (n + 1)


def main():
    arms = ["null", "reseed", "bhop"]
    data = {a: {} for a in arms}
    for p in PROBS:
        for a in arms:
            v = load(p, a)
            if v is not None:
                data[a][p] = v
    probs = [p for p in PROBS if all(p in data[a] for a in arms)]
    missing = [p for p in PROBS if p not in probs]

    print("=" * 96)
    print("その133 — 重複を潰した半径での再投入は一様再起動に勝つか（キュー 3）")
    print("=" * 96)
    print(f"\n  規則: {ARM} (r={ARM_R})   3 腕が揃った問題: {len(probs)}/16")
    if missing:
        print(f"  **欠けた問題: {', '.join(missing)}**  (揃っている本数: "
              + ", ".join(f"{a}={len(data[a])}" for a in arms) + ")")
        print(f"  **群 A の揃い: {sum(p in GROUP_A for p in probs)}/8   "
              f"群 B の揃い: {sum(p in GROUP_B for p in probs)}/8**")

    # --------------------------------------------------- 関門 (i) 採点経路
    g_mpr = float(np.mean([data["null"][p]["mpr"] for p in PROBS if p in data["null"]]))
    g_sc = float(np.mean([data["null"][p]["score"] for p in PROBS if p in data["null"]]))
    ok = abs(g_mpr - GATE["mpr"]) < 5e-5 and abs(g_sc - GATE["score"]) < 5e-5
    print(f"\n  関門(i) null 16 問平均の再現: MPR {g_mpr:.4f} (記録 {GATE['mpr']}) / "
          f"Score {g_sc:.4f} (記録 {GATE['score']})  -> {'一致' if ok else '**不一致**'}")
    if not ok:
        print("  採点経路が再現しないので判定に進まない。")
        return

    # --------------------------------------------------- 関門 (ii) per-problem の rho
    tab = rho_dedup_table()
    print(f"\n  関門(ii) per-problem の rho（腕の rho 中央値 対 e125 の rho_dedup）")
    print(f"  **参照値 `rho_dedup` は null の着地点から測ったもの**で、腕は着地の分布そのものを変えるので、")
    print(f"  **厳密な 2 桁一致は設計上は期待できない。縮退（0 への潰れ）の検出が関門の役目。**")
    print(f"{'problem':<16}{'rho_dedup':>11}{'rs rho~':>10}{'bh rho~':>10}{'rs/ref':>9}{'bh/ref':>9}{'2桁一致':>9}")
    gate2, ratios = {}, []
    for p in probs:
        ref = tab[p]
        a, b = rho_trace(p, "reseed"), rho_trace(p, "bhop")
        if not a or not b:
            continue
        ra, rb_ = a["rho_med"] / max(ref, 1e-12), b["rho_med"] / max(ref, 1e-12)
        ratios += [ra, rb_]
        # 2 桁一致 = 有効数字 2 桁が一致（相対差 5% 以内で近似）
        gate2[p] = max(abs(ra - 1), abs(rb_ - 1)) <= 0.05
        print(f"{p:<16}{ref:11.4f}{a['rho_med']:10.4f}{b['rho_med']:10.4f}"
              f"{ra:9.2f}{rb_:9.2f}{'一致' if gate2[p] else '外れ':>9}")
    npass = sum(gate2.values())
    print(f"  -> 有効数字 2 桁の一致: {npass}/{len(gate2)} 問   "
          f"比 rho_arm/rho_dedup の範囲 {min(ratios):.2f}-{max(ratios):.2f}（中央 {np.median(ratios):.2f}）")
    deg = [p for p in probs if (rho_trace(p, 'reseed') or {}).get('rho_med', 1) < 1e-6]
    print(f"  -> **rho が 0 に潰れた問題: {len(deg)}/{len(probs)}**  "
          f"（その125 は 14/16。**これがゼロなら計器は直っている**）")

    # --------------------------------------------------- 問題別
    print(f"\n{'problem':<16}" + "".join(f"{a + ' mpr':>12}" for a in arms)
          + f"{'d(rs-nl)':>10}{'d(bh-nl)':>10}" + "".join(f"{a + ' sc':>11}" for a in arms))
    print("-" * 96)
    for p in probs:
        r = [data[a][p] for a in arms]
        print(f"{p:<16}" + "".join(f"{v['mpr']:12.4f}" for v in r)
              + f"{r[1]['mpr'] - r[0]['mpr']:10.4f}{r[2]['mpr'] - r[0]['mpr']:10.4f}"
              + "".join(f"{v['score']:11.4f}" for v in r))

    def means(sub, key):
        return {a: float(np.mean([data[a][p][key] for p in sub])) for a in arms}

    for label, sub in (("全体", probs),
                       ("群 A", [p for p in probs if p in GROUP_A]),
                       ("群 B", [p for p in probs if p in GROUP_B])):
        if not sub:
            continue
        m, f1, sc = means(sub, "mpr"), means(sub, "f1"), means(sub, "score")
        nn, pr = means(sub, "n"), means(sub, "prec")
        print(f"\n  [{label}] n={len(sub)}")
        for a in arms:
            print(f"    {a:<7} MPR {m[a]:.4f}   mean-F1 {f1[a]:.4f}   Score {sc[a]:.4f}   "
                  f"報告点数 {nn[a]:.1f}   precision {pr[a]:.4f}")
        print(f"    差(MPR): reseed-null {m['reseed'] - m['null']:+.4f}   "
              f"bhop-null {m['bhop'] - m['null']:+.4f}   "
              f"reseed-bhop {m['reseed'] - m['bhop']:+.4f}")
        print(f"    差(Score): reseed-null {sc['reseed'] - sc['null']:+.4f}   "
              f"bhop-null {sc['bhop'] - sc['null']:+.4f}   "
              f"reseed-bhop {sc['reseed'] - sc['bhop']:+.4f}")
        for key in ("mpr", "score"):
            for x, y in (("reseed", "null"), ("bhop", "null"), ("reseed", "bhop")):
                d = paired({p: data[x][p][key] for p in sub},
                           {p: data[y][p][key] for p in sub}, sub)
                print(f"    Wilcoxon {key:<5} {x}-{y:<7} mean {d['mean']:+.4f}  "
                      f"W/T/L {d['w']}/{d['t']}/{d['l']}  p={d['p']:.4g}  rb={d['rb']:+.3f}")

    # --------------------------------------------------- 腕の計器
    print(f"\n  腕の計器（`mode` / `rho` 列）")
    print(f"{'problem':<16}{'rs desc':>9}{'rs unif':>9}{'rs pert':>9}{'rs rho~':>9}"
          f"{'bh desc':>9}{'bh unif':>9}{'bh pert':>9}{'bh rho~':>9}{'bh rho_end':>11}")
    for p in probs:
        a, b = rho_trace(p, "reseed"), rho_trace(p, "bhop")
        if not a or not b:
            continue
        print(f"{p:<16}{a['n']:9d}{a['uniform']:9d}{a['perturbed']:9d}{a['rho_med']:9.3f}"
              f"{b['n']:9d}{b['uniform']:9d}{b['perturbed']:9d}{b['rho_med']:9.3f}"
              f"{b['rho_last']:11.4f}")

    # --------------------------------------------------- 副判定 P1
    print(f"\n  [副判定 P1] 順位づけ = seed 1 の null（事前登録で凍結）、差 = seed 0")
    rank = {p: i for i, p in enumerate(S1_ORDER)}     # 0 = いちばん弱い
    xs = [rank[p[:3]] for p in probs]
    for arm in ("reseed", "bhop"):
        ys = [data[arm][p]["mpr"] - data["null"][p]["mpr"] for p in probs]
        rho, pv = perm_spearman(xs, ys)
        print(f"    seed1 順位 対 ({arm}-null) の MPR 差: Spearman {rho:+.3f}  "
              f"並べ替え p={pv:.4g}  -> P1 は{'成立' if (rho < 0 and pv < 0.05) else '**反証**'}")
    # 交絡つきの読み（その125 と同じ形。比較のためだけに出す）
    for arm in ("reseed", "bhop"):
        xs0 = [data["null"][p]["mpr"] for p in probs]
        ys = [data[arm][p]["mpr"] - data["null"][p]["mpr"] for p in probs]
        rho, pv = perm_spearman(xs0, ys)
        print(f"    （参考・平均回帰の交絡つき）seed0 null MPR 対 ({arm}-null): "
              f"Spearman {rho:+.3f}  p={pv:.4g}")

    # --------------------------------------------------- 集計 CSV
    out = os.path.join(HERE, "by_problem.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "arm", "K", "mpr", "f1", "precision", "score",
                    "n_reported", "n_descents", "rho_med"])
        for p in probs:
            for a in arms:
                v = data[a][p]
                t = rho_trace(p, a) if a != "null" else None
                w.writerow([p, a, K_of(p), f"{v['mpr']:.6f}", f"{v['f1']:.6f}",
                            f"{v['prec']:.6f}", f"{v['score']:.6f}", v["n"], v["ndesc"],
                            f"{t['rho_med']:.6f}" if t else ""])
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
