#!/usr/bin/env python3
"""その122 — PIN01 は代表的か（キュー 1）。

新 suite の測定 1122 件すべてが PIN01 の上に載っている。suite は 16 PID × 4 次元
× 15 インスタンス = 960 問（`core/benchmarks.py:731`）で、**公表値は D=10 の
15 インスタンス平均**。その106 以降 13 サイクルの結論が代表性の未確認な 1 本に
載っているので、**同じ `Restart-Lander` を PIN02 / PIN03 でも回して対にする。**

採点規則も採点コードも**その115 / その116 のものをそのまま import する。新しい規則は
1 つも定義しない**（`rule_indices` / `score` / `aggregate` / `paired` / `attribute`）。
違うのは入力（どのインスタンスの降下ダンプか）だけ。

  * PIN01 -> `analysis/mmo2024/e115/descents`（その115 の座標つき再 run、seed 0）
  * PIN02 / PIN03 -> `analysis/mmo2024/e122/descents`（本サイクル、seed 0）

**最近傍最適への帰属（`opt`）はここで計算する**（e116 と同じ手続き）。`opt` は
**採点にだけ**使い、規則の定義には使わない（その115 の教訓）。

使い方: python3 analysis/mmo2024/e122/analyze.py
"""
from __future__ import annotations

import csv
import os
import re
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MMO, "e115"))

from analyze import (LEVEL_NAMES, SPAN, aggregate, mean_of,        # noqa: E402
                     paired, read_dump, rule_indices)
from core.benchmarks import niching_by_name                        # noqa: E402

# その115 が選んだ合法な最良腕（oracle 上限と 16/16 厳密一致）。e116 と同一。
ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN
INSTANCES = ["PIN01", "PIN02", "PIN03"]
# その116 の seed 0・合法規則での PIN01 の値（採点経路の再現チェックに使う）
E116_PIN01 = dict(mpr=0.5644, score=0.6284)
PREREG_TIGHT, PREREG_WIDE = 0.05, 0.10   # 事前登録した帯
PUBLISHED_BEST = 0.6080                  # 競技優勝 TRDE-LR の Score（D=10、15 instance 平均）

_NAME_RE = re.compile(r"^M(\d{2})-D(\d{2})-(PIN\d{2})$")


def attribute(x: np.ndarray, optima: np.ndarray) -> np.ndarray:
    """各点を最近傍の最適に帰属させる（`core.runner.count_goptima_nn` と同順序）。
    **オラクル量。採点にだけ使う。** e116 の同名関数と同一。"""
    d = np.linalg.norm(x[:, None, :] - optima[None, :, :], axis=2)
    return np.argmin(d, axis=1)


_BENCH: dict = {}


def bench(prob):
    if prob not in _BENCH:
        b = niching_by_name(prob)
        _BENCH[prob] = (int(b.n_global_optima),
                        np.asarray(b.optima_pos, dtype=float))
    return _BENCH[prob]


def load_instance(dump_dir, want_pin):
    """1 インスタンス分の run を読む（1 問 1 要素、seed 0）。key は PID。"""
    runs = []
    if not os.path.isdir(dump_dir):
        return runs
    for fn in sorted(os.listdir(dump_dir)):
        if not fn.endswith((".csv", ".csv.gz")):
            continue
        prob = fn.split("_")[0]
        m = _NAME_RE.match(prob)
        if not m or m.group(3) != want_pin:
            continue
        f, _opt, x = read_dump(os.path.join(dump_dir, fn))
        if x is None:          # 座標の無いダンプは合法規則を当てられない
            continue
        K, opts = bench(prob)
        runs.append(dict(problem=f"M{m.group(1)}", instance=want_pin, seed=0,
                         K=K, f=f, x=x, opt=attribute(x, opts)))
    return runs


def a12(d):
    """対にした差から A12（対照は「差が正である確率」の形）。同点は 0.5 で数える。"""
    d = np.asarray(d, dtype=float)
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d)) if len(d) else float("nan")


def main() -> int:
    src = {"PIN01": os.path.join(MMO, "e115", "descents"),
           "PIN02": os.path.join(HERE, "descents"),
           "PIN03": os.path.join(HERE, "descents")}
    runs = {pin: load_instance(src[pin], pin) for pin in INSTANCES}

    print("=" * 90)
    print("その122 — PIN01 は代表的か（キュー 1、`Restart-Lander`・D=10・seed 0・正規予算 50 万）")
    print("=" * 90)
    for pin in INSTANCES:
        print(f"  {pin}  {len(runs[pin]):>2}/16 問")
    have = {pin: {r["problem"] for r in runs[pin]} for pin in INSTANCES}
    done = [pin for pin in INSTANCES if len(have[pin]) == 16]
    if "PIN01" not in done or len(done) < 2:
        print("\n  対にできるインスタンスが 2 本に満たない。集計しない。")
        return 1
    part = [pin for pin in INSTANCES if pin not in done and have[pin]]
    if part:
        print(f"\n  ** 部分結果のインスタンス: {', '.join(part)} "
              f"（{', '.join(str(len(have[p])) for p in part)} 問）——"
              f" 事前登録の打ち切り規則により 16 問平均には混ぜない。**")

    agg = {}
    for pin in INSTANCES:
        if not runs[pin]:
            continue
        agg[pin] = aggregate(
            runs[pin],
            lambda r: rule_indices(ARM, r["f"], r["K"], r["x"], ARM_R))
    probs16 = sorted(have["PIN01"])

    print(f"\n  規則: {ARM}  r = {ARM_R / SPAN:g} x span（その115 の合法な最良腕、"
          f"e116 と同一。新しい規則は定義していない）")

    # ---------------------------------------------------- 0. 採点経路の再現チェック
    print("\n## 0. 採点経路の再現チェック（PIN01 が その116 の値に戻るか）\n")
    got = {k: mean_of(agg["PIN01"], probs16, "mpr" if k == "mpr" else "score")
           for k in ("mpr", "score")}
    ok = all(abs(got[k] - E116_PIN01[k]) < 5e-4 for k in got)
    for k in ("mpr", "score"):
        print(f"  PIN01 {k:<6} 今回 {got[k]:.4f}  その116 {E116_PIN01[k]:.4f}  "
              f"差 {got[k] - E116_PIN01[k]:+.4f}")
    print("  ==> " + ("採点経路は再現した。インスタンス差の議論に入ってよい"
                      if ok else
                      "**再現しない。採点経路が違うのでインスタンス差の議論に入らない**"))

    # ---------------------------------------------------- 1. 16 問平均
    print("\n## 1. 16 問平均（5 水準等重み、seed 0 の 1 本）\n")
    print(f"{'instance':<10}{'MPR':>9}{'mean-F1':>10}{'Score':>9}{'n_rep':>9}"
          f"{'MPR-PIN01':>12}{'Score-PIN01':>13}")
    for pin in done:
        g = agg[pin]
        print(f"{pin:<10}{mean_of(g, probs16, 'mpr'):>9.4f}"
              f"{mean_of(g, probs16, 'f1'):>10.4f}"
              f"{mean_of(g, probs16, 'score'):>9.4f}"
              f"{mean_of(g, probs16, 'n'):>9.1f}"
              f"{mean_of(g, probs16, 'mpr') - mean_of(agg['PIN01'], probs16, 'mpr'):>+12.4f}"
              f"{mean_of(g, probs16, 'score') - mean_of(agg['PIN01'], probs16, 'score'):>+13.4f}")
    print(f"\n  参考: 競技優勝 TRDE-LR の Score {PUBLISHED_BEST:.4f}"
          f"（D=10、15 インスタンス平均）")

    # ---------------------------------------------------- 2. 対検定
    print("\n## 2. 16 問を対にした検定（両側 Wilcoxon exact、PIN01 を基準）\n")
    for pin in done[1:]:
        for key, lab in (("mpr", "MPR"), ("score", "Score"), ("f1", "mean-F1")):
            a = {p: agg[pin][p][key] for p in probs16}
            b = {p: agg["PIN01"][p][key] for p in probs16}
            r = paired(a, b, probs16)
            d = np.array([a[p] - b[p] for p in probs16])
            print(f"  {pin} - PIN01  {lab:<8}{r['mean']:+.4f}  "
                  f"{r['w']}/{r['l']}/{r['t']}  p={r['p']:.4g}  "
                  f"rb={r['rb']:+.3f}  A12={a12(d):.3f}  "
                  f"|max| {np.abs(d).max():.4f}")
        print()
    if len(done) == 3:
        print("  PIN02 - PIN03（インスタンスどうし）")
        for key, lab in (("mpr", "MPR"), ("score", "Score")):
            a = {p: agg["PIN02"][p][key] for p in probs16}
            b = {p: agg["PIN03"][p][key] for p in probs16}
            r = paired(a, b, probs16)
            print(f"    {lab:<8}{r['mean']:+.4f}  {r['w']}/{r['l']}/{r['t']}  "
                  f"p={r['p']:.4g}")
        rows = [[agg[pin][p]["mpr"] for p in probs16] for pin in done]
        fr = stats.friedmanchisquare(*rows)
        print(f"\n  Friedman（3 インスタンス × 16 問、MPR）: "
              f"chi2={fr.statistic:.4f}  p={fr.pvalue:.4g}")
        rows = [[agg[pin][p]["score"] for p in probs16] for pin in done]
        fr = stats.friedmanchisquare(*rows)
        print(f"  Friedman（同、Score）:                   "
              f"chi2={fr.statistic:.4f}  p={fr.pvalue:.4g}")

    # ---------------------------------------------------- 3. 問題別
    print("\n## 3. 問題別 MPR（1 seed の値。その111 により問題別の差は 1 seed では引用できない）\n")
    head = "".join(f"{p:>9}" for p in done)
    print(f"{'PID':<7}{'K':>4}{head}{'range':>9}{'|max-d|':>9}")
    for p in probs16:
        vs = [agg[pin][p]["mpr"] for pin in done]
        K = agg["PIN01"][p]
        kk = next(r["K"] for r in runs["PIN01"] if r["problem"] == p)
        rng = max(vs) - min(vs)
        md = max(abs(v - vs[0]) for v in vs)
        flag = "  <<" if md >= PREREG_WIDE else ""
        print(f"{p:<7}{kk:>4}" + "".join(f"{v:>9.4f}" for v in vs)
              + f"{rng:>9.4f}{md:>9.4f}{flag}")
        del K

    # ---------------------------------------------------- 4. 事前登録の判定
    print("\n## 4. 事前登録した棄却条件（主判定量 = 16 問平均 MPR）\n")
    base = mean_of(agg["PIN01"], probs16, "mpr")
    print(f"  PIN01 の 16 問平均 MPR = {base:.4f}"
          f"（その116 の記録 {E116_PIN01['mpr']:.4f}）\n")
    verdicts = []
    for pin in done[1:]:
        d = mean_of(agg[pin], probs16, "mpr") - base
        if abs(d) <= PREREG_TIGHT:
            v = f"±{PREREG_TIGHT} 以内 ＝ 代表的"
        elif abs(d) >= PREREG_WIDE:
            v = f"{PREREG_WIDE} 以上ずれた ＝ インスタンス差は結論より大きい"
        else:
            v = f"中間（{PREREG_TIGHT}-{PREREG_WIDE}）＝ 問題別の列挙が要る"
        verdicts.append((pin, d, v))
        print(f"  {pin}: 差 {d:+.4f}  ->  {v}")
    worst = max((abs(d) for _, d, _ in verdicts), default=0.0)
    print()
    if worst <= PREREG_TIGHT:
        print("  ==> **PIN01 は代表的。** ただし事前登録の追加条件により、"
              "16 問を対にした検定が有意でないことも要る（§2 を見ること）。")
        print("      status.md の留保は「1 インスタンス」から「3 インスタンスで一致」に狭められる。")
    elif worst >= PREREG_WIDE:
        print("  ==> **インスタンス差は結論より大きい。** 記録済みの新 suite の数値すべてに")
        print("      「PIN01 の値」と明記する必要がある。15 インスタンス全部を回すかの判断は")
        print("      status.md 経由でユーザーへ（**実行役は回さない。約 4.7 時間 ＝ 7 サイクル**）。")
    else:
        print("  ==> **中間。** §3 の問題別の表で、どの問題がずれたかを列挙する。")

    # ---------------------------------------------------- 5. 水準別
    print("\n## 5. 水準別 PR / F1（16 問平均）\n")
    print(f"{'instance':<10}{'量':<5}" + "".join(f"{l:>9}" for l in LEVEL_NAMES))
    for pin in done:
        g = agg[pin]
        pr = np.mean([g[p]["pr_lv"] for p in probs16], axis=0)
        f1 = np.mean([g[p]["f1_lv"] for p in probs16], axis=0)
        print(f"{pin:<10}{'PR':<5}" + "".join(f"{v:>9.4f}" for v in pr))
        print(f"{'':<10}{'F1':<5}" + "".join(f"{v:>9.4f}" for v in f1))

    out = os.path.join(HERE, "instances_d10.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "instance", "K", "n_reported", "mpr", "mean_f1", "score"])
        for pin in INSTANCES:
            if pin not in agg:
                continue
            for p in sorted(agg[pin]):
                g = agg[pin][p]
                kk = next(r["K"] for r in runs[pin] if r["problem"] == p)
                w.writerow([p, pin, kk, f"{g['n']:.1f}", f"{g['mpr']:.4f}",
                            f"{g['f1']:.4f}", f"{g['score']:.4f}"])
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
