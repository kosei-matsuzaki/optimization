#!/usr/bin/env python3
"""その177 — 2D BBOB-24 の基準（MC-ESO ＋ 比較 5 手法）を 3 指標で出す。

`scripts/analyze_quick.py` が出す全体表の補完として、**この回の docs に載せた
数値そのもの**（関数別 SR@1e-10 / evals_succ_mean と、MC-ESO が下回る関数の一覧、
Wilcoxon の勝敗内訳）を 1 本で出す。新しい統計量は定義していない。

使い方:
    python3 analysis/single/e177/analyze.py [summary.csv] [wilcoxon.csv]
既定は同ディレクトリの保存物を読む。
"""
from __future__ import annotations
import csv, os, sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "summary_base2d.csv")
WILCOXON = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "wilcoxon_base2d.csv")

REF = "MC-ESO"
METHODS = [REF, "CMA-ES", "IPOP-CMA-ES", "BIPOP-CMA-ES", "DE", "L-SHADE"]
LEVELS = ["sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10"]


def pct(s):
    s = (s or "").strip().rstrip("%")
    return float(s) / 100.0 if s not in ("", "N/A") else float("nan")


def num(s):
    s = (s or "").strip()
    if s in ("", "---", "N/A", "inf", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


rows = {}
with open(SUMMARY, newline="") as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["function"], {})[r["method"]] = r
funcs = sorted(rows)

print(f"functions: {len(funcs)}   methods: {', '.join(METHODS)}")

print("\n[1] 全体（24 関数平均）")
print(f"  {'method':14s} " + " ".join(f"{k.replace('sr_','SR@'):>10s}" for k in LEVELS)
      + f" {'evals_succ_mean':>16s}")
for m in METHODS:
    cells = [rows[fn][m] for fn in funcs if m in rows[fn]]
    srs = [mean(pct(c[k]) for c in cells) for k in LEVELS]
    ev = [num(c["evals_succ_mean"]) for c in cells]
    ev = [v for v in ev if v is not None]
    print(f"  {m:14s} " + " ".join(f"{100*v:9.2f}%" for v in srs)
          + f" {mean(ev):16.1f}  (成功のある関数 {len(ev)}/{len(cells)})")

print("\n[2] 関数別 SR@1e-10（%）と evals_succ_mean")
print(f"  {'function':22s} " + " ".join(f"{m:>13s}" for m in METHODS))
for fn in funcs:
    cells = []
    for m in METHODS:
        r = rows[fn].get(m)
        if r is None:
            cells.append("        —")
            continue
        e = num(r["evals_succ_mean"])
        cells.append(f"{100*pct(r['sr_1e-10']):5.0f}/{'---' if e is None else f'{e:.0f}'}")
    print(f"  {fn:22s} " + " ".join(f"{c:>13s}" for c in cells))

print("\n[3] MC-ESO がいずれかの手法に SR@1e-10 で下回る関数")
lose = []
for fn in funcs:
    ref = pct(rows[fn][REF]["sr_1e-10"])
    best_m, best_v = None, ref
    for m in METHODS[1:]:
        v = pct(rows[fn][m]["sr_1e-10"])
        if v > best_v + 1e-12:
            best_m, best_v = m, v
    if best_m:
        lose.append((fn, ref, best_m, best_v))
for fn, ref, m, v in lose:
    er = num(rows[fn][REF]["evals_succ_mean"])
    eb = num(rows[fn][m]["evals_succ_mean"])
    print(f"  {fn:22s} MC-ESO {100*ref:3.0f}% (evals {'---' if er is None else f'{er:.0f}'})"
          f"  <  {m} {100*v:3.0f}% (evals {'---' if eb is None else f'{eb:.0f}'})")
print(f"  計 {len(lose)} 関数: " + ", ".join(fn.split('-')[0] for fn, *_ in lose))

print("\n[4] 同点も含めた「MC-ESO が単独 1 位でない」関数")
tie = []
for fn in funcs:
    ref = pct(rows[fn][REF]["sr_1e-10"])
    if any(pct(rows[fn][m]["sr_1e-10"]) >= ref - 1e-12 for m in METHODS[1:]):
        tie.append(fn)
print(f"  計 {len(tie)} 関数: " + ", ".join(fn.split('-')[0] for fn in tie))

print("\n[5] Wilcoxon（reference = MC-ESO、両側 p<0.05 を有意とする）")
w = {}
with open(WILCOXON, newline="") as f:
    for r in csv.DictReader(f):
        w.setdefault(r["method"], []).append(r)
print(f"  {'method':14s} {'有意に MC-ESO が優':>18s} {'有意に MC-ESO が劣':>18s} {'全 tie':>8s}")
for m in METHODS[1:]:
    better = worse = alltie = 0
    for r in w.get(m, []):
        p = num(r["p_value_two_sided"])
        a12 = num(r["a12"])
        if int(r["tie_count"]) == int(r["n"]):
            alltie += 1
        if p is not None and p < 0.05 and a12 is not None:
            if a12 > 0.5:
                better += 1
            elif a12 < 0.5:
                worse += 1
    print(f"  {m:14s} {better:18d} {worse:18d} {alltie:8d}")
print("\n  有意に MC-ESO が劣る (function, method, p, A12):")
any_worse = False
for m in METHODS[1:]:
    for r in w.get(m, []):
        p, a12 = num(r["p_value_two_sided"]), num(r["a12"])
        if p is not None and p < 0.05 and a12 is not None and a12 < 0.5:
            any_worse = True
            print(f"    {r['function']:22s} {m:14s} p={p:.4g} A12={a12:.4f} ({r['a12_magnitude']})")
if not any_worse:
    print("    なし")
