#!/usr/bin/env python3
"""その178 — 5D BBOB-24 の現在地（MC-ESO ＋ 比較 5 手法）を 3 指標で出す。

その177（2D）の `analysis/single/e177/analyze.py` と同じ 5 節（全体 / 関数別 /
下回る関数 / 同点込み / Wilcoxon）に、この回の事前登録が要求した 2 節を足した:
  [6] BBOB 公式 5 群ごとの内訳
  [7] 仮想ベスト手法（5 手法の関数別ベスト、および 6 手法の関数別ベスト）との差
新しい統計量は定義していない。

使い方:
    python3 analysis/single/e178/analyze.py [summary.csv] [wilcoxon.csv]
既定は同ディレクトリの保存物（4 shard を連結したもの）を読む。
"""
from __future__ import annotations
import csv, os, sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "summary_base5d.csv")
WILCOXON = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "wilcoxon_base5d.csv")

REF = "MC-ESO"
METHODS = [REF, "CMA-ES", "IPOP-CMA-ES", "BIPOP-CMA-ES", "DE", "L-SHADE"]
LEVELS = ["sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10"]
# BBOB の公式 5 群（COCO の function group。境界は F05/F09/F14/F19）
GROUPS = [
    ("g1 separable            (F01-F05)", range(1, 6)),
    ("g2 low/moderate cond.   (F06-F09)", range(6, 10)),
    ("g3 high cond. unimodal  (F10-F14)", range(10, 15)),
    ("g4 multimodal, global   (F15-F19)", range(15, 20)),
    ("g5 multimodal, weak     (F20-F24)", range(20, 25)),
]


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


def fnum(fn):
    return int(fn[1:3])


rows = {}
with open(SUMMARY, newline="") as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["function"], {})[r["method"]] = r
funcs = sorted(rows, key=fnum)

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
          f"  <  {m} {100*v:3.0f}% (evals {'---' if eb is None else f'{eb:.0f}'})"
          f"   差 {100*(v-ref):+5.0f}pt")
print(f"  計 {len(lose)} 関数: " + ", ".join(fn.split('-')[0] for fn, *_ in lose))

print("\n[4] MC-ESO が単独 1 位の関数 / 単独最下位の関数")
solo_win, solo_lose = [], []
for fn in funcs:
    ref = pct(rows[fn][REF]["sr_1e-10"])
    others = [pct(rows[fn][m]["sr_1e-10"]) for m in METHODS[1:]]
    if all(ref > v + 1e-12 for v in others):
        solo_win.append(fn)
    if all(ref < v - 1e-12 for v in others):
        solo_lose.append(fn)
print(f"  単独 1 位 {len(solo_win)} 関数: " + (", ".join(f.split('-')[0] for f in solo_win) or "なし"))
print(f"  単独最下位 {len(solo_lose)} 関数: " + (", ".join(f.split('-')[0] for f in solo_lose) or "なし"))

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

print("\n[6] BBOB 公式 5 群ごとの SR@1e-10（%）")
print(f"  {'group':36s} " + " ".join(f"{m:>13s}" for m in METHODS))
for label, rng in GROUPS:
    sel = [fn for fn in funcs if fnum(fn) in rng]
    cells = []
    for m in METHODS:
        vals = [pct(rows[fn][m]["sr_1e-10"]) for fn in sel if m in rows[fn]]
        cells.append(f"{100*mean(vals):12.2f}%" if vals else "            —")
    print(f"  {label:36s} " + " ".join(f"{c:>13s}" for c in cells)
          + f"   (n={len(sel)})")

print("\n[7] 仮想ベスト手法との距離（SR@1e-10、24 関数平均）")
ref_mean = mean(pct(rows[fn][REF]["sr_1e-10"]) for fn in funcs)
vb5 = mean(max(pct(rows[fn][m]["sr_1e-10"]) for m in METHODS[1:]) for fn in funcs)
vb6 = mean(max(pct(rows[fn][m]["sr_1e-10"]) for m in METHODS) for fn in funcs)
print(f"  MC-ESO                              {100*ref_mean:7.2f}%")
print(f"  仮想ベスト（比較 5 手法の関数別ベスト） {100*vb5:7.2f}%"
      f"   MC-ESO との差 {100*(vb5-ref_mean):+6.2f}pt")
print(f"  仮想ベスト（6 手法の関数別ベスト）      {100*vb6:7.2f}%"
      f"   MC-ESO の上乗せ {100*(vb6-vb5):+6.2f}pt")
print(f"  MC-ESO の赤字（100% まで）              {100*(1-ref_mean):7.2f}pt"
      f"   うち比較手法が実測で到達している分 {100*(vb5-ref_mean if vb5>ref_mean else 0):.2f}pt")
print("\n  比較手法が実測で到達している赤字の関数別内訳（寄与 pt = (best5 − MC-ESO)/24）:")
contrib = []
for fn in funcs:
    r = pct(rows[fn][REF]["sr_1e-10"])
    b = max(pct(rows[fn][m]["sr_1e-10"]) for m in METHODS[1:])
    if b > r + 1e-12:
        contrib.append((100 * (b - r) / len(funcs), fn, 100 * r, 100 * b))
contrib.sort(reverse=True)
for c, fn, r, b in contrib:
    print(f"    {fn:22s} {c:6.3f}pt   MC-ESO {r:3.0f}% → best5 {b:3.0f}%")
print(f"    合計 {sum(c for c, *_ in contrib):.3f}pt")

print("\n[8] MC-ESO が下回る 13 関数の SR 梯子（MC-ESO、n=20）と median best_f")
LAD = ["sr_1e-1", "sr_1e-2", "sr_1e-3", "sr_1e-4", "sr_1e-5", "sr_1e-7", "sr_1e-10"]
print(f"  {'function':22s} " + " ".join(f"{k.replace('sr_',''):>6s}" for k in LAD)
      + f" {'median_best_f':>14s}")
for fn, *_ in lose:
    r = rows[fn][REF]
    print(f"  {fn:22s} " + " ".join(f"{100*pct(r[k]):5.0f}%" for k in LAD)
          + f" {r['median_best_f']:>14s}")

print("\n[9] 群ごとの赤字の内訳（比較手法が実測で到達している 22.29pt をどこが作るか）")
for label, rng in GROUPS:
    tot = sum(c for c, fn, *_ in contrib if fnum(fn) in rng)
    names = [fn.split('-')[0] for c, fn, *_ in contrib if fnum(fn) in rng]
    print(f"  {label:36s} {tot:6.3f}pt  ({', '.join(names) or 'なし'})")
