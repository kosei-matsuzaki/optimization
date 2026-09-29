#!/usr/bin/env python3
"""その182 — 学習 C の採用規則（cc_mu_frac）の腕を 10D BBOB-24 で 3 指標で読む。

節の構成は `analysis/single/e178/analyze.py` と同じで、この回の事前登録が要求した
2 点を足しただけ。新しい統計量は定義していない。
  [2b] 腕 − base の関数別差（SR@1e-10）＝ 改善・悪化の全列挙
  [3b] evals_succ_mean を「両手法とも成功のある関数」だけで平均し直した対照
  [10] 反証条件 (a)(b)(c) の機械判定

使い方:
    python3 analysis/single/e182/analyze.py [summary.csv] [wilcoxon.csv]
"""
from __future__ import annotations
import csv, os, sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "summary_ccmu_10d.csv")
WILCOXON = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "wilcoxon_ccmu_10d.csv")

REF = "MC-ESO"
ARM = "ccmu50"
METHODS = [REF, ARM, "CMA-ES", "IPOP-CMA-ES"]
LEVELS = ["sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10"]
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
    ev = [v for v in (num(c["evals_succ_mean"]) for c in cells) if v is not None]
    print(f"  {m:14s} " + " ".join(f"{100*v:9.2f}%" for v in srs)
          + f" {mean(ev):16.1f}  (成功のある関数 {len(ev)}/{len(cells)})")

print("\n[2] 関数別 SR@1e-10（%）/ evals_succ_mean")
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

print(f"\n[2b] 腕 {ARM} − base の関数別差（SR@1e-10、pt）—— 改善・悪化の全列挙")
up, down, same = [], [], []
for fn in funcs:
    b = pct(rows[fn][REF]["sr_1e-10"]); a = pct(rows[fn][ARM]["sr_1e-10"])
    d = 100 * (a - b)
    (up if d > 1e-9 else down if d < -1e-9 else same).append((d, fn, 100 * b, 100 * a))
up.sort(reverse=True); down.sort()
for label, lst in (("改善", up), ("悪化", down)):
    print(f"  {label} {len(lst)} 関数:")
    for d, fn, b, a in lst:
        print(f"    {fn:22s} {b:3.0f}% → {a:3.0f}%   {d:+6.1f}pt")
    if not lst:
        print("    なし")
print(f"  変化なし {len(same)} 関数: " + (", ".join(fn.split('-')[0] for _, fn, *_ in same) or "なし"))
net = (sum(d for d, *_ in up) + sum(d for d, *_ in down)) / len(funcs)
print(f"  正味 24 関数平均 {net:+.2f}pt")

print("\n[3] MC-ESO(base) がいずれかの手法に SR@1e-10 で下回る関数")
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
    print(f"  {fn:22s} base {100*ref:3.0f}%  <  {m} {100*v:3.0f}%   差 {100*(v-ref):+5.0f}pt")
print(f"  計 {len(lose)} 関数: " + ", ".join(fn.split('-')[0] for fn, *_ in lose))

print("\n[3b] evals_succ_mean —— 両手法とも成功のある関数だけで平均（同一関数集合）")
common = [fn for fn in funcs
          if num(rows[fn][REF]["evals_succ_mean"]) is not None
          and num(rows[fn][ARM]["evals_succ_mean"]) is not None]
if common:
    eb = mean(num(rows[fn][REF]["evals_succ_mean"]) for fn in common)
    ea = mean(num(rows[fn][ARM]["evals_succ_mean"]) for fn in common)
    print(f"  共通 {len(common)} 関数: base {eb:.1f}   {ARM} {ea:.1f}   差 {ea-eb:+.1f}"
          f" ({100*(ea-eb)/eb:+.1f}%)")
    print("  " + ", ".join(fn.split('-')[0] for fn in common))
else:
    print("  共通関数なし")

print("\n[5] Wilcoxon（reference = MC-ESO、両側 p<0.05）")
w = {}
with open(WILCOXON, newline="") as f:
    for r in csv.DictReader(f):
        w.setdefault(r["method"], []).append(r)
print(f"  {'method':14s} {'有意に base が優':>18s} {'有意に base が劣':>18s} {'全 tie':>8s}")
for m in METHODS[1:]:
    better = worse = alltie = 0
    for r in w.get(m, []):
        p, a12 = num(r["p_value_two_sided"]), num(r["a12"])
        if int(r["tie_count"]) == int(r["n"]):
            alltie += 1
        if p is not None and p < 0.05 and a12 is not None:
            better += a12 > 0.5
            worse += a12 < 0.5
    print(f"  {m:14s} {better:18d} {worse:18d} {alltie:8d}")
print(f"\n  腕 {ARM} の有意な行（p<0.05）:")
anyrow = False
for r in w.get(ARM, []):
    p, a12 = num(r["p_value_two_sided"]), num(r["a12"])
    if p is not None and p < 0.05 and a12 is not None:
        anyrow = True
        side = "base が優" if a12 > 0.5 else "腕が優"
        print(f"    {r['function']:22s} p={p:.4g} A12={a12:.4f} ({r['a12_magnitude']})  {side}")
if not anyrow:
    print("    なし")

print("\n[6] BBOB 公式 5 群ごとの SR@1e-10（%）")
print(f"  {'group':36s} " + " ".join(f"{m:>13s}" for m in METHODS))
for label, rng in GROUPS:
    sel = [fn for fn in funcs if fnum(fn) in rng]
    cells = []
    for m in METHODS:
        vals = [pct(rows[fn][m]["sr_1e-10"]) for fn in sel if m in rows[fn]]
        cells.append(f"{100*mean(vals):12.2f}%" if vals else "            —")
    print(f"  {label:36s} " + " ".join(f"{c:>13s}" for c in cells) + f"   (n={len(sel)})")

print("\n[7] 0% 対 100% の 2 関数（F07 / F12）を名指しで")
for key in ("F07", "F12"):
    fn = next((f for f in funcs if f.startswith(key)), None)
    if fn is None:
        continue
    print(f"  {fn:22s} " + "  ".join(
        f"{m} {100*pct(rows[fn][m]['sr_1e-10']):.0f}%" for m in METHODS))
    print(f"    梯子 base  " + " ".join(
        f"{k.replace('sr_','')}={100*pct(rows[fn][REF][k]):.0f}%" for k in LEVELS)
        + f"  median_best_f={rows[fn][REF]['median_best_f']}")
    print(f"    梯子 {ARM} " + " ".join(
        f"{k.replace('sr_','')}={100*pct(rows[fn][ARM][k]):.0f}%" for k in LEVELS)
        + f"  median_best_f={rows[fn][ARM]['median_best_f']}")

print("\n[8] 包絡線への上乗せ（比較 2 手法 = CMA-ES / IPOP-CMA-ES の関数別ベストを基準）")
cmp_m = METHODS[2:]
vb = mean(max(pct(rows[fn][m]["sr_1e-10"]) for m in cmp_m) for fn in funcs)
for m in (REF, ARM):
    mm = mean(pct(rows[fn][m]["sr_1e-10"]) for fn in funcs)
    vbp = mean(max([pct(rows[fn][x]["sr_1e-10"]) for x in cmp_m] + [pct(rows[fn][m]["sr_1e-10"])])
               for fn in funcs)
    print(f"  {m:14s} {100*mm:6.2f}%   包絡線 {100*vb:6.2f}% との差 {100*(mm-vb):+6.2f}pt"
          f"   包絡線への上乗せ {100*(vbp-vb):+6.2f}pt")

print("\n[10] 反証条件の機械判定（判定線は prereg.md のとおり、数値を見る前に固定）")
f07 = next(f for f in funcs if f.startswith("F07"))
f12 = next(f for f in funcs if f.startswith("F12"))
a07, a12_ = pct(rows[f07][ARM]["sr_1e-10"]), pct(rows[f12][ARM]["sr_1e-10"])
base_all = mean(pct(rows[fn][REF]["sr_1e-10"]) for fn in funcs)
arm_all = mean(pct(rows[fn][ARM]["sr_1e-10"]) for fn in funcs)
escaped = (a07 >= 0.05) or (a12_ >= 0.05)
notdown = arm_all >= base_all - 1e-12
print(f"  F07 腕 {100*a07:.0f}% / F12 腕 {100*a12_:.0f}%"
      f"   → 「0% を脱した（≥5%）」= {escaped}")
print(f"  24 関数平均 base {100*base_all:.2f}% → 腕 {100*arm_all:.2f}%"
      f"   → 「全体が下がらない」= {notdown}")
print(f"  (a) どちらも 0% のまま ＝ 飢餓は原因ではない : {'発火' if not escaped else '不発'}")
print(f"  (b) 片方が脱し全体が下がらない ＝ 軸は生きている : "
      f"{'発火' if (escaped and notdown) else '不発'}")
print(f"  (c) 局所改善なのに全体が下がる ＝ 5 件目の棄却 : "
      f"{'発火' if (escaped and not notdown) else '不発'}")
