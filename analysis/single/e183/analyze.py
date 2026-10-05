#!/usr/bin/env python3
"""その183 — cc_mu_frac の腕を 5D（ccmu50）と 10D（ccmu100）で 3 指標で読む。

その182 の analyze.py と同じ節構成で、違いは 3 つだけ:
  - 腕名と比較手法集合を argv で受ける（5D は 4 手法、10D は 2 手法）
  - [10] の機械判定を この回の事前登録 (a)(b)(c) に差し替えた
  - [11] を足した: 10D の ccmu100 と その182 の ccmu50 を同一 base 越しに並べる（反証条件 (c)）

使い方:
    python3 analysis/single/e183/analyze.py SUMMARY WILCOXON ARM "M1,M2,..."
"""
from __future__ import annotations
import csv, os, sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = sys.argv[1]
WILCOXON = sys.argv[2]
ARM = sys.argv[3] if len(sys.argv) > 3 else "ccmu50"
REF = "MC-ESO"
METHODS = (sys.argv[4].split(",") if len(sys.argv) > 4
           else [REF, ARM, "CMA-ES", "IPOP-CMA-ES"])
LEVELS = ["sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10"]
GROUPS = [
    ("g1 separable            (F01-F05)", range(1, 6)),
    ("g2 low/moderate cond.   (F06-F09)", range(6, 10)),
    ("g3 high cond. unimodal  (F10-F15)", range(10, 15)),
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


def load(path):
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d.setdefault(r["function"], {})[r["method"]] = r
    return d


rows = load(SUMMARY)
funcs = sorted(rows, key=fnum)
print(f"summary: {SUMMARY}")
print(f"functions: {len(funcs)}   methods: {', '.join(METHODS)}   arm: {ARM}")

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

if len(METHODS) > 2:
    print("\n[3] MC-ESO(base) がいずれかの比較手法に SR@1e-10 で下回る関数")
    lose = []
    for fn in funcs:
        ref = pct(rows[fn][REF]["sr_1e-10"])
        best_m, best_v = None, ref
        for m in METHODS[2:]:
            v = pct(rows[fn][m]["sr_1e-10"])
            if v > best_v + 1e-12:
                best_m, best_v = m, v
        if best_m:
            lose.append((fn, ref, best_m, best_v))
    for fn, ref, m, v in lose:
        print(f"  {fn:22s} base {100*ref:3.0f}%  <  {m} {100*v:3.0f}%   差 {100*(v-ref):+5.0f}pt")
    print(f"  計 {len(lose)} 関数 / 赤字 "
          f"{sum(100*(v-r) for _, r, _, v in lose)/len(funcs):.2f}pt: "
          + ", ".join(fn.split('-')[0] for fn, *_ in lose))

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
print(f"  腕 {ARM} の medium 以上だが有意に届かない行:")
for r in w.get(ARM, []):
    p, a12 = num(r["p_value_two_sided"]), num(r["a12"])
    if p is not None and p >= 0.05 and r.get("a12_magnitude") in ("medium", "large"):
        print(f"    {r['function']:22s} p={p:.4g} A12={a12:.4f} ({r['a12_magnitude']})")

print("\n[6] BBOB 公式 5 群ごとの SR@1e-10（%）")
print(f"  {'group':36s} " + " ".join(f"{m:>13s}" for m in METHODS))
for label, rng in GROUPS:
    sel = [fn for fn in funcs if fnum(fn) in rng]
    cells = []
    for m in METHODS:
        vals = [pct(rows[fn][m]["sr_1e-10"]) for fn in sel if m in rows[fn]]
        cells.append(f"{100*mean(vals):12.2f}%" if vals else "            —")
    print(f"  {label:36s} " + " ".join(f"{c:>13s}" for c in cells) + f"   (n={len(sel)})")

print("\n[7] 0% 対 100% の 2 関数（F07 / F12）を名指しで（この軸で追わないが、動いたかは記録する）")
for key in ("F07", "F12"):
    fn = next((f for f in funcs if f.startswith(key)), None)
    if fn is None:
        continue
    print(f"  {fn:22s} " + "  ".join(
        f"{m} {100*pct(rows[fn][m]['sr_1e-10']):.0f}%" for m in METHODS))
    for m in (REF, ARM):
        print(f"    梯子 {m:9s} " + " ".join(
            f"{k.replace('sr_',''):>6s}={100*pct(rows[fn][m][k]):3.0f}%" for k in LEVELS)
            + f"  median_best_f={rows[fn][m]['median_best_f']}")

if len(METHODS) > 2:
    print("\n[8] 包絡線への上乗せ（比較手法の関数別ベストを基準）")
    cmp_m = METHODS[2:]
    vb = mean(max(pct(rows[fn][m]["sr_1e-10"]) for m in cmp_m) for fn in funcs)
    for m in (REF, ARM):
        mm = mean(pct(rows[fn][m]["sr_1e-10"]) for fn in funcs)
        vbp = mean(max([pct(rows[fn][x]["sr_1e-10"]) for x in cmp_m]
                       + [pct(rows[fn][m]["sr_1e-10"])]) for fn in funcs)
        print(f"  {m:14s} {100*mm:6.2f}%   包絡線 {100*vb:6.2f}% との差 {100*(mm-vb):+6.2f}pt"
              f"   包絡線への上乗せ {100*(vbp-vb):+6.2f}pt")

print("\n[10] 反証条件の機械判定（判定線は e183/prereg.md のとおり、数値を見る前に固定）")
base_all = 100 * mean(pct(rows[fn][REF]["sr_1e-10"]) for fn in funcs)
arm_all = 100 * mean(pct(rows[fn][ARM]["sr_1e-10"]) for fn in funcs)
netd = arm_all - base_all
print(f"  24 関数平均 SR@1e-10: base {base_all:.2f}% → 腕 {arm_all:.2f}%  正味 {netd:+.2f}pt")
print(f"  (a) 正味 <= 0.00pt（軸を 5 件目の棄却として閉じる）      = {netd <= 0.0}")
print(f"  (b) 正味 >= +2.00pt（軸は次元をまたいで生きている）      = {netd >= 2.0}")
print(f"  どちらも不発（0 < 正味 < 2 ＝ 符号は正だが弱い）        = {0.0 < netd < 2.0}")

E182 = os.path.join(os.path.dirname(HERE), "e182", "summary_ccmu_10d.csv")
if ARM == "ccmu100" and os.path.exists(E182):
    print("\n[11] 反証条件 (c) —— 10D で ccmu100 と ccmu50 を同一 base 越しに並べる")
    old = load(E182)
    ofuncs = sorted(old, key=fnum)
    ob = 100 * mean(pct(old[fn]["MC-ESO"]["sr_1e-10"]) for fn in ofuncs)
    o50 = 100 * mean(pct(old[fn]["ccmu50"]["sr_1e-10"]) for fn in ofuncs)
    print(f"  その182: base {ob:.2f}%  ccmu50 {o50:.2f}%  ({o50-ob:+.2f}pt)")
    print(f"  この回 : base {base_all:.2f}%  ccmu100 {arm_all:.2f}%  ({netd:+.2f}pt)")
    print(f"  base の再現（決定的なはず）: {'一致' if abs(ob-base_all) < 1e-9 else f'差 {base_all-ob:+.4f}pt ＝ 不一致'}")
    print(f"  (c) ccmu100 > ccmu50 （効いているのは選抜ではなく本数そのもの） = {arm_all > o50}")
    print("\n  関数別（SR@1e-10 %）: base / ccmu50(その182) / ccmu100(この回)")
    for fn in funcs:
        b = 100 * pct(rows[fn][REF]["sr_1e-10"])
        a100 = 100 * pct(rows[fn][ARM]["sr_1e-10"])
        a50 = 100 * pct(old[fn]["ccmu50"]["sr_1e-10"]) if fn in old else float("nan")
        mark = "" if (a50 == a100 and a100 == b) else "   *"
        print(f"    {fn:22s} {b:5.0f} {a50:7.0f} {a100:8.0f}{mark}")
