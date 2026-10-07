#!/usr/bin/env python3
"""Job 9 — 包絡線への上乗せ: the e183 analyze.py [8] formula applied to the
5 comparison methods as the envelope and MC-ESO as the 6th method.

usage: envelope.py SUMMARY.csv [LABEL]
"""
import csv, sys
from statistics import mean

SUM = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else SUM
REF = "MC-ESO"
CMP = ["CMA-ES", "IPOP-CMA-ES", "BIPOP-CMA-ES", "DE", "L-SHADE"]
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
with open(SUM, newline="") as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["function"], {})[r["method"]] = r
funcs = sorted(rows, key=lambda s: int(s[1:3]))
print(f"=== {LABEL} ===")
print(f"functions: {len(funcs)}  ({', '.join(f.split('-')[0] for f in funcs)})")

print("\n[1] 全体平均 SR（%）と evals_succ_mean")
print(f"  {'method':14s} " + " ".join(f"{k.replace('sr_','SR@'):>10s}" for k in LEVELS)
      + f" {'evals_succ_mean':>16s}")
for m in [REF] + CMP:
    cells = [rows[fn][m] for fn in funcs if m in rows[fn]]
    srs = [mean(pct(c[k]) for c in cells) for k in LEVELS]
    ev = [v for v in (num(c["evals_succ_mean"]) for c in cells) if v is not None]
    ev_s = f"{mean(ev):16.1f}" if ev else f"{'---':>16s}"
    print(f"  {m:14s} " + " ".join(f"{100*v:9.2f}%" for v in srs)
          + f" {ev_s}  (成功のある関数 {len(ev)}/{len(cells)})")

print("\n[2] 関数別包絡線（比較 5 手法の関数別ベスト SR@1e-10）と MC-ESO")
print(f"  {'function':24s} {'envelope5':>10s} {'MC-ESO':>8s} {'envelope6':>10s} {'上乗せ':>8s}  best-of-5")
d_sum = 0.0
for fn in funcs:
    v5 = max(pct(rows[fn][m]["sr_1e-10"]) for m in CMP)
    best = [m for m in CMP if abs(pct(rows[fn][m]["sr_1e-10"]) - v5) < 1e-12]
    ref = pct(rows[fn][REF]["sr_1e-10"])
    v6 = max(v5, ref)
    d_sum += v6 - v5
    print(f"  {fn:24s} {100*v5:9.0f}% {100*ref:7.0f}% {100*v6:9.0f}% {100*(v6-v5):+7.0f}pt  "
          + ",".join(best))
e5 = mean(max(pct(rows[fn][m]["sr_1e-10"]) for m in CMP) for fn in funcs)
e6 = mean(max([pct(rows[fn][m]["sr_1e-10"]) for m in CMP] + [pct(rows[fn][REF]["sr_1e-10"])])
          for fn in funcs)
mm = mean(pct(rows[fn][REF]["sr_1e-10"]) for fn in funcs)
print(f"\n  包絡線（比較 5 手法）           {100*e5:6.2f}%")
print(f"  包絡線（MC-ESO を 6 手法目に）  {100*e6:6.2f}%")
print(f"  **包絡線への上乗せ            {100*(e6-e5):+6.2f}pt**")
print(f"  MC-ESO 単独 {100*mm:.2f}%   包絡線との差 {100*(mm-e5):+.2f}pt")
print(f"  上乗せが出た関数: " + (", ".join(
    fn.split('-')[0] for fn in funcs
    if pct(rows[fn][REF]["sr_1e-10"]) > max(pct(rows[fn][m]["sr_1e-10"]) for m in CMP) + 1e-12)
    or "なし"))
