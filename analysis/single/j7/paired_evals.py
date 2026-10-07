#!/usr/bin/env python3
"""Paired `evals_succ_mean`: the project rule averages only over the functions
where BOTH compared methods have successes (docs/experiments.md 評価方法論).
`scripts/analyze_quick.py` [1] prints an unpaired mean, so this prints the
paired pair-by-pair figure that the 3-metric report needs.

usage: paired_evals.py SUMMARY.csv [REF]
"""
import csv, sys
from statistics import mean

SUM = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else "MC-ESO"
CMP = ["CMA-ES", "IPOP-CMA-ES", "BIPOP-CMA-ES", "DE", "L-SHADE"]


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
print(f"paired evals_succ_mean (ref={REF}), {len(funcs)} functions: "
      + ", ".join(f.split('-')[0] for f in funcs))
print(f"  {'method':14s} {'method evals':>13s} {REF+' evals':>14s} {'common funcs':>13s}")
for m in CMP:
    pairs = [(num(rows[fn][m]["evals_succ_mean"]), num(rows[fn][REF]["evals_succ_mean"]))
             for fn in funcs if m in rows[fn] and REF in rows[fn]]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        print(f"  {m:14s} {'---':>13s} {'---':>14s} {0:>13d}")
        continue
    print(f"  {m:14s} {mean(a for a, _ in pairs):13.1f} "
          f"{mean(b for _, b in pairs):14.1f} {len(pairs):13d}")
