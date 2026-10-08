#!/usr/bin/env python3
"""Per-function SR@1e-10 change between two budgets, for every method.

The 3-metric report has to list every function whose SR@1e-10 changed, so this
diffs two summary.csv files function by function and method by method.

usage: per_function_sr.py OLD.csv NEW.csv [OLD_LABEL NEW_LABEL]
"""
import csv, sys

OLD, NEW = sys.argv[1], sys.argv[2]
OLD_L = sys.argv[3] if len(sys.argv) > 3 else "old"
NEW_L = sys.argv[4] if len(sys.argv) > 4 else "new"
METHODS = ["MC-ESO", "CMA-ES", "IPOP-CMA-ES", "BIPOP-CMA-ES", "DE", "L-SHADE"]


def pct(s):
    s = (s or "").strip().rstrip("%")
    return float(s) if s not in ("", "N/A") else None


def load(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["function"], r["method"])] = pct(r["sr_1e-10"])
    return out


old, new = load(OLD), load(NEW)
funcs = sorted({fn for fn, _ in new}, key=lambda s: int(s[1:3]))
print(f"SR@1e-10: {OLD_L} -> {NEW_L}   {len(funcs)} functions")
for m in METHODS:
    chg, same, miss = [], [], []
    for fn in funcs:
        a, b = old.get((fn, m)), new.get((fn, m))
        if a is None or b is None:
            miss.append(fn.split("-")[0])
        elif abs(a - b) > 1e-9:
            chg.append((fn, a, b))
        else:
            same.append((fn, b))
    up = sum(1 for _, a, b in chg if b > a)
    down = len(chg) - up
    print(f"\n  {m}: 改善 {up} / 悪化 {down} / 変化なし {len(same)}"
          + (f" / 欠落 {len(miss)} ({', '.join(miss)})" if miss else ""))
    for fn, a, b in chg:
        print(f"    {fn:24s} {a:5.0f} -> {b:5.0f}  ({b-a:+.0f}pt)")
    if same:
        print("    変化なし: " + ", ".join(f"{fn.split('-')[0]}({b:.0f})" for fn, b in same))
