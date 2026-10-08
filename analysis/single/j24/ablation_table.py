"""Per-function SR@1e-10 (and evals_succ_mean) for every arm in one run, plus
the diff of each arm against the reference arm.

usage: ablation_table.py SUMMARY.csv REF_METHOD [--csv OUT.csv]
"""
import csv, sys

path, ref = sys.argv[1], sys.argv[2]
out_csv = None
if "--csv" in sys.argv:
    out_csv = sys.argv[sys.argv.index("--csv") + 1]


def pct(s):
    s = (s or "").strip().rstrip("%")
    return float(s) if s not in ("", "N/A") else None


def num(s):
    s = (s or "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    # "inf" marks a function with no successful run: no evals_succ_mean exists.
    return v if v == v and abs(v) != float("inf") else None


sr, ev, methods, funcs = {}, {}, [], []
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        fn, m = r["function"], r["method"]
        sr[(fn, m)] = pct(r["sr_1e-10"])
        ev[(fn, m)] = num(r["evals_succ_mean"])
        if m not in methods:
            methods.append(m)
        if fn not in funcs:
            funcs.append(fn)
funcs.sort(key=lambda s: int(s[1:3]))
order = [ref] + [m for m in methods if m != ref]

w = max(len(m) for m in order) + 2
print(f"Per-function SR@1e-10 (%),  ref = {ref},  {len(funcs)} functions\n")
print("function".ljust(24) + "".join(m.rjust(w) for m in order)
      + "".join((f"d:{m}").rjust(w) for m in order[1:]))
print("-" * (24 + w * (2 * len(order) - 1)))
for fn in funcs:
    cells = [f"{sr[(fn, m)]:.0f}" if sr.get((fn, m)) is not None else "-" for m in order]
    diffs = []
    for m in order[1:]:
        a, b = sr.get((fn, ref)), sr.get((fn, m))
        diffs.append(f"{b - a:+.0f}" if (a is not None and b is not None) else "-")
    print(fn.ljust(24) + "".join(c.rjust(w) for c in cells)
          + "".join(d.rjust(w) for d in diffs))

print("\nOverall mean SR@1e-10 and diff vs ref:")
means = {}
for m in order:
    vals = [sr[(fn, m)] for fn in funcs if sr.get((fn, m)) is not None]
    means[m] = sum(vals) / len(vals)
    tag = "" if m == ref else f"   ({means[m] - means[ref]:+.2f}pt vs {ref})"
    print(f"  {m.ljust(14)} {means[m]:6.2f}%  (n={len(vals)}){tag}")

print("\nFunctions whose SR@1e-10 differs from ref:")
for m in order[1:]:
    chg = []
    for fn in funcs:
        a, b = sr.get((fn, ref)), sr.get((fn, m))
        if a is not None and b is not None and abs(a - b) > 1e-9:
            chg.append(f"{fn.split('-')[0]} {a:.0f}->{b:.0f}")
    print(f"  {m}: {len(chg)} changed" + (": " + ", ".join(chg) if chg else ""))

print("\nPaired evals_succ_mean (functions where ref and the arm both succeed):")
for m in order[1:]:
    common = [fn for fn in funcs
              if ev.get((fn, ref)) is not None and ev.get((fn, m)) is not None]
    if not common:
        print(f"  {m}: no common function")
        continue
    a = sum(ev[(fn, ref)] for fn in common) / len(common)
    b = sum(ev[(fn, m)] for fn in common) / len(common)
    print(f"  {m.ljust(14)} {b:8.1f}  vs ref {a:8.1f}  (nf={len(common)})")

if out_csv:
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["function"] + [f"sr1e10_{m}" for m in order]
                    + [f"diff_{m}_minus_{ref}" for m in order[1:]]
                    + [f"evals_{m}" for m in order])
        for fn in funcs:
            row = [fn]
            row += [sr.get((fn, m)) for m in order]
            row += [(sr[(fn, m)] - sr[(fn, ref)])
                    if (sr.get((fn, m)) is not None and sr.get((fn, ref)) is not None)
                    else None for m in order[1:]]
            row += [ev.get((fn, m)) for m in order]
            wr.writerow(row)
        wr.writerow(["MEAN"] + [round(means[m], 4) for m in order]
                    + [round(means[m] - means[ref], 4) for m in order[1:]]
                    + [None] * len(order))
    print(f"\nwrote {out_csv}")
