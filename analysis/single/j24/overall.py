"""Overall SR ladder + own/paired evals_succ_mean per method (ref = MC-ESO)."""
import csv, sys
path, ref = sys.argv[1], "MC-ESO"
LV = ["sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10"]


def pct(s):
    s = (s or "").strip().rstrip("%")
    return float(s) if s not in ("", "N/A") else None


def num(s):
    try:
        v = float((s or "").strip())
    except ValueError:
        return None
    return v if v == v and abs(v) != float("inf") else None


sr, ev, methods, funcs = {}, {}, [], []
for r in csv.DictReader(open(path, newline="")):
    fn, m = r["function"], r["method"]
    for lv in LV:
        sr[(fn, m, lv)] = pct(r[lv])
    ev[(fn, m)] = num(r["evals_succ_mean"])
    if m not in methods:
        methods.append(m)
    if fn not in funcs:
        funcs.append(fn)
order = [ref] + [m for m in methods if m != ref]
print(f"functions={len(funcs)}  methods={len(order)}\n")
hdr = ("method".ljust(12) + "".join(l.replace('sr_', 'SR@').rjust(10) for l in LV)
       + "evals_own".rjust(12) + "(nf)".rjust(6)
       + "evals_pair".rjust(12) + "ref_pair".rjust(10) + "(nf)".rjust(6))
print(hdr); print("-" * len(hdr))
for m in order:
    cells = []
    for lv in LV:
        v = [sr[(fn, m, lv)] for fn in funcs if sr.get((fn, m, lv)) is not None]
        cells.append(f"{sum(v)/len(v):.2f}%")
    own = [ev[(fn, m)] for fn in funcs if ev.get((fn, m)) is not None]
    line = m.ljust(12) + "".join(c.rjust(10) for c in cells)
    line += f"{sum(own)/len(own):.1f}".rjust(12) + f"{len(own)}".rjust(6)
    if m == ref:
        line += "-".rjust(12) + "-".rjust(10) + "-".rjust(6)
    else:
        common = [fn for fn in funcs
                  if ev.get((fn, m)) is not None and ev.get((fn, ref)) is not None]
        a = sum(ev[(fn, m)] for fn in common) / len(common)
        b = sum(ev[(fn, ref)] for fn in common) / len(common)
        line += f"{a:.1f}".rjust(12) + f"{b:.1f}".rjust(10) + f"{len(common)}".rjust(6)
    print(line)
print("\nevals_own  = averaged over functions where THIS method succeeds (nf)")
print("evals_pair = averaged over functions where BOTH it and MC-ESO succeed; "
      "ref_pair = MC-ESO on the same set")
