"""Wilcoxon win/loss counts with A12, per compared method (ref = MC-ESO)."""
import csv, sys, math
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
methods = []
for r in rows:
    if r["method"] not in methods:
        methods.append(r["method"])
for m in methods:
    better, worse, nan = [], [], 0
    mags = []
    for r in rows:
        if r["method"] != m:
            continue
        try:
            pv = float(r["p_value_two_sided"])
        except ValueError:
            pv = float("nan")
        if math.isnan(pv):
            nan += 1
            continue
        if pv < 0.05:
            a12 = float(r["a12"])
            tag = f'{r["function"].split("-")[0]}(A12={a12:.2f},{r["a12_magnitude"]})'
            mags.append(r["a12_magnitude"])
            (better if a12 > 0.5 else worse).append(tag)
    print(f"{m}: MC-ESO better {len(better)} / worse {len(worse)}  "
          f"(p=nan, all runs tied: {nan})")
    print(f"   better: {', '.join(better) or 'none'}")
    print(f"   worse : {', '.join(worse) or 'none'}")
    if mags:
        from collections import Counter
        print(f"   A12 magnitudes of the {len(mags)} significant: "
              f"{dict(Counter(mags))}")
