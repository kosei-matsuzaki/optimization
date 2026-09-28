#!/usr/bin/env python3
"""e181: compare MC-ESO / BBOB-24 dim2 across three arms.

  OLD  : commit 25418d9 (the commit that recorded pin 93.54 / 798), current libs
  HEAD : current HEAD, current libs  (the environment's base, 92.08 / 677.7)
  LIBS : current HEAD, ioh 0.3.18 + numpy 1.26.4

Prints the three headline numbers, the per-function SR@1e-10 table, and where
the arms disagree. No extra optimisation runs: reads the summary.csv files.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path

ARMS = sys.argv[1:]  # label=path/to/dim2/summary.csv


def read(path: str) -> dict[str, dict]:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["method"] != "MC-ESO":
                continue
            out[row["function"]] = row
    return out


def pct(s: str) -> float:
    return float(s.rstrip("%"))


data = {}
for a in ARMS:
    label, path = a.split("=", 1)
    data[label] = read(path)

labels = list(data)
funcs = sorted(data[labels[0]])
LEVELS = ["sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10"]

print("=" * 78)
print("headline (24-function mean, n=20 per function, 5000 evals, dim 2)")
print("=" * 78)
hdr = f"{'arm':<6}" + "".join(f"{l.replace('sr_','SR@'):>10}" for l in LEVELS) + f"{'evals_succ_mean':>17}"
print(hdr)
for lab in labels:
    d = data[lab]
    cells = []
    for lv in LEVELS:
        cells.append(f"{sum(pct(d[f][lv]) for f in funcs)/len(funcs):>10.2f}")
    # evals_succ_mean: unweighted mean over the functions that have one
    ev = [float(d[f]["evals_succ_mean"]) for f in funcs
          if d[f]["evals_succ_mean"] not in ("", "N/A", "---", "inf")]
    print(f"{lab:<6}" + "".join(cells) + f"{sum(ev)/len(ev):>17.1f}"
          + f"   (n_func with evals={len(ev)})")

print()
print("=" * 78)
print("per-function SR@1e-10 (%) and evals_succ_mean")
print("=" * 78)
print(f"{'function':<24}" + "".join(f"{l+' SR':>12}" for l in labels)
      + "".join(f"{l+' ev':>12}" for l in labels) + "  flag")
ndiff_sr = ndiff_ev = 0
for f in funcs:
    srs = [pct(data[l][f]["sr_1e-10"]) for l in labels]
    evs = [data[l][f]["evals_succ_mean"] for l in labels]
    flag = ""
    if len(set(srs)) > 1:
        flag += " SR-DIFF"; ndiff_sr += 1
    if len(set(evs)) > 1:
        flag += " EV-DIFF"; ndiff_ev += 1
    print(f"{f:<24}" + "".join(f"{s:>12.0f}" for s in srs)
          + "".join(f"{e:>12}" for e in evs) + " " + flag)
print()
print(f"functions where SR@1e-10 differs between arms : {ndiff_sr} / {len(funcs)}")
print(f"functions where evals_succ_mean differs       : {ndiff_ev} / {len(funcs)}")

# ── full-row identity check ───────────────────────────────────────────────
# SR alone would hide a difference that does not cross a threshold. Compare
# every column the runner writes, cell by cell, across the arms.
SKIP = {"mean_time_s"}   # wall clock, not a result
if len(labels) > 1:
    ref = labels[0]
    cols = [c for c in data[ref][funcs[0]] if c not in SKIP]
    print()
    print("=" * 78)
    print(f"cell-by-cell identity against arm {ref} "
          f"({len(funcs)} functions x {len(cols)} columns)")
    print("=" * 78)
    for lab in labels[1:]:
        diffs = []
        for f in funcs:
            if f not in data[lab]:
                continue
            for c in cols:
                a, b = data[ref][f].get(c), data[lab][f].get(c)
                if a != b:
                    diffs.append((f, c, a, b))
        ncmp = len([f for f in funcs if f in data[lab]]) * len(cols)
        print(f"{lab:<6} {len(diffs):>4} differing cells out of {ncmp}")
        for f, c, a, b in diffs[:20]:
            print(f"       {f:<24}{c:<20}{ref}={a!r}  {lab}={b!r}")
