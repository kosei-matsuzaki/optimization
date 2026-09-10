#!/usr/bin/env python3
"""Entry 106 -- MC-ESO's measured MPR on the GECCO'2024 suite against the null.

Reads the per-problem CSVs this cycle wrote and pairs them, problem by problem,
with the memoryless restart-lander null already in the record (entry 103, 500
draws). Pre-registration and rejection conditions: prereg.md.

    python3 analysis/mmo2024/e106/analyze.py [--csv out.csv]
"""
import argparse
import csv
import pathlib
import sys

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
PROBS = [f"M{i:02d}-D10-PIN01" for i in range(1, 17)]

# Entry 103, `== per problem, 400 -> 500 draws`, the n=500 column. `mpr` is the
# memoryless ceiling (n' i.i.d. restarts inside the same budget); `sup` is
# support/K, i.e. what perfect redundancy avoidance over the same landing
# distribution would score. Means over the 16: 0.5529 and 0.7006.
NULL = {
    "M01-D10-PIN01": (0.5898, 0.8000), "M02-D10-PIN01": (0.4615, 0.7400),
    "M03-D10-PIN01": (0.5296, 0.8300), "M04-D10-PIN01": (0.6333, 0.8000),
    "M05-D10-PIN01": (0.6742, 0.8000), "M06-D10-PIN01": (0.3823, 0.6000),
    "M07-D10-PIN01": (0.4959, 0.6000), "M08-D10-PIN01": (0.6421, 0.8000),
    "M09-D10-PIN01": (0.4593, 0.5400), "M10-D10-PIN01": (0.6309, 0.8400),
    "M11-D10-PIN01": (0.5188, 0.6000), "M12-D10-PIN01": (0.7464, 0.8000),
    "M13-D10-PIN01": (0.7975, 0.9000), "M14-D10-PIN01": (0.2979, 0.3000),
    "M15-D10-PIN01": (0.4185, 0.6600), "M16-D10-PIN01": (0.5692, 0.6000),
}
PUBLISHED_D10 = 0.651                       # GECCO'2024 deck, 16-problem mean
EPS_COLS = ["pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]


def load(method):
    """{problem: (per-seed 5-level-mean MPR, per-seed per-eps array, K, |rep|)}."""
    out = {}
    for nm in PROBS:
        p = HERE / "by_problem" / f"{nm}.csv"
        if not p.exists():
            continue
        rows = [r for r in csv.DictReader(p.open()) if r["method"] == method]
        if not rows:
            continue
        per_eps = np.array([[float(r[c]) for c in EPS_COLS] for r in rows])
        out[nm] = (per_eps.mean(axis=1), per_eps,
                   int(rows[0]["n_optima"]),
                   float(np.mean([float(r["n_reported"]) for r in rows])))
    return out


def paired(a, b):
    """Wilcoxon signed-rank on paired per-problem values + rank-biserial."""
    d = np.asarray(a) - np.asarray(b)
    nz = d[d != 0]
    if nz.size == 0:
        return float("nan"), 1.0, 0.0, (0, 0, len(d))
    w, p = stats.wilcoxon(a, b, zero_method="wilcox")
    r = stats.rankdata(np.abs(nz))
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return float(w), float(p), float(rb), (int((d > 0).sum()),
                                           int((d < 0).sum()),
                                           int((d == 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="MC-ESO")
    ap.add_argument("--csv", type=pathlib.Path,
                    default=HERE / "mceso_vs_null_d10.csv")
    args = ap.parse_args()

    got = load(args.method)
    missing = [n for n in PROBS if n not in got]
    print(f"== entry 106: {args.method} at D=10, PIN01, budget 500,000 "
          f"(5-level mean MPR on the reported set)")
    print(f"problems finished: {len(got)}/16"
          + (f"   MISSING: {', '.join(missing)}" if missing else ""))
    if not got:
        sys.exit("no per-problem CSV found -- did run.sh finish a problem?")
    nseed = {len(v[0]) for v in got.values()}
    print(f"seeds per problem: {sorted(nseed)}")

    print(f"\n{'problem':<16}{'K':>3}{'|rep|':>7}{'MPR':>9}{'sd':>8}"
          f"{'null mpr':>10}{'diff':>9}{'null sup':>10}{'diff':>9}"
          f"   per-eps 1e-1..1e-5")
    names = [n for n in PROBS if n in got]
    m, n0, ns = [], [], []
    for nm in names:
        vals, per_eps, K, rep = got[nm]
        mu, sd = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        a, b = NULL[nm]
        m.append(mu); n0.append(a); ns.append(b)
        cells = " ".join(f"{v:5.3f}" for v in per_eps.mean(axis=0))
        print(f"{nm:<16}{K:>3}{rep:>7.0f}{mu:>9.4f}{sd:>8.4f}"
              f"{a:>10.4f}{mu - a:>+9.4f}{b:>10.4f}{mu - b:>+9.4f}   {cells}")

    m, n0, ns = np.array(m), np.array(n0), np.array(ns)
    print(f"\n{'mean':<16}{'':>3}{'':>7}{m.mean():>9.4f}{'':>8}"
          f"{n0.mean():>10.4f}{m.mean() - n0.mean():>+9.4f}"
          f"{ns.mean():>10.4f}{m.mean() - ns.mean():>+9.4f}")
    print(f"{'published 0.651':<16}{'':>3}{'':>7}{'':>9}{'':>8}"
          f"{'':>10}{m.mean() - PUBLISHED_D10:>+9.4f} (method - published)")

    print("\n== paired tests, unit = problem (two-sided Wilcoxon, alpha=0.05)")
    for lbl, ref in (("vs null mpr  (memoryless)", n0),
                     ("vs null sup  (no redundancy)", ns)):
        w, p, rb, (up, dn, tie) = paired(m, ref)
        verdict = "significant" if p < 0.05 else "NOT significant"
        print(f"   {lbl:<30} W={w:>7.1f}  p={p:.4g}  rank-biserial={rb:+.3f}"
              f"  win/loss/tie={up}/{dn}/{tie}  -> {verdict}")

    print("\n== prereg branches (prereg.md)")
    mm = m.mean()
    if mm > 0.7006:
        print(f"   mean {mm:.4f} > 0.7006  -> BRANCH 1: repulsion widens the "
              "support itself; the 0.299 reachability estimate must be redone")
    elif mm < 0.5529:
        print(f"   mean {mm:.4f} < 0.5529  -> BRANCH 2: MC-ESO has a memory and "
              "still loses to the memoryless null; the loss is not redundancy")
    else:
        rec = (mm - 0.5529) / (0.7006 - 0.5529)
        print(f"   0.5529 <= mean {mm:.4f} <= 0.7006  -> BRANCH 3: the split "
              f"survives; MC-ESO has recovered {rec:.0%} of the 0.148 "
              "redundancy loss")

    with args.csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "n_reported_mean", "seeds", "mpr_mean",
                    "mpr_sd", "null_mpr", "null_sup"]
                   + [f"mpr_{c}" for c in EPS_COLS])
        for nm in names:
            vals, per_eps, K, rep = got[nm]
            a, b = NULL[nm]
            w.writerow([nm, K, f"{rep:.1f}", len(vals), f"{vals.mean():.6f}",
                        f"{vals.std(ddof=1) if len(vals) > 1 else 0:.6f}",
                        a, b] + [f"{v:.6f}" for v in per_eps.mean(axis=0)])
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
