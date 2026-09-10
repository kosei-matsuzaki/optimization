#!/usr/bin/env python3
"""Entry 107 -- is MC-ESO's loss on the GECCO'2024 suite the reporting rule or
the search?

Reads the per-problem CSVs this cycle wrote. Each run is scored three ways off
the *same* evaluations (`current` / `reselect` / `history`), so every comparison
below is paired at the run level and cost no extra evaluations. Pre-registration
and rejection conditions: prereg.md.

    python3 analysis/mmo2024/e107/analyze.py [--csv out.csv]
"""
import argparse
import csv
import pathlib
import sys

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
PROBS = [f"M{i:02d}-D10-PIN01" for i in range(1, 17)]
RULES = ["current", "reselect", "history"]
EPS_COLS = ["pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]

# Entry 103's memoryless restart-lander null at n=500 draws, per problem:
# (`mpr`, `sup`). Means over the 16: 0.5529 and 0.7006. Entry 106's table.
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
# Entry 106's *seed-0* `current` rows, per problem, 1e-1..1e-5. Seed 0 is one of
# entry 106's three and the optimiser is seeded deterministically, so this
# entry's `current` rows must be bit-equal to these. Anything else means the
# scoring changes made this cycle moved the historical number, and the run is
# not a re-score of the same thing.
E106_SEED0 = {
    "M01-D10-PIN01": [0.65, 0.25, 0.05, 0.05, 0.05],
    "M02-D10-PIN01": [0.45, 0.25, 0.05, 0.05, 0.05],
    "M03-D10-PIN01": [0.05, 0.05, 0.05, 0.05, 0.05],
    "M04-D10-PIN01": [0.10, 0.10, 0.05, 0.05, 0.05],
    "M05-D10-PIN01": [0.05, 0.05, 0.05, 0.05, 0.05],
    "M06-D10-PIN01": [0.00, 0.00, 0.00, 0.00, 0.00],
    "M07-D10-PIN01": [0.05, 0.05, 0.05, 0.00, 0.00],
    "M08-D10-PIN01": [0.05, 0.05, 0.05, 0.05, 0.00],
    "M09-D10-PIN01": [0.60, 0.30, 0.10, 0.10, 0.10],
    "M10-D10-PIN01": [0.50, 0.30, 0.10, 0.10, 0.10],
    "M11-D10-PIN01": [0.20, 0.10, 0.10, 0.10, 0.10],
    "M12-D10-PIN01": [0.10, 0.10, 0.10, 0.10, 0.10],
    "M13-D10-PIN01": [0.10, 0.10, 0.10, 0.10, 0.10],
    "M14-D10-PIN01": [0.20, 0.20, 0.10, 0.10, 0.10],
    "M15-D10-PIN01": [0.10, 0.10, 0.10, 0.10, 0.10],
    "M16-D10-PIN01": [0.10, 0.10, 0.10, 0.10, 0.00],
}


def load():
    """{rule: {problem: (5-level mean per seed, per-eps mean, K, |rep|)}}."""
    # run.sh writes one CSV per problem under by_problem/ so a cut-short cycle
    # still leaves finished problems behind (entry 85's rule); those are merged
    # into rules_runs_d10.csv and the directory dropped (entry 106's pattern).
    rows = []
    for p in sorted((HERE / "by_problem").glob("*.csv")):
        rows += list(csv.DictReader(p.open()))
    merged = HERE / "rules_runs_d10.csv"
    if not rows and merged.exists():
        rows = list(csv.DictReader(merged.open()))
    if not rows:
        sys.exit("no run table found -- did run.sh finish a problem?")
    out = {r: {} for r in RULES}
    for rule in RULES:
        for nm in PROBS:
            sel = [r for r in rows
                   if r["function"] == nm and r["rule"] == rule]
            if not sel:
                continue
            per_eps = np.array([[float(r[c]) for c in EPS_COLS] for r in sel])
            out[rule][nm] = (per_eps.mean(axis=1), per_eps.mean(axis=0),
                             int(sel[0]["n_optima"]),
                             float(np.mean([float(r["n_reported"])
                                            for r in sel])))
    return out, rows


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
    ap.add_argument("--csv", type=pathlib.Path,
                    default=HERE / "rules_d10.csv")
    args = ap.parse_args()
    got, rows = load()

    names = [n for n in PROBS if n in got["current"]]
    missing = [n for n in PROBS if n not in names]
    seeds = sorted({len(v[0]) for v in got["current"].values()})
    print("== entry 107: MC-ESO at D=10, PIN01, budget 500,000, one set of "
          "runs scored three ways")
    print(f"problems finished: {len(names)}/16"
          + (f"   MISSING: {', '.join(missing)}" if missing else ""))
    print(f"seeds per problem: {seeds}   (rules are paired within the run)")

    # -- wiring check: `current` must reproduce entry 106's seed-0 rows exactly
    bad = [nm for nm in names
           if not np.allclose(got["current"][nm][1], E106_SEED0[nm], atol=1e-9)]
    print("wiring check vs entry 106 seed 0: "
          + ("all 16 rows bit-equal" if not bad and len(names) == 16
             else f"{len(names) - len(bad)}/{len(names)} equal"
                  + (f"   DIFFER: {', '.join(bad)}" if bad else "")))

    # -- per problem, the three rules side by side, at the level the loss lives
    print(f"\n{'problem':<16}{'K':>3}"
          f"{'cur@1e-1':>10}{'resel@1e-1':>12}{'hist@1e-1':>11}"
          f"{'cur MPR':>9}{'resel':>8}{'hist':>8}"
          f"{'null mpr':>10}{'|hist|':>9}")
    cols = {r: [] for r in RULES}
    e1 = {r: [] for r in RULES}
    for nm in names:
        line = f"{nm:<16}{got['current'][nm][2]:>3}"
        for r in RULES:
            e1[r].append(got[r][nm][1][0])
        for r in RULES:
            line += f"{got[r][nm][1][0]:>{10 if r == 'current' else (12 if r == 'reselect' else 11)}.4f}"
        for r in RULES:
            v = float(got[r][nm][0].mean())
            cols[r].append(v)
            line += f"{v:>{9 if r == 'current' else 8}.4f}"
        line += f"{NULL[nm][0]:>10.4f}{got['history'][nm][3]:>9.0f}"
        print(line)

    cols = {r: np.array(v) for r, v in cols.items()}
    e1 = {r: np.array(v) for r, v in e1.items()}
    print(f"\n{'mean':<16}{'':>3}"
          f"{e1['current'].mean():>10.4f}{e1['reselect'].mean():>12.4f}"
          f"{e1['history'].mean():>11.4f}"
          f"{cols['current'].mean():>9.4f}{cols['reselect'].mean():>8.4f}"
          f"{cols['history'].mean():>8.4f}"
          f"{np.mean([NULL[n][0] for n in names]):>10.4f}")

    # -- accuracy ladder for each rule
    NULL_EPS = np.array([0.7437, 0.7219, 0.7156, 0.6906, 0.6312])
    print("\n== accuracy ladder, 16-problem mean")
    print(f"{'rule':<10}" + "".join(f"{c.replace('pr_',''):>9}"
                                    for c in EPS_COLS)
          + f"{'1e-1 -> 1e-5':>15}{'retained':>10}")
    for r in RULES:
        v = np.array([got[r][n][1] for n in names]).mean(axis=0)
        print(f"{r:<10}" + "".join(f"{x:>9.4f}" for x in v)
              + f"{v[0] - v[-1]:>+15.4f}{v[-1] / v[0] if v[0] else 0:>10.1%}")
    print(f"{'null sup':<10}" + "".join(f"{x:>9.4f}" for x in NULL_EPS)
          + f"{NULL_EPS[0] - NULL_EPS[-1]:>+15.4f}"
          f"{NULL_EPS[-1] / NULL_EPS[0]:>10.1%}")

    # -- paired tests between the rules (same runs)
    print("\n== paired tests between rules, unit = problem "
          "(two-sided Wilcoxon, alpha=0.05)")
    for lbl, a, b in (("reselect - current  @1e-1", e1["reselect"], e1["current"]),
                      ("history  - current  @1e-1", e1["history"], e1["current"]),
                      ("history  - reselect @1e-1", e1["history"], e1["reselect"]),
                      ("reselect - current  5-level", cols["reselect"], cols["current"]),
                      ("history  - current  5-level", cols["history"], cols["current"])):
        w, p, rb, (up, dn, tie) = paired(a, b)
        print(f"   {lbl:<28} delta={np.mean(a) - np.mean(b):+.4f}  W={w:>6.1f}"
              f"  p={p:.4g}  rb={rb:+.3f}  win/loss/tie={up}/{dn}/{tie}"
              f"  -> {'significant' if p < 0.05 else 'NOT significant'}")

    # -- the pre-registered branches
    print("\n== prereg branches (prereg.md), read on the 16-problem mean @1e-1")
    rs, hs = float(e1["reselect"].mean()), float(e1["history"].mean())
    if rs <= 0.30:
        print(f"   reselect {rs:.4f} <= 0.30  -> BRANCH 1: the reporting rule "
              "is NOT the bottleneck; the loss is in the search")
    elif rs > 0.50:
        print(f"   reselect {rs:.4f} > 0.50   -> BRANCH 2: entry 106 measured "
              "the reporting rule, not the search; entry 106 + status.md need "
              "correcting")
    else:
        print(f"   0.30 < reselect {rs:.4f} <= 0.50 -> BRANCH 3: partial split")
    print(f"   rho-free check: history (supremum over ALL reporting rules) "
          f"= {hs:.4f}"
          + ("  <= 0.30 -> branch 1 holds for every rho and every selection "
             "rule" if hs <= 0.30 else
             "  > 0.30 -> the ceiling is above the threshold; the cap/selection "
             "is implicated, see the reselect-vs-history row"))

    with args.csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "rule", "seeds", "n_reported_mean",
                    "mpr_5level"] + [f"mpr_{c}" for c in EPS_COLS])
        for nm in names:
            for r in RULES:
                vals, per_eps, K, rep = got[r][nm]
                w.writerow([nm, K, r, len(vals), f"{rep:.1f}",
                            f"{vals.mean():.6f}"]
                           + [f"{v:.6f}" for v in per_eps])
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
