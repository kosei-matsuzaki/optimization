#!/usr/bin/env python3
"""Entry 109 -- NMMSO on the GECCO'2024 suite at D=10, full budget (500,000
evaluations), 3 seeds, scored the same way entries 106/107 scored MC-ESO.

Turns the suite's only comparison (MC-ESO vs our own memoryless null) into a
ranking. Pre-registration and rejection conditions: prereg.md.

    python3 analysis/mmo2024/e109/analyze.py [--csv out.csv]
"""
import argparse
import csv
import pathlib
import sys

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
PROBS = [f"M{i:02d}-D10-PIN01" for i in range(1, 17)]
EPS_COLS = ["pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]

# Entry 106's per-problem MC-ESO seed means (3 seeds, same budget, same scorer)
# and entry 103's 500-draw memoryless-null values, from
# analysis/mmo2024/e106/mceso_vs_null_d10.csv:
#   problem -> (K, MC-ESO 5-level mpr, MC-ESO pr@1e-1, null mpr, null sup)
REF = {
    "M01-D10-PIN01": (20, 0.200000, 0.550000, 0.5898, 0.80),
    "M02-D10-PIN01": (20, 0.160000, 0.450000, 0.4615, 0.74),
    "M03-D10-PIN01": (20, 0.056667, 0.083333, 0.5296, 0.83),
    "M04-D10-PIN01": (20, 0.080000, 0.116667, 0.6333, 0.80),
    "M05-D10-PIN01": (20, 0.050000, 0.050000, 0.6742, 0.80),
    "M06-D10-PIN01": (20, 0.110000, 0.183333, 0.3823, 0.60),
    "M07-D10-PIN01": (20, 0.043333, 0.050000, 0.4959, 0.60),
    "M08-D10-PIN01": (20, 0.040000, 0.050000, 0.6421, 0.80),
    "M09-D10-PIN01": (10, 0.220000, 0.500000, 0.4593, 0.54),
    "M10-D10-PIN01": (10, 0.140000, 0.233333, 0.6309, 0.84),
    "M11-D10-PIN01": (10, 0.113333, 0.166667, 0.5188, 0.60),
    "M12-D10-PIN01": (10, 0.106667, 0.133333, 0.7464, 0.80),
    "M13-D10-PIN01": (10, 0.100000, 0.100000, 0.7975, 0.90),
    "M14-D10-PIN01": (10, 0.046667, 0.066667, 0.2979, 0.30),
    "M15-D10-PIN01": (10, 0.086667, 0.100000, 0.4185, 0.66),
    "M16-D10-PIN01": (10, 0.080000, 0.100000, 0.5692, 0.60),
}
PUBLISHED_BEST = 0.651  # GECCO'2024 result sheet, 16-problem mean, D=10 (entry 92)


def load(max_seed):
    """{problem: (per-seed 5-level means, per-eps mean, K, |rep| mean, n_seeds)}

    ``max_seed`` keeps the seed sets balanced across problems. This cycle was
    cut after two complete seeds to stay inside the window, but two seed-2 runs
    (M05, M06) were already in flight and landed anyway; they are kept in the
    stored table and excluded from the headline so the 16-problem mean is not a
    mix of 2-seed and 3-seed estimates.
    """
    rows = []
    for p in sorted((HERE / "by_run").glob("*.csv")):
        rows += list(csv.DictReader(p.open()))
    merged = HERE / "nmmso_runs_d10.csv"
    if not rows and merged.exists():
        rows = list(csv.DictReader(merged.open()))
    if not rows:
        sys.exit("no run table found -- did run.sh finish a shard?")
    extra = [r for r in rows if int(r["seed"]) > max_seed]
    if extra:
        print("runs held out of the headline to keep seeds balanced: "
              + ", ".join(f"{r['function']} seed {r['seed']} "
                          f"(1e-1 {float(r['pr_1e-1']):.4f})" for r in extra))
    rows = [r for r in rows if int(r["seed"]) <= max_seed]
    out = {}
    for nm in PROBS:
        sel = [r for r in rows if r["function"] == nm]
        if not sel:
            continue
        per_eps = np.array([[float(r[c]) for c in EPS_COLS] for r in sel])
        out[nm] = (per_eps.mean(axis=1), per_eps.mean(axis=0),
                   int(sel[0]["n_optima"]),
                   float(np.mean([float(r["n_reported"]) for r in sel])),
                   len(sel))
    return out, rows


def paired(a, b):
    """Wilcoxon signed-rank on paired per-problem values + rank-biserial."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
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
    ap.add_argument("--csv", type=pathlib.Path, default=HERE / "nmmso_d10.csv")
    ap.add_argument("--max-seed", type=int, default=1,
                    help="highest seed index in the headline (default 1 = the "
                         "two balanced seeds this cycle finished)")
    args = ap.parse_args()
    got, rows = load(args.max_seed)

    names = [n for n in PROBS if n in got]
    missing = [n for n in PROBS if n not in names]
    print("== entry 109: NMMSO at D=10, PIN01, budget 500,000, reporting rule "
          "'current'")
    print(f"problems finished: {len(names)}/16"
          + (f"   MISSING: {', '.join(missing)}" if missing else ""))
    print(f"runs: {len(rows)}   seeds per problem: "
          f"{sorted({got[n][4] for n in names})}")

    # -- NMMSO does not reproduce at a fixed seed, so the spread is reported,
    #    not hidden inside a mean (prereg's threat 1).
    print(f"\n{'problem':<16}{'K':>3}{'NMMSO@1e-1':>12}{'NMMSO MPR':>11}"
          f"{'sd':>8}{'MC-ESO MPR':>12}{'null mpr':>10}{'null sup':>10}"
          f"{'|rep|':>7}")
    nm_mpr, nm_e1, mc_mpr, mc_e1, nu_mpr, nu_sup = [], [], [], [], [], []
    volatile = []
    for nm in names:
        vals, per_eps, K, rep, ns = got[nm]
        K_ref, mc, mce1, nu, nus = REF[nm]
        assert K == K_ref, f"{nm}: K {K} != {K_ref}"
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        m = float(vals.mean())
        if len(vals) > 1 and sd > m > 0:
            volatile.append(nm)
        nm_mpr.append(m); nm_e1.append(float(per_eps[0]))
        mc_mpr.append(mc); mc_e1.append(mce1)
        nu_mpr.append(nu); nu_sup.append(nus)
        print(f"{nm:<16}{K:>3}{per_eps[0]:>12.4f}{m:>11.4f}{sd:>8.4f}"
              f"{mc:>12.4f}{nu:>10.4f}{nus:>10.4f}{rep:>7.0f}")
    print(f"{'mean':<16}{'':>3}{np.mean(nm_e1):>12.4f}{np.mean(nm_mpr):>11.4f}"
          f"{'':>8}{np.mean(mc_mpr):>12.4f}{np.mean(nu_mpr):>10.4f}"
          f"{np.mean(nu_sup):>10.4f}")
    print("problems whose seed spread exceeds their own mean: "
          + (", ".join(volatile) if volatile else "none"))

    # -- accuracy ladder
    print("\n== accuracy ladder, 16-problem mean")
    print(f"{'method':<12}" + "".join(f"{c.replace('pr_', ''):>9}"
                                      for c in EPS_COLS)
          + f"{'1e-1 -> 1e-5':>15}{'retained':>10}")
    v = np.array([got[n][1] for n in names]).mean(axis=0)
    print(f"{'NMMSO':<12}" + "".join(f"{x:>9.4f}" for x in v)
          + f"{v[0] - v[-1]:>+15.4f}{v[-1] / v[0] if v[0] else 0:>10.1%}")
    # entry 107's MC-ESO ladder (`current`, seed 0..2 mean) and entry 103's null
    MC = np.array([0.1833, 0.1396, 0.0771, 0.0698, 0.0646])
    NU = np.array([0.7437, 0.7219, 0.7156, 0.6906, 0.6312])
    print(f"{'MC-ESO':<12}" + "".join(f"{x:>9.4f}" for x in MC)
          + f"{MC[0] - MC[-1]:>+15.4f}{MC[-1] / MC[0]:>10.1%}")
    print(f"{'null sup':<12}" + "".join(f"{x:>9.4f}" for x in NU)
          + f"{NU[0] - NU[-1]:>+15.4f}{NU[-1] / NU[0]:>10.1%}")

    # -- paired tests, unit = problem
    print("\n== paired tests, unit = problem "
          f"(n={len(names)}, two-sided Wilcoxon, alpha=0.05)")
    for lbl, a, b in (
            ("NMMSO - MC-ESO   5-level", nm_mpr, mc_mpr),
            ("NMMSO - MC-ESO   @1e-1", nm_e1, mc_e1),
            ("NMMSO - null mpr 5-level", nm_mpr, nu_mpr),
            ("NMMSO - null sup 5-level", nm_mpr, nu_sup)):
        w, p, rb, (up, dn, tie) = paired(a, b)
        print(f"   {lbl:<26} delta={np.mean(a) - np.mean(b):+.4f}  W={w:>6.1f}"
              f"  p={p:.4g}  rb={rb:+.3f}  win/loss/tie={up}/{dn}/{tie}"
              f"  -> {'significant' if p < 0.05 else 'NOT significant'}")
    print(f"   level only, not tested: published best (GECCO'2024 sheet, "
          f"16-problem mean) = {PUBLISHED_BEST:.3f}, "
          f"NMMSO = {np.mean(nm_mpr):.4f}  "
          f"(delta {np.mean(nm_mpr) - PUBLISHED_BEST:+.4f})")

    # -- the threat found while reading the table: NMMSO reports as few as 10
    #    points where the cap is max(100,2K), so `current` might be hiding what
    #    its search touched. Entry 107 closed this for MC-ESO, not for NMMSO.
    #    Four problems with the tightest reported sets were re-run and scored
    #    under all three rules; `history` is the supremum over every reporting
    #    rule, so equality there settles it rho-free.
    probe = HERE / "nmmso_rules_probe_d10.csv"
    if probe.exists():
        pr = list(csv.DictReader(probe.open()))
        print("\n== reporting-rule probe (separate runs, seed 0, all three "
              "rules off the same run)")
        print(f"{'problem':<16}{'|cur|':>7}{'cur@1e-1':>10}"
              f"{'resel@1e-1':>12}{'hist@1e-1':>11}{'cur MPR':>9}"
              f"{'resel':>8}{'hist':>8}")
        gaps = []
        for nm in sorted({r["function"] for r in pr}):
            g = {r["rule"]: r for r in pr if r["function"] == nm}
            def mpr(r):
                return float(np.mean([float(r[c]) for c in EPS_COLS]))
            print(f"{nm:<16}{int(g['current']['n_reported']):>7}"
                  f"{float(g['current']['pr_1e-1']):>10.4f}"
                  f"{float(g['reselect']['pr_1e-1']):>12.4f}"
                  f"{float(g['history']['pr_1e-1']):>11.4f}"
                  f"{mpr(g['current']):>9.4f}{mpr(g['reselect']):>8.4f}"
                  f"{mpr(g['history']):>8.4f}")
            gaps.append(mpr(g["history"]) - mpr(g["current"]))
        print(f"   history - current, 5-level, over {len(gaps)} problems: "
              f"max {max(gaps):+.4f}, mean {np.mean(gaps):+.4f}"
              + ("   -> the reported set discards nothing: no reporting rule, "
                 "no rho and no cap can raise NMMSO here"
                 if max(gaps) == 0 else ""))

    # -- the pre-registered branches, read on the 16-problem mean @1e-1
    print("\n== prereg branches (prereg.md), read on the 16-problem mean @1e-1")
    e1 = float(np.mean(nm_e1))
    if e1 <= 0.30:
        print(f"   NMMSO {e1:.4f} <= 0.30  -> BRANCH 1: the narrow support set "
              "is NOT MC-ESO-specific; every method we hold is narrow at D=10, "
              "and the mechanism has to come from queue item 2, not from "
              "comparing methods")
    elif e1 >= 0.60:
        print(f"   NMMSO {e1:.4f} near the null's 0.7437 -> BRANCH 2: there is "
              "real room inside the restart-and-descend family to widen the "
              "support set")
    else:
        print(f"   0.30 < NMMSO {e1:.4f} < 0.60 -> BRANCH 3: partial; report "
              "where it falls and let the review role place the mechanism")

    with args.csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "seeds", "n_reported_mean", "nmmso_mpr",
                    "nmmso_mpr_sd", "mceso_mpr", "null_mpr", "null_sup"]
                   + [f"nmmso_{c}" for c in EPS_COLS])
        for nm in names:
            vals, per_eps, K, rep, ns = got[nm]
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            w.writerow([nm, K, ns, f"{rep:.1f}", f"{vals.mean():.6f}",
                        f"{sd:.6f}", f"{REF[nm][1]:.6f}", f"{REF[nm][3]:.4f}",
                        f"{REF[nm][4]:.2f}"]
                       + [f"{x:.6f}" for x in per_eps])
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
