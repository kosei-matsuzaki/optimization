#!/usr/bin/env python3
"""Which clause released the hunts behind the residual step? (research_loop question 1)

Entry 51 left one step in the N08-Shubert3D profile: PR 0.860 at eps=1e-4 against
0.754 at eps=1e-5, i.e. ~8 of the 81 peaks are only reported to within
(1e-5, 1e-4]. Entry 53 rejected the obvious cause (the release level L sitting on
the deepest scored threshold — moving L a decade down left the step in the same
place at the same size). This script counts what has never been counted: for each
peak that scores at 1e-4 but not at 1e-5, **where the reported point that covers
it came from, and which release clause ended the hunt that produced it.**

The scoring rule matters for the reading (``core/runner.count_goptima``): seeds
are picked rho-greedily from the *whole* fitness-sorted reported set and only
then filtered by accuracy, so a shallow point can occupy a niche and block a
deeper one nearby. That gives three competing explanations for one step seed,
and this script separates them per seed:

  depth   the reported set holds nothing deeper than 1e-5 within rho of it
          -> the hunt really did stop shallow; the release clause is the story.
  block   a point with f <= 1e-5 exists within rho but lost the rho-greedy walk
          -> a reporting/selection problem, not a depth problem.
  budget  the covering point is a live population member (source "pop")
          -> that hunt was still running when the budget ran out; no clause
          released it at all.

Pre-registered rejection condition (research_loop question 1): if the step seeds
that trace back to a released hunt were released **by the level clause** in the
majority, the stagnation-clause explanation cannot hold and the question folds
over to the coverage/reporting axis.

Usage:
  python3 scripts/e54_release_probe.py --func N08-Shubert3D --seeds 15 \
      --evals-frac 1.0 --out analysis/hm/e54
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME            # noqa: E402
from core.runner import NICHE_ACCURACIES, _seed_indices           # noqa: E402
from core.optimizers.mceso_release_probe import ReleaseProbeMCESO  # noqa: E402

# The adoption candidate (entry 46 corner, entry 44 clamp). Same keywords as the
# "MC-ESO-rel" arm of scripts/niching_baseline.py, so the peak ratios this script
# prints must reproduce analysis/hm/e51 exactly -- that is the identity check.
ARM = {"rel_level": 1e-5, "fis_floor": 1e-12}


def score_run(opt, res, b, accuracies) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Replicate core.runner._niching_counts / count_goptima on one run, keeping
    the provenance tag attached to every reported point.

    Returns (peaks covered per accuracy, per-seed rows, per-step-seed rows).
    """
    k, rho = b.n_global_optima, b.niche_rho
    cap = max(100, 2 * k)
    X = np.asarray(res.final_solutions, dtype=float)
    tags = opt.solution_tags()
    assert len(tags) == len(X), (len(tags), len(X))
    F = np.array([float(b.func(x)) for x in X])
    if len(F) > cap:                                  # same cap, same tie order
        keep = np.argsort(F)[:cap]
        X, F, tags = X[keep], F[keep], [tags[i] for i in keep]
    order = np.argsort(F)
    sx, sf = X[order], F[order]
    stags = [tags[i] for i in order]

    seed_idx = _seed_indices(sx, rho)
    counts = np.zeros(len(accuracies))
    for j, a in enumerate(accuracies):
        c = 0
        for i in seed_idx:
            if sf[i] <= a:
                c += 1
                if c == k:
                    break
        counts[j] = c

    seed_rows = [{"i": i, "f": float(sf[i]), "source": stags[i]["source"],
                  "rec": stags[i]["rec"]} for i in seed_idx]
    # Step seeds: counted at 1e-4, not at 1e-5. (Both levels are inside the k cap
    # here -- N08 reports ~40 seeds under 1e-4 against k=81 -- but assert it.)
    assert counts[-1] < k and counts[-2] < k, "k cap reached; step seeds ill-defined"
    step = []
    for r in seed_rows:
        if 1e-5 < r["f"] <= 1e-4:
            # Is a deeper point available within rho of this seed, i.e. did the
            # rho-greedy walk block it, or is the set genuinely shallow here?
            d = np.linalg.norm(sx - sx[r["i"]], axis=1)
            near_deep = int(np.sum((d <= rho) & (sf <= 1e-5)))
            kind = ("budget" if r["source"] == "pop"
                    else "block" if near_deep else "depth")
            step.append({**r, "near_deep": near_deep, "kind": kind})
    return counts, seed_rows, step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--func", default="N08-Shubert3D")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--seed-lo", type=int, default=0,
                    help="first seed index; seeds run are [lo, seeds). Splitting "
                         "the range over processes keeps the seed numbering "
                         "(seed*100) identical to entry 51 / 53, so the arms "
                         "stay paired.")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--evals-frac", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("analysis/hm/e54"))
    args = ap.parse_args()

    b = NICHING_BENCHMARKS_BY_NAME[args.func]
    budget = max(1000, int(b.suite_max_evals * args.evals_frac))
    args.out.mkdir(parents=True, exist_ok=True)
    acc = NICHE_ACCURACIES

    pr_fh = open(args.out / f"{args.func}{args.tag}_pr.csv", "w", newline="")
    pr_w = csv.writer(pr_fh)
    pr_w.writerow(["function", "method", "rule", "seed", "evals", "n_optima",
                   "n_reported"]
                  + [f"pr_{a:.0e}".replace("e-0", "e-") for a in acc])
    st_fh = open(args.out / f"{args.func}{args.tag}_step.csv", "w", newline="")
    st_w = csv.writer(st_fh)
    st_w.writerow(["seed", "f", "source", "kind", "near_deep", "released",
                   "level_ok", "sigma_floor_ok", "stagnated", "basin_best",
                   "level", "sigma_over_floor", "no_improve", "stag_need",
                   "spill_i", "spill_evals", "binding", "hunt_start",
                   "first_sigma", "first_level", "first_stag"])
    sp_fh = open(args.out / f"{args.func}{args.tag}_spill.csv", "w", newline="")
    sp_w = csv.writer(sp_fh)
    sp_w.writerow(["seed", "spill_i", "evals", "released", "pre_exhausted",
                   "level_ok", "sigma_floor_ok", "stagnated", "basin_best",
                   "level", "sigma_over_floor", "no_improve", "stag_need",
                   "basin_switch", "pop_best", "survived", "binding",
                   "hunt_start", "first_sigma", "first_level", "first_stag"])

    print(f"{args.func}  K={b.n_global_optima}  rho={b.niche_rho}  "
          f"budget={budget}  seeds={args.seeds}")
    print(f"{'seed':>4}{'|rep|':>7}"
          + "".join(f"{f'{a:.0e}'.replace('e-0', 'e-'):>9}" for a in acc)
          + f"{'step':>6}{'depth':>7}{'block':>7}{'budget':>7}"
          + f"{'spills':>8}{'lvl%':>6}{'s':>6}")
    tot = {"step": 0, "depth": 0, "block": 0, "budget": 0}
    for s in range(args.seed_lo, args.seeds):
        t0 = time.time()
        opt = ReleaseProbeMCESO(b, seed=s * 100, **ARM)
        res = opt.optimize(budget)
        counts, seed_rows, step = score_run(opt, res, b, acc)
        pr_w.writerow([args.func, "MC-ESO-rel-probe", "current", s, budget,
                       b.n_global_optima, len(res.final_solutions)]
                      + [f"{c / b.n_global_optima:.4f}" for c in counts])
        recs = opt.spill_records()
        survived = {t["rec"]["spill_i"] for t in opt.solution_tags()
                    if t["rec"] is not None}
        for r in recs:
            sp_w.writerow([s, r["spill_i"], r["evals"], r.get("released"),
                           r.get("pre_exhausted"), r.get("level_ok"),
                           r.get("sigma_floor_ok"), r.get("stagnated"),
                           f"{r.get('basin_best', float('nan')):.6e}",
                           f"{r.get('level', float('nan')):.3e}",
                           f"{r.get('sigma_over_floor', float('nan')):.3f}",
                           r.get("no_improve"), f"{r.get('stag_need', 0):.0f}",
                           r["basin_switch"], f"{r['pop_best']:.6e}",
                           r["spill_i"] in survived, r.get("binding"),
                           r.get("hunt_start"), r.get("first_sigma"),
                           r.get("first_level"), r.get("first_stag")])
        for r in step:
            rc = r["rec"] or {}
            st_w.writerow([s, f"{r['f']:.6e}", r["source"], r["kind"],
                           r["near_deep"], rc.get("released"), rc.get("level_ok"),
                           rc.get("sigma_floor_ok"), rc.get("stagnated"),
                           f"{rc.get('basin_best', float('nan')):.6e}",
                           f"{rc.get('level', float('nan')):.3e}",
                           f"{rc.get('sigma_over_floor', float('nan')):.3f}",
                           rc.get("no_improve"), f"{rc.get('stag_need', 0):.0f}",
                           rc.get("spill_i"), rc.get("evals"),
                           rc.get("binding"), rc.get("hunt_start"),
                           rc.get("first_sigma"), rc.get("first_level"),
                           rc.get("first_stag")])
        kinds = {k: sum(1 for r in step if r["kind"] == k)
                 for k in ("depth", "block", "budget")}
        tot["step"] += len(step)
        for k in kinds:
            tot[k] += kinds[k]
        lvl = [r for r in recs if r.get("released") and r.get("level_ok")]
        rel = [r for r in recs if r.get("released")]
        print(f"{s:>4}{len(res.final_solutions):>7}"
              + "".join(f"{c / b.n_global_optima:>9.3f}" for c in counts)
              + f"{len(step):>6}{kinds['depth']:>7}{kinds['block']:>7}"
              f"{kinds['budget']:>7}{len(recs):>8}"
              f"{(100 * len(lvl) / max(1, len(rel))):>6.0f}"
              f"{time.time() - t0:>6.0f}")
        pr_fh.flush(); st_fh.flush(); sp_fh.flush()

    print(f"\ntotal step seeds {tot['step']}  "
          f"depth {tot['depth']}  block {tot['block']}  budget {tot['budget']}")
    pr_fh.close(); st_fh.close(); sp_fh.close()
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
