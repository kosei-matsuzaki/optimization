#!/usr/bin/env python3
"""Do the sets that real niching/QD methods report pay off under unmodelled criteria?

scripts/complementarity_test.py answered the question for a *stand-in* for
geometric diversity (farthest-point selection from a multistart pool). The
obvious objection is that no one runs that rule — so this scores the sets that
the actual methods report.

Each method optimises the nominal objective f with a fixed budget and hands back
its reported solution set (OptimizeResult.final_solutions — the same set the
CEC2013 peak-ratio metric scores). The set is truncated to its K best by nominal
f, which is what a practitioner presenting K options would do. Then the set is
scored on held-out scenarios q(x) = f(x) + eps * (w . x), which no method ever
saw: the unmodelled criterion the whole diverse-solutions literature invokes as
its justification.

Two reference rules are computed from a multistart pool of the same budget:

  quality     the K best local minima by nominal f — no diversity at all
  complement  K chosen greedily to minimise regret on TRAINING scenarios; this
              one sees scenarios, so it is an upper reference, not a competitor

Usage:
  python3 scripts/scenario_value.py [--funcs ...] [--budget 5000] [--k 5]
"""
from __future__ import annotations
import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.benchmarks import (BENCHMARKS_BY_NAME, _BBOB_SPECS,        # noqa: E402
                             _make_bbob, make_benchmark_by_name)
from core.optimizers import (MultiChannelEpidemicOptimizer,          # noqa: E402
                             NCDEOptimizer, RingPSOOptimizer, NMMSOOptimizer,
                             MultistartNelderMeadOptimizer,
                             IPOPCMAESOptimizer, DEOptimizer, PSOOptimizer,
                             MAPElitesOptimizer, MCESOCrowding)

_METHODS = {
    "MC-ESO":     (MultiChannelEpidemicOptimizer, {}),
    "NMMSO":      (NMMSOOptimizer, {}),
    "NCDE":       (NCDEOptimizer, {}),
    "r3pso":      (RingPSOOptimizer, {}),
    "NM-Restart": (MultistartNelderMeadOptimizer, {}),
    "IPOP-CMA-ES": (IPOPCMAESOptimizer, {}),
    "DE":         (DEOptimizer, {}),
    # Two controlled contrasts for the attraction hypothesis: methods whose
    # population is pulled toward an incumbent lose ground as the unmodelled
    # criterion grows, methods without such a pull do not.
    #   PSO vs r3pso      global best vs ring neighbourhood best (topology)
    #   DE vs Crowding-DE parent replacement vs nearest replacement, same mutation
    "PSO":        (PSOOptimizer, {}),
    "Crowding-DE": (NCDEOptimizer, {"m": 30}),
    # Quality-diversity, the framing this study kept invoking without measuring.
    # Descriptor = first two coordinates (see MAPElitesOptimizer), the standard
    # convention for continuous test functions.
    "MAP-Elites": (MAPElitesOptimizer, {}),
    # The design principle under test: same search, nearest-neighbour
    # replacement instead of rollback against the host a child replaced.
    "MC-ESO-crowd": (MCESOCrowding, {}),
}
_FUNCS = ["F03-RastriginSep", "F15-RastriginRot", "F21-Gallagher101", "F17-SchafferF7"]


_POOL_BASE_STARTS = 50          # the size every earlier measurement used


def _multistart_pool(f, lo, hi, dim, rng, n_starts, tol, extra_rng=None):
    """Local minima found by L-BFGS-B from random starts, deduplicated.

    The first `_POOL_BASE_STARTS` starts come from `rng`, any beyond that from
    `extra_rng`. That split matters: `rng` goes on to draw the scenarios, so
    taking the extra starts from a separate stream keeps the scenario set --
    and the first 50 starts -- identical whatever `--pool-starts` is. A 50 vs
    150 comparison is then paired on scenarios rather than confounded with a
    re-draw of them, and the default reproduces the earlier files exactly.
    """
    found: list[np.ndarray] = []
    for i in range(n_starts):
        src = rng if i < _POOL_BASE_STARTS else extra_rng
        r = minimize(f, src.uniform(lo, hi, dim), method="L-BFGS-B",
                     bounds=[(lo, hi)] * dim, options={"maxfun": 2000})
        x = np.clip(r.x, lo, hi)
        if all(float(np.linalg.norm(x - y)) > tol for y in found):
            found.append(x)
    return np.array(found)


def _greedy_complement(vals, opt, k):
    chosen, cur = [], np.full(vals.shape[1], np.inf)
    for _ in range(min(k, len(vals))):
        best_i, best_score = None, np.inf
        for i in range(len(vals)):
            if i in chosen:
                continue
            score = float(np.mean(np.minimum(cur, vals[i]) - opt))
            if score < best_score:
                best_i, best_score = i, score
        chosen.append(best_i)
        cur = np.minimum(cur, vals[best_i])
    return chosen


def _spread_within(pool, f_pool, spread, k, tau):
    """The K furthest-apart solutions that are still within tau of the best.

    The alternative to reporting the K best. Losses under a forbidden region
    come entirely from scenarios that delete every reported point, and how often
    that happens is set by how far apart the points are, so this trades a
    bounded amount of nominal quality for distance. tau is in units of the
    pool's own spread of local optima, so tau=0 collapses to the K best.
    """
    ok = np.flatnonzero(f_pool <= f_pool.min() + tau * spread)
    if len(ok) <= k:
        return list(np.argsort(f_pool)[:k])
    order = ok[np.argsort(f_pool[ok])]
    chosen = [int(order[0])]                       # always keep the best one
    while len(chosen) < k:
        d = np.min([np.linalg.norm(pool[ok] - pool[c], axis=1) for c in chosen],
                   axis=0)
        nxt = int(ok[int(np.argmax(d))])
        if nxt in chosen:
            break
        chosen.append(nxt)
    return chosen


def _spread_within_n(pool, f_pool, spread, k, n_min, tau_max=float("inf")):
    """Widen the tolerance band until it holds n_min candidates, then pick.

    The fixed-tau rule misfires whenever the band holds no more than K
    candidates: it falls back silently to the top K, so the comparison is a tie
    by construction. That happened in 78 of 120 cells at dim2 with 50 starts and
    in 105 of 120 at dim5, i.e. the width that leaves something in the band is a
    property of how densely the pool samples the local optima, not of the
    dimension. Fixing the *count* instead of the width is therefore the version
    that needs no per-dimension retuning. tau_max caps how much nominal quality
    may be surrendered to reach the count; without a cap the band can widen to
    the whole pool on a function whose optima are far apart in value.

    Returns (indices, tau_eff) so the caller can record which width was used.
    """
    order = np.argsort(f_pool)
    if len(order) >= n_min:
        tau_eff = float(f_pool[order[n_min - 1]] - f_pool[order[0]]) / spread
        tau_eff *= 1.0 + 1e-9          # keep the n_min-th candidate inside <=
    else:
        tau_eff = float("inf")         # pool too small; band is everything
    tau_eff = min(tau_eff, tau_max)
    return _spread_within(pool, f_pool, spread, k, tau_eff), tau_eff


def _make_scenarios(args, name, f, lo, hi, span, dim, spread, rng, n_all):
    """Return q(x, s): the objective as it turns out to be under scenario s.

    Three shapes of unmodelled criterion, all revealed only after the solution
    set is fixed. Strength is scaled to the spread of the local optima, so
    `--tilt` means the same thing across shapes: how far the criterion can
    reorder solutions the nominal objective ranks close together.
    """
    if args.scenario == "tilt":
        eps = args.tilt * spread / (span * dim ** 0.5)
        W = rng.normal(size=(n_all, dim))
        W /= np.linalg.norm(W, axis=1, keepdims=True)

        def q(x, s):
            return f(x) + eps * float(np.dot(W[s], np.asarray(x, float)))
        return q

    if args.scenario == "instance":
        # The family's own shift and rotation: the problem you face is instance
        # 2+s, the one you optimised was instance 1. No strength parameter — the
        # benchmark itself decides how far instances sit from each other.
        spec = next((sp for sp in _BBOB_SPECS if sp[1] == name), None)
        if spec is None:
            raise SystemExit(f"instance scenarios need a BBOB function, got {name}")
        variants = [_make_bbob(spec[0], spec[1], spec[2], dim, instance=2 + s)
                    for s in range(n_all)]

        def q(x, s):
            return float(variants[s].func(np.clip(np.asarray(x, float), lo, hi)))
        return q

    # constraint: a half-space turns out to be forbidden, priced so that any
    # solution inside it is worse than any solution outside.
    A = rng.normal(size=(n_all, dim))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    mid = (lo + hi) / 2.0
    cut = getattr(args, "cut", 0.25)
    off = rng.uniform(-cut, cut, n_all) * span        # cuts through the middle
    penalty = args.tilt * 10.0 * max(spread, 1e-12)

    def violates(x, s):
        z = np.asarray(x, float) - mid
        return float(np.dot(A[s], z)) - float(off[s]) > 0

    def q(x, s):
        z = np.asarray(x, float) - mid
        viol = float(np.dot(A[s], z)) - float(off[s])
        return f(x) + (penalty * (1.0 + viol / span) if viol > 0 else 0.0)

    # Exposed so the caller can record whether a whole reported set landed
    # inside the forbidden half-space, which is the loss with no ceiling.
    q.violates = violates
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--funcs", type=str, default=",".join(_FUNCS))
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dim", type=int, default=2,
                    help="dimension of the BBOB functions. The scenario models "
                         "are all defined at any dimension, so this only "
                         "rebuilds the benchmarks. Custom (C*) benchmarks are "
                         "2-D only and are rejected here.")
    ap.add_argument("--methods", type=str, default=None,
                    metavar="A,B,...",
                    help="run only these methods ('none' runs none). Comparisons "
                         "between reference rules are unaffected by this: every "
                         "regret is measured against the same per-scenario "
                         "best_known, so dropping methods shifts a scenario's "
                         "regrets by one common constant and paired differences "
                         "between rules are unchanged. Use it to sweep tau "
                         "without paying for the methods.")
    ap.add_argument("--pool-starts", type=int, default=_POOL_BASE_STARTS,
                    help="L-BFGS-B restarts behind the reference pool. Raising it "
                         "gives the rules more candidates to choose from, which "
                         "is what decides whether a narrow tolerance band has "
                         "anything in it at all.")
    ap.add_argument("--scenario", choices=["tilt", "instance", "constraint"],
                    default="tilt",
                    help="what the unmodelled criterion looks like. tilt: an unknown "
                         "linear bias. instance: the problem you face is another "
                         "instance of the same family (BBOB's own shift/rotation). "
                         "constraint: a region of the domain turns out to be "
                         "forbidden. All three are revealed only after the set is "
                         "reported, which is the setting the field's justification "
                         "describes.")
    ap.add_argument("--tilt", type=float, default=1.0)
    ap.add_argument("--tau", type=float, nargs="*",
                    default=[0.1, 0.25, 0.5, 1.0],
                    help="quality tolerances, in units of the pool's spread of "
                         "local optima, for the 'spread@T' reference rule: report "
                         "the K furthest-apart solutions no worse than tau below "
                         "the best. These cost no extra evaluations, so several "
                         "are scored at once.")
    ap.add_argument("--tau-n", type=int, nargs="*", default=[],
                    metavar="N",
                    help="adaptive variant of the same rule: instead of a fixed "
                         "width, widen the band until it holds N candidates, then "
                         "pick the K furthest apart inside it (rule 'spreadN@N'). "
                         "N must exceed K or the rule degenerates to the top K. "
                         "The width needed to hold N candidates is set by the "
                         "pool, so this needs no per-dimension retuning.")
    ap.add_argument("--tau-n-caps", type=float, nargs="*", default=[],
                    metavar="TAU",
                    help="also score each --tau-n level with the band capped at "
                         "this width (rule 'spreadNc{TAU}@N'), which bounds the "
                         "nominal quality the adaptive rule may give away on a "
                         "function whose optima are far apart in value.")
    ap.add_argument("--postprocess", type=float, default=None,
                    metavar="TAU",
                    help="also score, for every method, the set obtained by "
                         "applying the same distance-within-tolerance rule to "
                         "that method's own reported solutions (suffix '+s'). "
                         "This is the rule as a proposal: the search is "
                         "untouched and only the choice of what to report "
                         "changes.")
    ap.add_argument("--cut", type=float, default=0.25,
                    help="constraint only: how far off centre the forbidden "
                         "half-space may be cut, as a fraction of the span. Larger "
                         "values let a scenario forbid more of the domain, so a "
                         "reported set is likelier to be wiped out entirely.")
    ap.add_argument("--n-train", type=int, default=15)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dedup", type=float, default=0.05,
                    help="drop reported points within this fraction of the span of "
                         "a better one before taking the top K. Without it a "
                         "converged population reports K near-duplicates, which is "
                         "not what anyone would present.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write one row per (function, method, seed, scenario) so "
                         "the comparison can be tested pairwise instead of on means")
    ap.add_argument("--raw", action="store_true",
                    help="also score the un-deduplicated top-K, to show what the "
                         "duplicates cost")
    args = ap.parse_args()

    if args.methods is None:
        methods = dict(_METHODS)
    elif args.methods.strip().lower() == "none":
        methods = {}
    else:
        want = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in want if m not in _METHODS]
        if unknown:
            raise SystemExit(f"unknown method(s): {', '.join(unknown)}. "
                             f"known: {', '.join(_METHODS)}")
        methods = {m: _METHODS[m] for m in want}

    writer = None
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fh = open(args.csv, "w", newline="")
        writer = csv.writer(fh)
        cols = ["function", "method", "seed", "scenario", "regret", "spread"]
        if args.scenario == "constraint":
            cols.append("wiped")     # every reported point is inside the region
        if args.tau_n:
            # The width the adaptive rule actually chose on this (function,
            # seed); blank for every rule whose width was fixed in advance.
            cols.append("tau_eff")
        writer.writerow(cols)
    print(f"reported sets under an unmodelled criterion   budget={args.budget}  "
          f"K={args.k}  scenario={args.scenario}  tilt={args.tilt}  "
          f"train/test={args.n_train}/{args.n_test}  "
          f"seeds={args.seeds}  dim={args.dim}  pool-starts={args.pool_starts}  "
          f"methods={len(methods)}")
    head = (f"{'function':<20}{'method':<13}{'|report|':>9}{'regret':>10}"
            f"{'vs quality':>12}")
    print(head)
    print("-" * len(head))

    for name in [s.strip() for s in args.funcs.split(",")]:
        if args.dim == 2:
            b = BENCHMARKS_BY_NAME[name]
        elif name.startswith("C"):
            raise SystemExit(f"{name} is 2-D only; --dim {args.dim} needs a BBOB function")
        else:
            b = make_benchmark_by_name(name, args.dim)
        lo, hi = b.bounds
        span, dim = hi - lo, b.dim

        def f(x):
            return float(b.func(np.clip(np.asarray(x, float), lo, hi)))

        rows: dict[str, list[float]] = {}
        sizes: dict[str, list[int]] = {}
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            pool = _multistart_pool(f, lo, hi, dim, rng, args.pool_starts,
                                    0.01 * span,
                                    np.random.default_rng(90000 + seed))
            f_pool = np.array([f(x) for x in pool])
            spread = float(np.percentile(f_pool, 75) - f_pool.min()) or 1.0
            n_all = args.n_train + args.n_test
            q = _make_scenarios(args, name, f, lo, hi, span, dim, spread,
                                rng, n_all)

            def score(points):
                """Mean held-out regret of the best-of-set, plus its own values."""
                P = np.asarray(points, dtype=float)
                v = np.array([[q(x, s) for s in range(n_all)] for x in P])
                return v

            pool_vals = score(pool)

            # Reference per scenario: best of everything evaluated so far, filled
            # in again below once each method's points are known.
            best_known = pool_vals.min(axis=0)

            sets: dict[str, np.ndarray] = {}
            for m, (cls, kw) in methods.items():
                kwargs = dict(kw)
                if cls is IPOPCMAESOptimizer:
                    kwargs["sigma0"] = 0.2 * span
                res = cls(b, seed=seed * 100, **kwargs).optimize(args.budget)
                rep = np.asarray(res.final_solutions or [res.best_x], dtype=float)
                order = np.argsort([f(x) for x in rep])
                if args.raw:
                    sets[m + " (raw)"] = rep[order[:args.k]]
                # Deduplicate the way a peak-ratio count would: walk in quality
                # order and keep a point only if it is far from every better one.
                kept: list[np.ndarray] = []
                for i in order:
                    x = rep[i]
                    if all(float(np.linalg.norm(x - y)) > args.dedup * span
                           for y in kept):
                        kept.append(x)
                    if len(kept) == args.k:
                        break
                sets[m] = np.array(kept)
                sizes.setdefault(m, []).append(len(rep))
                if args.postprocess is not None and len(rep) > args.k:
                    # The same rule applied to what the method itself returned,
                    # rather than to the offline pool. This is the form a
                    # proposal would take: leave the search alone and change
                    # only which of its solutions get reported.
                    f_rep = np.array([f(x) for x in rep])
                    idx = _spread_within(rep, f_rep, spread, args.k,
                                         args.postprocess)
                    sets[f"{m}+s"] = rep[idx]

            # Reference rules from the multistart pool, same budget in spirit.
            sets["quality"] = pool[np.argsort(f_pool)[:args.k]]
            tr = slice(0, args.n_train)
            idx = _greedy_complement(pool_vals[:, tr], pool_vals[:, tr].min(axis=0),
                                     args.k)
            sets["complement"] = pool[idx]
            for tau in args.tau:
                sets[f"spread@{tau:g}"] = pool[_spread_within(pool, f_pool, spread,
                                                              args.k, tau)]
            tau_eff: dict[str, float] = {}
            for n_min in args.tau_n:
                for cap in [float("inf")] + list(args.tau_n_caps):
                    idx_n, t_eff = _spread_within_n(pool, f_pool, spread, args.k,
                                                    n_min, cap)
                    tag = ("spreadN" if cap == float("inf")
                           else f"spreadNc{cap:g}") + f"@{n_min}"
                    sets[tag] = pool[idx_n]
                    tau_eff[tag] = t_eff

            vals = {m: score(P) for m, P in sets.items()}
            for v in vals.values():
                best_known = np.minimum(best_known, v.min(axis=0))
            te = slice(args.n_train, n_all)
            for m, v in vals.items():
                per_scen = np.min(v[:, te], axis=0) - best_known[te]
                rows.setdefault(m, []).append(float(np.mean(per_scen)))
                if writer is not None:
                    P = sets[m]
                    # Mean pairwise distance of the reported set, span-relative.
                    # Constant across scenarios, but carried per row so that any
                    # slice of the file can be read on its own.
                    pd = [float(np.linalg.norm(x - y)) / span
                          for x, y in itertools.combinations(P, 2)]
                    sp = float(np.mean(pd)) if pd else 0.0
                    for j, val in enumerate(per_scen):
                        row = [name, m, seed, args.n_train + j,
                               f"{float(val):.6g}", f"{sp:.5f}"]
                        if args.scenario == "constraint":
                            sc = args.n_train + j
                            row.append(int(all(q.violates(x, sc) for x in P)))
                        if args.tau_n:
                            t = tau_eff.get(m)
                            row.append("" if t is None else f"{t:.6g}")
                        writer.writerow(row)

        base = float(np.mean(rows["quality"]))
        order = sorted(rows, key=lambda m: float(np.mean(rows[m])))
        for m in order:
            r = float(np.mean(rows[m]))
            n_rep = int(np.mean(sizes[m.replace(" (raw)", "")]))                 if m.replace(" (raw)", "") in sizes else args.k
            ratio = r / base if base > 0 else float("nan")
            print(f"{name:<20}{m:<13}{n_rep:>9}{r:>10.3f}{ratio:>11.2f}x")
        print()

    if writer is not None:
        fh.close()
        print(f"per-scenario rows written to {args.csv}")
    print("regret: mean over held-out scenarios of the best-of-K, against the best "
          "value anything reached (lower is better).")
    print("vs quality: ratio to simply reporting the K best local minima. >1 means "
          "the method's reported set is worse than not diversifying at all.")
    print("'complement' sees training scenarios, so it is an upper reference.")


if __name__ == "__main__":
    main()
