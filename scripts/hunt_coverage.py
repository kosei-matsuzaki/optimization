#!/usr/bin/env python3
"""Which optima do MC-ESO's hunts keep re-landing in, and why those?

Reads a hunt dump that carries endpoint coordinates
(``diagnose_niching.py --hunt-csv`` on a function whose optima are known
analytically) and maps every hunt endpoint onto the nearest true global optimum.
Then it asks whether the optima that get hit repeatedly are the ones with the
*widest basins* — which is what a restart rule that draws uniformly from the box
and repels with a single fixed radius would do on a landscape whose basins are
not all the same size.

Vincent, f = 1 - mean_i sin(10 log x_i), has its optima at
    x_i = exp((2 pi n + pi/2) / 10),  n = 0..5   on [0.25, 10],
i.e. **log-spaced**: consecutive optima are 0.10 apart near x = 0.25 and 3.3
apart near x = 10, a 33x range in one dimension. A uniform draw therefore lands
in the wide basins far more often than in the narrow ones, and a repel radius
fixed at 0.02 * span = 0.195 (``mceso._diversified_reseed``) is smaller than a
wide basin, so it does not push the draw out of a basin already drilled.

Shubert's optima are spaced ~0.88 apart throughout, so the same fixed radius
behaves consistently across the domain — the contrast case.

Usage (MC-ESO hunt dump):
        python3 scripts/hunt_coverage.py analysis/hm/hunts_n07_xy.csv

Method-agnostic mode (entry 64).  ``--run`` drops the MC-ESO hunt dump and
asks the same question of *any* optimizer, by reading its evaluation history:
every evaluated point at or below the scoring accuracy is attributed to its
nearest true optimum (the rule ``core.runner.optima_found_mask`` already uses),
which turns "which optima does this method reach, and how often" into something
that can be asked of NMMSO and NM-Restart as well as of MC-ESO.

Two counters per (method, seed, optimum) come out of that, because raw point
counts are not comparable across methods -- a method that parks a swarm on an
optimum accrues thousands of points for one arrival:
  * ``pts``      — evaluated points at the accuracy, attributed to this optimum.
  * ``arrivals`` — *entries* into the optimum's eps-ball, i.e. runs of
    consecutive attributed points in evaluation order, counted once each.  This
    is the history-side analogue of a hunt endpoint.  It is still inflated for
    any method that interleaves several populations in one evaluation stream,
    so coverage (reached at all: density-free) is the primary read.

Usage:
  python3 scripts/hunt_coverage.py --run --func N09-Vincent3D \
      --methods MC-ESO,NM-Restart,NMMSO --seeds 15 --evals-frac 1.0 \
      --csv analysis/hm/e64/visits.csv
  python3 scripts/hunt_coverage.py --analyze analysis/hm/e64/visits.csv
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Vincent's 1-D optimum positions on [0.25, 10]: sin(10 log x) = 1, i.e.
# log x = (pi/2 + 2 pi n) / 10.  log(0.25) = -1.386 and log(10) = 2.303 admit
# n = -2 .. 3 — six roots per axis, which is what makes K = 6**dim (36 and 216).
# Taking only n >= 0 leaves four in-box roots and two outside, so the nearest-
# optimum mapping below silently attaches endpoints to points that do not exist.
_V1D = np.array([np.exp((2.0 * np.pi * n + np.pi / 2.0) / 10.0)
                 for n in range(-2, 4)])
assert _V1D.min() > 0.25 and _V1D.max() < 10.0, _V1D


def vincent_optima(dim: int) -> np.ndarray:
    """The full grid of global optima (6**dim of them)."""
    grids = np.meshgrid(*([_V1D] * dim), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def true_optima(b) -> np.ndarray:
    """The benchmark's global optima, for any registered niching function (e87).

    Vincent's grid is generated analytically above because ``optima_pos`` for
    the hand-written N04-N10 carries only the closed-form list; the CEC2013
    composition functions registered through ``ioh`` (N11-N20) carry the full
    ``prob.optima`` positions, so those are read straight off the benchmark.
    """
    if "Vincent" in b.name:
        return vincent_optima(b.dim)
    if not b.optima_pos:
        raise SystemExit(f"{b.name}: no optima positions on the benchmark")
    opts = np.asarray(b.optima_pos, dtype=float)
    assert len(opts) == b.n_global_optima, (len(opts), b.n_global_optima)
    return opts


def basin_width(opts: np.ndarray) -> np.ndarray:
    """Basin-width proxy: distance from each optimum to its nearest neighbour."""
    d = np.linalg.norm(opts[:, None, :] - opts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


# ── method-agnostic mode: attribute a run's evaluation history ──────────────
def attribute_history(hx, hf, opts: np.ndarray, eps: float,
                      radius: float) -> tuple[np.ndarray, np.ndarray]:
    """(pts, arrivals) per optimum for one run's evaluation history.

    Same attribution rule as ``core.runner.optima_found_mask``: a point counts
    for optimum k if ``f <= eps`` and k is its nearest optimum within
    ``radius``.  ``arrivals`` collapses each maximal run of consecutive
    attributed points on the same optimum into one event, so a swarm that sits
    on an optimum for 10,000 evaluations scores one arrival, not 10,000 points.
    """
    K = len(opts)
    pts = np.zeros(K, dtype=np.int64)
    arrivals = np.zeros(K, dtype=np.int64)
    X = np.asarray(hx, dtype=float)
    F = np.asarray(hf, dtype=float)
    idx = np.flatnonzero(F <= eps)
    if idx.size == 0:
        return pts, arrivals
    lab = np.full(len(F), -1, dtype=np.int64)
    for a in range(0, idx.size, 20000):                 # chunked: (M, K) is big
        sl = idx[a:a + 20000]
        d = np.linalg.norm(X[sl][:, None, :] - opts[None, :, :], axis=2)
        j = np.argmin(d, axis=1)
        ok = d[np.arange(len(sl)), j] <= radius
        lab[sl[ok]] = j[ok]
    seq = lab[lab >= 0]
    np.add.at(pts, seq, 1)
    # an arrival = a labelled point whose previous *labelled or unlabelled*
    # position carried a different label, i.e. the trajectory entered the ball.
    prev = np.concatenate(([-2], lab[:-1]))
    starts = lab[(lab >= 0) & (prev != lab)]
    np.add.at(arrivals, starts, 1)
    return pts, arrivals


def _diagnostic_arms() -> dict:
    """Arms that are not in the baseline table because they are not methods.

    ``MC-ESO-place`` is MC-ESO with the best-of-n_pop race over restart draws
    removed (the population commits to one anchor; run sigma stays at base,
    entry 40's `commit_place_r010`).  It is the *within-method* manipulation of
    the mechanism this entry is testing: if the width bias is a property of
    best-of-n restart selection, taking the race out has to move the bias.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.optimizers.mceso_commit_reseed import CommitReseedMCESO
    return {"MC-ESO-place": (CommitReseedMCESO,
                             {"commit_sigma_mode": "place",
                              "commit_sigma_ratio": 0.1})}


_DIAGNOSTIC_ARMS: dict = _diagnostic_arms()


def _run_one(args_tuple):
    name, method, seed, budget, eps_list = args_tuple
    import time
    from core.benchmarks import niching_by_name
    from scripts.niching_baseline import _METHODS
    b = niching_by_name(name)
    cls, kw = dict(_METHODS, **_DIAGNOSTIC_ARMS)[method]
    t0 = time.time()
    r = cls(b, seed=seed, **kw).optimize(budget)
    opts = true_optima(b)
    span = b.bounds[1] - b.bounds[0]
    radius = max(0.5, 0.02 * span)                      # core.runner rule
    out = []
    for eps in eps_list:
        pts, arr = attribute_history(r.history_x, r.history_f, opts, eps, radius)
        out.append((eps, pts, arr))
    return name, method, seed, budget, len(r.history_f), time.time() - t0, out


def run_mode(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="hunt_coverage.py --run")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--func", default="N09-Vincent3D")
    ap.add_argument("--methods", default="MC-ESO,NM-Restart,NMMSO")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--evals-frac", type=float, default=1.0)
    ap.add_argument("--eps", default="1e-3,1e-5")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--csv", type=Path, required=True)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.benchmarks import niching_by_name
    b = niching_by_name(a.func)
    opts = true_optima(b)          # e87: any registered niching function
    assert len(opts) == b.n_global_optima, (len(opts), b.n_global_optima)
    width = basin_width(opts)
    budget = max(1000, int(b.suite_max_evals * a.evals_frac))
    eps_list = [float(s) for s in a.eps.split(",")]
    jobs = [(a.func, m, s * 100, budget, eps_list)
            for m in a.methods.split(",") for s in range(a.seeds)]

    a.csv.parent.mkdir(parents=True, exist_ok=True)
    from multiprocess import Pool
    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function", "method", "seed", "evals", "eps", "opt",
                    "width", "pts", "arrivals"])
        with Pool(a.procs) as pool:
            for name, m, seed, bud, nev, secs, out in pool.imap_unordered(
                    _run_one, jobs):
                for eps, pts, arr in out:
                    for j in range(len(opts)):
                        w.writerow([name, m, seed, nev, f"{eps:g}", j,
                                    f"{width[j]:.6f}", pts[j], arr[j]])
                fh.flush()
                cov = {f"{eps:g}": int((p > 0).sum()) for eps, p, _ in out}
                print(f"{m:<12} seed {seed:>4}  {nev} evals  {secs:6.1f}s  "
                      f"covered {cov}", flush=True)
    print(f"rows written to {a.csv}")


_NULL_EPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)   # the GECCO'2024 scoring accuracies


def _null_descent(args_tuple):
    """One draw of the *restart-lander* null: uniform start, isotropic descent.

    The null a coverage claim has to beat is not "uniform points" but "uniform
    points that then descend the way this optimiser descends".  MC-ESO reseeds
    a hunt at a draw from the box and drills it with an isotropic step
    (sigma_init = ``sigma`` x span); later hunts are cut at a fixed length
    (1499 evaluations on N18, e76).  Reproducing exactly that, minus the repel
    rule and minus the best-of-n_pop race over draws, gives the landing
    distribution the landscape alone forces.  ``iso=False`` turns the full
    covariance on as a sensitivity check: if the null moves, the null is a
    statement about the descent model, not about basin volume.
    """
    name, k, budget, sigma0, iso = args_tuple
    import cma
    from core.benchmarks import niching_by_name
    b = niching_by_name(name)
    opts = true_optima(b)
    lo, hi = b.bounds
    rng = np.random.default_rng(1_000_000 + k)
    x0 = rng.uniform(lo, hi, size=b.dim)
    j0 = int(np.argmin(np.linalg.norm(opts - x0, axis=1)))
    o = {"bounds": [lo, hi], "maxfevals": budget, "seed": k + 1, "verbose": -9,
         "tolfun": 0, "tolfunhist": 0, "tolx": 0}
    if iso:
        o["CMA_on"] = 0                       # step size only, no rotation
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, o)
    best_f, best_x, used = float("inf"), x0, 0
    # Evaluations spent when this descent first got inside each scoring
    # accuracy, and where it stood at that moment (entry 92).  A class ceiling
    # read off `used` alone fixes the price of a restart at what a descent that
    # runs to convergence costs, so it understates what the class can score at
    # the loose accuracies -- a member is free to cut the descent the moment it
    # is inside eps and spend the rest on more restarts.  -1 = never reached.
    first = {e: (-1, -1) for e in _NULL_EPS}
    while not es.stop() and used < budget:
        xs = es.ask()
        fs = [float(b.func(np.asarray(x))) for x in xs]
        used += len(xs)
        es.tell(xs, fs)
        i = int(np.argmin(fs))
        if fs[i] < best_f:
            best_f, best_x = fs[i], np.asarray(xs[i], dtype=float)
        for e in _NULL_EPS:
            if first[e][0] < 0 and best_f <= e:
                dd = np.linalg.norm(opts - best_x, axis=1)
                first[e] = (used, int(np.argmin(dd)))
    d = np.linalg.norm(opts - best_x, axis=1)
    j = int(np.argmin(d))
    return (k, j0, j, float(d[j]), best_f, used,
            [first[e] for e in _NULL_EPS])


def null_mode(argv: list[str]) -> None:
    """Volume-proportional nulls for a landing distribution (entry 87).

    ``geo``   -- uniform draws attributed to the nearest optimum (the Voronoi
                 volume share; this is the null entry 64 used on N09).
    ``desc``  -- uniform draws *descended* first (see ``_null_descent``).
    """
    ap = argparse.ArgumentParser(prog="hunt_coverage.py --null")
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--func", default="N18-CF3-10D")
    ap.add_argument("--geo-draws", type=int, default=1_000_000)
    ap.add_argument("--descents", type=int, default=0)
    ap.add_argument("--budget", type=int, default=1499,
                    help="evaluations per descent (N18 hunt length, e76)")
    ap.add_argument("--sigma-ratio", type=float, default=0.2,
                    help="sigma0 / span; MC-ESO's `sigma` default")
    ap.add_argument("--full-cov", action="store_true",
                    help="sensitivity: run the descent with covariance on")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--geo-csv", type=Path, default=None)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.benchmarks import niching_by_name
    b = niching_by_name(a.func)
    opts = true_optima(b)
    K = len(opts)
    lo, hi = b.bounds
    print(f"{a.func}: dim {b.dim}, box [{lo}, {hi}], K = {K}")

    # ── geometric (Voronoi) null ────────────────────────────────────────────
    rng = np.random.default_rng(0)
    cnt = np.zeros(K, dtype=np.int64)
    done = 0
    while done < a.geo_draws:
        m = min(20000, a.geo_draws - done)
        X = rng.uniform(lo, hi, size=(m, b.dim))
        d = np.linalg.norm(X[:, None, :] - opts[None, :, :], axis=2)
        np.add.at(cnt, np.argmin(d, axis=1), 1)
        done += m
    share = cnt / cnt.sum()
    se = np.sqrt(share * (1 - share) / a.geo_draws)
    print(f"\n== geometric (Voronoi) null, {a.geo_draws} uniform draws")
    for j in range(K):
        print(f"  opt {j}: share {share[j]:.4f} +- {1.96 * se[j]:.4f} (95% CI)")
    if a.geo_csv:
        a.geo_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.geo_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["opt", "cnt", "share"])
            for j in range(K):
                w.writerow([j, int(cnt[j]), f"{share[j]:.6f}"])
        print(f"  counts written to {a.geo_csv}")

    if a.descents <= 0:
        return
    # ── restart-lander null: uniform draw, then descend ─────────────────────
    sigma0 = a.sigma_ratio * (hi - lo)
    print(f"\n== descent null: {a.descents} draws, {a.budget} evals each, "
          f"sigma0 = {sigma0:g}, {'full covariance' if a.full_cov else 'isotropic'}")
    jobs = [(a.func, k, a.budget, sigma0, not a.full_cov)
            for k in range(a.descents)]
    from multiprocess import Pool
    rows = []
    t0 = time.time()
    with Pool(a.procs) as pool:
        for k, j0, j, dist, f, used, first in pool.imap_unordered(_null_descent, jobs):
            rows.append((k, j0, j, dist, f, used)
                        + tuple(v for pair in first for v in pair))
    print(f"  {len(rows)} descents in {time.time() - t0:.1f}s")
    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            # `ev_<eps>` / `opt_<eps>`: evaluations spent when the descent first
            # got inside that accuracy, and the optimum it was nearest then
            # (-1 = never).  Dumps written before entry 92 stop at `evals`.
            cols = ["draw", "start_opt", "land_opt", "dist", "best_f", "evals"]
            for e in _NULL_EPS:
                cols += [f"ev_{e:g}", f"opt_{e:g}"]
            w.writerow(cols)
            w.writerows(rows)
        print(f"  rows written to {a.csv}")
    land = np.zeros(K, dtype=np.int64)
    np.add.at(land, np.array([r[2] for r in rows]), 1)
    ds = np.array([r[3] for r in rows])
    fsv = np.array([r[4] for r in rows])
    p = land / land.sum()
    sed = np.sqrt(p * (1 - p) / len(rows))
    for j in range(K):
        print(f"  opt {j}: share {p[j]:.4f} +- {1.96 * sed[j]:.4f}  "
              f"(geo {share[j]:.4f})")
    print(f"  descent endpoint: median dist {np.median(ds):.4f}, "
          f"median f {np.median(fsv):.4g}, frac f <= 0.1 "
          f"{float((fsv <= 0.1).mean()):.3f}")


def analyze_mode(paths: list[str]) -> None:
    """Per-seed width statistics per method, then across-method comparison."""
    from scipy.stats import mannwhitneyu, spearmanr
    rows = []
    for p in paths:                     # row-level dumps are stored gzipped
        opener = ((lambda q: __import__("gzip").open(q, "rt", newline=""))
                  if p.endswith(".gz") else (lambda q: open(q, newline="")))
        with opener(p) as fh:
            rows += list(csv.DictReader(fh))
    eps_vals = sorted({r["eps"] for r in rows}, key=float, reverse=True)
    methods = sorted({r["method"] for r in rows})
    for eps in eps_vals:
        print(f"\n=== eps = {eps} " + "=" * 52)
        # `pts/arr` is the residency probe of entry 65: how many evaluations at
        # the accuracy each entry into an eps-ball consumes.  `q4 pts%` is the
        # widest-quartile share of that residency, i.e. the same concentration
        # statistic as `q4 arr%` but weighted by budget spent rather than by
        # entries.  Read both against the caveat above: methods that interleave
        # populations in one evaluation stream inflate arrivals, so compare
        # arms of the same method first and across methods only qualitatively.
        print(f"{'method':<12}{'seeds':>6}{'cov':>7}{'cov%':>7}{'A12 w':>8}"
              f"{'rho':>7}{'q4 arr%':>9}{'q1 cov%':>9}{'q4 cov%':>9}"
              f"{'pts/arr':>9}{'q4 pts%':>9}")
        print("-" * 92)
        stats: dict[str, dict[str, list[float]]] = {}
        for m in methods:
            sel = [r for r in rows if r["method"] == m and r["eps"] == eps]
            if not sel:
                continue
            seeds = sorted({int(r["seed"]) for r in sel})
            acc = {"cov": [], "a12": [], "rho": [], "q4arr": [],
                   "q1cov": [], "q4cov": [], "ppa": [], "q4pts": []}
            for s in seeds:
                sr = [r for r in sel if int(r["seed"]) == s]
                sr.sort(key=lambda r: int(r["opt"]))
                w = np.array([float(r["width"]) for r in sr])
                arr = np.array([float(r["arrivals"]) for r in sr])
                pts = np.array([float(r["pts"]) for r in sr])
                hit = pts > 0
                acc["cov"].append(hit.sum())
                if hit.any() and (~hit).any():
                    u, _ = mannwhitneyu(w[hit], w[~hit], alternative="greater")
                    acc["a12"].append(u / (hit.sum() * (~hit).sum()))
                elif hit.all():
                    acc["a12"].append(float("nan"))
                if arr.sum() > 0:
                    acc["rho"].append(spearmanr(w, arr).statistic)
                q = np.quantile(w, [0.25, 0.75])
                q1, q4 = w <= q[0], w >= q[1]
                acc["q4arr"].append(arr[q4].sum() / max(arr.sum(), 1))
                acc["q1cov"].append(hit[q1].mean())
                acc["q4cov"].append(hit[q4].mean())
                acc["ppa"].append(pts.sum() / max(arr.sum(), 1))
                acc["q4pts"].append(pts[q4].sum() / max(pts.sum(), 1))
            stats[m] = acc
            K = len({int(r["opt"]) for r in sel})
            md = lambda k: (np.nanmedian(acc[k]) if len(acc[k]) else float("nan"))
            print(f"{m:<12}{len(seeds):>6}{md('cov'):>7.1f}"
                  f"{100 * md('cov') / K:>7.1f}{md('a12'):>8.2f}"
                  f"{md('rho'):>7.2f}{100 * md('q4arr'):>9.1f}"
                  f"{100 * md('q1cov'):>9.1f}{100 * md('q4cov'):>9.1f}"
                  f"{md('ppa'):>9.1f}{100 * md('q4pts'):>9.1f}")
        # across-method: is the width bias the same in every method?
        print("\n  across-method (Mann-Whitney on the per-seed statistic):")
        ms = [m for m in methods if m in stats]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                for key, lbl in (("a12", "A12(width|reached)"),
                                 ("rho", "Spearman(width,arrivals)"),
                                 ("q4arr", "widest-quartile arrival share"),
                                 ("ppa", "pts per arrival (residency)"),
                                 ("q4pts", "widest-quartile pts share")):
                    x = [v for v in stats[ms[i]][key] if np.isfinite(v)]
                    y = [v for v in stats[ms[j]][key] if np.isfinite(v)]
                    if len(x) < 3 or len(y) < 3:
                        continue
                    u, p = mannwhitneyu(x, y, alternative="two-sided")
                    print(f"    {ms[i]:>11} vs {ms[j]:<11} {lbl:<30}"
                          f"{np.median(x):>7.2f}{np.median(y):>7.2f}"
                          f"   p = {p:.3g}   A12 = {u / (len(x) * len(y)):.2f}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        return run_mode(sys.argv[1:])
    if len(sys.argv) > 1 and sys.argv[1] == "--null":
        return null_mode(sys.argv[1:])
    if len(sys.argv) > 2 and sys.argv[1] == "--analyze":
        return analyze_mode(sys.argv[2:])
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)

    rows = []
    for p in sys.argv[1:]:
        # Row-level dumps are stored gzipped (the repository rule); read both.
        opener = ((lambda q: __import__("gzip").open(q, "rt", newline=""))
                  if p.endswith(".gz") else (lambda q: open(q, newline="")))
        with opener(p) as fh:
            for r in csv.DictReader(fh):
                xs = [float(r[k]) for k in r if k.startswith("x") and k[1:].isdigit()]
                rows.append((r["function"], int(r["seed"]), float(r["f"]), np.array(xs)))

    name = rows[0][0]
    dim = len(rows[0][3])
    if "Vincent" not in name:
        raise SystemExit(f"{name}: only Vincent's optima are hard-coded here")
    opts = vincent_optima(dim)

    print(f"{name}: {len(rows)} hunt endpoints over "
          f"{len({s for _, s, _, _ in rows})} seeds, K = {len(opts)}")
    print(f"Optimum spacing along one axis: "
          f"{np.diff(_V1D).min():.3f} (narrowest) .. {np.diff(_V1D).max():.3f} "
          f"(widest), ratio {np.diff(_V1D).max() / np.diff(_V1D).min():.0f}x")
    print("MC-ESO's post-exhaustion repel radius is 0.02 * span = "
          f"{0.02 * (10.0 - 0.25):.3f}\n")

    # ── which optima get hit, how often ─────────────────────────────────────
    hit = Counter()
    for _, seed, f, x in rows:
        j = int(np.argmin(np.linalg.norm(opts - x, axis=1)))
        hit[(seed, j)] += 1

    per_seed: dict[int, Counter] = {}
    for (seed, j), c in hit.items():
        per_seed.setdefault(seed, Counter())[j] += c

    print(f"{'seed':>5}{'hunts':>7}{'optima hit':>12}{'max repeats':>13}"
          f"{'hunts wasted':>14}")
    print("-" * 51)
    for seed in sorted(per_seed):
        c = per_seed[seed]
        n = sum(c.values())
        print(f"{seed:>5}{n:>7}{len(c):>12}{max(c.values()):>13}"
              f"{n - len(c):>14}")

    # ── are the repeatedly-hit optima the wide-basin ones? ──────────────────
    # Basin width proxy: distance to the nearest other optimum. On a log-spaced
    # landscape this varies by design, and it is exactly what a fixed repel
    # radius cannot adapt to.
    d = np.linalg.norm(opts[:, None, :] - opts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    width = d.min(axis=1)

    total = Counter()
    for (_, j), c in hit.items():
        total[j] += c
    seeds = len(per_seed)

    print("\nHits vs. basin width (nearest-neighbour distance), optima pooled "
          "over seeds.")
    print(f"{'basin width band':>22}{'optima':>8}{'hits':>7}{'hits/optimum':>14}"
          f"{'covered':>9}")
    print("-" * 60)
    qs = np.quantile(width, [0.0, 0.25, 0.5, 0.75, 1.0])
    for a, b in zip(qs[:-1], qs[1:]):
        sel = [j for j in range(len(opts))
               if (a <= width[j] < b) or (b == qs[-1] and width[j] == b)]
        if not sel:
            continue
        h = sum(total.get(j, 0) for j in sel)
        cov = sum(1 for j in sel if total.get(j, 0) > 0)
        print(f"{f'{a:.2f} - {b:.2f}':>22}{len(sel):>8}{h:>7}"
              f"{h / len(sel) / seeds:>14.2f}{f'{cov}/{len(sel)}':>9}")

    w_hit = np.array([width[j] for j in range(len(opts)) if total.get(j, 0)])
    w_miss = np.array([width[j] for j in range(len(opts)) if not total.get(j, 0)])
    print(f"\nmedian basin width, optima ever reached : {np.median(w_hit):.3f} "
          f"(n={len(w_hit)})")
    if len(w_miss):
        print(f"median basin width, optima never reached: "
              f"{np.median(w_miss):.3f} (n={len(w_miss)})")
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(w_hit, w_miss, alternative="greater")
        a12 = u / (len(w_hit) * len(w_miss))
        print(f"Mann-Whitney (reached wider than missed): p = {p:.2g}, "
              f"A12 = {a12:.2f}")
        print("\nA12 well above 0.5 means the hunts are landing in the wide "
              "basins and missing the\nnarrow ones — a uniform reseed with one "
              "fixed repel radius, on a log-spaced landscape.")
    else:
        print("every optimum was reached at least once by some seed.")


if __name__ == "__main__":
    main()
