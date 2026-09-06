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
    from core.benchmarks import NICHING_BENCHMARKS_BY_NAME
    from scripts.niching_baseline import _METHODS
    b = NICHING_BENCHMARKS_BY_NAME[name]
    cls, kw = dict(_METHODS, **_DIAGNOSTIC_ARMS)[method]
    t0 = time.time()
    r = cls(b, seed=seed, **kw).optimize(budget)
    opts = vincent_optima(b.dim)
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
    from core.benchmarks import NICHING_BENCHMARKS_BY_NAME
    b = NICHING_BENCHMARKS_BY_NAME[a.func]
    if "Vincent" not in a.func:
        raise SystemExit(f"{a.func}: only Vincent's optima are hard-coded here")
    opts = vincent_optima(b.dim)
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
