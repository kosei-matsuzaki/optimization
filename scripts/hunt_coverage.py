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

Usage:  python3 scripts/hunt_coverage.py analysis/hm/hunts_n07_xy.csv
"""
from __future__ import annotations
import csv
import sys
from collections import Counter

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


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)

    rows = []
    for p in sys.argv[1:]:
        with open(p) as fh:
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
