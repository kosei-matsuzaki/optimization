#!/usr/bin/env python3
"""Diagnostic: which f-independent structural signals separate the per-landscape
optimal channel route?

Runs *unmodified* base MC-ESO (SR untouched) over ALL 35 benchmarks (BBOB-24 +
Custom-11, dim 2) and post-processes the recorded per-generation populations +
dynamics into a rich set of candidate signals, tabulated against each function's
known optimal air-budget route (from the four rejected channel-schedule variants,
2026-07). The point is to find a small set of signals + thresholds that routes
the *priority* functions correctly while not mis-routing the maintain ones.

Optimal routes (air budget destination):
  • droplet : F11 F12 F13 F14   (ill-conditioned valleys/ridges)
  • close   : F04 F16           (separable / axis-aligned)
  • keep-air: F15 F17 F19 F20 F24 C05 C11  (multimodal / rotated-smooth: air escape)

Signals (median over the EXPLORATION phase — σ_global above the drilling
threshold — then mean over runs):
  cond    log10(λmax/λmin) of the population covariance     (conditioning)
  PR      participation ratio (Σλ)²/Σλ² ∈ [1, dim]          (spectrum spread)
  algD    max|comp| of the dominant eigenvector             (axis-align, dominant)
  algA    mean_j max_i|V_ij| over all eigenvectors          (axis-align, overall)
  offd    RMS off-diagonal of the correlation matrix ∈ [0,1] (rotation/coupling)
  divs    mean(std)/span                                    (spread / convergence)
  kurt    mean per-dim excess kurtosis                      (peakedness; multimodal↓)
  mgap    max per-dim normalized nearest-gap                (cluster/multimodality)
  nelt    mean niched-elite count                           (multimodality proxy)
  nelX    max niched-elite count                            (multimodality proxy)
  spil    spillover (restart) count                         (stagnation/multimodality)

Usage:
    .venv/bin/python3 scripts/measure_channel_signals.py [--n-runs 8] [--max-evals 5000] [--csv path]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.benchmarks import BENCHMARKS_BY_NAME
from core.optimizers.mceso import MultiChannelEpidemicOptimizer

# function-number prefix -> (known optimal route, priority)
#   priority: ★ improve-target, · maintain, ✗ deprioritized (regression allowed)
ROUTES: dict[str, tuple[str, str]] = {
    "F11": ("droplet", "★"), "F12": ("droplet", "★"),
    "F13": ("droplet", "★"), "F14": ("droplet", "★"),
    "F02": ("droplet?", "·"), "F10": ("droplet?", "·"),
    "F04": ("close", "★"), "F16": ("close", "★"),
    "F06": ("close?", "★"), "F18": ("?", "★"), "F19": ("keep-air", "★"),
    "F15": ("keep-air", "·"), "F20": ("keep-air", "·"), "C05": ("keep-air", "·"),
    "F17": ("keep-air", "✗"), "F24": ("keep-air", "✗"), "C11": ("keep-air", "✗"),
}
_ROUTE_ORDER = ["droplet", "droplet?", "close", "close?", "?", "keep-air", "-"]


def _excess_kurtosis(x: np.ndarray) -> float:
    m = x.mean(); v = x.var()
    if v < 1e-300:
        return 0.0
    return float(np.mean((x - m) ** 4) / (v ** 2) - 3.0)


def _max_norm_gap(x: np.ndarray) -> float:
    s = np.sort(x); rng = s[-1] - s[0]
    if rng < 1e-300 or len(s) < 2:
        return 0.0
    return float(np.max(np.diff(s)) / rng)


def _gen_signals(pop: np.ndarray, span: float) -> tuple | None:
    """Per-generation covariance/geometry signals, or None if degenerate."""
    if pop.shape[0] < 2 or pop.shape[1] < 2:
        return None
    dim = pop.shape[1]
    cov = np.cov(pop, rowvar=False)
    ev, V = np.linalg.eigh(cov)
    ev = np.maximum(ev, 1e-300)
    cond = np.log10(ev[-1] / ev[0])
    pr = (ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-300)
    algD = float(np.max(np.abs(V[:, -1])))
    algA = float(np.mean(np.max(np.abs(V), axis=0)))
    std = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    corr = cov / np.outer(std, std)
    iu = np.triu_indices(dim, k=1)
    offd = float(np.sqrt(np.mean(corr[iu] ** 2))) if len(iu[0]) else 0.0
    divs = float(np.mean(std) / span)
    kurt = float(np.mean([_excess_kurtosis(pop[:, d]) for d in range(dim)]))
    mgap = float(np.max([_max_norm_gap(pop[:, d]) for d in range(dim)]))
    return cond, pr, algD, algA, offd, divs, kurt, mgap


def _run_signals(res, span: float, drill: float) -> tuple | None:
    pops = res.history_pop[1:]
    sig = res.history_sigma_global
    nel = res.history_n_elite
    nis = res.history_no_improve
    acc = [[] for _ in range(8)]
    nelits = []
    for g in range(min(len(pops), len(sig))):
        if sig[g] <= drill * span:
            continue
        s = _gen_signals(np.asarray(pops[g]), span)
        if s is None:
            continue
        for i in range(8):
            acc[i].append(s[i])
        if g < len(nel):
            nelits.append(nel[g])
    if not acc[0]:
        return None
    med = [float(np.median(a)) for a in acc]
    nelt = float(np.mean(nelits)) if nelits else np.nan
    nelX = float(np.max(nel)) if len(nel) else np.nan
    # spillover count: no_improve builds up then resets sharply toward 0.
    spil = 0
    for g in range(len(nis) - 1):
        if nis[g] >= 250 and nis[g + 1] < 50:
            spil += 1
    return (*med, nelt, nelX, float(spil))


def main(n_runs: int, max_evals: int, csv_path: str | None) -> None:
    drill = 1e-3
    names = sorted(BENCHMARKS_BY_NAME,
                   key=lambda n: (n[0], int(n[1:3]) if n[1:3].isdigit() else 99))
    hdr = ["function", "route", "pri", "cond", "PR", "algD", "algA",
           "offd", "divs", "kurt", "mgap", "nelt", "nelX", "spil"]
    results = []
    for name in names:
        bench = BENCHMARKS_BY_NAME[name]
        if bench.dim != 2:
            continue
        span = bench.bounds[1] - bench.bounds[0]
        pre = name[:3]
        route, pri = ROUTES.get(pre, ("-", "·"))
        runs = []
        for seed in range(n_runs):
            res = MultiChannelEpidemicOptimizer(bench, seed=seed).optimize(max_evals=max_evals)
            r = _run_signals(res, span, drill)
            if r is not None:
                runs.append(r)
        if not runs:
            continue
        agg = np.nanmean(np.array(runs), axis=0)
        results.append([name, route, pri] + [float(v) for v in agg])

    results.sort(key=lambda r: (_ROUTE_ORDER.index(r[1]) if r[1] in _ROUTE_ORDER else 99, -r[3]))

    print(f"measure_channel_signals  n_runs={n_runs} max_evals={max_evals} dim=2 "
          f"(base MC-ESO, unmodified) — all 35 functions\n")
    print(f"{'function':22s}{'route':10s}{'pri':4s}"
          f"{'cond':>6s}{'PR':>6s}{'algD':>6s}{'algA':>6s}{'offd':>6s}"
          f"{'divs':>7s}{'kurt':>7s}{'mgap':>6s}{'nelt':>6s}{'nelX':>6s}{'spil':>6s}")
    print("-" * 116)
    cur = None
    for r in results:
        if r[1] != cur:
            print(); cur = r[1]
        print(f"{r[0]:22s}{r[1]:10s}{r[2]:4s}"
              f"{r[3]:6.2f}{r[4]:6.2f}{r[5]:6.3f}{r[6]:6.3f}{r[7]:6.3f}"
              f"{r[8]:7.4f}{r[9]:7.2f}{r[10]:6.3f}{r[11]:6.2f}{r[12]:6.1f}{r[13]:6.1f}")
    print("\nlegend: cond=log10(λmax/λmin) PR=[1,2] algD/algA=axis-align[.71,1] "
          "offd=RMS offdiag corr[0,1] divs=spread/span kurt=excess kurtosis "
          "mgap=max norm gap nelt/nelX=niche mean/max spil=#spillover")

    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(hdr)
            for r in results:
                w.writerow(r)
        print(f"\nsaved CSV → {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=8)
    ap.add_argument("--max-evals", type=int, default=5000)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()
    main(args.n_runs, args.max_evals, args.csv)
