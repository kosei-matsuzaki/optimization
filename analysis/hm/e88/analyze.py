"""Is the F14-F20 coverage ceiling a property of the restart-lander *class*,
on the whole function set, or only on N18 (entry 87)?

Entry 87 answered this on N18-CF3-10D: the descent null (uniform draw ->
isotropic descent at MC-ESO's own sigma0 -> nearest optimum) reaches exactly the
4 optima MC-ESO reaches and exactly the 4 NMMSO reaches, and the two it never
reaches (gopt 2, gopt 3) get 0 of 2000 draws and 0 of 300 ten-times-budget
draws.  That made ``PRtrue@1e-1 = 4/6`` a property of the landscape under this
class of search, not a defect of MC-ESO's restart rule.

What was left open is whether N18 is special.  This entry runs the same two
nulls on the six functions of the set entry 87 did not touch (N14, N15, N16,
N17, N19, N20).  **No optimizer runs at all** -- the nulls are the whole
measurement, which is what makes the question affordable (entry 87: 2000
descents x 1499 evaluations = 101 s on 4 cores).

Two statistics, both reported for every function, because they answer different
questions and the queue's phrase "the number of distinct optima the desc null
reaches" is ambiguous between them:

  per-run   The expected number of distinct optima one *run's worth* of restart
            landings covers: n = suite budget / descent budget = 400000/1499 =
            267 multinomial draws from the null's landing distribution, 20000
            simulations.  This is the quantity ``PR x K`` measures, because
            PR@1e-5 is a per-run average, so this is the primary read.
  union     Distinct optima that are the nearest endpoint of *any* of the 2000
            descents.  2000 descents is ~7.5 runs' worth, so this is the class's
            absolute reach: an optimum that takes 0 of 2000 has no basin of
            attraction under uniform-restart-plus-isotropic-descent at all.

Rejection condition, as written in the queue before any number was read: if on
any function the desc null's reach exceeds the published best PR@1e-5 x K, then
the restart-lander class still has room on that function -- coverage headroom is
real somewhere in F14-F20 and the axis is not closed.  If every function falls
below, the F14-F20 coverage ceiling is the class ceiling, and the cheap screen
the overview asked for ("does a candidate function set have headroom?") can be
written as "run the desc null and compare its reach with the published best".

Disclosure: N14's landing shares were on screen (the six runs are sequential)
when the two statistics above were separated.  Both are reported for all six
functions and the verdict is stated under each, so the split cannot be chosen
after the fact.

Published best PR@1e-5: HillVallEA GECCO'18 arXiv 1810.07085 table 4, the "best
of 6 methods" row of docs/related_work.md.  Entry 71 obtained it; queue item 4
notes it is a single source.

Usage: python3 analysis/hm/e88/analyze.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260908)
NSIM = 20000
SUITE_BUDGET = 400_000

# function -> (K, published best PR@1e-5)
FUNCS = {
    "N14-CF3-3D": (6, 0.793),
    "N15-CF4-3D": (8, 0.750),
    "N16-CF3-5D": (6, 0.677),
    "N17-CF4-5D": (8, 0.745),
    "N19-CF4-10D": (8, 0.512),
    "N20-CF4-20D": (8, 0.465),
}
# entry 87's own numbers, carried in for the comparison row (not re-run here).
N18 = {"K": 6, "pub": 0.667, "union": 4, "per_run": 4.0}


def _find(stem: str) -> Path:
    for p in (HERE / stem, HERE / (stem + ".gz")):
        if p.exists():
            return p
    raise SystemExit(f"missing: {stem}")


def per_run_distinct(p: np.ndarray, n: int) -> tuple[float, float, float]:
    """Distinct optima covered by n independent landings from p.

    ``p`` may sum to less than 1: the deficit is the probability that a descent
    ends at no optimum at all (see ``landing_probs``), and it is carried as an
    extra multinomial category that contributes to no coverage.
    """
    q = np.append(p, max(0.0, 1.0 - p.sum()))
    draws = RNG.multinomial(n, q, size=NSIM)[:, :len(p)]
    d = (draws > 0).sum(axis=1).astype(float)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def landing_probs(t: pd.DataFrame, K: int, eps: float | None) -> np.ndarray:
    """Per-optimum landing probability of one descent.

    ``eps = None`` is nearest-attribution alone (entry 87's read, and the
    generous one: it counts an endpoint for its nearest optimum however far away
    it stopped).  A number restricts to endpoints that are actually *at* that
    accuracy, which is what ``PR@eps`` counts -- and it is the only fair
    comparison against a published ``PR@1e-5``.  On N20-CF4-20D the two differ
    completely: nearest-attribution says the class reaches 4 of 8 optima, and no
    descent at all gets within 1e-1 of any of them.
    """
    sel = t if eps is None else t[t.best_f <= eps]
    c = np.zeros(K, dtype=np.int64)
    if len(sel):
        for j, n in sel.land_opt.value_counts().items():
            c[int(j)] = int(n)
    return c / len(t)


def report(func: str, budget: int, tag: str) -> dict:
    K, pub = FUNCS[func]
    t = pd.read_csv(_find(f"null_iso_b{budget}_{func}.csv"))
    n_draws = len(t)
    cnt = np.zeros(K, dtype=np.int64)
    v = t.land_opt.value_counts()
    for j, c in v.items():
        cnt[int(j)] = int(c)
    p = cnt / cnt.sum()
    union = int((cnt > 0).sum())

    # accuracy-restricted reach: an endpoint only counts for optimum j if it is
    # actually at the accuracy (the scoring rule's f threshold), which is what
    # PR@eps counts.  Nearest-attribution alone is the generous read.
    reach_eps = {}
    for eps in (1e-1, 1e-3, 1e-5):
        sub = t[t.best_f <= eps]
        reach_eps[eps] = int(sub.land_opt.nunique()) if len(sub) else 0

    # A restart lander spends the suite budget on landings, so the number of
    # landings is set by what a descent *actually costs*, not by the cap: CMA-ES
    # stops itself when it has converged, and on the 3D functions it does so
    # well under 1499 evaluations.  Using the cap would understate the number of
    # restarts and so understate the class.
    n_hunts = max(1, int(SUITE_BUDGET / t.evals.mean()))
    mean, lo, hi = per_run_distinct(p, n_hunts)
    # the comparison that matches PR@1e-5: only endpoints at the accuracy count.
    pr = {e: per_run_distinct(landing_probs(t, K, e), n_hunts)
          for e in (1e-1, 1e-5)}
    thr = pub * K
    print(f"\n{'=' * 78}\n== {func}  ({tag}: {n_draws} descents x {budget} evals, K = {K})")
    print("  landing share: " + "  ".join(f"{j}:{p[j]:.4f}" for j in range(K)))
    print(f"  never landed on            : {[j for j in range(K) if cnt[j] == 0]}")
    print(f"  <= 3 of {n_draws} draws        : "
          f"{[j for j in range(K) if 0 < cnt[j] <= 3]}")
    print(f"  endpoint quality           : median f {t.best_f.median():.4g}, "
          f"frac f <= 1e-1 {float((t.best_f <= 0.1).mean()):.3f}, "
          f"frac f <= 1e-5 {float((t.best_f <= 1e-5).mean()):.3f}")
    print(f"  evals per descent          : median {t.evals.median():.0f}, "
          f"mean {t.evals.mean():.0f}, cap {budget} "
          f"({'capped' if t.evals.max() >= budget else 'CMA stopped itself'})")
    print(f"  reach restricted to f<=eps : "
          + "  ".join(f"{e:g}:{reach_eps[e]}" for e in (1e-1, 1e-3, 1e-5)))
    print(f"  UNION reach ({n_draws} draws)   : {union} / {K}")
    print(f"  PER-RUN distinct (n={n_hunts:>4})  : {mean:.2f}  [{lo:.0f}, {hi:.0f}]"
          f"   (nearest-attribution, no accuracy check)")
    for e in (1e-1, 1e-5):
        m2, l2, h2 = pr[e]
        print(f"  PER-RUN distinct @ f<={e:g}   : {m2:.2f}  [{l2:.0f}, {h2:.0f}]")
    print(f"  threshold  pub x K         : {pub:.3f} x {K} = {thr:.2f}")
    print(f"  VERDICT (primary, @1e-5)   : "
          f"{'REJECT (headroom)' if pr[1e-5][0] > thr else 'no'}"
          f"   |  @1e-1 : {'above thr' if pr[1e-1][0] > thr else 'no'}"
          f"   |  nearest-attr : {'above thr' if mean > thr else 'no'}")
    return {"func": func, "K": K, "pub": pub, "thr": thr, "union": union,
            "per_run": mean, "per_run_lo": lo, "per_run_hi": hi,
            "pr1": pr[1e-1][0], "pr1_lo": pr[1e-1][1], "pr1_hi": pr[1e-1][2],
            "pr5": pr[1e-5][0], "pr5_lo": pr[1e-5][1], "pr5_hi": pr[1e-5][2],
            "reach_1e1": reach_eps[1e-1], "reach_1e5": reach_eps[1e-5],
            "n_hunts": n_hunts, "budget": budget, "n_draws": n_draws,
            "med_f": float(t.best_f.median()),
            "frac_le_1e1": float((t.best_f <= 0.1).mean()),
            "shares": p}


def transition(func: str, budget: int) -> None:
    """Voronoi cell -> basin of attraction, the map entry 87 read on N18."""
    K = FUNCS[func][0]
    t = pd.read_csv(_find(f"null_iso_b{budget}_{func}.csv"))
    m = pd.crosstab(t.start_opt, t.land_opt).reindex(
        index=range(K), columns=range(K), fill_value=0)
    keep = np.trace(m.to_numpy()) / max(1, m.to_numpy().sum())
    print(f"\n-- {func}: start -> land, row-normalised "
          f"(draws keeping their own start optimum: {keep:.4f})")
    print(m.div(m.sum(axis=1).replace(0, 1), axis=0).round(3).to_string())


def main() -> None:
    # Two budget allocations of the same 400000 evaluations.  `short` is
    # MC-ESO's own observed hunt length (entry 76/87); `long` lets each descent
    # run until CMA-ES's own stopping rules fire, which costs restarts.  The
    # class ceiling is the better of the two, so both have to be read.
    short = {f: report(f, 1499, "short descents = MC-ESO's hunt length")
             for f in FUNCS}
    long_ = {f: report(f, 14990, "long descents = CMA-ES stops itself")
             for f in FUNCS}

    print(f"\n{'=' * 78}\n== SUMMARY: distinct optima one run of the restart-lander "
          f"class covers,\n   at the accuracy the published number is quoted at "
          f"(PR@1e-5), under both allocations\n   of the same 400000 evaluations. "
          f"'thr' = published best PR@1e-5 x K.")
    print(f"\n{'func':<13}{'K':>3}{'pub':>7}{'thr':>7} |"
          f"{'n':>5}{'pr@1e-5':>9}{'pr@1e-1':>9} |"
          f"{'n':>5}{'pr@1e-5':>9}{'pr@1e-1':>9} |{'best':>7}{'verdict':>9}")
    print(f"{'':<30} |{'---- short descents ----':^23} |"
          f"{'---- long descents ----':^23} |")
    print("-" * 100)
    for f in FUNCS:
        s, l = short[f], long_[f]
        best = max(s["pr5"], l["pr5"])
        print(f"{f:<13}{s['K']:>3}{s['pub']:>7.3f}{s['thr']:>7.2f} |"
              f"{s['n_hunts']:>5}{s['pr5']:>9.2f}{s['pr1']:>9.2f} |"
              f"{l['n_hunts']:>5}{l['pr5']:>9.2f}{l['pr1']:>9.2f} |"
              f"{best:>7.2f}{('REJECT' if best > s['thr'] else 'no'):>9}")
    print(f"{'N18-CF3-10D':<13}{N18['K']:>3}{N18['pub']:>7.3f}"
          f"{N18['pub'] * N18['K']:>7.2f} |{267:>5}{N18['per_run']:>9.2f}"
          f"{N18['per_run']:>9.2f} |{'26':>5}{'4.00':>9}{'4.00':>9} |"
          f"{4.00:>7.2f}{'(e87)':>9}")
    print("\n  REJECT = the class already covers more optima than the published "
          "best does,\n  i.e. coverage is not what stops a method on that "
          "function -- headroom is real there.")
    for f in FUNCS:
        transition(f, 1499)


if __name__ == "__main__":
    main()
