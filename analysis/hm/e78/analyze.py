#!/usr/bin/env python3
"""Does entry 77's `basin_reset` arm cost anything on the BBOB-24 dim2 gate?

Entry 77 bought depth on N18-CF3-10D (PR@1e-5 0.167 -> 0.292, 12 seeds, no
loss) by resetting the stagnation counter on the *basin's* own progress. That
counter also drives the restart path, the sigma control and `_basin_exhausted`,
so the arm is not a release-clause knob and the cheap proxies do not apply:

  * `post_gain > 0` is inadmissible (entry 56): it only sees improvements after
    the first exhaustion, and this knob is live from evaluation one.
  * `best_f` byte equality is inadmissible as a *proxy* for the pins (entry 56
    again): the proxy holds only where the knob cannot fire before the answer is
    banked. The pre-flight probe already saw the histories diverge in 7 of 8
    runs, so the arm has to be judged at level (3) of entry 61's hierarchy --
    the pins themselves, measured directly on both arms.

Pre-registered, before the 480 cells were run:

  H0  the arm does not cost BBOB-24 dim2 performance.
  Rejected if SR@1e-10 falls below base's 92.08% (this environment's value,
      entries 26/29) or `evals_succ_mean` worsens. Either way the per-function
      breakdown is reported, so a rejection names the functions that pay.

Both arms share the seed, so every cell is a pair. Reported per budget:

  SR@1e-10       both arms, plus the discordant cells (won / lost) and an exact
                 two-sided sign test on them -- the paired form of the SR
                 comparison, which the marginal rates alone do not give.
  evals_succ     paired Wilcoxon over the cells where *both* arms reached
                 1e-10, plus each arm's own mean over its own successes (the
                 CLAUDE.md aggregate, which is not paired and moves with the
                 success set).
  per function   CLAUDE.md requires the function-level list; a mean that hides
                 one function paying is not a pass.

Usage: python3 analysis/hm/e78/analyze.py analysis/hm/e78/gate5k_c*.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from math import comb

import numpy as np
from scipy.stats import wilcoxon

THRESH_NAME = "1e-10"
BASE_SR_PIN = 0.9208        # this environment's base, entries 26/29
BASE_EV_PIN = 677.7


def _load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                rows.append(dict(
                    function=r["function"], seed=int(r["seed"]),
                    div_ev=int(r["div_ev"]), bank_ev=int(r["bank_ev"]),
                    base_f=float(r["base_best_f"]), var_f=float(r["var_best_f"]),
                    base_ev=int(r["base_succ_ev"]), var_ev=int(r["var_succ_ev"]),
                    n_fire=int(r["n_fire"]), same=r["same"] == "True"))
    return rows


def _sign_test(won: int, lost: int) -> float:
    """Exact two-sided binomial test on the discordant cells (p = 1/2)."""
    n = won + lost
    if n == 0:
        return 1.0
    k = min(won, lost)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def report(rows, label: str) -> None:
    n = len(rows)
    base_ok = np.array([r["base_f"] <= 1e-10 for r in rows])
    var_ok = np.array([r["var_f"] <= 1e-10 for r in rows])
    won = int((var_ok & ~base_ok).sum())
    lost = int((~var_ok & base_ok).sum())

    print(f"\n=== {label}: {n} paired cells "
          f"({len({r['function'] for r in rows})} functions x "
          f"{len({r['seed'] for r in rows})} seeds) ===")

    fires = np.array([r["n_fire"] for r in rows])
    diverged = sum(1 for r in rows if r["div_ev"] >= 0)
    same_f = sum(1 for r in rows if r["same"])
    print(f"  firing        : {int((fires > 0).sum())}/{n} cells fire "
          f"(median {np.median(fires):.0f}, max {fires.max()}) "
          f"-- a tie here is not an unfired knob")
    print(f"  gate level (1): histories byte-identical in {n - diverged}/{n} cells")
    print(f"  gate level (2): best_f byte-identical in {same_f}/{n} cells")

    print(f"  SR@{THRESH_NAME}     : base {base_ok.mean():.4f}  "
          f"variant {var_ok.mean():.4f}   "
          f"(won {won}, lost {lost}, p = {_sign_test(won, lost):.4f})")

    b_s = [r["base_ev"] for r in rows if r["base_ev"] >= 0]
    v_s = [r["var_ev"] for r in rows if r["var_ev"] >= 0]
    print(f"  evals_succ_mean: base {np.mean(b_s):.1f} (n={len(b_s)})  "
          f"variant {np.mean(v_s):.1f} (n={len(v_s)})")
    both = [(r["base_ev"], r["var_ev"]) for r in rows
            if r["base_ev"] >= 0 and r["var_ev"] >= 0]
    if both:
        a = np.array([x for x, _ in both], float)
        c = np.array([y for _, y in both], float)
        d = c - a
        try:
            p = wilcoxon(a, c).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        print(f"  paired evals   : {len(both)} cells reached {THRESH_NAME} in both; "
              f"variant - base = {d.mean():+.1f} (median {np.median(d):+.1f}, "
              f"faster {int((d < 0).sum())} / slower {int((d > 0).sum())} / "
              f"tie {int((d == 0).sum())}, p = {p:.4f})")

    by = defaultdict(list)
    for r in rows:
        by[r["function"]].append(r)
    print(f"\n  {'function':<24}{'SRb':>6}{'SRv':>6}{'won':>5}{'lost':>6}"
          f"{'ev_b':>9}{'ev_v':>9}{'d_ev':>8}{'fire':>7}{'tie_f':>7}")
    print("  " + "-" * 89)
    hurt, helped = [], []
    for f in sorted(by):
        rs = by[f]
        bo = np.array([r["base_f"] <= 1e-10 for r in rs])
        vo = np.array([r["var_f"] <= 1e-10 for r in rs])
        w = int((vo & ~bo).sum())
        l = int((~vo & bo).sum())
        bs = [r["base_ev"] for r in rs if r["base_ev"] >= 0]
        vs = [r["var_ev"] for r in rs if r["var_ev"] >= 0]
        mb = np.mean(bs) if bs else float("nan")
        mv = np.mean(vs) if vs else float("nan")
        pair = [(r["base_ev"], r["var_ev"]) for r in rs
                if r["base_ev"] >= 0 and r["var_ev"] >= 0]
        dev = (np.mean([y - x for x, y in pair]) if pair else float("nan"))
        tf = sum(1 for r in rs if r["same"])
        fr = int(np.median([r["n_fire"] for r in rs]))
        print(f"  {f:<24}{bo.mean():>6.2f}{vo.mean():>6.2f}{w:>5}{l:>6}"
              f"{mb:>9.1f}{mv:>9.1f}{dev:>+8.1f}{fr:>7}{tf:>7}")
        if l > w or (l == w and pair and dev > 0):
            hurt.append(f)
        if w > l:
            helped.append(f)
    print(f"\n  functions where the arm loses cells : {hurt or 'none'}")
    print(f"  functions where the arm gains cells : {helped or 'none'}")

    verdict = "PASS" if (var_ok.mean() >= base_ok.mean()
                         and np.mean(v_s) <= np.mean(b_s)) else "REJECT"
    print(f"\n  pre-registered verdict: {verdict} "
          f"(SR {var_ok.mean():.4f} vs base {base_ok.mean():.4f}; "
          f"evals {np.mean(v_s):.1f} vs {np.mean(b_s):.1f})")


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    rows = _load(paths)
    label = "BBOB-24 dim2, 5000 evals" if "5k" in paths[0] else "BBOB-24 dim2"
    if "20k" in paths[0]:
        label = "BBOB-24 dim2, 20000 evals"
    report(rows, label)
    print(f"\n  reference pins (CLAUDE.md / this environment): "
          f"SR {BASE_SR_PIN:.4f}, evals_succ_mean {BASE_EV_PIN:.1f}")


if __name__ == "__main__":
    main()
