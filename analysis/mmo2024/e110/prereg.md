# entry 110 — pre-registration (written before the runs finished)

**Question (queue 1).** 0.5529 is an estimate; no run has ever produced it.
Chain the null's descents into one run, score it on the path MC-ESO and NMMSO
were scored on, and see whether the estimate survives.

## What is run

`Restart-Lander` (`core/optimizers/restart_lander.py`), new this cycle, is the
memoryless multistart null as an optimizer: uniform draw → isotropic CMA descent
(`CMA_on = 0`, `tolfun = tolfunhist = tolx = 0`, per-descent cap 12500) → the
descent's best point joins the reported set → repeat until the suite budget is
spent. Parameters are entry 103's (`sigma0 = 0.1 × span`, cap 12500 = budget/40),
because 0.5529 is entry 103's number.

- suite GECCO'2024, D=10, PIN01, M01–M16, budget 5e5, `--report-rule current`
- driver `scripts/niching_baseline.py`, scorer `core.runner._niching_counts`
  (`count_goptima_nn`, reported set capped at `max(100, 2K)`), MPR = mean over
  the five accuracies 1e-1 … 1e-5 — identical to entries 106 and 109
- 1 seed this cycle (the queue allows splitting the pre-registered 2 seeds over
  two cycles; entry 107's measured 28 min for 16 runs at this budget leaves no
  room for both in one 40-minute frame)
- comparison unit = problem, n = 16, paired against entry 103's per-problem
  estimate and against entries 106/109's per-problem MC-ESO/NMMSO values;
  two-sided Wilcoxon signed-rank at α = 0.05 with rank-biserial as effect size

## Identity check (run first, passed)

`analysis/mmo2024/e110/identity_check.py` forces the optimizer's draws 0–3 to be
`_null_descent`'s draws 0–3 on M09-D10-PIN01 and compares against the saved dump
`analysis/mmo2024/e98/M09-D10-PIN01_sig100200.csv.gz`: **evals, best_f and
landing optimum match exactly on all four** (4190 / 12060 / 9660 / 10760
evaluations; best_f 1.42e-14 / 4.26e-14 / 5.68e-14 / 1.42e-14; landings 8/9/9/9).
So the run and the estimate share the descent, and any difference between them
is one of the three things this entry is about.

## Branches (from the queue, fixed before the numbers)

- **branch 1 — 16-problem mean MPR ≥ 0.50.** The estimate is confirmed by a run.
  A baseline that beats both methods on all 16 problems is in hand.
- **branch 2 — < 0.40.** The estimate was overstated; say where, from the four
  named candidates.
- **branch 3 — 0.40–0.50.** Separate the sources before closing.

**What would refute the hypothesis** ("the null really scores ≈ 0.55 as a run"):
a 16-problem mean below 0.40, or a paired loss against entry 103's per-problem
estimates that is significant with a large negative rank-biserial.

## The four candidates, and how this cycle tells them apart

The per-descent dump (`descents/`) carries landing optimum, best_f, evaluations
and CMA's stop reason for every descent of every run, so all four are separable
from **one** set of runs at zero extra evaluations:

1. **report cap 100.** The estimate credits every restart; the scorer keeps only
   the 100 best-by-f reported points. Measured by scoring the run's own reported
   set twice, capped and uncapped.
2. **descent-cost averaging.** The estimate buys `n' = budget / mean cost`
   restarts. The run's actual descent count is recorded; compare.
3. **i.i.d. approximation.** The estimate draws `n'` landings independently from
   a 500-descent empirical distribution. Recomputing entry 103's multinomial
   formula on *this run's own* landings isolates the formula from the sample.
4. **early `es.stop()`.** With all three tolerances at 0 it is not obvious what
   ends a descent. Stop reasons are counted per problem (the identity check
   already shows `tolflatfitness` on M09).
