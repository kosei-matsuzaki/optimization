# entry 111 — pre-registration (written before the runs finished)

**Question (queue 1).** Entry 110 ran the memoryless multistart null
(`Restart-Lander`) on 16 problems with **one** seed and closed the body of the
question on branch 1 (16-problem mean MPR 0.5606). The queue left the
**pre-registered second seed** open, for one stated reason: the per-problem
values (M01 0.72 … M15 0.30) are what a paper table would carry, and entry 109
showed on NMMSO that a single-seed per-problem value cannot be quoted alone.

## What is run

Identical to entry 110 in every respect except the seed index:
`analysis/mmo2024/e111/run.sh 1 1` — seed index 1 (optimiser seed 100), against
entry 110's seed index 0 (optimiser seed 0).

- suite GECCO'2024, D=10, PIN01, M01–M16, budget 5e5, `--report-rule current`
- driver `scripts/niching_baseline.py`, scorer `core.runner._niching_counts`
  (`count_goptima_nn`, reported set capped at `max(100, 2K)`), MPR = mean over
  the five accuracies 1e-1 … 1e-5 — identical to entries 106, 109 and 110
- comparison unit = problem, n = 16; two-sided Wilcoxon signed-rank at α = 0.05
  with rank-biserial as effect size

## Branches (from the queue, fixed before the numbers)

- **branch A — seed 1's 16-problem mean MPR ≥ 0.50.** Entry 110's branch 1 was
  not a one-seed fluke. The question is settled and may be deleted; the
  per-problem table becomes a 2-seed mean and the log states its spread.
- **branch B — seed 1's 16-problem mean MPR < 0.50.** Entry 110's branch 1 was a
  one-seed hit. Say so and return to entry 110's branch-3 procedure (separating
  the four candidate sources).

**What would refute the hypothesis** ("the null really scores ≈ 0.55 as a run,
and entry 110's per-problem values are readable"): a 16-problem mean below 0.50
on the second seed, or a paired seed0-vs-seed1 difference that is significant
with a large rank-biserial (which would mean the seed, not the problem, carries
the signal).

## The second thing this cycle gets for free

Entry 110 bootstrapped **over a run's own descents** (cap re-applied each
resample) and said in its own code comment that this "is not a substitute for a
second seed — it holds the descent count fixed". With the second seed in hand
that claim becomes testable at zero extra evaluations:

- **coverage**: how many of the 16 problems have seed 1's MPR inside seed 0's
  95% descent-bootstrap interval. If the interval is an honest stand-in for
  seed-to-seed spread, ≈ 15/16 should fall inside; materially fewer means the
  descent bootstrap **understates** run-to-run spread and must not be quoted as
  a per-problem uncertainty.
- **width**: mean bootstrap interval width against the mean |seed0 − seed1| gap.

This is pre-registered as a *description*, not a test: n = 2 seeds cannot
estimate a variance, so the coverage count is reported with that caveat and no
branch hangs on it.
