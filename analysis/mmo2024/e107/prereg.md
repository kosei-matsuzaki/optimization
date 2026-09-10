# Entry 107 — pre-registration (written before any full-budget result was read)

**Question (queue item 1)**: MC-ESO loses 0.56 to the memoryless null on the
GECCO'2024 suite (entry 106). Is the loss in the **reporting rule** (the search
touches the basins, the reported set throws them away) or in the **search** (it
never gets there)? The queue asks for zero extra evaluations.

## What is measured

- Problems: `M01..M16-D10-PIN01`. K = 20 (M01-M08) / 10 (M09-M16).
- Budget: `--evals-frac 1.0` = 500,000 evaluations, the suite's own.
- Method: MC-ESO, defaults untouched.
- Seeds: **1** (seed index 0), scaled down from entry 106's 3. Entry 106 took
  66 min of wall clock for 48 runs 4-wide; 16 runs is ~22 min and fits the
  40-minute window, 32 would not. The comparison this entry exists for is
  **within a run** — the same evaluations scored under different rules — so it
  is paired at the run level and the seed count only affects how well each
  problem's level is pinned down, not the contrast. Seed 0 is one of entry
  106's three, so the `current` rows must reproduce its seed-0 values; that is
  the wiring check.
- Metric: MPR by nearest optimum (entry 106's `count_goptima_nn`), at
  `NICHE_ACCURACIES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)`.

## The three scorings (same runs, no extra evaluations)

1. **`current`** — the method's own reported set, capped at max(100, 2K) = 100.
   This is entry 106's number.
2. **`reselect`** — the rho-greedy pick from the run's own evaluation history,
   same cap of 100 (entry 28's rule, `scripts/diagnose_niching.reselect_from_history`).
   The new suite ships no rho, so it is taken as **half the smallest distance
   between two global optima** — the largest radius that cannot merge two
   distinct optima, i.e. the choice most favourable to coverage. That uses the
   optima positions, so on this suite `reselect` is a **diagnostic ceiling for a
   capped reporting rule**, not an output rule a method could adopt blind.
3. **`history`** — the **entire** evaluation history, uncapped. Not a legal
   answer, and not meant as one: every reporting rule reports a subset of the
   points the run evaluated, so this is the **supremum over all of them**. It is
   rho-free, which is what makes the verdict independent of the choice in 2.

## What would refute what

The queue's rejection conditions are copied unchanged, read on the
**16-problem mean at 1e-1** (the level entry 106 showed the loss already exists
at: MC-ESO 0.1833 vs null `S_obs/K` 0.7437):

1. **`reselect` mean@1e-1 does not exceed 0.30** → the reporting rule is not the
   bottleneck; the loss is in the search, and no cycle is spent tuning the
   reporting rule.
2. **`reselect` mean@1e-1 exceeds 0.5** → entry 106's 0.1021 was measuring the
   reporting rule rather than the search, and entry 106's section plus
   status.md's wording have to be corrected (status.md is the review role's, so
   it is raised, not edited).
3. Between 0.30 and 0.5 → the split is partial; report the size of each part
   and let the review role decide where the mechanism goes.

**The rho-free strengthening**: `history` bounds `reselect` from above by
construction. If `history` mean@1e-1 is itself under 0.30, condition 1 fires
**for every possible rho and every possible selection rule**, and the "maybe the
rho was wrong" objection cannot be raised. If `history` is high while `reselect`
is low, the loss is specifically in the *cap of 100 / the selection*, which is a
third answer neither the queue nor entry 106 anticipated, and it is reported as
such.

Per-problem values are listed in full (entry 28's rule: no single-number
verdicts), and the three rules are compared with a two-sided Wilcoxon signed
rank over the 16 problems with rank-biserial correlation (entry 83's rule).

## Cost

Entry 106: 48 runs, 4-wide, 66 min. This is 16 runs, 4-wide, so ~22 min, plus
scoring. Scoring is the added part and it is cheap: a 25,000-evaluation probe
(M01, M09) put all three rules at ~1 s on top of the run. `history` scores the
stored `history_f` rather than re-calling the objective — recomputing it would
be another full budget of evaluations per run.
