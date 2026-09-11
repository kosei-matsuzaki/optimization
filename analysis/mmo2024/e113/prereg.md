# Entry 113 — pre-registration (queue 2: the breakdown of the 4.5x loss)

Written and committed **before** the run's dumps were read. Entry 113 claimed
queue 2 at 2026-09-11 18:32 UTC, following the 方針 of 2026-09-11 (2)
("次はキュー 2 —— 4.5 倍差の内訳").

## Question

MC-ESO loses to the memoryless multistart null (`Restart-Lander`) by 0.1021 vs
0.5478 in MPR (4.5x) and 0.0638 vs 0.3639 in Score (5.7x) on the GECCO'2024
suite, D=10, 16 problems, 500k evaluations. About 9/10 of the MPR gap is on the
coverage side (@1e-1: 0.1833 vs 0.7437). **Which stage loses it — the number of
hunts, the selection of where they go, or the descent itself?**

## Design

* **Unit.** The null's hunt = one uniform draw + one CMA descent (one row of
  `analysis/mmo2024/e110/descents/*.csv.gz`). MC-ESO has no such object, so the
  matching unit is **the search segment between two spillovers**: it opens with
  a fresh set of `n_pop` re-seed draws and closes when the population stalls and
  is re-seeded. `core/optimizers/mceso_traced.py` records one row per segment
  (`segment, evals, best_f, land_opt, dist, basin_switch`) and one row per
  re-seed draw (`segment, slot, f, land_opt, dist, kind`), with the same column
  names and the same nearest-optimum attribution as entry 110's dump.
* **No default is changed.** `TracedMCESO` overrides two hooks, both calling
  `super()`, and consumes no RNG and no evaluation. Identity against the base
  class is checked at the same seed (`identity_check.py`).
* **Pairing.** 16 problems, seed offset 0 — the same problems, the same budget,
  the same scorer and the same optimiser seed as entry 110's null run, so every
  comparison is paired problem by problem.
* **Coverage.** `detected(ε) = |{distinct land_opt : best_f ≤ ε}|`, exactly the
  quantity entry 112's identity check validated against the scorer's `pr × K`.
  Reported at all five levels; the headline is ε = 1e-1 (the coverage end) with
  the 5-level mean alongside.
* **Matched counts.** The null's descent dump is truncated to its first
  `n_MC` descents (its draws are i.i.d. uniform, so a prefix is a valid
  sub-sample) and re-scored, giving coverage at equal hunt counts.
* **Seeds.** 1 seed. Entry 111 measured the seed-to-seed pairing on this suite
  at −0.0256, 5/6/5, p=0.4236 — seeds carry no systematic signal — but entry 111
  also showed **per-problem values from a single seed move by 0.15-0.30**, so
  per-problem numbers here are stated as one-seed observations, and only the
  16-problem aggregate is claimed.

## Pre-registered branches

Let `n_MC` and `n_null` be the per-problem hunt counts, and `cov_null(n_MC)` the
null's coverage truncated to `n_MC` descents.

* **Branch A — the descent.** `n_MC ≥ n_null / 3` on a majority of the 16
  problems **and** `cov_null(n_MC) < 1.5 × cov_MC`. Then the loss is neither the
  number of hunts nor where they are placed: one MC-ESO hunt simply does not
  reach the bottom of its basin. Write that and pass to queue 3 (the depth
  side), per the queue's own instruction.
* **Branch B — the count.** `n_MC < n_null / 3` on a majority of problems. Then
  MC-ESO spends its budget on too few basins, and the per-hunt comparison is
  secondary. The next step is where the evaluations go instead.
* **Branch C — the placement/selection.** `n_MC ≥ n_null / 3` on a majority but
  `cov_null(n_MC) ≥ 1.5 × cov_MC`. Then at equal hunt counts the null still
  covers more: the difference is where MC-ESO's hunts land, i.e. the draw stage
  (reservoir re-ignition / herd-immunity repulsion) or the selection over the
  `n_pop` draws — entry 22's finding on N07, tested here for the first time on
  a suite where coverage is the binding constraint.

Branches B and C can both hold; if so, both are reported, and the count is
stated first because it bounds the other.

## What would refute the hypothesis behind this question

The standing reading (status.md, "いちばん弱い環 2") is that MC-ESO's per-hunt
search is the problem. **That is refuted if, at matched hunt counts, MC-ESO's
coverage is within 1.5x of the null's** — in that case a single MC-ESO hunt is
as good as a single null descent and the entire 4.5x is hunt count, i.e. an
accounting fact about where the 500k evaluations go, not a search defect.

Symmetrically, the claim "the loss is the number of hunts" is refuted if
`n_MC ≥ n_null` on a majority of problems.

## Scale-down rule (if the 40-minute budget binds)

The number of problems is cut, not the seeds and not the budget — the same rule
entries 100 and 105 used. Problems are dropped from M16 downwards, and the log
says which ran. The null side costs nothing (it is a re-read of stored dumps).
