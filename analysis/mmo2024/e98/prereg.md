# e98 pre-registration (written before any number of the new arms was read)

**Question (queue item 2, as entry 97 rewrote it).** Entry 97 established that the
class ceiling of entries 88/92/93/94 *does* move with `sigma0`, so those values are
"the ceiling at `sigma0 = 0.2 x span`", not the ceiling of the class. Question 2's
remaining step is therefore to sweep `sigma0` and take the per-problem maximum over
the sweep, so the sentence "route (C)'s class ceiling stays under the published
best" can be stated without a `sigma0` qualifier.

**Sweep points and the scale-down.** Question 2 names four ratios
(0.05 / 0.1 / 0.2 / 0.4) and says explicitly that four points do not fit one cycle
("1 サイクルに入るのは 8 問 × 1-2 腕まで。掃引 4 点は 2 サイクルに割ること").
This cycle runs the two **new small** points, **0.05 and 0.1**, on group B
(M09-M16), D=10, PIN=1, 200 draws each, descent cap 12500 = suite budget / 40 --
the exact configuration of the saved dumps, so every arm pairs draw-for-draw.

- `0.2` is not re-run: entries 94 (M09-M12, M14-M16) and 91 (M13) are that arm.
- `popsig` (sigma0 = 3.108 median, i.e. ratio ~0.31) is entry 97's saved dump and
  enters the per-problem maximum as a fifth point at no cost.
- **`0.4` is deferred to the next cycle, and the small side is taken first on
  purpose.** Entry 97 measured the large side: raising sigma0 from 2.0 to 3.108
  lowered `mpr` on 7 of 8 problems and cost 4 distinct optima net. Entry 95
  measured the same direction in 2D and found the coverage gain came from sigma0
  going *below* `0.2 x span`. So the ceiling, if it rises anywhere in this sweep,
  rises on the small side; 0.4 is on the side already measured as falling. Cost
  also points the same way (small sigma0 descends into the nearest basin and
  stops; entry 97's large-sigma0 arm took 2x the isotropic arm's wall clock).

**Arms.** `hunt_coverage.py --null --sigma-ratio {0.05, 0.1} --descents 200
--budget 12500`, group B, one CSV per problem per ratio. Nothing else changes:
`--sigma-ratio` feeds `sigma0 = ratio x span` and leaves `_null_descent`'s start
point stream (`default_rng(1_000_000 + k)`) and CMA seed (`k + 1`) alone, so draw
k of every arm begins at the same x0 with the same seed.

**Identity check (run before the new arms, as entry 97 requires).** Four draws of
M09-D10-PIN01 at `--sigma-ratio 0.2 --budget 12500` reproduced e94's saved dump
exactly on `land_opt`, `best_f`, `evals`, `ev_1e-05`, `opt_1e-05` (draws 0-3).

**Metrics.** The same four ceilings, imported from entries 92/93 rather than
reimplemented: `mpr`, `mpr_earlystop`, `mpr_sup`, `mpr_sup_chao1`. The headline
statistic of this cycle is the **per-problem maximum over the sweep points**
(0.05, 0.1, 0.2, popsig) for each ceiling, compared against the published best at
D=10 (RR-CMA-ES, MPR 0.651). Paired diagnostics per arm against the 0.2 baseline:
bootstrap CI on the paired `mpr` difference (same resampled draw indices score both
arms), count of draws whose landing optimum moved, McNemar counts on `f <= 1e-5`,
and the set of optima reached at 1e-5.

**Refutation conditions.**
1. *Primary (question 2's own).* If, on any one problem, any of the four ceilings
   at any swept `sigma0` exceeds 0.651 **newly** -- i.e. excluding M13, which entry
   91 already recorded as the one D=10 problem sitting above the published mean in
   the isotropic arm -- then route (C)'s premise depends on the choice of `sigma0`
   and breaks. The finding goes to the review cycle in the log, **not** back into
   MC-ESO (standing instruction, item 3).
2. *Secondary.* If a small-sigma0 arm's paired `mpr` difference against 0.2 is
   positive with a bootstrap CI excluding zero on any problem, then the ceiling
   rises on the small side even where no published value is cleared, and the sweep
   has to be extended (finer points below 0.05) before the word "class ceiling" is
   used unqualified.
3. *A null needs a firing check.* If an arm changes no landings and no
   `f <= 1e-5` outcomes, the result is "the arm did not fire", not "the step size
   does not matter" (entry 86's third form). `landing_moved` and the McNemar counts
   are reported for exactly this.

**Prediction.** Small sigma0 buys coverage and loses depth -- the mirror image of
entry 97. Concretely: `hit@1e-5` falls (a short step cannot cross the composite's
inner structure inside 12500 evaluations), the number of *distinct* optima reached
rises on some problems, and `mpr` -- which multiplies the two -- moves less than
either. The per-problem maximum over the sweep is predicted to stay under 0.651 on
all seven non-M13 problems, i.e. no refutation. The mechanism, if this is right, is
the one entries 89/95/97 have now hit three times: the knob moves depth, the metric
is coverage, and the two do not share a sign.
