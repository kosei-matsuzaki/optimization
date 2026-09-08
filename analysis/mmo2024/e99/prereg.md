# e99 pre-registration (written before any number of the new arm was read)

**Question (queue item 2, as entry 98 rewrote it).** Entry 98 swept `sigma0` on
group B (M09-M16, K=10) and found the class ceiling rises on the small side:
`mpr` mean 0.508 -> 0.542 at sigma = 0.1, and the margin of the most permissive
ceiling in the only comparable unit (the 16-problem mean, since per-problem
published values do not exist -- entry 92 §6) shrank from 0.070 to 0.017.
That number still carries group A (M01-M08, K=20) **unswept**, at entry 92/91's
isotropic values. Entry 98 wrote the extrapolation explicitly: if group A rose by
the same +0.106 per problem, the 16-problem mean would be 0.687 and clear the
published best 0.651. **This cycle measures group A instead of extrapolating it.**

**Arm.** `hunt_coverage.py --null --sigma-ratio 0.1 --descents 200 --budget
12500`, group A, D=10, PIN=1 -- the exact configuration of the saved isotropic
dumps (entry 92 for M02-M08, entry 91 for M01), so every draw pairs. `sigma0 =
0.1 x span = 1.0` against the isotropic `0.2 x span = 2.0`.

**Why sigma = 0.1 and only that point.** Entry 98's own reading: the sweep
maximum over four points is the maximum of four estimates and is biased upward by
winner's curse (visibly so for Chao1). The unbiased reading is "pick one `sigma0`
for the whole suite", and on group B the best such point is 0.1 (mean `mpr` 0.542
vs 0.520 at 0.05, 0.508 at 0.2, 0.442 at popsig). Group A at K=20 costs ~3x group
B per problem, so one arm on all eight problems is 45-50 min -- outside one
cycle. Queue item 2 therefore splits it M01-M04 / M05-M08. **This cycle runs
M01-M04 and continues into M05-M08 only if the clock allows; whatever is
finished is reported and the rest is named as the next step.**

**Scale-down declared in advance.** No reduction in draws (200) or budget
(12500): those must match the saved dumps or the pairing is lost. The scale-down
axis is the number of problems, and it is reported per problem, so a partial
group A is still an exact paired measurement on the problems it covers.

**Identity check (run before the new arm, as entries 97/98 require).** Four draws
of M01-D10-PIN01 at `--sigma-ratio 0.2 --budget 12500` reproduced entry 91's
saved dump exactly on `land_opt`, `best_f`, `evals` (draws 0-3). M01's dump
predates entry 92 and has no `ev_<eps>` / `opt_<eps>` columns, so its
`mpr_earlystop` is nan -- unknown, not below; the other three ceilings are
unaffected.

**Metrics.** The same four ceilings, imported from entries 92/93 rather than
reimplemented: `mpr`, `mpr_earlystop`, `mpr_sup`, `mpr_sup_chao1`. Per-arm paired
diagnostics against the isotropic baseline: bootstrap CI on the paired `mpr`
difference (the same resampled draw indices score both arms), count of draws
whose landing optimum moved, McNemar counts on `f <= 1e-5`, and the number of
distinct optima reached at 1e-5. Headline: the **16-problem mean** of each
ceiling with group B at entry 98's sigma = 0.1 values and group A at this
cycle's, against the published best at D=10 (RR-CMA-ES, MPR 0.651).

**Refutation conditions.**
1. *Primary (queue item 2's own).* If the 16-problem mean of **any** of the four
   ceilings exceeds 0.651 once group A is swept, route (C)'s premise -- that the
   class ceiling of "uniform restart + local descent" sits under the published
   best -- depends on the choice of `sigma0` and breaks. The finding goes to the
   review cycle in the log, **not** back into MC-ESO (standing instruction, item
   3). Partial coverage counts here only if the mean already clears 0.651 with
   the unmeasured problems held at their isotropic values.
2. *Secondary.* If group A's mean per-problem gain is within noise of group B's
   +0.106, entry 98's extrapolation was sound and the margin is a matter of how
   many problems are swept, which has to be said whenever "class ceiling" is
   used. If group A's gain is materially smaller (or negative), the extrapolation
   was not sound and the margin stands wider than 0.017.
3. *A null needs a firing check.* If the arm changes no landings and no
   `f <= 1e-5` outcomes on a problem, that problem's result is "the arm did not
   fire", not "the step size does not matter" (entry 86's third form).
   `landing_moved` and the McNemar counts are reported for exactly this.

**Prediction.** Group A gains **less** than group B did, and the 16-problem mean
stays under 0.651 on all four ceilings. Two reasons, both from the record.
(a) The mechanism entry 98 measured is that a smaller `sigma0` trades depth for
coverage: `hit@1e-5` fell 0.754 -> 0.673 while distinct optima rose 41 -> 42 of
80. Group A has K=20 per problem against group B's 10, so its isotropic
per-problem coverage share is already spread thinner and each newly reached
optimum is worth half as much in `mpr`. (b) Group B's +0.106 is dominated by two
problems (M10 +0.155, M12 +0.039 are the only CIs excluding zero); a mean driven
by 2 of 8 cells is not a rate that transfers. Concretely predicted: group A mean
`mpr` rises by less than +0.05 per problem, `hit@1e-5` falls, and the
`mpr_sup_chao1` 16-problem mean lands between 0.634 (group A unchanged) and
0.651. **What would refute this**: a group A gain at or above +0.106 per problem,
which puts the Chao1 mean at or above 0.687.
