# e97 pre-registration (written before any number of the new arm was read)

**Question (queue item 2, as entry 95 rewrote it).** Re-draw the class ceiling of
entries 92-94 on the GECCO'2024/'2025 suite with the arm whose descent step size
comes from the population's own spread instead of `0.2 x span`. Entry 95 found the
only coverage gain on the CF3 family came from that step size, not from the
hill-valley test, and that the gain fell to zero by D=5. The new suite is D >= 5,
so the ceiling should not move -- which would make the entry-88/92/93/94 ceilings
independent of `sigma0` and strengthen route (C)'s premise.

**Scope, and the scale-down.** Group B (M09-M16), D=10, PIN=1, 8 problems, 200
draws each, descent cap 12500 = suite budget / 40 -- the exact configuration of
the saved isotropic dumps, so the arms pair. Question 2 itself specifies "group B,
8 problems, 2 arms" first; the isotropic arm is not re-run because entry 94's
dumps already are it (identity-checked, below). Group A (K=20) and D=20 are not
in this cycle: entry 92 measured group A at 200 draws as 33 minutes on its own.

**Arms.**
- `iso` -- uniform draw, isotropic CMA descent, `sigma0 = 0.2 x span = 2.0`.
  Reused from `e94/` (M09-M12, M14-M16) and `e91/` (M13).
- `popsig` -- same start point, same CMA seed, `sigma0 = 0.5 x` the
  nearest-neighbour distance inside a 200-draw population. This is exactly the
  singleton branch of entry 95's `_cluster_sigma`, i.e. its `split` arm, written
  into the per-draw dump form the ceiling estimators read (`--pop-sigma`).
  Evaluating the population costs 200 evaluations per block of 200 descents and
  is charged as +1 evaluation per draw, so the arms stay budget-matched.

**Pairing is exact, not statistical.** `--pop-sigma` leaves `_null_descent`'s
start-point stream and CMA seed alone, so draw k of both arms begins at the same
x0 with the same seed and only `sigma0` differs.

**Identity check (run before the new arm).** Four draws of M09-D10-PIN01 at
`--sigma-ratio 0.2`, `--budget 12500` reproduced e94's saved dump exactly
(`land_opt`, `best_f`, `evals`, `ev_1e-05`, `opt_1e-05` all equal on draws 0-3).

**Metrics.** The four ceilings of entries 92/93/94, imported from those scripts:
`mpr`, `mpr_earlystop`, `mpr_sup`, `mpr_sup_chao1`. Plus per-draw paired
diagnostics that say whether the arm fired at all: how many draws changed their
landing optimum, and a McNemar count on "reached f <= 1e-5".

**Refutation conditions, in the order question 2 states them.**
1. *Primary (the question's own).* If any of the four ceilings for `popsig`
   exceeds the published best at D=10 (RR-CMA-ES, MPR 0.651) on any one problem,
   route (C)'s premise breaks at D=10. The finding then goes to the review cycle
   in the log -- **not** back into MC-ESO (the standing instruction, item 3).
2. *Secondary.* If the paired `popsig - iso` difference in `mpr` is positive with
   a bootstrap CI excluding zero on any problem, then "the ceiling does not depend
   on `sigma0`" is false at D=10 even where no published value is cleared, and
   entries 88/92/93/94 have to be re-labelled as ceilings at one `sigma0`.
3. *A null needs a firing check.* If the arm changes no landings and no
   `f <= 1e-5` outcomes, the result is "the arm did not fire", not "the step size
   does not matter" -- entry 86's third form. `landing_moved` and the McNemar
   counts are reported for exactly this.

**Prediction.** No movement. In 10 dimensions the nearest-neighbour distance of
200 uniform points in `[-5,5]^10` is of order 8-9, so `popsig`'s `sigma0` is
about 4 -- *larger* than the isotropic 2.0, not smaller. The 2D gain came from
`sigma0` shrinking below `0.2 x span`; the mechanism reverses with dimension,
which is the same reason entry 95 saw the gain vanish by D=5.
