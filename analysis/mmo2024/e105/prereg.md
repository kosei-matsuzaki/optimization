# e105 pre-registration (written before any number of the new arm was read)

**Question (queue item 2, step (2) -- the sigma side of the sweep).** Entry 104
closed the *draw-count* sweep under its own stopping rule. What queue item 2
still names as open is the `sigma0` sweep, with two unmeasured points: **0.4**
(large side, both groups) and **group A at 0.05**. Entry 98 swept group B
(M09-M16, K=10) over {0.05, 0.1, 0.2, 0.31}; entry 99 added group A (M01-M08,
K=20) at 0.1 only. So group B's sweep maximum is **interior** (0.1 beats both
0.05 and 0.2), while group A's is at the **edge of what has been measured** --
its best point, 0.1, is the smallest sigma ever run on those eight problems.

**This cycle measures group A at sigma = 0.05.** That is the point that brackets
group A's maximum, and it is on the side where coverage is known to rise, i.e.
the only remaining side on which the class ceiling could still move up. The 0.4
point is deliberately *not* the first choice: entry 97 already measured the large
side (popsig, median 0.31 x span) and found the ceiling **falls** on 7 of 8
problems, so a fifth point further out is a confirmation, not a test. It is
covered by the stopping rule below and by the leftover-clock clause.

**Arm.** `hunt_coverage.py --null --sigma-ratio 0.05 --descents 200 --budget
12500`, group A (M01-M08), D=10, PIN=1, `--procs 4` -- the exact configuration of
the saved dumps (entry 91 for M01, entry 92 for M02-M08 at sigma 0.2; entry 99 at
sigma 0.1), so **every draw pairs across all three sigma points**. `sigma0 = 0.05
x span = 0.5`.

**STOPPING RULE for the sigma sweep, declared before any result is read.**
Entry 103's lesson was that a sweep which can always take one more point needs
its retreat condition written on the *count*, not on the outcome. So:

- The swept set is **{0.05, 0.1, 0.2} for all 16 problems (+0.31 for group B)**.
- **At most one further sigma point may ever be measured**, and only if the small
  side is demonstrably still rising for group A, defined as **both**:
  (i) group A mean `mpr`(0.05) - mean `mpr`(0.1) >= **+0.035** (the size of the
  group-B gain that motivated the whole sweep: 0.508 -> 0.542 from 0.2 to 0.1),
  **and** (ii) the paired Wilcoxon signed-rank over the 8 problems has
  **p < 0.05**. If that fires, the one extra point is **sigma = 0.025**.
- **If it does not fire, the sigma sweep is closed here** and the ceiling is
  quoted as "maximised over sigma0 in {0.05, 0.1, 0.2}, 200 draws".
  **sigma = 0.4 is then not measured at all**: its direction is established by
  entry 97 (7/8 down at 0.31) and a point that can only lower the maximum cannot
  change any statement that takes the maximum.
- **Leftover-clock clause.** If group A finishes with more than ~12 minutes of
  the 40-minute budget left, sigma = 0.4 is run on **group B only** (the cheaper
  group) purely to close the bracket on the large side for the group that has the
  fuller sweep. A partial 0.4 arm is reported per problem and is never used to
  raise a maximum.

**Scale-down declared in advance.** Draws (200) and budget (12500) may not be
reduced -- they must match the saved dumps or the pairing is lost. The only
scale-down axis is the **number of problems**, reported per problem, so a partial
group A is still an exact paired measurement on the problems it covers. Problems
run in the order M01..M08.

**Identity check (run before the new arm; entries 97/98/99 require it).** Done.
`--sigma-ratio 0.1 --budget 12500` on M01-D10-PIN01, draws 0-3, reproduced entry
99's saved dump **exactly** on `start_opt` / `land_opt` / `dist` / `best_f` /
`evals` / `ev_1e-05` / `opt_1e-05` (all 4 draws, all 7 columns). The scoring path
and the start-point stream are therefore unchanged from e99.

**Metrics.** The same four ceilings, imported from e92/e93 via e99's `analyze.py`
rather than reimplemented: `mpr`, `mpr_earlystop`, `mpr_sup`, `mpr_sup_chao1`.
**`mpr_sup_chao1` is reported as an instrument only and is not compared with the
published best** (entry 104 §6 settled this: it fell while `S_obs` rose).
Headline: the **16-problem mean** of `mpr` and `mpr_sup` with group B at entry
98's per-problem sweep maximum and group A at this cycle's, against the published
best at D=10 (RR-CMA-ES, MPR 0.651). Per-problem paired diagnostics against
sigma = 0.1: `mpr` difference, count of draws whose landing optimum moved,
McNemar counts on `f <= 1e-5`, distinct optima reached at 1e-5, mean evals.

**Refutation conditions.**
1. *Primary.* If the 16-problem mean of `mpr` or `mpr_sup` **exceeds 0.651** once
   group A's 0.05 point is folded in at the per-problem sweep maximum, route
   (C)'s premise -- that this class's ceiling sits under the published best --
   depends on `sigma0` and breaks. That goes to the review cycle in the log,
   **not** back into MC-ESO (standing instruction, item 3).
2. *Secondary (the bracket).* If group A's `mpr` at 0.05 is **below** its 0.1
   value, group A's maximum is interior exactly as group B's is, and the sweep is
   closed with a bracketed maximum on all 16 problems. If it is **above**, the
   small side is not bracketed and the stopping rule decides whether one more
   point is taken.
3. *A null needs a firing check.* If the arm moves no landings and changes no
   `f <= 1e-5` outcome on a problem, that problem reads "the arm did not fire",
   not "step size does not matter" (entry 86's third form). `landing_moved` and
   the McNemar counts are reported for exactly this.

**Prediction.** Group A's `mpr` at 0.05 comes out **below** its 0.1 value, i.e.
the maximum is interior at 0.1 for group A as it is for group B, so the sweep
closes without the extra point; and the 16-problem means of `mpr` and `mpr_sup`
stay **under 0.651**. Reasons from the record: (a) the mechanism is a depth-for-
coverage trade (`hit@1e-5` fell 0.754 -> 0.673 -> 0.586 across 0.2 -> 0.1 -> 0.05
on group B) and on group B that trade had already turned unprofitable by 0.05
(mean `mpr` 0.542 -> 0.520); (b) group A has K=20 against group B's 10, so each
optimum is worth half as much in `mpr` while the depth loss is charged in full;
(c) the main indicator has been the stable one throughout (entry 104 §7: 200
extra draws moved `mpr` by +0.0021), so a 16-problem `mpr` near 0.55 is not
plausibly pushed past 0.651 by one sigma point.
**What would refute this:** group A mean `mpr`(0.05) at or above `mpr`(0.1) +
0.035 with a significant paired test -- which would also trip the stopping rule
and buy the sigma = 0.025 point.
