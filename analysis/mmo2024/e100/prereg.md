# e100 pre-registration (written before any number of the 400-draw arm was read)

Written while the arm was already computing, but before any of its output was
opened. The identity checks below are the only numbers seen at writing time, and
they concern the wiring, not the result.

**Question (queue item 2, as entry 99 rewrote it).** Entry 99 swept group A at
`sigma0 = 0.1 x span` and found the 16-problem mean of the most permissive
ceiling, `mpr_sup_chao1`, at **0.6549** against the published best at D=10
(RR-CMA-ES, MPR **0.651**) -- an excess of **+0.0039**, the first time the
pre-registered refutation condition of route (C)'s premise fired on real
coverage rather than on an estimator artifact (84% of the rise was optima
actually landed on). But `mpr_sup` is `S_obs / K`, a **richness** statistic, and
richness is monotone non-decreasing in the number of draws. Entry 94 already
recorded that the bare support set has **not converged at 200 draws**. So 0.6549
may be "the coverage of 200 draws", not "the ceiling of the class" -- and if so,
the phrase *class ceiling* is not defined without naming a draw count.

**The measurement.** Extend every `sigma0 = 0.1` dump of the new suite from 200
to **400 draws**, all 16 problems, D=10, PIN=1, budget 12500 (= suite budget
5e5 / 40) -- the configuration of the saved dumps, unchanged. Re-derive the four
ceilings (`mpr`, `mpr_earlystop`, `mpr_sup`, `mpr_sup_chao1`) as 16-problem
means at n = 400 and compare to the same means at n = 200.

**Cost, and how both groups fit one cycle.** Queue item 2 budgeted this as one
group per cycle (~35-41 min for 400 draws) and told the runner to split it A / B.
That estimate assumes 400 draws must be *recomputed*. They need not: draw k is a
closed form of k alone (start point `default_rng(1_000_000 + k)`, CMA seed
`k + 1`), so draws 200-399 can be computed alone and concatenated onto the saved
dump. This cycle added `--descent-start` to `hunt_coverage.py` for exactly that.
Going 200 -> 400 therefore costs the 200 *new* draws (~21 min group A + ~17 min
group B = ~38 min, from entry 99 §8's measured rates), and **both groups fit**.
This matters for the question itself, not just for the clock: `S_obs` is monotone
in n, so a 16-problem mean mixing group A at 400 with group B at 200 would rise
for a bookkeeping reason and could not be compared with the 200-draw mean at all.

**Identity checks (run before the arm, as entries 97/98/99 require).**
Both passed, against entry 99's saved `M01-D10-PIN01_sig100200.csv.gz`:
1. `--descent-start 0 --descents 4` reproduces draws 0-3 exactly on `start_opt`,
   `land_opt`, `best_f`, `evals`.
2. `--descent-start 2 --descents 2` reproduces draws 2-3 exactly -- i.e. the
   offset is a pure index shift, not a re-seeding.
A third check runs after the arm: the first 200 rows of each concatenated
400-draw dump must be byte-equal to the saved 200-draw dump, and the four
ceilings recomputed on that prefix must reproduce entry 99's table.

**Scale-down declared in advance.** No reduction in draws per problem (the point
of the cycle is the draw count) or in budget (12500, or the pairing with the
saved dumps is lost). The scale-down axis is **problems**: they run group B first
then group A, and a problem that does not finish is **held at its 200-draw
value** in the 16-problem mean. That is the conservative direction for the
refutation condition below -- holding a problem at 200 can only make the
400-draw mean look *less* like it grew.

**Free extra, from the same dump: the rarefaction curve.** With 400 draws in
hand, each ceiling can be recomputed on the first n draws for
n = 50, 100, 200, 300, 400 at no further compute. Two points can only say
"rose / fell"; the curve says whether `S_obs` and the Chao1 correction have
*converged*, which is the actual question behind "is this a ceiling or an
observation". Reported per ceiling as a 16-problem mean.

**What each ceiling should do if nothing is wrong.** `mpr` and `mpr_earlystop`
are built from *proportions* (`p = counts / n_draws`) and a restart count
`n = budget / mean(evals)`; both converge, so these two should be **flat** in n
up to noise. `mpr_sup = S_obs / K` is richness and can only **rise**. Chao1 is
built to estimate the asymptotic support, so it should be **flat** if it is
working.

**Refutation conditions.**
1. *Primary (queue item 2's own).* If the 400-draw 16-problem `mpr_sup_chao1`
   mean's excess over 0.651 **widens** beyond +0.0039, the excess is driven by
   draw count: "class ceiling" is then undefined without an n, and the proposal
   to restate it as "coverage at N draws" goes **to the review cycle in the log**
   (standing instruction, item 3), not back into MC-ESO. If it **shrinks below
   0.651**, entry 99's excess was a small-sample artifact of the estimator and
   route (C)'s premise stands on all four readings.
2. *Secondary, and the sharper one.* The two outcomes above are not the only
   ones, and queue item 2's phrasing hides a third: Chao1 is **designed** to be
   n-stable. If `mpr_sup_chao1` is flat from 200 to 400 while `mpr_sup` rises
   toward it, that is neither "draw-count artifact" nor "estimator artifact" --
   it is the estimator working, and 0.6549 is a genuine estimate of the
   asymptotic support ceiling. This must be reported as its own outcome and not
   forced into the binary.
3. *A null needs a firing check.* If the 200 new draws add no optima on a
   problem, that is "the support is saturated on this problem", not "the arm did
   not fire" -- the firing evidence is the count of newly reached optima and the
   number of draws landing on an optimum not seen in the first 200. Reported per
   problem for exactly this.

**Prediction.** `mpr_sup` rises and `mpr_sup_chao1` rises **less**, staying in
0.645-0.665; `mpr` and `mpr_earlystop` move by less than 0.01 in either
direction. Reasons from the record: entry 94 measured the bare support still
climbing at 200 draws, so `S_obs` has room; but entry 99 §6 found the Chao1
correction term already carrying only 0.0261 of the 0.6549, and f2 = 0 blow-ups
confined to M05, M08 and one level of M12, which means the correction is
computed under f2 >= 1 on most problems and should behave. Concretely: the
excess over 0.651 stays positive and **does not shrink to zero**, so route (C)'s
premise remains broken on the most permissive reading only, exactly as entry 99
left it. **What would refute this**: `mpr_sup_chao1` at 400 draws below 0.651, or
above 0.675.

**What this cycle will not do.** It will not re-tune anything, will not touch
MC-ESO defaults, and will not reorder the queue. Whichever way the result falls,
it goes into the log and `acceptance_topology.md`, and the review cycle decides.
