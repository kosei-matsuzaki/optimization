# e101 pre-registration (written while the arm computes, before any of its output was opened)

**Question (queue item 2, step (1), verbatim).** Entry 100 extended the
`sigma0 = 0.1 x span` dumps of the new suite (Ahrari MMO 2024, D=10, PIN=1,
budget 12500) from 200 to 400 draws, but hit its 40-minute budget after **10 of
16 problems** (group B's eight, plus M01 and M02). The remaining six — **M03,
M04, M05, M06, M07, M08**, all group A (K=20) — were held at their 200-draw
value inside the 16-problem mean. Entry 100 therefore reported
`mpr_sup_chao1` = **0.7014** at n = 400 (excess over the published best 0.651 of
**+0.0504**) and stated explicitly that this is a **lower bound**, because a
held problem contributes no growth. This cycle finishes those six problems and
fixes the 16-problem mean at a single draw count.

**The measurement.** Re-run `analysis/mmo2024/e100/run.sh` unchanged. It skips
every problem whose dump already exists and computes draws 200-399 for M03-M08
with `hunt_coverage.py --null --geo-draws 20000 --descents 200
--descent-start 200 --budget 12500 --sigma-ratio 0.1 --procs 4`. Score with
`analysis/mmo2024/e100/analyze.py` unchanged. **No new script, no new arm, no
parameter changed** — the point of the cycle is that the 16-problem mean stops
mixing two draw counts. Optimization runs: **zero**.

**Why the dumps land in `e100/` and not `e101/`.** They are the same 400-draw
set as entry 100's, produced by the same command with the same seeds, and
`analyze.py` finds them by that path. Splitting one set of 16 dumps across two
directories to satisfy a naming convention would break the scorer and make the
record harder to read, not easier. `e101/` carries this pre-registration and the
finalized ceiling tables.

**Refutation condition (queue item 2 wrote it; restated here unchanged).**
With all six problems added, if the 16-problem mean `mpr_sup_chao1` falls
**below 0.651**, entry 100's conclusion (the excess over the published best is
driven by draw count) was an artifact of averaging over the ten problems that
happened to finish, and this cycle records that correction. If it stays
**above** 0.651, entry 100's reading stands on the complete set.

**Secondary outcomes that must be reported separately and not forced into that
binary** (carried over from entry 100's prereg §Refutation 2-3):

1. If `mpr_sup_chao1` is **flat** from the 10-problem to the 16-problem set
   while `mpr_sup` rises, that is the estimator behaving, not an artifact.
2. **A null needs a firing check.** If the 200 new draws on a problem add no
   optima, that is saturation of that problem's support, not a dead arm. The
   evidence is the per-problem count of optima at `f <= 1e-5` at 200 vs 400 and
   the number of draws landing on an optimum unseen in the first 200
   (`analyze.py` prints both).
3. The **paired unit** (problems holding both 200 and 400 draws) is the
   like-for-like average and grows from 10 to 16 problems this cycle; it is the
   number to compare against entry 100's paired table, not the 16-problem mean,
   which entry 100 computed with six problems held at 200.

**Prediction.** The six added problems are group A (K=20), the same group whose
two finished members (M01, M02) each bought 2 new optima in draws 200-399. So
`mpr_sup` should rise a little further and `mpr_sup_chao1` should **stay above
0.651**; the excess should land in **+0.03 to +0.07**, i.e. near entry 100's
+0.0504 rather than collapsing. **What would refute this**: a 16-problem mean
below 0.651, or above +0.10 excess.
Entry 100's mechanism (§7) says the Chao1 correction jumps where a problem has
`f1 >= 2` singletons and `f2 = 0`; six more problems entering that branch could
push the mean up sharply, so the upper refutation line is not decorative.

**Scale-down declared in advance.** If the clock runs out, problems finish in
the order M03..M08 and any unfinished problem is again **held at its 200-draw
value**, with the count of extended problems reported alongside every mean. No
reduction in draws per problem or in budget: either would break the pairing with
the saved dumps, which is the whole measurement.

**What this cycle will not do.** No MC-ESO default is touched, no arm is tuned,
the queue is not reordered, and `docs/status.md` is not edited. The proposal
that queue item 2 step (2) asks for — restating "class ceiling" as "coverage at
N draws" — goes into the log for the review cycle, as the standing instruction
requires.
