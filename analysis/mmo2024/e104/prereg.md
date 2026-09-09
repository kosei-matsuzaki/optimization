# e104 pre-registration (written before any draw past 500 was computed or opened)

**Question (queue item 2, step (2), verbatim).** Entry 103 §4 found the 100-draw
window too fine to test: of 240 optima across the 16 problems, the 100 new draws
bought **5**, and **8 of 16 problems were tied in both compared windows**, which
by itself nearly kills a signed-rank test. The queue's own next step, written by
entry 103 and left as the top open question, is therefore:

> 窓を 200 draw に広げて対比較をやり直す。始点: `analysis/mmo2024/e103/run.sh 500`
> を 16 問に回す。そのうえで **200→400 の増分と 400→600 の増分を問題単位で対にする**。

This cycle runs draws 500-599 on all 16 problems, giving n=600, and compares the
**200→400** increment against the **400→600** increment, **paired per problem**.

**Refutation condition (pre-registered by the queue itself; restated verbatim in
substance, with the numbers fixed to what the record already states).** The
16-problem mean of `S_obs/K` was **0.628750 at n=200** and **0.681250 at n=400**,
so the reference increment is **Δ(200→400) = +0.0525**.

- **No deceleration** if Δ(400→600) of the 16-problem mean `S_obs/K` is
  **≥ +0.0526**, i.e. the 200-draw window buys as much as the previous one.
- **Deceleration** if it is **≤ +0.0263**, i.e. at most half.
- **Between the two: neither branch fires, and this is not a "measure one more
  point" outcome.** The queue pre-committed the stopping rule and this cycle
  honours it: *"あいだの不感帯なら「本数の掃引では決着しない」と確定させて掃引を
  打ち切り、俯瞰へ上げる"*. A dead band here **closes the draw-count sweep** and
  the finding escalates to the review cycle. **The retreat condition is on the
  draw count, not on the result** (entry 103's lesson: a count sweep can always
  afford one more point, so the stopping rule must not be readable off the
  answer).

**Paired test, decided in advance.** Per problem, `S_obs/K` at n=200, 400, 600;
increments `a_i = S_i(400) − S_i(200)` and `b_i = S_i(600) − S_i(400)`; Wilcoxon
signed-rank on `(a_i − b_i)`, α = 0.05, two-sided, with the win/tie/loss counts
and rank-biserial effect size reported alongside. **The mean and the test are
reported together and neither alone decides**: entry 103 showed a mean drop of
0.0263 → 0.0194 that the paired test could not distinguish from noise, and the
mean is the pre-registered branch quantity while the test says whether the
branch value is systematic. **Ties are reported as counts, not dropped
silently** — the tie count is the diagnostic that killed entry 103's test, and
doubling the window is precisely the intervention meant to reduce it. If ties
still dominate (≥ 8 of 16), that is itself reportable as "the sweep cannot be
tested at this resolution" and supports the dead-band stopping rule.

**The measurement.** `analysis/mmo2024/e103/run.sh 500` — i.e.
`scripts/hunt_coverage.py --null --geo-draws 20000 --descents 100
--descent-start 500 --budget 12500 --sigma-ratio 0.1 --procs 4` on all 16
problems (M01-M16, D=10, PIN01). **No new arm, no MC-ESO default touched, zero
optimization runs**, only the draw window moves. Scoring reuses
`analysis/mmo2024/e103/analyze.py`, which already declares the `draws500to599`
chunk and imports the four ceiling estimators from entries 92/93 — one scoring
code path for the whole theme. The paired test is a small addition
(`analysis/mmo2024/e104/paired.py`) reading the same concatenated dumps.

**Identity check (must pass or the cycle reports nothing).** The concatenated
dump's first 200 / 400 / 500 draws must reproduce the ceiling tables already in
the record — 0.5393/0.5864/0.6287/0.6549 at n=200, 0.5509/0.6093/0.6813/0.7276
at n=400, 0.5529/0.6133/0.7006/0.7357 at n=500 — and the draw indices must be
0..n−1 with no gap and no repeat on all 16 problems. `analyze.py` asserts both.

**Cost and the scale-down rule.** Entry 103 measured the 100-draw chunk over all
16 problems at **21.8 min**; this chunk is the same size, so ~22 min, inside the
40-minute budget. **If the clock runs out mid-chunk, the 600-draw mean is not
reported** — the finished problems are listed per problem and the mean stays at
500, the same whole-set-or-nothing rule entry 101 §1 established to stop the
16-problem mean from mixing two draw counts.

**Prediction.** Under the null of a smooth rarefaction curve, Δ(400→600) is
roughly the sum of the two 100-draw windows around it. The last measured window
was +0.0194 and the curve is flattening, so the prediction is
**+0.030 to +0.040 — inside the dead band**, meaning the sweep closes without
either branch firing. **What would refute this**: ≥ +0.0526 (the support is not
flattening at all and the flattening seen at 500 was noise) or ≤ +0.0263 (a real
deceleration, and a saturation scale finally exists so a level may be quoted).

**Secondary outcomes, reported separately and not folded into the binary.**

1. **Firing check.** Per problem, distinct optima at `f ≤ 1e-5` at 400 vs 600,
   and how many of the 200 new draws land on an optimum unseen in the first 400.
   A problem adding none while still reaching `f ≤ 1e-5` is *support saturation
   of that problem*, not a dead measurement.
2. **`mpr_sup_chao1` and its correction term** are printed as instruments only.
   Entries 101 §2 and 103 §6 showed the correction term is non-monotone
   (0.0411 → 0.0323 → 0.0261 → 0.0369 → 0.0464 → 0.0351); this cycle adds one
   point and **makes no level claim from it**.
3. `mpr` (the primary metric of the theme) and `mpr_earlystop` are printed at
   every n so the record's 400- and 500-draw statements stay checkable. **This
   cycle makes no claim about route (C): its premise is untouched either way.**

**What this cycle will not do.** No MC-ESO default is touched, no arm is tuned,
the queue is not reordered, `docs/status.md` is not edited. The proposal queue
item 2 step (1) asks for — restating "class ceiling" in draw-count terms — goes
into the log for the review cycle, for the **fourth** time.
