# e103 pre-registration (written before any draw past 400 was computed or opened)

**Question (queue item 2, step (2), verbatim).** "支持集合が 800 draw でも減速
しないかを見る。" Entry 101 §3 measured the bare support statistic `S_obs / K`
rising by **+0.0499 (100→200), +0.0263 (200→300), +0.0263 (300→400)** per 100
draws, 16-problem mean — the last two increments **equal**, i.e. no sign of
deceleration. If the support does not saturate, then no draw-independent number
may be quoted as "the class ceiling"; only "the coverage observed at N draws".

**Refutation condition (queue item 2 wrote it; restated here, adapted to the
scaled-down window below).** The queue's form: *if the increment past 400 is
**not smaller** than the increment over 200→400, the support does not saturate
in this range and "coverage at N draws" stays the only sayable form; if it **is
smaller**, a saturation scale exists and a level may finally be quoted.*
Because this cycle measures a 100-draw window, the comparison is made on the
**per-100-draw increment**, which is exactly the unit entry 101 §3 reported:

- **No deceleration** (queue's first branch) if Δ(400→500) of the 16-problem
  mean `S_obs/K` is **≥ 0.0263** — the value both preceding 100-draw windows
  took.
- **Deceleration** (queue's second branch) if it is **≤ 0.0132**, i.e. at most
  half the preceding increment.
- Between the two, **neither branch fires** and this cycle records the value and
  says the question needs the 500→600 window, which is the next chunk.

Pre-registering a dead band is deliberate: a single 100-draw window carries
integer-count noise (the mean increment of 0.0263 is about **6 newly covered
optima across the whole 16-problem set**), and entry 101's own lesson —
"推定量の補正項が n の或る区間で縮んでも、収束の証拠にはならない" — cuts both
ways. One window is not allowed to settle the question in either direction; it
is allowed to fire only the "not decelerating" branch, which is the branch the
prior two windows already support.

**The measurement.** `analysis/mmo2024/e103/run.sh 400` runs
`scripts/hunt_coverage.py --null --geo-draws 20000 --descents 100
--descent-start 400 --budget 12500 --sigma-ratio 0.1 --procs 4` on **all 16
problems** (M01-M16, D=10, PIN01). No new arm, no parameter changed other than
the draw window, **zero optimization runs**. `--descent-start` is a pure index
shift (draw k is a closed form of k alone: start point `default_rng(1_000_000 +
k)`, CMA seed `k + 1`), identity-checked by entry 100 against the saved dumps
and re-checked here (below). Scored by `analysis/mmo2024/e103/analyze.py`, which
**imports the four ceiling estimators from entries 92/93** rather than
reimplementing them, so this theme keeps one scoring code path.

**Identity check (must pass or the cycle reports nothing).** The concatenated
dump's first 200 and first 400 draws must reproduce entry 99/98's and entry
101's ceiling tables exactly. If they do not, the concatenation is wrong and no
number from this cycle is reportable.

**Scale-down, declared in advance and unprompted.** The queue's step (2) names
`--descent-start 400 --descents 400` (i.e. 400 → 800) and estimates one group
per cycle, two cycles. That estimate is for **200** new draws per problem (entry
101's group-A mean of 204 s per problem per 200 draws → 27 min for 8 problems);
**400 new draws on 8 problems is ~54 min**, over the 40-minute budget, and one
group alone would put the 16-problem mean back to mixing two draw counts — the
exact defect entry 101 §1 had just repaired. **This cycle therefore takes 100
draws on all 16 problems (400 → 500) instead of 400 draws on 8.** The whole
16-problem set moves together or the cycle reports no mean. If the clock allows
after chunk 400-499 is complete on all 16, chunk 500-599 is started under the
same rule: **the 600-draw mean is reported only if all 16 problems reach 600**;
otherwise the finished problems are noted per-problem and the mean stays at 500.

**Prediction.** The two preceding windows were equal at +0.0263 and entry 101
§3 found no mechanism for them to shrink here (new optima keep arriving as
single landings). So Δ(400→500) lands in **+0.020 to +0.030** and the "no
deceleration" branch fires. **What would refute this**: ≤ 0.0132 (deceleration),
or a value above +0.035 (acceleration, which no mechanism predicts and which
would suggest the concatenation or the draw indexing is wrong — check the
identity check before believing it).

**Secondary outcomes reported separately, not folded into that binary.**

1. **Firing check.** Per-problem count of distinct optima at `f ≤ 1e-5` at 400
   vs 500, and how many of the 100 new draws land on an optimum unseen in the
   first 400. A problem adding none is *saturation of that problem's support*,
   not a dead arm; the two are distinguished by whether any draw of that problem
   reached `f ≤ 1e-5` at all.
2. **`mpr_sup_chao1` is reported but is not the quantity under test.** Entry 101
   §2 showed it is a step function of the singleton count (two problems drove
   72% of its rise). The queue's step (2) asks about the **bare support**, so
   `S_obs/K` is the primary; the Chao1 reading and its correction term are
   printed as instruments, and the U-turn entry 101 found (correction/K
   0.0411 → 0.0323 → 0.0261 → 0.0369 → 0.0464) is extended by one point.
3. The main metric `mpr` and `mpr_earlystop` are printed at every n so the
   400-draw statements in the record stay checkable, but this cycle makes no
   claim about route (C): its premise is untouched either way.

**What this cycle will not do.** No MC-ESO default is touched, no arm is tuned,
the queue is not reordered, `docs/status.md` is not edited. The proposal queue
item 2 step (1) asks for — restating "class ceiling" in draw-count terms — goes
into the log for the review cycle, for the **third** time.
