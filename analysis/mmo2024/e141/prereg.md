# Entry 141 — pre-registration (queue 2, step (a): the population-size audit of NCDE / r3pso)

Written **before** any run of this cycle. Claimed 2026-09-18 18:30 UTC.

## What the queue asked for, and what is actually runnable this cycle

Queue 2(a) is "照合してから CEC2013 で関門を通す": pin the published population size
for `NCDE` (Qu, Suganthan & Liang 2012) and `r3pso` (Li 2010), then gate the
corrected wiring against **published per-problem PR** on N15 / N17 / N18 / N19 / N20.

**The published-value half of that gate cannot be closed this cycle.** Two
findings, both cheap and both recorded before the run:

1. **Neither method is in the CEC2013 competition result table**, so the
   comparator table this project already holds does not carry them.
   `analysis/mmo2024/e134/refs/fieldsend2014_nmmso_cec.txt.gz` Table V lists 15
   comparison algorithms (A-NSGA-II, CMA-ES, CrowdingDE, dADE/nrand/{1,2},
   DECG, DELG, DELS-aj, DE/nrand/{1,2}, IPOP-CMA-ES, NEA1, NEA2, N-VMO,
   PNA-NSGA-II) and **neither `NCDE` nor `r3pso` appears**. Both papers predate
   the CEC2013 suite (2012 / 2010) and used their own function sets.
2. **The two original papers are behind this sandbox's egress.** `WebSearch`
   returns them; the one full-text PDF it surfaced
   (`web.xidian.edu.cn/xlwang/files/20150312_174832.pdf`, Li 2010) is
   `EGRESS_BLOCKED`, as is every IEEE mirror. The CI `fetch_refs` route is the
   designated path and is left as the next step.
   (Entry 134's rule was applied first: `ls analysis/*/refs/ analysis/*/*/refs/`
   — six files, none of them these two papers.)

**Secondary-source fact that was obtainable** (`WebSearch`, 2026-09-18):
NCDE's neighbourhood size `m` is defined as **10% of the population** (20% for
NSDE) — i.e. a *fraction*, not the absolute constant 6 the code ships.

## So what this cycle measures instead, and why it is the same question

The reason queue 2(a) exists is one hypothesis, not one table:

> **H**: `ncde.py:36` `n_pop=30` and `r3pso.py:36` `n_particles=30` are
> D-independent hard-codings of the same shape as the NMMSO `swarm_size=10`
> defect (entry 136), and they weaken these baselines at the dimensions where
> the head claim is made.

H is testable **without** the published constant, by asking whether the
population size bites at all over a decade around the shipped value. If PR is
flat from 30 to 300 on these functions, then whatever the published constant
turns out to be, the recorded numbers do not move and step (b) may run on the
shipped default. If it is not flat, the published constant must be pinned
before (b), and the "old default" bookkeeping of 方針欄 2026-09-11 (3) is owed.

**Deviation from the queue text, stated on purpose** (the queue says that if the
budget binds, drop a method rather than functions): the 5 functions are kept,
both methods are kept, and what is dropped instead is the *number of population
levels* — one decade, two levels (30 vs 300), rather than a sweep. The queue's
"keep 5 functions" rule protects the published-value gate, which is not the
measurement running here.

## Design

- **Functions** (the queue's five, unchanged): `N15-CF4-3D`, `N17-CF4-5D`,
  `N18-CF3-10D`, `N19-CF4-10D`, `N20-CF4-20D`.
- **Budget**: the suite's own, `--evals-frac 1.0` = 400,000 evaluations per run.
- **Seeds**: **10 per cell** (`--seeds 10`, seed index i = optimiser seed i*100).
  n=5 is explicitly disallowed by entry 137's lesson (1 run takes 1/K-sized
  steps; K=6 and K=8 here, so 1/6 and 1/8).
- **Arms** (`scripts/niching_baseline.py` `_METHODS`, diagnostic rows added;
  `core/optimizers/` untouched, the existing `NCDE` / `r3pso` rows unchanged):
  `NCDE` (n_pop=30, m=6) / `NCDE-p300` (n_pop=300, m=60 — the shipped 1/5
  fraction held) / `r3pso` (30) / `r3pso-p300` (300).
- **Reporting rule**: `current` (the method's own `final_solutions`) — the rule
  every recorded NCDE / r3pso number in this project was scored under.
- **Total**: 5 x 4 x 10 = **200 runs**. Wall-clock probe (1 seed, N19, 4e4
  evals, extrapolated x10): NCDE 34 s (300: 46 s), r3pso 18 s per full run
  => ~5,900 s serial, 4 workers.

## Statistic, fixed now

Pairing unit = **(function, seed)**, 50 pairs per method, arm minus shipped
default. `scipy.stats.wilcoxon` two-sided, alpha = 0.05, plus the mean
difference and rank-biserial correlation. No new statistic is defined.

**Primary accuracy: PR@1e-5** (the level the published CEC2013 tables and the
entry 137 NMMSO gate use, so this cycle's numbers stay comparable to the gate
that follows). PR@1e-1 ... 1e-4 are reported as the breakdown.

## Refutation conditions, fixed now

- **(R1) H is refuted for a method** if its 50 paired PR@1e-5 differences give
  **p >= 0.05 and |mean difference| < 0.05**. Then the shipped 30 is *not* a
  NMMSO-type defect on these functions, the recorded values stand, and step (b)
  runs on the shipped default.
- **(R2) H is supported for a method** if the difference is **positive with
  p < 0.05**. Then the published constant must be pinned (CI `fetch_refs`)
  before (b), and the "which recorded numbers are old-default" line of
  方針欄 2026-09-11 (3) is owed in the same commit as any default change.
- **(R3) The awkward third case** — significant but **negative** (300 is worse)
  — is recorded as-is and *not* rewritten into a recommendation: it would mean
  the shipped 30 is favourable to these baselines, which strengthens rather
  than weakens the head claim, and the published constant still has to be
  pinned before anything is changed.
- A result that is significant but with |mean| < 0.05, or non-significant with
  |mean| >= 0.05, is written down as **undecided at n=50** and the next step is
  more seeds, not a re-read of the band. **The band is fixed here and is not
  loosened after seeing the arms** (entry 136's lesson).

## What is deliberately not touched

MC-ESO is not run and not read (案 (C) / 方針欄). No default in
`core/optimizers/` is changed by this cycle — the audit arms live in the
driver's `_METHODS` only, so no recorded number changes meaning.
