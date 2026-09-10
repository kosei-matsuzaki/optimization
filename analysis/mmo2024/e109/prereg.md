# Entry 109 — pre-registration (written before any NMMSO full-budget result was read)

**Question (queue item 1)**: on the GECCO'2024/'2025 suite at D=10, MC-ESO
touches 49 of 240 optima with 500,000 evaluations per problem (entry 107) and
loses 16/16 to the memoryless restart-lander null (entry 106). Every number so
far is **MC-ESO against a null we built ourselves** — one against one, not a
ranking. Entry 83's rule says we cannot write "MC-ESO is last" until a second
method has run on the same suite. NMMSO is the only published-strength niching
method installed here (RR-CMA-ES, queue item 3, is not obtained yet), and it is
the leader on the old CEC2013 target set.

## What is measured

- Problems: `M01..M16-D10-PIN01`. K = 20 (M01-M08) / 10 (M09-M16), 240 optima.
- Budget: `--evals-frac 1.0` = the suite's own 500,000 evaluations per run.
- Method: **NMMSO**, defaults untouched (`scripts/niching_baseline.py:_METHODS`).
- Seeds: **3** (indices 0,1,2) — the same three as entry 106's MC-ESO, so the
  comparison is paired problem by problem on matched seed sets. The queue's
  note that **NMMSO does not reproduce run to run even at a fixed seed** (set
  iteration order is object-id dependent, see acceptance_topology.md's
  environment section) is why one seed may not be quoted alone; three seeds also
  give a per-problem SD, which is the honest way to report a method whose seed
  does not pin the run.
- Reporting rule: `current` (the method's own reported set, cap max(100,2K)).
  Entry 107 showed that on this suite the reporting rule is not the bottleneck
  **for MC-ESO**; that is not established for NMMSO, so if the window allows,
  the same runs get re-scored under `all` afterwards rather than re-run.
- Metric: MPR by nearest optimum (`count_goptima_nn`) at
  `NICHE_ACCURACIES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)`; the suite's headline
  number is the 5-level mean, the queue's branch condition is read at 1e-1.

## Comparisons (all paired, unit = problem, n = 16)

1. **NMMSO vs MC-ESO** (entry 106's per-problem seed means,
   `analysis/mmo2024/e106/mceso_vs_null_d10.csv`). Two-sided Wilcoxon signed
   rank, alpha = 0.05, rank-biserial correlation, win/loss/tie listed (entry 83).
2. **NMMSO vs the memoryless null `mpr`** (entry 103's 500-draw lander, same
   file). Same test.
3. **NMMSO vs the null's support ceiling `null_sup`** — the value a perfectly
   non-redundant member of the restart-and-descend family would reach.
4. Published best on this suite is **MPR 0.651** (GECCO'2024 result sheet,
   16-problem mean, D=10, no per-problem values exist — entry 92). It is a
   single external number, so it is compared as a level only, never tested.

## What would refute what (copied from the queue, unchanged)

Read on the **16-problem mean MPR@1e-1**:

1. **NMMSO mean@1e-1 does not exceed 0.30** → the narrow support set is **not a
   MC-ESO-specific weakness**; at D=10 it is common to every method we hold.
   The mechanism then cannot come from comparing methods and has to come from
   queue item 2 (the structure of the optima that are never landed on). Write
   that and close the question.
2. **NMMSO mean@1e-1 lands near the null's 0.7437** → there is real room inside
   the restart-and-descend family to widen the support set, and the family is
   not the limit; that is direct evidence for the mechanism living in how the
   restarts are placed.
3. Between the two → report where it falls and let the review role decide.

Independently of the branch, the **NMMSO vs MC-ESO** test decides whether
"MC-ESO is last on this suite" becomes a statement we may write.

## Threats to this measurement, stated in advance

- **NMMSO's irreproducibility**: seed 0 of this cycle is not bit-comparable to
  any stored NMMSO row. The wiring check is therefore not bit-equality (entry
  107 could use that; this cannot). Instead: the per-problem SD over 3 seeds is
  printed, and any problem whose SD exceeds its own mean is flagged so a large
  spread cannot hide inside a 16-problem average.
- **A stub `pynmmso` silently produces nothing**: `niching_baseline.py` swallows
  a failed NMMSO import (`except Exception: pass`). The run script asserts
  `NMMSO in _METHODS` **and** that an instantiated `Nmmso` does not raise,
  before any measurement starts (the environment section's stub raises on
  `__init__` by design).
- **Entry 104's trap**: shard logs are kept, and completion is judged by
  **counting output CSVs (48 expected)**, never by a log line. The trap has a
  second form found while this cycle was in flight: `niching_baseline.py` opens
  its `--csv` when it starts, so all 48 files exist at 0 bytes within seconds
  and a plain `ls | wc -l` reads as "finished". **Count non-empty shards**
  (`find ... -size +0c`).

## Cost

Probe at 1/10 budget on this image: M09 (K=10) 10.8 s, M01 (K=20) 15.6 s for
50,000 evaluations, i.e. ~0.19 ms per evaluation, near-linear against the 3.2 s
seen at 10,000. Full budget is therefore ~95 s (K=10) to ~140 s (K=20) per run;
48 runs 4-wide is ~25 minutes on this 4-core container. Shards are one run per
(problem, seed) so a cycle cut short leaves whole runs behind, and seed 0 is
ordered first so the worst case still yields a complete 16-problem set.
