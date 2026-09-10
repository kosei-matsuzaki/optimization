# Entry 106 — pre-registration (written before any full-budget result was read)

**Question (queue item 1)**: run MC-ESO (and NMMSO) on the 16 GECCO'2024 problems at
D=10 / PIN01 / normal budget and read the 5-level mean MPR against the memoryless
restart-lander null.

## What is measured

- Problems: `M01..M16-D10-PIN01`. K = 20 (M01-M08) / 10 (M09-M16).
- Budget: `suite_max_evals` = floor(50000 x D) = 500,000, i.e. `--evals-frac 1.0`.
- Metric: `pr.mean()` over `NICHE_ACCURACIES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)`,
  scored on the reported set (final population + restart archives, capped at
  max(100, 2K)), same type as the published value.
- Scoring rule: **nearest optimum**, not the CEC2013 rho filter. The new suite
  defines PR purely in objective space and ships no rho
  (`external/mmo2024/docs/competition_setup_TR2024001.txt` §4), and the null in
  entries 98-105 attributes each descent to its nearest optimum
  (`scripts/hunt_coverage.py:265-268`). Scoring the method any other way would
  make the comparison that this question exists for impossible.
- Seeds: 3 (scaled down from the 5 the queue asks for; see cost below).
- Methods: MC-ESO. **NMMSO is deferred to the next cycle** — it now imports
  (see the log entry), but 16 x 3 runs of it do not fit beside MC-ESO's in one
  40-minute window.

## Comparison and test

Paired on the problem (n = 16), against the null values already in the record
(`analysis/mmo2024/e103/scored.txt`, n = 500 draws, per-problem `mpr` and `sup`):

- `mpr` (memoryless, i.i.d. restarts) — 16-problem mean **0.5530**
- `mpr_sup` (support / K, perfect redundancy avoidance) — mean **0.7006**
- published best at D=10 (GECCO'2024 results deck, 16-problem mean) — **0.651**

Two-sided Wilcoxon signed-rank, alpha = 0.05, with rank-biserial correlation.
Per-problem values are listed in full (entry 28's rule: no single-number verdicts).

## What would refute what

The queue registers the rejection conditions; they are copied here unchanged.

1. **MC-ESO mean MPR > 0.7006** → the reading "the memoryless null's support set
   is the family's reachability" is wrong; repulsion widens the support itself,
   and the 0.299 reachability estimate has to be redone.
2. **MC-ESO mean MPR < 0.5530** → MC-ESO carries a memory and still loses to a
   memoryless null, so losses other than redundancy (reporting, depth) dominate
   on this suite and must be decomposed first.
3. **Between the two** → the three-way split in status.md (1) survives, and the
   gap between MC-ESO and 0.5530 measures how much of the 0.148 redundancy loss
   its repulsion has already recovered.

Because 1 and 2 are two-sided and the interval between them is where the
hypothesis lives, no outcome is unfalsifiable.

## Cost (probe, this container, 4 cores)

1 seed at `--evals-frac 0.1` (50,000 evaluations): M01 **22 s**, M09 **16 s**.
Linear extrapolation to 500,000 gives ~190 s per run, so 16 x 3 = 48 runs is
~9,100 core-seconds ≈ 38 min at 4-wide. 5 seeds would be ~63 min and would not
fit. Per-problem CSVs (`by_problem/<name>.csv`) so a cut-short run still leaves
finished problems behind (entry 85).
