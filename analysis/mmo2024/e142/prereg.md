# Entry 142 — pre-registration (queue 1: the method axis, n=2 -> n=4)

Written **before** any run of this cycle. Claimed 2026-09-19 00:30 UTC.
Wall-clock probe only (5% budget, M09, 1 seed) was run before this file; **no
scored run was.**

## The question, unchanged from the queue

The head claim ("on the new suite at D=10 a memoryless multistart beats
dedicated niching methods") rests on **two** niching methods: MC-ESO (this
project's) and NMMSO (repaired in entry 137, re-measured in entry 139). 16/16 is
the *problem* axis; the *method* axis is n=2. This cycle runs the two remaining
external niching baselines the repository already holds, `NCDE` and `r3pso`, on
the same suite, same budget, same scorer, so the method axis becomes n=4.

Entry 141 closed step (a) and its two findings are taken as given here:
`NCDE`'s shipped `n_pop=30` is **not** a defect at the primary level
(+0.0150, p=0.348), so it runs at its shipped default; `r3pso`'s shipped
`n_particles=30` **is** below Li (2010) §VI-B's published band (300-800 for
D=8-20, i.e. roughly `40*D`), so `r3pso` runs as **two** arms.

## Design

- **Problems**: `M01-D10-PIN01` .. `M16-D10-PIN01` (the whole suite), D=10,
  instance PIN01. Submitted interleaved group A (M01-M08, K=20) / group B
  (M09-M16, K=10), entry 133's rule, so a truncated run is not group-skewed.
- **Budget**: `--evals-frac 1.0` = the suite's own 500,000 evaluations.
- **Seed**: **seed 0 only**, matching the conditions under which the
  comparators were recorded (MC-ESO 0.1385 / NMMSO-10D 0.4144 / `Restart-Lander`
  0.6284 are all seed 0, PIN01).
- **Arms** (`scripts/niching_baseline.py` `_METHODS`; `core/optimizers/` is
  **not touched**, the shipped `NCDE` / `r3pso` rows are **unchanged**):
  - `NCDE` — shipped default (`n_pop=30`, `m=6`).
  - `r3pso` — shipped default (`n_particles=30`).
  - `r3pso-p400` — **new diagnostic row**, `n_particles=400` = Li's band read
    at D=10 (`40*D`). Defaults are not changed, so every recorded `r3pso`
    number keeps its meaning (方針欄 2026-09-11 (3)).
- **Scoring**: runs are executed with `--report-rule current` and
  `REPORT_SET_DUMP` set; the **reporting rule is applied offline** by
  `rule_indices("eps_loose+dedup", r = 0.05 * SPAN)` imported from
  `analysis/mmo2024/e115/analyze.py`, together with `score` / `paired` /
  `aggregate` / `mean_of`. This is exactly the path entries 116 / 139 used.
  **No new statistic is defined.**
- **Total**: 16 x 3 = **48 runs**, 4 workers.

## Gate, run before any judgement

`Restart-Lander`'s 16 stored seed-0 descents (`e115/descents/`) are rescored
through the same code path and **must match `e116/ranking_d10.csv` (`arm=legal`)
to 4 digits on all 16 problems**. If they do not, the scorer is not the one the
comparators were taken on and this cycle reports the mismatch instead of a
judgement.

## Judgement, fixed now

**Primary**: the 16-problem mean Score (PR and mean-F1 averaged, the suite's
official ranking metric, 方針欄 2026-09-11 (1)) of each of `NCDE`, `r3pso`(30),
`r3pso`(400) against `Restart-Lander` **0.6284 (seed 0)**; 0.6238 (3-seed) is
quoted alongside but the paired comparison uses the seed-0 series, since that is
the one measured on the same runs. **MPR and mean-F1 are reported separately for
every arm** so it is visible which half moved.

- If **all three** arms fall below the null, the method axis of the head claim
  becomes **n=4** (MC-ESO, NMMSO, NCDE, r3pso).

**Refutation (a)**: if **any** arm's 16-problem mean Score exceeds
`Restart-Lander`, the head claim weakens from "niching methods lose" to "these
methods lose". The execution role does **not** rewrite the claim: it reports how
many arms beat the null on how many problems and hands it to the review cycle.

**Refutation (b)**: if `r3pso`(30) and `r3pso`(400) **split in sign** relative to
the null on the 16-problem mean, the method-axis claim is wiring-dependent for
that method; that arm is then not used as support for the method axis and the
dependence itself is handed up.

**Per-problem wins/losses are listed for every arm** (entries 139 / 140: a
16-problem mean loss coexists with individual problems won — whether "all 16"
can be written is decided there, not by the mean).

## Stated limitations, before the numbers

- **n=1 seed.** Entry 135's lesson ("a 1-seed screen is for wall-clock only")
  is acknowledged and *not* evaded: the pairing unit here is the **problem**
  (n=16 pairs), not the seed, which is the same unit entries 116 / 139 used,
  and the comparators are all seed 0. A per-arm seed spread is **not** measured
  this cycle and no claim is made about it. Any arm that lands within the
  null's seed-to-seed spread (16-problem mean Score SD over seeds = 0.0066 for
  the null, entry 131) would need seeds before its sign is claimed.
- `NCDE` / `r3pso` have **no published per-problem comparator** on any suite
  this project holds (entry 141 (a-1)), so the "matches the published table"
  gate that validated NMMSO cannot be run for them. Their wiring is audited
  only through entry 141's population sweep.

## Budget contingency, decided now

The probe (5% budget, M09-D10-PIN01, 1 seed, serial) gives **NCDE 11 s,
`r3pso` 8 s, `r3pso-p400` 7 s**, i.e. roughly 220 / 160 / 140 s per full run
=> 8,320 s serial => ~35 min at 4 workers. That is at the edge of the cycle's
compute window. The job list is therefore ordered **all 16 problems x
{`NCDE`, `r3pso`} first (32 runs), then the 16 `r3pso-p400` runs**, so that if
the window binds, what is lost is the third arm and **not** problems — the
queue's rule (drop a method, never functions, or Score stops being comparable).

## Retention, decided now

48 report-set dumps are folded into a single `report_sets.csv.gz` with
`problem` / `method` / `seed` columns in this same cycle, and the per-problem
dumps deleted, **after** checking that `analyze.py`'s output is byte-identical
before and after the fold. The fold script and the `rm` are **separate command
lines** (entry 131's accident). A closed route is deleted in the same commit to
keep `analysis/` under the 400-file guideline.
