# Entry 143 — pre-registration (queue 1, remainder: the published-band `r3pso`)

Written **before** any run of this cycle. Claimed 2026-09-19 06:30 UTC.
No scored run of this cycle preceded this file.

## The question, as the queue left it

Entry 142 closed the method axis at n=4 (`NCDE` 0.1171 and `r3pso`(30) 0.1468
both 0/16 against the null's 0.6284). But the fourth arm's number was taken at
`n_particles=30`, which entry 141 itself established is **below the lower end of
Li (2010) §VI-B's published band** (300-800 for D=8-20). Entry 142's budget
bound, so its `r3pso-p400` arm covered only M01 / M09 / M10 and was dropped from
the judgement by the pre-registered escape hatch (drop an arm, never problems).
On those 3 problems 400 beats 30 by **+0.2028** Score — same direction and same
order of magnitude as the NMMSO `swarm_size` defect (+0.2596).

So the standing claim "`r3pso` loses 16/16" rests on wiring this project has
already written down as defective. This cycle completes the arm.

## Design — identical to entry 142 in every respect except the problem list

- **Problems**: the 13 missing ones, `M02`-`M08` and `M11`-`M16`, D=10, PIN01.
  M01 / M09 / M10 are already in `e142/report_sets.csv.gz` and are **not**
  re-run; the same dump, the same scorer, the same rule.
- **Budget**: `--evals-frac 1.0` = the suite's 500,000 evaluations.
- **Seed**: seed 0 only, matching every comparator (MC-ESO 0.1385,
  NMMSO-10D 0.4144, `Restart-Lander` 0.6284, `NCDE` 0.1171, `r3pso`(30) 0.1468).
- **Arm**: `r3pso-p400` from `scripts/niching_baseline.py` `_METHODS`
  (`n_particles=400` = `40*D` at D=10). **`core/optimizers/` is not touched** and
  no default is changed (方針欄 2026-09-11 (3)), so every recorded `r3pso`
  number keeps its meaning.
- **Scoring**: runs execute with `--report-rule current` + `REPORT_SET_DUMP`;
  the reporting rule `eps_loose+dedup`, r = 0.05*span, is applied **offline** by
  `rule_indices` / `score` / `aggregate` / `paired` / `mean_of` imported from
  `analysis/mmo2024/e115/analyze.py` through `e142/analyze.py`, unchanged.
  **No new statistic is defined this cycle.**
- **Submission order**: group A (M02-M08, K=20) and group B (M11-M16, K=10)
  interleaved (entry 133's rule), 4 workers, so a bound budget does not skew the
  arm toward one group.

## Gate, run before any judgement

Unchanged from entry 142 and re-run here: `Restart-Lander`'s 16 stored seed-0
descents (`e115/descents/`) are rescored through the same path and must match
`e116/ranking_d10.csv` (`arm=legal`) to 4 digits on all 16 problems. If they do
not, this cycle reports the mismatch instead of a judgement.

**Second gate, specific to this cycle**: the three `r3pso-p400` problems entry
142 already scored (M01 0.5133 / M09 0.3886 / M10 0.3808 per
`e142/scored.txt`, read from that file after this prereg was first written and
corrected here before any result of this cycle existed) must come out
**unchanged to 4 digits** after the new rows are folded in. This is what proves the fold appended and did not perturb.

## Judgement, fixed now

**Primary**: the 16-problem mean Score (the suite's official ranking metric;
方針欄 2026-09-11 (1)) of `r3pso-p400` against `Restart-Lander` **0.6284**
(seed 0), paired by problem, two-sided exact Wilcoxon, alpha = 0.05, with the
rank-biserial effect size. MPR and mean-F1 are reported separately so it is
visible which half moved.

- If `r3pso-p400` falls **below** the null on the 16-problem mean, the method
  axis n=4 **holds on published-setting wiring** and the head claim stops
  resting on a configuration this project called defective.

**Refutation (a)** (pre-registered, unchanged from the queue): if
`r3pso-p400`'s 16-problem mean Score **exceeds** the null, the head claim weakens
from "niching methods lose" to "these methods lose". The execution role does
**not** rewrite the claim — it reports on how many problems the arm beat the null
and hands it to the review cycle.

**Refutation (b)**: if `r3pso`(30) and `r3pso`(400) **split in sign** relative to
the null on the 16-problem mean, `r3pso` is not usable as support for the method
axis and the dependence itself is handed up.

**Per-problem wins/losses are listed for every arm** (entries 139 / 140).

**What would have refuted the hypothesis**: the hypothesis under test is "the
30-particle defect does not change `r3pso`'s verdict against the null". A single
problem where `r3pso-p400` exceeds the null's Score is already reportable; the
16-problem mean crossing 0.6284 refutes it outright. The 3-problem partial
(+0.2028 for 400 over 30) is large enough that the arm could plausibly land near
`NMMSO-10D` (0.4144) — that is a real possibility this design can see, not a
foregone conclusion.

## Stated limitations, before the numbers

- **n=1 seed**, as in entries 116 / 139 / 142: the pairing unit is the
  **problem** (n=16), and every comparator is seed 0. The null's seed-to-seed SD
  on the 16-problem mean is 0.0066 (entry 131); any arm landing inside that band
  is reported without a sign claim.
- `r3pso` has **no published per-problem comparator** on any suite this project
  holds (entry 141 (a-1)), so the "matches the published table" gate that
  validated NMMSO cannot be run for it. Its wiring is audited only against Li
  (2010) §VI-B's stated population band.
- 400 = `40*D` is one reading of a 300-800 band. Nothing here measures whether
  the arm is at its own optimum inside that band.

## Retention, decided now

The 13 dumps are appended to `e142/report_sets.csv.gz` in this same cycle (the
queue's instruction; `e142/analyze.py` then sees all 16 problems for the arm and
activates it). The append is verified **row by row against the source dumps**
before the per-run dumps are deleted, and the fold script and the `rm` are on
**separate command lines** (entry 131's accident). A closed route is deleted in
the same commit to keep `analysis/` under the 400-file guideline.
