# e121 pre-registration — is the thin basin *reachable*, or is the operator locked out?

Queue item 1 (claimed 2026-09-13 18:30 UTC). Written and committed **before** the runs.

## What entry 119 left open

Entry 119 established that the 55 clustered optima of group A (M01-M08) are not
"already harvested": corrected for Voronoi cell volume they take **13x fewer
descents than isolated optima** (`enrichment` 0.0770 vs 1.1468, p=7.84e-20,
`>=1.0` in 0/55 vs 82/185). The basins are thin. It did **not** say why.

Two readings, and they point at different arms:

* **(α) resolution.** The basin exists at a scale the deployed descent overshoots.
  Then an arm that *re-searches the neighbourhood of a landing at a smaller scale*
  is worth one cycle.
* **(β) the operator is locked out.** No start distance gets in. Then the arm has to
  change the descent itself (step size schedule, anisotropy, a different operator),
  and **arms that only re-allocate restarts are off the candidate list.**

## Design

**Targets.** The 55 cluster members of group A from `e119/enrichment_d10.csv`
(`clustered == 1`), and a problem-matched control of 55 isolated optima
(`clustered == 0`) — the same count per problem as that problem's members,
drawn with `np.random.default_rng(20260913)`. Group B has no members, so this
measures 8 of the 16 problems by construction.

**Start points.** For each target `z` and radius `r ∈ {0.0, 0.1, 0.25, 0.5, 1.0, 2.0}`,
one start `x0 = clip(z + r·u, lo, hi)` with `u` uniform on the unit 10-sphere.
`r = 0.0` is the **calibration level**, not a data point: it is the ceiling of the
measurement. Radii are in the suite's own units (box `[-5, 5]`, span 10).

**Descent.** Byte-for-byte `Restart-Lander`'s descent operator (isotropic CMA-ES,
`CMA_on = 0`, `tolfun = tolfunhist = tolx = 0`, bounds `[-5, 5]`), capped at
**3125 evaluations** = 12500/4, the budget the queue entry allows.

**Two σ0 conditions**, because the deployed σ0 confounds the two readings:

| condition | σ0 | what it asks |
|---|---|---|
| `deployed` | `0.1 · span = 1.0` (the null's default) | can the operator *as deployed* enter from distance r? |
| `matched` | `max(r, 0.1) / 2` | can the operator enter *at all*, if its scale is handed to it? |

The deployed σ0 is 1.0, i.e. **4x larger than r = 0.25 and 10x larger than r = 0.1**.
Without the `matched` arm a flat deployed curve cannot distinguish "there is no basin"
from "the initial cloud is wider than the basin", which is exactly the (α)/(β) split.

**Reached (primary).** `land_opt == target` **and** `dist(best_x, z) < 0.1` —
nearest-optimum credit as everywhere else in this project (`land_opt`), plus
entry 119's own `NEAR = 0.1` convergence convention.
**Reported alongside (secondary):** `land_opt == target` alone, and `best_f`.

**Why not depth 1e-5.** A pilot descent started *on* an optimum with σ0 = 0.05
reached only `best_f = 3.9e-5` in 3125 evaluations. At this budget a depth
criterion would measure the budget, not the basin. The distance criterion is
reported instead and the `r = 0.0` level prices the residual.

**n.** 110 targets × 6 radii × 2 conditions = **1320 descents**, ~0.8 s each,
4-wide ⇒ ~5 min. Curves are aggregated over the 55 targets per (r, condition),
so each point is a binomial with n = 55 (±0.13 at p = 0.5).

## Pre-registered decision rule (from the queue entry, on the `deployed` curve)

* **P(reach | r = 0.25) ≥ 0.5 for members ⇒ (α).** Write that the basin is there and
  only the resolution is missing, and that "re-search the neighbourhood of a landing
  finely" is worth one arm.
* **P(reach | r = 0.1) < 0.2 for members ⇒ (β).** Write that this descent operator
  cannot enter the members, that an arm must change the descent side, and that
  **restart-allocation arms are dropped from the candidate list.**
* **In between:** report the shape — the half-reach radius `r_half` — for members
  and isolated separately.

**What would refute the whole framing.** If the **isolated** control curve is also
flat and low under `deployed`, the measurement is about the 3125-eval budget and
not about the members at all, and nothing above may be concluded. The isolated
control is therefore the falsifier, not decoration: members must be *worse than
matched isolated optima at the same r*, tested paired by problem (Wilcoxon over the
8 problems on per-problem reach rate, plus a Mann-Whitney over targets and A12).

**The `matched` curve answers the mechanism**, and is read only after the above:
members reachable under `matched` but not under `deployed` ⇒ (α) with a price tag;
members unreachable under both ⇒ (β).

## Before any arm (not this cycle)

The queue entry's own rule: the arm nearest this measurement is
**MSC-CMA-ES (arXiv 2606.15830)** — nearest-better clustering to identify basins,
then restart with a locally rescaled step size and population. That is close enough
to "re-search a landing's neighbourhood at a local scale" that, per the 2026-09-08
novelty rule, the matching and non-matching steps must be written out before an arm
is built. **This cycle measures only; it builds no arm and does not touch `mceso.py`.**
