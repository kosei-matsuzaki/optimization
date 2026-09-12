# Entry 117 — pre-registration

Queue 2 (研究ループ「未解決の問い」2 番) = the explicit execution of 方針欄
2026-09-11 decision (3). What is being decided is *adoption*, not a research
question, so most of the conditions were fixed by the user before this cycle
started; they are restated here verbatim so the verdicts can be checked against
them.

## Fixed by the user (方針欄 2026-09-11 (3)) — not chosen by this cycle

- The gate is **`best_f` byte-identity on BBOB-24 dim2**. The `CLAUDE.md` pin
  (SR@1e-10 93.5% / `evals_succ_mean` 798) is **not** the control: it does not
  reproduce in this container (0.9208 / 677.7, twice), and a criterion that does
  not reproduce cannot be re-run.
- An arm that is byte-identical **may be adopted**.
- An arm that is not byte-identical is read against **this environment's
  re-measured base**; if it lowers the pin it is rejected.
- Every adopted arm needs one line naming **which recorded numbers become
  old-default**. An arm for which that line cannot be written is not adopted.

## What this cycle measured

Only `commit_place_r010` had never been through the gate, so the measurement is
one pair: `./run.sh quick --all --n-runs 20 --max-evals 5000 --methods
MC-ESO,commit_place_r010` (24 functions x 20 seeds = 480 paired cells).

Refutation conditions, set before the run:

1. **Gate.** One cell where the arm's `best_f` differs *and* SR@1e-10 falls
   below this environment's base 0.9208, or `evals_succ_mean` rises above 677.7,
   rejects the arm. (Byte-identity plus unchanged pins = pass.)
2. **Gate power** (entry 29's lesson, added by this cycle because the user's
   definition asserts byte-identity == "never fires"): if the committed restart
   fires in **zero** cells, the tie is type (ii) — the gate measured nothing —
   and "safe" is not a finding. If it fires and the first deviation ever precedes
   the evaluation at which `best_f` reached its final value, the arm *could* have
   moved the pin and the tie needs explaining rather than accepting.

## Composite arm (`comp4`), measured second

The four gate-passing arms loaded at once — c=1.0 (`rel_level=1e-5`,
`fis_floor=1e-12`), `_sig10`, `_fl08`, `soltrim_rho` — with commit deliberately
off, because `commit_place_r010` was still inside the gate when this was
launched. Paired by seed against base on the CEC2013 functions the adoption
procedure names, at the suite's own budget.

Refutation conditions, set before scoring:

3. **The stack must not be worse than base at the judgement levels**
   (eps <= 1e-3, entry 28). A significant paired loss at 1e-3 or 1e-5 means the
   four arms interact and cannot be adopted as a stack — the untested
   `_fl08` x `soltrim_rho` interaction is the named suspect.
4. **Class identity.** `comp4_off` (same class, all four layers disabled) must
   reproduce base exactly on `best_f` / `hunts` / `spillover` / `basin_switch` /
   `visited` / `reported` / `landed`. If it does not, any difference the
   composite shows is an artifact of the variant class and nothing else can be
   read off it.

Scaled down from the procedure's N06 / N08 / N09 to **N06 (12 seeds) and N08
(8 seeds)**; N09-Vincent3D (400k x 2 arms) did not fit the 40-minute frame and
is the recorded next step. Single-arm decomposition is *not* measured: the
`e60` / `e63` single-arm CSVs were deleted in earlier consolidations, so
"is this gain just `_fl08`?" cannot be answered from disk.
