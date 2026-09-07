#!/usr/bin/env python3
"""Entry 81: re-score every stored niching arm on the two PR components.

PR = coverage - report loss, identically:

    PR       = reported / K
    coverage = visited  / K              (distinct global optima the run touched)
    loss     = (visited - reported) / K  (touched but not in the reported set)

Entry 80 found an arm (`basin_reset` on N09-Vincent3D) whose two components each
moved 15/15 seeds in the same direction and by nearly the same amount, leaving
PR flat: a real price paid, invisible to the metric the queue judges on. The
open question is whether that cancellation is specific to that arm or a
property of the coverage-limited regime. This re-reads the stored CSVs of the
last cycles -- no new runs -- and counts the arms it happens to.

Pre-registered classification, per (contrast, function, eps) cell on the paired
seeds (paired Wilcoxon, zsplit, two-sided):

  flat PR         p(PR) >= 0.05
  moving component p(component) < 0.05 AND |mean delta| >= 0.02  (2 points of K)
  SILENT TRADE    flat PR AND at least one moving component
  CANCELLING      silent trade where both components move, same sign
                  (= the entry-80 signature: PR is flat *because* they cancel)

Refutation: zero silent-trade cells outside `basin_reset` means the
cancellation is that arm's own and the two components do not need to be
reported alongside PR as a rule.

Multiplicity is reported both raw and Holm-corrected within the family of
component tests, since ~200 tests at alpha=0.05 buy ~10 false positives.
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.hunt_confound import load, paired  # noqa: E402

HM = os.path.join(os.path.dirname(__file__), "..")


def P(*parts: str) -> str:
    return os.path.join(HM, *parts)


def merged(func: str, *paths: str):
    """load() over several shards of the same arm (e57/e58/e60/e62/e63 split
    their 15 seeds across three files)."""
    out: dict = {}
    for p in paths:
        for eps, per_seed in load(p, func).items():
            out.setdefault(eps, {}).update(per_seed)
    return out


# (function, budget-tag, contrast label, arm paths, reference label, ref paths,
#  cycle tag: "core" = the e63/e68/e69 re-read the question names)
N09F = ("N09-Vincent3D", "400k")
N07F = ("N07-Vincent2D", "200k")
N06F = ("N06-Shubert2D", "200k")
N08F = ("N08-Shubert3D", "400k")

n09_base = [P("e80", "n09_base.csv")]                       # == e58 base, verified
n07_base = [P("e57", f"n07_full_base_cap_s{i}.csv") for i in (0, 5, 10)]
n06_base = [P("e61", "n06_full_base.csv")]
n08_base = [P("e61", "n08_full_base.csv")]
cp09 = [P("e62", f"n09_full_cp_s{i}.csv") for i in (0, 5, 10)]
cps09 = [P("e63", f"n09_full_cp_soltrim_s{i}.csv") for i in (0, 5, 10)]
cps07 = [P("e63", f"n07_full_cp_soltrim_s{i}.csv") for i in (0, 5, 10)]
ct09 = [P("e68", "n09_full_ct.csv")]
cts09 = [P("e68", "n09_full_ctsoltrim.csv")]
rel09 = [P("e69", "n09_full_ct_soltrim_rel.csv")]

CONTRASTS = [
    # --- N09-Vincent3D, full budget (coverage-limited) ---
    (*N09F, "soltrim_rho / base", [P("e60", f"n09_full_soltrim_rho_s{i}.csv") for i in (0, 5, 10)], "base", n09_base, "e60"),
    (*N09F, "solcap2000 / base", [P("e60", f"n09_full_solcap2000_s{i}.csv") for i in (0, 5, 10)], "base", n09_base, "e60"),
    (*N09F, "commit_place / base", cp09, "base", n09_base, "e62"),
    (*N09F, "cp_soltrim / base", cps09, "base", n09_base, "core"),
    (*N09F, "cp_soltrim / commit_place", cps09, "commit_place", cp09, "core"),
    (*N09F, "commit_tight / base", ct09, "base", n09_base, "core"),
    (*N09F, "ct_soltrim / commit_tight", cts09, "commit_tight", ct09, "core"),
    (*N09F, "ct_soltrim / cp_soltrim", cts09, "cp_soltrim", cps09, "core"),
    (*N09F, "ct_sol_rel / ct_soltrim", rel09, "ct_soltrim", cts09, "core"),
    (*N09F, "ct_sol_rel_fl08 / ct_sol_rel", [P("e69", "n09_full_ct_soltrim_rel_fl08.csv")], "ct_sol_rel", rel09, "core"),
    (*N09F, "basin_reset / base", [P("e80", "n09_basin_reset.csv")], "base", n09_base, "e80"),
    # --- N07-Vincent2D, full budget (coverage-limited) ---
    (*N07F, "soltrim_rho / base", [P("e60", f"n07_full_soltrim_rho_s{i}.csv") for i in (0, 5, 10)], "base", n07_base, "e60"),
    (*N07F, "commit_place / base", [P("e62", "n07_full_cp.csv")], "base", n07_base, "e62"),
    (*N07F, "cp_soltrim / base", cps07, "base", n07_base, "core"),
    (*N07F, "cp_soltrim / commit_place", cps07, "commit_place", [P("e62", "n07_full_cp.csv")], "core"),
    # --- N06 / N08 Shubert, full budget (depth-limited) ---
    (*N06F, "soltrim_rho / base", [P("e61", "n06_full_rho.csv")], "base", n06_base, "e61"),
    (*N06F, "basin_reset / base", [P("e80", "n06_basin_reset.csv")], "base", n06_base, "e80"),
    (*N08F, "soltrim_rho / base", [P("e61", "n08_full_rho.csv")], "base", n08_base, "e61"),
    (*N08F, "basin_reset / base", [P("e80", "n08_basin_reset.csv")], "base", n08_base, "e80"),
    (*N08F, "lvl_c100_fl08 / lvl_c100", [P("e55", "n08_level_rel_c100_fl08.csv")], "lvl_c100", [P("e55", "n08_level_rel_c100.csv")], "e55"),
    (*N08F, "lvl_c100_sig10 / lvl_c100", [P("e55", "n08_level_rel_c100_sig10.csv")], "lvl_c100", [P("e55", "n08_level_rel_c100.csv")], "e55"),
]
# --- tenth-budget decomposition sets (both regimes), same three arms each ---
for fn, tag, stem in (("N09-Vincent3D", "40k", "n09"), ("N08-Shubert3D", "40k", "n08"),
                      ("N07-Vincent2D", "20k", "n07"), ("N06-Shubert2D", "20k", "n06")):
    for arm in ("commit_place_r010", "commit_tight", "sigma_only"):
        CONTRASTS.append((fn, tag, f"{arm} / base", [P(f"{stem}_decomp_{arm}.csv")],
                          "base", [P(f"{stem}_decomp_base.csv")], "e40"))

MOVE = 0.02   # 2 points of K; below this a "significant" shift is not a price
ALPHA = 0.05


def holm(pvals: list[float]) -> list[bool]:
    """Holm-Bonferroni at ALPHA over one family; returns per-test rejection."""
    order = np.argsort(pvals)
    m = len(pvals)
    rej = [False] * m
    for rank, idx in enumerate(order):
        if pvals[idx] <= ALPHA / (m - rank):
            rej[idx] = True
        else:
            break
    return rej


def main() -> None:
    rows = []
    for func, tag, label, arm_paths, ref_label, ref_paths, cycle in CONTRASTS:
        arm = merged(func, *arm_paths)
        ref = merged(func, *ref_paths)
        for eps in sorted(set(arm) & set(ref), reverse=True):
            seeds = sorted(set(arm[eps]) & set(ref[eps]))
            if len(seeds) < 5:
                print(f"SKIP {func} {label} eps={eps:g}: {len(seeds)} shared seeds")
                continue
            cell = {"func": func, "budget": tag, "contrast": label,
                    "ref": ref_label, "eps": eps, "cycle": cycle,
                    "n_seeds": len(seeds)}
            for col, name in (("pr", "pr"), ("visited", "cov"), ("loss", "loss")):
                x = np.array([arm[eps][s][col] for s in seeds])
                y = np.array([ref[eps][s][col] for s in seeds])
                wtl, p, a = paired(x, y)
                cell[f"{name}_ref"] = round(float(np.mean(y)), 4)
                cell[f"{name}_arm"] = round(float(np.mean(x)), 4)
                cell[f"{name}_d"] = round(float(np.mean(x - y)), 4)
                cell[f"{name}_wtl"] = wtl
                cell[f"{name}_p"] = float(p)
                cell[f"{name}_a12"] = round(a, 2)
            rows.append(cell)

    # Holm within each contrast's own family (its eps levels x 2 components).
    # A single family over all 109 cells cannot be passed by anything: the
    # two-sided Wilcoxon on 15 seeds floors at p ~ 6e-4, below 0.05/218, so a
    # global correction would also reject the 15/0/0 results this project
    # already treats as established. The family that one adoption call spans is
    # one arm on one function, so that is the family corrected here.
    fam: dict = {}
    for r in rows:
        fam.setdefault((r["func"], r["contrast"]), []).append(r)
    for cells in fam.values():
        rej = holm([c[f"{k}_p"] for c in cells for k in ("cov", "loss")])
        for i, c in enumerate(cells):
            c["cov_holm"], c["loss_holm"] = rej[2 * i], rej[2 * i + 1]

    for r in rows:
        flat = r["pr_p"] >= ALPHA
        movers = [c for c in ("cov", "loss")
                  if r[f"{c}_p"] < ALPHA and abs(r[f"{c}_d"]) >= MOVE]
        movers_h = [c for c in ("cov", "loss")
                    if r[f"{c}_holm"] and abs(r[f"{c}_d"]) >= MOVE]
        r["silent"] = bool(flat and movers)
        r["silent_holm"] = bool(flat and movers_h)
        r["cancelling"] = bool(r["silent"] and len(movers) == 2
                               and np.sign(r["cov_d"]) == np.sign(r["loss_d"]))
        r["movers"] = "+".join(movers) or "-"
        # Which channel a *visible* PR move came through. Entry 60's design
        # rule says a report-side gain only exists while the reported set is
        # under the cap, so an arm that moves PR with no coverage gain is
        # buying something structurally different from one that moves coverage.
        pr_moves = r["pr_p"] < ALPHA and abs(r["pr_d"]) >= MOVE
        cov_moves = "cov" in movers
        loss_moves = "loss" in movers
        r["channel"] = ("-" if not pr_moves else
                        "mixed" if cov_moves and loss_moves else
                        "coverage" if cov_moves else
                        "report" if loss_moves else "sub-threshold")

    out = os.path.join(os.path.dirname(__file__), "cells.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} cells "
          f"({len({(r['func'], r['budget'], r['contrast']) for r in rows})} contrasts) "
          f"-> {out}\n")
    hdr = (f"{'func':<15}{'bud':>5}{'eps':>8}  {'contrast':<28}"
           f"{'dPR':>8}{'pPR':>7}{'dcov':>8}{'pcov':>7}"
           f"{'dloss':>8}{'ploss':>7}  flags")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flags = []
        if r["silent"]:
            flags.append("SILENT:" + r["movers"])
        if r["cancelling"]:
            flags.append("CANCEL")
        if r["silent_holm"]:
            flags.append("holm")
        print(f"{r['func']:<15}{r['budget']:>5}{r['eps']:>8.0e}  {r['contrast']:<28}"
              f"{r['pr_d']:>8.3f}{r['pr_p']:>7.3f}{r['cov_d']:>8.3f}{r['cov_p']:>7.3f}"
              f"{r['loss_d']:>8.3f}{r['loss_p']:>7.3f}  {' '.join(flags)}")

    judged = [r for r in rows if r["eps"] <= 1e-3]     # entry 28's rule
    def count(rs, key):
        return sum(1 for r in rs if r[key])
    print(f"\nsilent-trade cells: {count(rows, 'silent')}/{len(rows)} all eps, "
          f"{count(judged, 'silent')}/{len(judged)} at judging eps (<=1e-3); "
          f"Holm-surviving {count(rows, 'silent_holm')}/{len(rows)} and "
          f"{count(judged, 'silent_holm')}/{len(judged)}")
    print(f"cancelling (entry-80 signature): {count(rows, 'cancelling')} all eps, "
          f"{count(judged, 'cancelling')} at judging eps")
    arms = {(r["func"], r["contrast"]) for r in judged if r["silent"]}
    print(f"distinct arms with a silent trade at judging eps: {len(arms)}")
    for a in sorted(arms):
        print(f"  {a[0]}  {a[1]}")

    print("\nchannel of every *visible* PR move at judging eps "
          "(what the second component buys even when PR is not blind):")
    chan: dict = {}
    for r in judged:
        if r["channel"] in ("-",):
            continue
        chan.setdefault(r["channel"], []).append(
            f"{r['func'].split('-')[0]} {r['budget']} {r['contrast']} "
            f"eps={r['eps']:g} dPR={r['pr_d']:+.3f} "
            f"dcov={r['cov_d']:+.3f} dloss={r['loss_d']:+.3f}")
    absorbed = [r for r in judged
                if r["cov_p"] < ALPHA and r["cov_d"] >= MOVE]
    if absorbed:
        print("\nabsorption of a coverage gain by the report trim "
              "(dloss / dcov at judging eps; 1.0 = the whole gain is eaten, "
              "as in entry 80; <=0 = nothing lost):")
        for r in sorted(absorbed, key=lambda r: -r["loss_d"] / r["cov_d"]):
            print(f"    {r['loss_d'] / r['cov_d']:>+7.2f}  "
                  f"{r['func'].split('-')[0]} {r['budget']} {r['contrast']} "
                  f"eps={r['eps']:g}  dcov={r['cov_d']:+.3f} "
                  f"dloss={r['loss_d']:+.3f} -> dPR={r['pr_d']:+.3f}")

    for k in ("report", "coverage", "mixed", "sub-threshold"):
        v = chan.get(k, [])
        print(f"  {k:<14} {len(v):>3} cells")
        for line in v:
            print(f"      {line}")


if __name__ == "__main__":
    main()
