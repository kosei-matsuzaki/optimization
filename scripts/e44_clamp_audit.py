#!/usr/bin/env python3
"""Entry 44: does a clamp that only stops the endpoint keep the variant intact?

Entry 37 required a clamp on `hunt_level_tol = rel_level / f_init_scale` because
one run in 960 (F07-StepEllipsoidal seed 19) starts on `f = 0`, pins
`f_init_scale` at its floor 1e-300 and sends the quotient to 1e294. Entry 43
then showed the clamp entry 37 wrote down -- `min(quotient, the shipped default
1e-6)` -- binds on *every* problem with `f_init_scale < 1`, which is where the
variant's only significant gain lives, and returns it to base there.

This compares the two clamp shapes that touch only the endpoint against the
stored base and unclamped arms of entry 37 (`analysis/hm/relgate24_*.csv.gz`),
cell by cell (function x seed x budget). Three questions, one table each:

  safety   does the clamped arm's `best_f` still match base in every cell?
  binding  in which cells does the clamp actually change `hunt_level_tol`?
  endpoint at the endpoint cell, does the clamped arm behave like base (the
           level clause stops releasing everything) rather than like the
           unclamped quotient?

Usage:
  python3 scripts/e44_clamp_audit.py --arms fis,cap
"""
from __future__ import annotations
import argparse
import csv
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HM = ROOT / "analysis" / "hm"
E44 = HM / "e44"
KEY = ("function", "seed", "evals")


def _load_stored(arm: str) -> dict:
    out = {}
    for tag, ev in (("5k", "5000"), ("20k", "20000")):
        with gzip.open(HM / f"relgate24_{arm}_{tag}.csv.gz", "rt") as fh:
            for r in csv.DictReader(fh):
                out[(r["function"], r["seed"], ev)] = r
    return out


def _load_arm(arm: str) -> dict:
    out = {}
    with gzip.open(E44 / f"bbob_{arm}.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            out[(r["function"], r["seed"], r["evals"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="fis,cap")
    args = ap.parse_args()

    base, rel = _load_stored("base"), _load_stored("rel")
    print(f"stored base cells {len(base)}, stored unclamped-rel cells {len(rel)}\n")

    for arm in args.arms.split(","):
        cells = _load_arm(arm)
        common = sorted(set(cells) & set(base))
        # 1. safety: best_f against base, exact string compare on the printed
        #    repr (entry 37's "byte-identical" standard).
        bad = [k for k in common if cells[k]["best_f"] != base[k]["best_f"]]
        # 2. binding: where the clamp moved hunt_level_tol off the raw quotient.
        bind = [k for k in common
                if float(cells[k]["eff_tol"]) != float(rel[k]["eff_tol"])]
        # 3. did the clamped arm stay identical to the unclamped arm elsewhere?
        drift = [k for k in common if k not in set(bind)
                 and cells[k]["n_level_release"] != rel[k]["n_level_release"]]
        print(f"=== arm {arm}: {len(common)} cells")
        print(f"  best_f != base                    : {len(bad)}")
        print(f"  clamp binds (eff_tol != unclamped) : {len(bind)}")
        print(f"  non-binding cells that still drift : {len(drift)}")
        for k in bind:
            c, r_, b = cells[k], rel[k], base[k]
            print(f"    bind {k[0]:<22} seed={k[1]:<3} ev={k[2]:<6} "
                  f"fis={float(c['f_init_scale']):<10.3g} "
                  f"tol {float(r_['eff_tol']):.4g} -> {float(c['eff_tol']):.4g} "
                  f"(base {float(b['eff_tol']):.4g}) | "
                  f"lvl_rel base={b['n_level_release']} "
                  f"unclamped={r_['n_level_release']} clamped={c['n_level_release']} | "
                  f"best_f base={b['best_f']} clamped={c['best_f']}")
        for k in bad[:10]:
            print(f"    MISMATCH {k}: base={base[k]['best_f']} arm={cells[k]['best_f']}")
        print()


if __name__ == "__main__":
    main()
