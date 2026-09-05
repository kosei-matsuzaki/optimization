#!/usr/bin/env python3
"""Pair a `gate_power.py` probe arm against base on BBOB-24 dim2.

Written for entry 47 (is the c = 1.0 arm as safe on BBOB-24 dim2 as c = 0.1
was?) and generalised in entry 56 so any arm can be run through the same gate
without a new throwaway script -- pass the arm's per-run CSVs with `--arm`.

Entry 37 measured the eps-relative release rule at c = 0.1 (`rel_level = 1e-6`)
and got `best_f` byte-identical to base in all 960 cells. Entry 46 then found
c = 0.1 sits inside the corner: c = 1.0 (`L = eps_target = 1e-5`) pays a
strictly smaller coarse-level cost on both Shubert functions. c = 1.0 releases
hunts ten times earlier, so the window in which a release could still move
`best_f` is wider -- entry 37's lesson is not to settle that by argument.

This reads the per-run probe CSVs and reports, per function and per budget:
  same_best_f  cells where the arm's best_f equals base's, bit for bit
  post_gain>0  runs where best_so_far improved after the first exhaustion
               (the only route by which any post-exhaustion rule can reach the
               scored answer)
  lvl_rel      mean level-clause releases per run: a tie with zero firings is
               entry 33's case (a), which is not evidence of safety.
  SR@1e-10     both arms' own success rate, the number CLAUDE.md pins. `best_f`
               equality makes the two identical by construction, so a gap here
               is the rejection condition firing, not a separate finding.

Usage (entry 47's own arms are the defaults):
  python3 scripts/e47_analyze.py
  python3 scripts/e47_analyze.py --label sig10 \
      --arm 5000 analysis/hm/e56/sig10_5k.csv \
      --arm 20000 analysis/hm/e56/sig10_20k.csv
"""
from __future__ import annotations
import argparse
import csv
import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E47 = ROOT / "analysis/hm/e47"

#  base arms are reused across entries (37 / 44 / 47 / 56): same seeds, same
#  budgets, and base is always the shipped optimiser, so re-running it would
#  only burn the budget on numbers already on disk.
BASE = {5000: "analysis/hm/relgate24_base_5k.csv.gz",
        20000: "analysis/hm/relgate24_base_20k.csv.gz"}
DEFAULT_ARMS = {5000: E47 / "bbob_c100_5k.csv.gz",
                20000: E47 / "bbob_c100_20k.csv.gz"}


def load(path: Path) -> dict:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return {(r["function"], int(r["seed"])): r for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=2, action="append", metavar=("EVALS", "CSV"),
                    help="budget and per-run CSV for the arm under test; repeat "
                         "per budget. Omit for entry 47's own arm.")
    ap.add_argument("--label", default="c=1.0")
    ap.add_argument("--sr-thresh", type=float, default=1e-10)
    args = ap.parse_args()
    arms = ({int(e): Path(p) for e, p in args.arm} if args.arm else DEFAULT_ARMS)

    tot = {"cells": 0, "same": 0, "gain": 0, "exh": 0}
    print(f"arm: {args.label}")
    for ev in sorted(arms):
        base = load(ROOT / BASE[ev])
        arm_path = arms[ev] if arms[ev].is_absolute() else ROOT / arms[ev]
        if not arm_path.exists():
            sys.exit(f"missing {arm_path}")
        arm = load(arm_path)
        print(f"\n=== {ev} evals ===")
        print(f"{'function':<24}{'n':>4}{'same_best_f':>13}{'post_gain>0':>13}"
              f"{'lvl_rel_base':>14}{'lvl_rel_arm':>13}{'med_eff_tol':>13}")
        print("-" * 94)
        n_ok = {"base": 0, "arm": 0, "n": 0}
        for name in sorted({k[0] for k in arm}):
            keys = sorted(k for k in arm if k[0] == name)
            same = gain = exh = 0
            lb = la = 0.0
            tols = []
            for k in keys:
                b, a = base[k], arm[k]
                same += float(a["best_f"]) == float(b["best_f"])
                if a["exhausted"] == "True":
                    exh += 1
                    gain += float(a["post_gain"]) > 0
                lb += float(b["n_level_release"])
                la += float(a["n_level_release"])
                tols.append(float(a["eff_tol"]))
                n_ok["base"] += float(b["best_f"]) <= args.sr_thresh
                n_ok["arm"] += float(a["best_f"]) <= args.sr_thresh
                n_ok["n"] += 1
            n = len(keys)
            tols.sort()
            print(f"{name:<24}{n:>4}{same:>13}{gain:>13}"
                  f"{lb / n:>14.1f}{la / n:>13.1f}{tols[n // 2]:>13.3g}")
            tot["cells"] += n
            tot["same"] += same
            tot["gain"] += gain
            tot["exh"] += exh
        print(f"SR@{args.sr_thresh:g}: base {n_ok['base'] / n_ok['n']:.4f}  "
              f"arm {n_ok['arm'] / n_ok['n']:.4f}  (n = {n_ok['n']})")
    print(f"\nTOTAL: {tot['same']}/{tot['cells']} cells with best_f == base, "
          f"{tot['gain']} runs with post_gain > 0 "
          f"(of {tot['exh']} that exhausted)")


if __name__ == "__main__":
    main()
