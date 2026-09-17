#!/usr/bin/env python3
"""e137: does the shipped NMMSO reproduce Fieldsend 2014 table IV once
``swarm_size`` is the published ``10*D``? Applies the gate registered in
prereg.md (and in queue item 2 before that).

Gate, verbatim from the queue: 4 or more of the 5 CEC2013 functions inside
+-0.10 of the published PR@1e-5 means the wiring passes and the new-suite
re-measurement (c) may proceed. 2 or more outside means ``swarm_size`` was only
part of the cause.

``stats`` and the published table are imported from entry 136's analyze.py
rather than redefined -- same estimator, same constants, no new statistic
(prereg.md section 2).
"""
from __future__ import annotations
import csv
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "e136_analyze", HERE.parent / "e136" / "analyze.py")
_e136 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e136)
stats, PUBLISHED, BAND = _e136.stats, _e136.PUBLISHED, _e136.MOVE

ORDER = ["N15-CF4-3D", "N17-CF4-5D", "N18-CF3-10D", "N19-CF4-10D",
         "N20-CF4-20D"]
# The same functions before the fix (swarm_size = 10), recomputed from the
# stored CSVs, for the before/after column. NOT paired by seed -- different
# seed bands -- so this column is a shift in means, not a paired test.
#   N15 0.6667 (e136 base_N15, n=3)   N17 0.4167 (e136 base_N17, n=3)
#   N18 0.4000 (e135 n18, n=15)       N19 0.0917 (e136 base_N19, n=15;
#                                      e135 n19 gave 0.0333 on other seeds)
#   N20 0.0000 (e135 n20, n=10)
OLD = {"N15-CF4-3D": 0.6667, "N17-CF4-5D": 0.4167, "N18-CF3-10D": 0.4000,
       "N19-CF4-10D": 0.0917, "N20-CF4-20D": 0.0000}


def load() -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for p in sorted(HERE.glob("gate_*.csv")):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["rule"] != "current" or r["method"] != "NMMSO":
                    continue
                by.setdefault(r["function"], []).append(r)
    return by


def main() -> None:
    by = load()
    if not by:
        sys.exit("no gate CSVs yet")

    print("e137 -- NMMSO with swarm_size = 10*D (published setting), CEC2013, "
          "official budget 400,000\n")
    print(f"{'function':<14}{'D':>3}{'K':>3}{'n':>3}{'PR@1e-5':>9}{'SD':>8}"
          f"{'SE':>8}{'pub':>7}{'diff':>8}{'|rep|':>7}{'old':>8}"
          f"{'gain':>8}  verdict")
    print("-" * 104)
    inside = 0
    lines = []
    for f in ORDER:
        rs = by.get(f, [])
        if not rs:
            print(f"{f:<14}  -- missing --")
            continue
        pr5 = [float(r["pr_1e-5"]) for r in rs]
        rep = [int(r["n_reported"]) for r in rs]
        m, sd, se = stats(pr5)
        pub = PUBLISHED[f]
        diff = m - pub
        ok = abs(diff) <= BAND
        inside += int(ok)
        dim = int(f.split("-")[-1].rstrip("D"))
        verdict = "inside +-0.10" if ok else "OUTSIDE"
        if diff > BAND:
            verdict = "OUTSIDE (over)"
        print(f"{f:<14}{dim:>3}{int(rs[0]['n_optima']):>3}{len(rs):>3}"
              f"{m:>9.4f}{sd:>8.4f}{se:>8.4f}{pub:>7.3f}{diff:>+8.4f}"
              f"{stats([float(v) for v in rep])[0]:>7.1f}"
              f"{OLD[f]:>8.4f}{m - OLD[f]:>+8.4f}  {verdict}")
        lines.append((f, m, pub, diff, ok))

    print("\nall five accuracy levels (mean over seeds):")
    lv = ["pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]
    print(f"{'function':<14}" + "".join(f"{l.replace('pr_', ''):>9}"
                                        for l in lv))
    for f in ORDER:
        rs = by.get(f, [])
        if rs:
            print(f"{f:<14}" + "".join(
                f"{stats([float(r[l]) for r in rs])[0]:>9.4f}" for l in lv))

    print(f"\ninside the +-{BAND:.2f} band: {inside} of {len(lines)}")
    if inside >= 4:
        print("VERDICT: the wiring PASSES the gate (>=4 of 5 inside). "
              "Queue item 2 (c) -- re-measure the new suite -- may proceed.")
    else:
        print(f"VERDICT: refutation (a) FIRES ({len(lines) - inside} of "
              f"{len(lines)} outside the band). `swarm_size` is only part of "
              "the cause; max_evol/tol_val were checked at run time and match "
              "the paper (100 / 1e-6), so the remainder is elsewhere.")


if __name__ == "__main__":
    main()
