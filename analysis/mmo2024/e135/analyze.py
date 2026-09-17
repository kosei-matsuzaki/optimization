#!/usr/bin/env python3
"""e135: does this harness's NMMSO reproduce the published CEC2013 PR at D=10?

Reads the per-seed CSVs written by scripts/niching_baseline.py (one per
function, official budget, rule=current) and compares the mean `pr_1e-5`
against the published table this project recovered in entry 134 (Fieldsend
2014, table IV; 50 runs, official budget).

The pre-registered decision is in prereg.md: |measured - published| <= 0.10 on
all three functions clears the wiring. This script only reports; it does not
pick the threshold.

The published numbers are 50-run means and ours are 10-15, so the mean is
printed with its standard error and with a normal-approximation 95% interval,
and the gap is stated in units of that SE as well as in absolute PR.
"""
from __future__ import annotations
import csv
import math
import sys
from pathlib import Path

# Published PR@1e-5, Fieldsend 2014 table IV, 50 runs at the official budget.
# Recovered by entry 134 (ORE Exeter handle 10871/15247).
PUBLISHED = {
    "N18-CF3-10D": 0.633,
    "N19-CF4-10D": 0.443,
    "N20-CF4-20D": 0.178,
}
TOL = 0.10          # pre-registered decision half-width
HERE = Path(__file__).resolve().parent


def read(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["rule"] != "current" or r["method"] != "NMMSO":
                continue
            rows.setdefault(r["function"], []).append(r)
    return rows


def stats(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd, sd / math.sqrt(n) if n else 0.0


def main() -> None:
    paths = sorted(HERE.glob("n*.csv"))
    if not paths:
        sys.exit("no per-function CSVs in " + str(HERE))
    by_func: dict[str, list[dict]] = {}
    for p in paths:
        for f, rs in read(p).items():
            by_func.setdefault(f, []).extend(rs)

    levels = ["pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]
    print("NMMSO, CEC2013 niching, official budget (400,000 evaluations), "
          "rule=current, rho path\n")
    print(f"{'function':<14}{'n':>3}{'evals':>8}{'K':>3}{'|rep|':>7}"
          + "".join(f"{l.replace('pr_',''):>8}" for l in levels)
          + f"{'SE(1e-5)':>10}{'pub':>7}{'gap':>8}{'gap/SE':>8}  verdict")
    print("-" * 108)

    verdicts = {}
    for name in sorted(by_func):
        rs = by_func[name]
        prs = {l: [float(r[l]) for r in rs] for l in levels}
        m5, sd5, se5 = stats(prs["pr_1e-5"])
        pub = PUBLISHED.get(name)
        rep = sum(int(r["n_reported"]) for r in rs) / len(rs)
        gap = m5 - pub if pub is not None else float("nan")
        v = ("within" if abs(gap) <= TOL
             else ("HIGH" if gap > 0 else "LOW -- refutation (a)"))
        verdicts[name] = (m5, sd5, se5, pub, gap, v, len(rs))
        print(f"{name:<14}{len(rs):>3}{int(rs[0]['evals']):>8}"
              f"{int(rs[0]['n_optima']):>3}{rep:>7.1f}"
              + "".join(f"{stats(prs[l])[0]:>8.3f}" for l in levels)
              + f"{se5:>10.4f}{pub:>7.3f}{gap:>+8.3f}"
              + (f"{gap / se5:>8.2f}" if se5 > 0 else f"{'inf':>8}")
              + f"  {v}")

    print("\n95% normal-approximation interval on the measured mean PR@1e-5, "
          "against the published point:")
    for name, (m5, sd5, se5, pub, gap, v, n) in sorted(verdicts.items()):
        lo, hi = m5 - 1.96 * se5, m5 + 1.96 * se5
        covers = "covers" if lo <= pub <= hi else "EXCLUDES"
        print(f"  {name:<14} {m5:.3f} +- {se5:.4f}  "
              f"[{lo:.3f}, {hi:.3f}]  {covers} published {pub:.3f}  "
              f"(SD {sd5:.3f}, n={n})")

    bad = [k for k, v in verdicts.items() if v[5].startswith("LOW")]
    high = [k for k, v in verdicts.items() if v[5] == "HIGH"]
    print()
    if bad:
        print("PRE-REGISTERED REFUTATION (a) FIRED on: " + ", ".join(bad))
    elif high:
        print("Refutation (b) (measured high) on: " + ", ".join(high)
              + " -- conditions differ, no conclusion this cycle.")
    else:
        print(f"All {len(verdicts)} functions within +-{TOL:.2f} of the "
              "published values: the wiring is sound at D=10.")


if __name__ == "__main__":
    main()
