#!/usr/bin/env python3
"""e136: read the arm CSVs and apply the decision registered in prereg.md.

Decision (registered, and taken verbatim from queue item 2):
  * an arm that moves mean PR@1e-5 on N19 by >= +0.10 against `base` is a
    wrapper defect;
  * all three arms inside +-0.05 of `base` means the wrapper is innocent and
    the claim moves to the limited form (c);
  * in between is "acts, but does not explain".

Welch's t is printed for reference only -- the registered threshold is on the
difference in means, not on a p-value, because n=5 per arm was chosen for the
0.10 effect and the per-run statistic takes only the values 0, 0.125, 0.250.
The `base` arm is also checked against entry 135's 15 seeds (refutation (ii)).
"""
from __future__ import annotations
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E135 = HERE.parent / "e135"
LEVELS = ["pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]
PUBLISHED = {          # Fieldsend 2014 table IV, 50 runs, official budget
    "N15-CF4-3D": 0.668,
    "N17-CF4-5D": 0.538,
    "N18-CF3-10D": 0.633,
    "N19-CF4-10D": 0.443,
    "N20-CF4-20D": 0.178,
}
MOVE, SAME = 0.10, 0.05        # registered thresholds


def stats(v: list[float]) -> tuple[float, float, float]:
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, sd, sd / math.sqrt(n)


def welch(a: list[float], b: list[float]) -> float:
    ma, sa, _ = stats(a)
    mb, sb, _ = stats(b)
    se = math.sqrt(sa ** 2 / len(a) + sb ** 2 / len(b))
    return (ma - mb) / se if se > 0 else float("nan")


def load(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    rows: list[dict] = []
    for p in sorted(HERE.glob("*_N*.csv")):
        rows.extend(load(p))
    if not rows:
        sys.exit("no arm CSVs yet")

    n19 = [r for r in rows if r["function"] == "N19-CF4-10D"]
    arms = sorted({r["arm"] for r in n19})

    print("e136 -- N19-CF4-10D, official budget 400,000, one wrapper detail off "
          "per arm\n")
    print(f"{'arm':>8}{'n':>3}" + "".join(f"{l.replace('pr_', ''):>8}"
                                          for l in LEVELS)
          + f"{'SD':>8}{'SE':>8}{'sup':>7}{'near.1':>8}{'|rep|':>7}"
          + f"{'req-bud':>9}{'clip':>6}{'modes':>7}{'sec':>7}")
    print("-" * 104)
    by_arm: dict[str, list[float]] = {}
    for a in arms:
        rs = [r for r in n19 if r["arm"] == a]
        pr5 = [float(r["pr_1e-5"]) for r in rs]
        by_arm[a] = pr5
        m, sd, se = stats(pr5)
        print(f"{a:>8}{len(rs):>3}"
              + "".join(f"{stats([float(r[l]) for r in rs])[0]:>8.4f}"
                        for l in LEVELS)
              + f"{sd:>8.4f}{se:>8.4f}"
              + f"{stats([float(r['hist_sup_pr_1e-5']) for r in rs])[0]:>7.3f}"
              + f"{stats([float(r['near_0p1']) for r in rs])[0]:>8.2f}"
              + f"{stats([float(r['n_reported']) for r in rs])[0]:>7.1f}"
              + f"{stats([float(r['requests']) - float(r['budget']) for r in rs])[0]:>9.1f}"
              + f"{stats([float(r['clipped']) for r in rs])[0]:>6.1f}"
              + f"{stats([float(r['modes_finite']) for r in rs])[0]:>7.1f}"
              + f"{stats([float(r['seconds']) for r in rs])[0]:>7.0f}")

    # refutation (ii): does this file's `base` reproduce entry 135's 15 seeds?
    e135 = []
    p = E135 / "n19.csv"
    if p.exists():
        e135 = [float(r["pr_1e-5"]) for r in load(p)
                if r["rule"] == "current" and r["method"] == "NMMSO"]
    if e135 and "base" in by_arm:
        m1, _, _ = stats(by_arm["base"])
        m0, sd0, _ = stats(e135)
        ok = abs(m1 - m0) <= 0.10
        print(f"\nfidelity gate (refutation (ii)): base {m1:.4f} (n="
              f"{len(by_arm['base'])}) vs entry 135 {m0:.4f} (n={len(e135)}, "
              f"SD {sd0:.4f})  diff {m1 - m0:+.4f}  -> "
              + ("PASS, arms are comparable" if ok
                 else "FAIL, the copied wrapper is not faithful"))
        pooled = by_arm["base"] + e135
    else:
        pooled = by_arm.get("base", [])

    print(f"\nregistered decision, against base pooled with entry 135 "
          f"(n={len(pooled)}, mean {stats(pooled)[0]:.4f}):")
    verdicts = {}
    for a in arms:
        if a == "base":
            continue
        d = stats(by_arm[a])[0] - stats(pooled)[0]
        v = ("WRAPPER DEFECT (>= +0.10)" if d >= MOVE else
             "acts but does not explain (+0.05..+0.10)" if d >= SAME else
             "innocent (within +-0.05)" if abs(d) <= SAME else
             f"moves the wrong way ({d:+.4f})")
        verdicts[a] = (d, v)
        print(f"  {a:>8}  diff {d:+.4f}  Welch t {welch(by_arm[a], pooled):+.2f}"
              f"   -> {v}")

    pub = PUBLISHED["N19-CF4-10D"]
    if "nocut" in by_arm:
        hit = stats(by_arm["nocut"])[0] >= pub - 0.10
        print(f"\nrefutation (i): nocut {stats(by_arm['nocut'])[0]:.4f} vs "
              f"published {pub:.3f} -> "
              + ("FIRED -- entry 135's budget conclusion was wrong"
                 if hit else "not fired"))

    print()
    if all("innocent" in v for _, v in verdicts.values()):
        print("ALL THREE WRAPPER DETAILS INNOCENT: the shortfall is in the "
              "search, i.e. in pynmmso itself -> limited claim (c).")
    else:
        print("At least one arm moved: " + "; ".join(
            f"{a} {d:+.4f}" for a, (d, _) in verdicts.items() if "innocent" not in _))

    # dimension ladder (prereg section 4)
    ladder = [r for r in rows if r["function"] != "N19-CF4-10D"]
    if ladder:
        print("\nOther CEC2013 functions, against Fieldsend 2014 table IV "
              "(arm is stated -- `base` is swarm_size 10, `sw10d` is 10*D):")
        print(f"{'function':<14}{'arm':>7}{'D':>3}{'n':>3}{'PR@1e-5':>10}"
              f"{'SD':>8}{'SE':>8}{'pub':>8}{'gap':>9}  verdict")
        for f, a in sorted({(r["function"], r["arm"]) for r in ladder}):
            rs = [r for r in ladder
                  if r["function"] == f and r["arm"] == a]
            v = [float(r["pr_1e-5"]) for r in rs]
            m, sd, se = stats(v)
            pub = PUBLISHED.get(f, float("nan"))
            gap = m - pub
            print(f"{f:<14}{a:>7}{rs[0]['dim']:>3}{len(rs):>3}{m:>10.4f}{sd:>8.4f}"
                  f"{se:>8.4f}{pub:>8.3f}{gap:>+9.4f}  "
                  + ("within +-0.10" if abs(gap) <= 0.10
                     else ("LOW" if gap < 0 else "HIGH")))


if __name__ == "__main__":
    main()
