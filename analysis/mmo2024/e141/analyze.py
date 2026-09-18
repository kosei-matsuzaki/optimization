#!/usr/bin/env python3
"""Entry 141 — does the D-independent hard-coded population of 30 bite?

Reads the raw per-run CSVs written by scripts/niching_baseline.py, pairs each
audit arm against its shipped default on (function, seed), and applies the
statistic fixed in prereg.md: Wilcoxon signed-rank on PR@1e-5 over the 50 pairs,
with the mean difference and the rank-biserial correlation beside it.

No new statistic is defined here (queue rule); scipy's wilcoxon is the same test
entry 115's analyze.py uses, and rank-biserial is read off its statistic.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
# The 20 per-(function, arm) CSVs this cycle's run.sh writes are folded into one
# gzipped table in the same cycle (CLAUDE.md retention rule); `raw/` is read
# only if it is still there, so the script works during a run and after the fold.
RAW = HERE / "raw"
RUNS = HERE / "runs.csv.gz"
ACC = ["pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]
PRIMARY = "pr_1e-5"
PAIRS = [("NCDE", "NCDE-p300"), ("r3pso", "r3pso-p300")]


def load() -> pd.DataFrame:
    if RAW.is_dir():
        frames = [pd.read_csv(p) for p in sorted(RAW.glob("*.csv"))
                  if p.stat().st_size > 0]
    elif RUNS.exists():
        frames = [pd.read_csv(RUNS)]
    else:
        sys.exit("neither raw/ nor runs.csv.gz is present")
    if not frames:
        sys.exit("no rows")
    df = pd.concat(frames, ignore_index=True)
    return df[df["rule"] == "current"]


def rank_biserial(d: np.ndarray) -> float:
    """r_b = (W+ - W-) / (W+ + W-) over the non-zero differences."""
    nz = d[d != 0]
    if nz.size == 0:
        return 0.0
    r = pd.Series(np.abs(nz)).rank().to_numpy()
    return float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())


def main() -> None:
    df = load()
    print(f"cells: {len(df)} rows, "
          f"{df['function'].nunique()} functions x {df['method'].nunique()} arms "
          f"x {df['seed'].nunique()} seeds\n")

    # per-(function, arm) means, all accuracy levels -- the breakdown a reader
    # needs to see which function moved, not just the pooled sign.
    piv = (df.groupby(["function", "method"])[ACC].mean().round(4))
    print(piv.to_string(), "\n")

    rows = []
    for base, arm in PAIRS:
        b = df[df["method"] == base].set_index(["function", "seed"]).sort_index()
        a = df[df["method"] == arm].set_index(["function", "seed"]).sort_index()
        common = b.index.intersection(a.index)          # prereg: paired only
        if len(common) == 0:
            print(f"{base}: no completed pairs yet"); continue
        for acc in ACC:
            d = (a.loc[common, acc] - b.loc[common, acc]).to_numpy()
            if np.allclose(d, 0):
                p, stat = 1.0, 0.0
            else:
                stat, p = wilcoxon(d, alternative="two-sided")
            wins = int((d > 0).sum()); ties = int((d == 0).sum())
            rows.append(dict(base=base, arm=arm, acc=acc, n_pairs=len(common),
                             base_mean=round(b.loc[common, acc].mean(), 4),
                             arm_mean=round(a.loc[common, acc].mean(), 4),
                             mean_diff=round(float(d.mean()), 4),
                             w_l_t=f"{wins}/{len(d)-wins-ties}/{ties}",
                             p=float(f"{p:.5g}"),
                             rb=round(rank_biserial(d), 3)))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    print("\n-- prereg verdict on the primary level "
          f"({PRIMARY}), R1/R2/R3 of prereg.md --")
    for base, arm in PAIRS:
        r = out[(out.base == base) & (out.acc == PRIMARY)]
        if r.empty:
            continue
        r = r.iloc[0]
        md, p = r["mean_diff"], r["p"]
        if p >= 0.05 and abs(md) < 0.05:
            v = "R1 fired: H refuted -- the shipped 30 is not a NMMSO-type defect here"
        elif p < 0.05 and md > 0:
            v = "R2 fired: H supported -- pin the published constant before step (b)"
        elif p < 0.05 and md < 0:
            v = "R3 fired: 300 is WORSE -- the shipped 30 is favourable to this baseline"
        else:
            v = "undecided at this n (prereg: more seeds, do not loosen the band)"
        print(f"  {base}: mean_diff={md:+.4f} p={p:.5g} n={int(r['n_pairs'])} -> {v}")

    piv.to_csv(HERE / "per_function_means.csv")
    out.to_csv(HERE / "paired.csv", index=False)


if __name__ == "__main__":
    main()
