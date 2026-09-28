#!/usr/bin/env python3
"""Paired Wilcoxon on best_f, seed by seed, between the committed default
(softmax_beta=5.0, arm HEAD) and softmax_beta=0 (arm SM0, = the pre-2026-08-24
parent selection).  Reference = MC-ESO (the default), alpha = 0.05, A12 shown,
per the project's three-metric rule.  Reads only the per-run stats already
written by quick_check.py -- no extra optimisation runs.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
from scipy.stats import wilcoxon

DEF_DIR, SM0_DIR = Path(sys.argv[1]), Path(sys.argv[2])


def runs(d: Path, func: str) -> dict[int, float]:
    with open(d / "stats" / f"{func}.csv", newline="") as fh:
        return {int(r["seed"]): float(r["best_f"]) for r in csv.DictReader(fh)}


funcs = sorted(p.stem for p in (DEF_DIR / "stats").glob("*.csv"))
print(f"{'function':<24}{'n':>3}{'def<b0':>8}{'b0<def':>8}{'tie':>5}"
      f"{'p_two_sided':>13}{'A12':>7}  verdict")
wins = losses = 0
for f in funcs:
    a, b = runs(DEF_DIR, f), runs(SM0_DIR, f)
    seeds = sorted(set(a) & set(b))
    x = [a[s] for s in seeds]      # default (beta=5)
    y = [b[s] for s in seeds]      # beta=0
    nd = sum(1 for i, j in zip(x, y) if i < j)   # default better (smaller f)
    nb = sum(1 for i, j in zip(x, y) if j < i)
    tie = len(seeds) - nd - nb
    # A12: probability a random default run beats a random beta=0 run
    gt = sum((1.0 if i < j else 0.5 if i == j else 0.0) for i in x for j in y)
    a12 = gt / (len(x) * len(y))
    if tie == len(seeds):
        p, verdict = 1.0, "identical"
    else:
        p = float(wilcoxon(x, y, zero_method="wilcox").pvalue)
        if p < 0.05:
            verdict = "DEFAULT WINS" if nd > nb else "DEFAULT LOSES"
            wins += nd > nb
            losses += nb > nd
        else:
            verdict = "-"
    print(f"{f:<24}{len(seeds):>3}{nd:>8}{nb:>8}{tie:>5}{p:>13.3e}{a12:>7.2f}  {verdict}")
print()
print(f"significant at alpha=0.05: default wins {wins}, default loses {losses}")
