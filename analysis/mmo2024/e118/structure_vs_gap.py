"""e118 middle branch: does the RL-minus-niching gap track the placement structure?

The pre-registration's middle branch (5-11 problems significant) asks for this:
if the gap does not correlate with whether a problem has structure, then
structure is not the rate-limiting thing.
Reads e116/ranking_d10.csv (legal rule, seed 0) and e118/structure_d10.csv.
"""
import csv
import pathlib

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

HERE = pathlib.Path(__file__).resolve().parent
rank = {}
with open(HERE.parent / "e116/ranking_d10.csv") as f:
    for r in csv.DictReader(f):
        if r["arm"] != "legal":
            continue
        rank.setdefault(r["problem"][:3], {})[r["method"]] = (
            float(r["mpr"]), float(r["mean_f1"]), float(r["score"]))

st = {}
with open(HERE / "structure_d10.csv") as f:
    for r in csv.DictReader(f):
        st[r["pid"]] = r

print("pid  sig_a  ratio_a  ratio_b   RL_score  MC_score  NM_score  RL-MC   RL-NM")
gapA, gapB, ratios, gaps = [], [], [], []
for pid in [f"M{p:02d}" for p in range(1, 17)]:
    rl, mc, nm = rank[pid]["Restart-Lander"], rank[pid]["MC-ESO"], rank[pid]["NMMSO"]
    g_mc, g_nm = rl[2] - mc[2], rl[2] - nm[2]
    sig = st[pid]["sig_a"] == "True"
    ra = float(st[pid]["T_a"]) / float(st[pid]["null_med"])
    (gapA if sig else gapB).append(g_mc)
    ratios.append(ra); gaps.append(g_mc)
    print(f"{pid}  {str(sig):5s}  {ra:.3f}    {float(st[pid]['ratio']):.3f}   "
          f"{rl[2]:.4f}    {mc[2]:.4f}    {nm[2]:.4f}   {g_mc:+.4f} {g_nm:+.4f}")

mw = mannwhitneyu(gapA, gapB, alternative="two-sided")
a12 = float(mw.statistic / (len(gapA) * len(gapB)))
sp = spearmanr(ratios, gaps)
print(f"\nRL-MC score gap: structured (n={len(gapA)}) mean {np.mean(gapA):+.4f} vs "
      f"unstructured (n={len(gapB)}) mean {np.mean(gapB):+.4f}; "
      f"MWU p={mw.pvalue:.4g}, A12={a12:.3f}")
print(f"Spearman(clustering ratio T/null, RL-MC gap) rho={sp.statistic:+.3f} p={sp.pvalue:.4g}")
print(f"all 16 gaps positive: {all(g > 0 for g in gaps)}  min={min(gaps):+.4f}")
