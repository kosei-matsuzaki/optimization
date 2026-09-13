"""e118 (c) continued: the 1e-5 ('deep') definition, plus the absorption check.

Two things the first pass left open:
  * the queue's six low-support problems are the ones low on the DEEP (f <= 1e-5)
    definition, so repeat the placement comparison with that definition;
  * `land_opt` credits the nearest optimum, so within a tight pair only one of
    the two can be credited.  Count how often an un-landed optimum's own nearest
    neighbour was landed -- that share is definitional, not a search failure.
"""
import csv
import gzip
import pathlib

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIDS = [f"M{p:02d}" for p in range(1, 17)]


def read_rl(pid):
    p = ROOT / f"analysis/mmo2024/e115/descents/{pid}-D10-PIN01_seed0.csv.gz"
    with gzip.open(p, "rt") as f:
        return [(int(r["land_opt"]), float(r["best_f"])) for r in csv.DictReader(f)]


z = np.load(HERE / "optima_d10_pin01.npz")
rows, pu, pl, absorbed, tot = [], [], [], 0, 0
print("pid   K  deep  n_un  nn_un   nn_la   MWU_p   absorbed/un")
for pid in PIDS:
    pts = z[pid]; K = pts.shape[0]
    d = cdist(pts, pts); np.fill_diagonal(d, np.inf)
    nn, nn_idx = d.min(axis=1), d.argmin(axis=1)
    deep = {k for k, v in read_rl(pid) if v <= 1e-5}
    un = sorted(set(range(K)) - deep)
    la = sorted(deep)
    ab = sum(1 for i in un if nn_idx[i] in deep)
    absorbed += ab; tot += len(un)
    pu += list(nn[un]); pl += list(nn[la])
    if un and la:
        mw = mannwhitneyu(nn[un], nn[la], alternative="two-sided")
        p, a, b = float(mw.pvalue), float(nn[un].mean()), float(nn[la].mean())
    else:
        p = float("nan"); a = float(nn[un].mean()) if un else float("nan")
        b = float(nn[la].mean()) if la else float("nan")
    rows.append(dict(pid=pid, K=K, deep_support=len(deep) / K, n_unlanded=len(un),
                     unlanded=";".join(map(str, un)), nn_unlanded=a, nn_landed=b,
                     mwu_p=p, absorbed=ab))
    print(f"{pid}  {K:2d}  {len(deep)/K:.2f}   {len(un):2d}  {a:7.4f} {b:7.4f}  {p:.4f}   {ab}/{len(un)}")

with open(HERE / "landing_deep_d10.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

mw = mannwhitneyu(pu, pl, alternative="two-sided")
print(f"\npooled deep: {len(pu)} un-reached vs {len(pl)} reached, "
      f"NN mean {np.mean(pu):.4f} vs {np.mean(pl):.4f}, p={mw.pvalue:.4g}, "
      f"A12={mw.statistic/(len(pu)*len(pl)):.3f}")
print(f"absorption: {absorbed}/{tot} = {absorbed/tot:.3f} of un-reached optima "
      f"have their own nearest neighbour in the reached set")
# group split
for lab, sl in (("A (PID 1-8)", slice(0, 8)), ("B (PID 9-16)", slice(8, 16))):
    r = rows[sl]
    print(f"{lab}: deep support mean {np.mean([x['deep_support'] for x in r]):.3f}, "
          f"absorbed {sum(x['absorbed'] for x in r)}/{sum(x['n_unlanded'] for x in r)}")
