"""e119 robustness (POST-HOC -- not the pre-registered decision, read as such).

Three things the first pass left open, each a threat to the reading of H1/H3:

  1. H1 is degenerate.  At the pre-registered tau = 0.6*null_med single linkage
     chains every group-A problem into exactly ONE cluster (sizes 5-8) and group B
     into none, so n=8, every point comes from one problem, and shuffling cluster
     labels within a problem is a no-op (the null band collapsed onto the observed
     value).  Sweep tau downward and report where, if anywhere, the design has power.

  2. H3's `attract` is confounded by Voronoi volume.  `land_opt` credits the NEAREST
     optimum, so a packed optimum owns a small cell and an isolated one owns a large
     cell.  Descents-per-optimum therefore measures cell volume as much as basin
     quality.  Control it: estimate each optimum's cell volume by Monte Carlo and
     report ENRICHMENT = observed share / cell-volume share.  Enrichment >= 1 for
     clustered optima is what reading (i) (a uniform sprinkle already harvests the
     cluster, because clusters are easy to hit by volume) actually predicts.

  3. H2 is dominated by a tautology: a deep descent sits ON an optimum, so its
     distance to the nearest optimum is 0 by construction.  Re-read H2 on the
     shallow (best_f > 1e-5) descents only.
"""
import csv
import gzip
import pathlib

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, spearmanr

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIDS = [f"M{p:02d}" for p in range(1, 17)]
RNG = np.random.default_rng(20260913)
LO, HI, D = -5.0, 5.0, 10
N_MC = 200000          # uniform points for the Voronoi cell-volume estimate
NEAR = 0.1             # a descent "converged onto" its credited optimum if dist < NEAR


def read_rl(pid):
    p = ROOT / f"analysis/mmo2024/e115/descents/{pid}-D10-PIN01_seed0.csv.gz"
    with gzip.open(p, "rt") as f:
        return [(int(r["land_opt"]), float(r["best_f"]), float(r["dist"]),
                 np.array([float(r[f"x{j}"]) for j in range(D)]))
                for r in csv.DictReader(f)]


def components(members, dmat, tau):
    rem, comps = set(members), []
    while rem:
        seed = rem.pop()
        comp, stack = [seed], [seed]
        while stack:
            u = stack.pop()
            for v in [v for v in list(rem) if dmat[u, v] < tau]:
                rem.discard(v)
                comp.append(v)
                stack.append(v)
        comps.append(sorted(comp))
    return comps


z = np.load(ROOT / "analysis/mmo2024/e118/optima_d10_pin01.npz")
nullmed = {r["pid"]: float(r["null_med"])
           for r in csv.DictReader(open(ROOT / "analysis/mmo2024/e118/structure_d10.csv"))}

OPT, DUMP, DMAT, NN = {}, {}, {}, {}
for pid in PIDS:
    OPT[pid] = z[pid]
    DUMP[pid] = read_rl(pid)
    d = cdist(z[pid], z[pid])
    np.fill_diagonal(d, np.inf)
    DMAT[pid], NN[pid] = d, d.min(axis=1)

# ---------------------------------------------------------------- 1. tau sweep
print("=== 1. tau sweep (POST-HOC): does H1 have power at any threshold?")
print("frac   tau_A  ncl_A ncl_B  sizes(min-max)  n_pts  rho      p       verdict")
sweep_rows = []
for frac in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
    cl = []
    for pid in PIDS:
        tau = frac * nullmed[pid]
        memb = [i for i in range(len(OPT[pid])) if NN[pid][i] < tau]
        hits = np.zeros(len(OPT[pid]), dtype=int)
        for lo, _bf, _d, _x in DUMP[pid]:
            hits[lo] += 1
        for c in components(memb, DMAT[pid], tau):
            if len(c) >= 2:
                cl.append(dict(pid=pid, frac=frac, size=len(c),
                               reach=float(np.mean(hits[c] > 0))))
    nA = sum(1 for c in cl if c["pid"] in PIDS[:8])
    nB = len(cl) - nA
    if len(cl) >= 3:
        rho, p = spearmanr([c["size"] for c in cl], [c["reach"] for c in cl])
        sizes = f"{min(c['size'] for c in cl)}-{max(c['size'] for c in cl)}"
    else:
        rho, p, sizes = float("nan"), float("nan"), "-"
    verdict = "(i)" if (p == p and p < 0.05 and rho > 0) else "(ii)/ns"
    print(f"{frac:.2f}  {frac*nullmed['M01']:5.2f}  {nA:4d} {nB:5d}  {sizes:14s} "
          f"{len(cl):5d}  {rho:7.4f} {p:7.4f}  {verdict}")
    sweep_rows.append(dict(frac=frac, n_cl_A=nA, n_cl_B=nB, n_cl=len(cl),
                           rho=rho, p=p, verdict=verdict))

# --------------------------------------------- 2. Voronoi-controlled enrichment
print("\n=== 2. enrichment = observed descent share / Voronoi cell-volume share")
print("    (reading (i) predicts enrichment >= 1 for clustered optima)")
print("pid   cell_cl  cell_iso | enr_cl  enr_iso | enr_cl(near)  enr_iso(near)")
rows = []
for pid in PIDS:
    pts, K = OPT[pid], len(OPT[pid])
    tau = 0.60 * nullmed[pid]
    memb = [i for i in range(K) if NN[pid][i] < tau]
    in_cl = {i for c in components(memb, DMAT[pid], tau) if len(c) >= 2 for i in c}

    U = RNG.uniform(LO, HI, size=(N_MC, D))
    own = cdist(U, pts).argmin(axis=1)
    cell = np.bincount(own, minlength=K) / N_MC          # Voronoi volume share

    hits = np.zeros(K)
    hits_near = np.zeros(K)
    for lo, _bf, dd, _x in DUMP[pid]:
        hits[lo] += 1
        if dd < NEAR:
            hits_near[lo] += 1
    sh = hits / hits.sum()
    sh_near = hits_near / max(hits_near.sum(), 1)
    enr = np.divide(sh, cell, out=np.full(K, np.nan), where=cell > 0)
    enr_near = np.divide(sh_near, cell, out=np.full(K, np.nan), where=cell > 0)

    cl = sorted(in_cl)
    iso = [i for i in range(K) if i not in in_cl]
    for i in range(K):
        rows.append(dict(pid=pid, idx=i, clustered=int(i in in_cl), nn=NN[pid][i],
                         cell=cell[i], hits=int(hits[i]), hits_near=int(hits_near[i]),
                         enrich=enr[i], enrich_near=enr_near[i]))
    f = lambda v, s: (np.mean(v[s]) if len(s) else float("nan"))
    print(f"{pid}  {f(cell,cl):7.4f}  {f(cell,iso):8.4f} | {f(enr,cl):6.3f}  {f(enr,iso):7.3f} "
          f"| {f(enr_near,cl):12.3f}  {f(enr_near,iso):13.3f}")

with open(HERE / "enrichment_d10.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(HERE / "tau_sweep_d10.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(sweep_rows[0]))
    w.writeheader()
    w.writerows(sweep_rows)

cl_r = [r for r in rows if r["clustered"] == 1]
is_r = [r for r in rows if r["clustered"] == 0]
for lab, key in (("all descents", "enrich"), ("converged only (dist<0.1)", "enrich_near")):
    a = np.array([r[key] for r in cl_r], float)
    b = np.array([r[key] for r in is_r], float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    mw = mannwhitneyu(a, b, alternative="two-sided")
    print(f"\n  enrichment, {lab}: clustered mean {a.mean():.4f} (median {np.median(a):.4f}) "
          f"vs isolated {b.mean():.4f} (median {np.median(b):.4f})")
    print(f"    MWU p={mw.pvalue:.4g}  A12={mw.statistic/(len(a)*len(b)):.3f}; "
          f"clustered >= 1.0 in {int((a >= 1).sum())}/{len(a)}, isolated {int((b >= 1).sum())}/{len(b)}")

cc = np.array([r["cell"] for r in cl_r]); ci = np.array([r["cell"] for r in is_r])
print(f"\n  Voronoi cell share: clustered {cc.mean():.5f} vs isolated {ci.mean():.5f} "
      f"(ratio {ci.mean()/cc.mean():.2f}x) -> this is the confound H3 was carrying")

# ------------------------------------------------------- 3. H2 without the tautology
print("\n=== 3. H2 re-read on SHALLOW descents only (best_f > 1e-5; deep ones sit on an optimum)")
U = RNG.uniform(LO, HI, size=(N_MC // 4, D))
sh_all, dp_all, uni_all = [], [], []
nsig = 0
for pid in PIDS:
    pts = OPT[pid]
    X = np.array([x for _lo, _bf, _d, x in DUMP[pid]])
    deep = np.array([bf <= 1e-5 for _lo, bf, _d, _x in DUMP[pid]])
    dl = cdist(X, pts).min(axis=1)
    du = cdist(U, pts).min(axis=1)
    if (~deep).any():
        mw = mannwhitneyu(dl[~deep], du, alternative="two-sided")
        if mw.pvalue < 0.05 and dl[~deep].mean() < du.mean():
            nsig += 1
        sh_all += list(dl[~deep])
    dp_all += list(dl[deep])
    uni_all += list(du)
sh_all, uni_all = np.array(sh_all), np.array(uni_all)
mw = mannwhitneyu(sh_all, uni_all, alternative="two-sided")
print(f"    shallow n={len(sh_all)}: mean {sh_all.mean():.4f}, median {np.median(sh_all):.4f} "
      f"vs uniform mean {uni_all.mean():.4f}, median {np.median(uni_all):.4f}")
print(f"    MWU p={mw.pvalue:.4g}  A12={mw.statistic/(len(sh_all)*len(uni_all)):.3f}; "
      f"{nsig}/16 problems significantly closer than uniform")
print(f"    deep n={len(dp_all)}: mean {np.mean(dp_all):.3g} (tautological -- a deep descent IS at an optimum)")
