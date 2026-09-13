"""e119 (queue 1): is the group-A clustering already spent by `Restart-Lander`?

e118 found (a) group A (PID 1-8) is over-concentrated 8/8 and (b) the structure is
uncorrelated with the Score gap to MC-ESO.  Two readings survive both facts:

  (i)  saturation -- a uniform sprinkle already harvests the cluster, because a
       cluster is easier to hit by volume, so no memory/archive arm can add to it;
  (ii) inert -- the cluster is there but too weak for anyone to use.

Pre-registered decision (prereg.md): Spearman(cluster size, reach rate) over all
clusters pooled.  Significantly positive -> (i); otherwise -> (ii).

Inputs are stored only: e118's optima cache and e115's descent dumps.  Zero runs.
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
TAU_FRAC = 0.6          # prereg: tau = 0.6 * uniform-CSR null median
N_UNIF = 20000          # prereg: uniform points for H2
N_PERM = 2000           # prereg: permutations for the power control


def read_rl(pid):
    """Descent dump: (land_opt, best_f, landing coordinates)."""
    p = ROOT / f"analysis/mmo2024/e115/descents/{pid}-D10-PIN01_seed0.csv.gz"
    with gzip.open(p, "rt") as f:
        out = []
        for r in csv.DictReader(f):
            out.append((int(r["land_opt"]), float(r["best_f"]),
                        np.array([float(r[f"x{j}"]) for j in range(D)])))
    return out


def components(members, dmat, tau):
    """Single-linkage components of `members` at threshold tau (list of index lists)."""
    rem, comps = set(members), []
    while rem:
        seed = rem.pop()
        comp, stack = [seed], [seed]
        while stack:
            u = stack.pop()
            near = [v for v in list(rem) if dmat[u, v] < tau]
            for v in near:
                rem.discard(v)
                comp.append(v)
                stack.append(v)
        comps.append(sorted(comp))
    return comps


z = np.load(HERE / "optima_d10_pin01.npz") if (HERE / "optima_d10_pin01.npz").exists() \
    else np.load(ROOT / "analysis/mmo2024/e118/optima_d10_pin01.npz")
nullmed = {r["pid"]: float(r["null_med"])
           for r in csv.DictReader(open(ROOT / "analysis/mmo2024/e118/structure_d10.csv"))}

clusters, per_opt, land_rows = [], [], []
h2_rows = []
print("pid   K  tau    memb  ncl  sizes            reach_cl  reach_iso  att_cl  att_iso")
for pid in PIDS:
    pts = z[pid]
    K = pts.shape[0]
    tau = TAU_FRAC * nullmed[pid]
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    members = [i for i in range(K) if nn[i] < tau]
    comps = [c for c in components(members, d, tau) if len(c) >= 2]
    in_cluster = {i for c in comps for i in c}

    dump = read_rl(pid)
    hits = np.zeros(K, dtype=int)
    for lo, _bf, _x in dump:
        if 0 <= lo < K:
            hits[lo] += 1

    # --- H2: landing coords vs uniform, distance to nearest global optimum ---
    X = np.array([x for _lo, _bf, x in dump])
    deep = np.array([bf <= 1e-5 for _lo, bf, _x in dump])
    U = RNG.uniform(LO, HI, size=(N_UNIF, D))
    d_land = cdist(X, pts).min(axis=1)
    d_unif = cdist(U, pts).min(axis=1)
    mw = mannwhitneyu(d_land, d_unif, alternative="two-sided")
    h2_rows.append(dict(pid=pid, n_land=len(d_land), land_mean=d_land.mean(),
                        unif_mean=d_unif.mean(), p=float(mw.pvalue),
                        a12=float(mw.statistic) / (len(d_land) * N_UNIF),
                        land_deep_mean=float(d_land[deep].mean()) if deep.any() else float("nan"),
                        land_shallow_mean=float(d_land[~deep].mean()) if (~deep).any() else float("nan"),
                        n_deep=int(deep.sum())))

    for c in comps:
        clusters.append(dict(pid=pid, size=len(c),
                             reach=float(np.mean(hits[c] > 0)),
                             attract=float(hits[c].sum()) / len(c),
                             members=";".join(map(str, c))))
    for i in range(K):
        per_opt.append(dict(pid=pid, idx=i, nn=float(nn[i]), tau=tau,
                            clustered=int(i in in_cluster), hits=int(hits[i]),
                            reached=int(hits[i] > 0)))

    iso = [i for i in range(K) if i not in in_cluster]
    r_cl = np.mean([hits[i] > 0 for i in in_cluster]) if in_cluster else float("nan")
    r_is = np.mean([hits[i] > 0 for i in iso]) if iso else float("nan")
    a_cl = np.mean([hits[i] for i in in_cluster]) if in_cluster else float("nan")
    a_is = np.mean([hits[i] for i in iso]) if iso else float("nan")
    print(f"{pid}  {K:2d} {tau:5.2f}  {len(in_cluster):3d}  {len(comps):3d}  "
          f"{str([len(c) for c in comps]):16s} {r_cl:8.3f}  {r_is:9.3f}  {a_cl:6.2f}  {a_is:6.2f}")

for name, rows in (("clusters_d10.csv", clusters), ("per_optimum_d10.csv", per_opt),
                   ("landing_vs_uniform_d10.csv", h2_rows)):
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

# ---------------- H1 (primary, pre-registered) ----------------
size = np.array([c["size"] for c in clusters], dtype=float)
reach = np.array([c["reach"] for c in clusters], dtype=float)
rho, p = spearmanr(size, reach)
print(f"\n=== H1 (PRIMARY) n_clusters={len(clusters)}  "
      f"sizes {int(size.min())}-{int(size.max())} (mean {size.mean():.2f})")
print(f"    Spearman(size, reach) rho={rho:.4f}  p={p:.4g}  -> "
      f"{'(i) saturation' if (p < 0.05 and rho > 0) else '(ii) inert'}")
for lab, sl in (("A (PID 1-8)", slice(0, 8)), ("B (PID 9-16)", slice(8, 16))):
    sub = [c for c in clusters if c["pid"] in PIDS[sl]]
    if len(sub) >= 3:
        r2, p2 = spearmanr([c["size"] for c in sub], [c["reach"] for c in sub])
        print(f"    {lab}: n={len(sub)} rho={r2:.4f} p={p2:.4g}")
    else:
        print(f"    {lab}: n={len(sub)} (too few to correlate)")

# ---------------- H3 (mechanism split) ----------------
cl_rows = [r for r in per_opt if r["clustered"] == 1]
is_rows = [r for r in per_opt if r["clustered"] == 0]
a_cl = np.array([r["hits"] for r in cl_rows], float)
a_is = np.array([r["hits"] for r in is_rows], float)
r_cl = np.array([r["reached"] for r in cl_rows], float)
r_is = np.array([r["reached"] for r in is_rows], float)
mw_a = mannwhitneyu(a_cl, a_is, alternative="two-sided")
mw_r = mannwhitneyu(r_cl, r_is, alternative="two-sided")
print(f"\n=== H3 clustered n={len(cl_rows)}  isolated n={len(is_rows)}")
print(f"    attract (descents per optimum): {a_cl.mean():.3f} vs {a_is.mean():.3f}  "
      f"p={mw_a.pvalue:.4g}  A12={mw_a.statistic/(len(a_cl)*len(a_is)):.3f}")
print(f"    reach   (fraction reached)    : {r_cl.mean():.3f} vs {r_is.mean():.3f}  "
      f"p={mw_r.pvalue:.4g}  A12={mw_r.statistic/(len(r_cl)*len(r_is)):.3f}")

# ---------------- H2 pooled ----------------
print("\n=== H2 pooled over 16 problems (per-problem rows in landing_vs_uniform_d10.csv)")
lm = np.array([r["land_mean"] for r in h2_rows])
um = np.array([r["unif_mean"] for r in h2_rows])
sig = sum(1 for r in h2_rows if r["p"] < 0.05 and r["land_mean"] < r["unif_mean"])
print(f"    landing mean {lm.mean():.4f} vs uniform mean {um.mean():.4f}; "
      f"{sig}/16 problems significantly closer than uniform")
print(f"    deep landings {np.nanmean([r['land_deep_mean'] for r in h2_rows]):.4f} vs "
      f"shallow {np.nanmean([r['land_shallow_mean'] for r in h2_rows]):.4f}  "
      f"(deep n = {sum(r['n_deep'] for r in h2_rows)} of {sum(r['n_land'] for r in h2_rows)})")

# ---------------- power control (read only if H1 non-significant) ----------------
obs = rho
null_rho = []
by_pid = {}
for c in clusters:
    by_pid.setdefault(c["pid"], []).append(c)
member_pool = {pid: [r for r in per_opt if r["pid"] == pid and r["clustered"] == 1]
               for pid in PIDS}
for _ in range(N_PERM):
    sizes, reaches = [], []
    for pid, cs in by_pid.items():
        pool = member_pool[pid][:]
        RNG.shuffle(pool)
        k = 0
        for c in cs:
            grp = pool[k:k + c["size"]]
            k += c["size"]
            sizes.append(c["size"])
            reaches.append(np.mean([g["reached"] for g in grp]))
    null_rho.append(spearmanr(sizes, reaches)[0])
null_rho = np.array([x for x in null_rho if np.isfinite(x)])
lo, hi = np.percentile(null_rho, [2.5, 97.5])
print(f"\n=== power control ({len(null_rho)} permutations of cluster labels within problem)")
print(f"    null rho 95% band [{lo:.4f}, {hi:.4f}]; observed {obs:.4f}  "
      f"-> {'outside' if (obs < lo or obs > hi) else 'inside'} the band")
print(f"    smallest |rho| this design can call significant: about {max(abs(lo), abs(hi)):.3f}")
