"""e118: (i) power control for the group A/B split, (ii) part (c) un-landed optima.

(i) Group A (PID 1-8) has K=20 and group B (PID 9-16) has K=10, so the split
    found in analyze.py could be power rather than placement.  Sub-sample 10 of
    the 20 optima in group A and re-run (a) and (b) at group B's sample size.
(ii) Part (c): which optima were never landed on, and where they sit in the
    placement.  Reads saved dumps only -- zero additional evaluations.
"""
import csv
import gzip
import pathlib

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, mannwhitneyu

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RNG = np.random.default_rng(20260913)
N_NULL, N_UNIF, N_SUB = 1000, 2000, 200
PIDS = [f"M{p:02d}" for p in range(1, 17)]


def nn_mean(pts):
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def clark_evans_p(pts, lo, hi, null_cache={}):
    K, D = pts.shape
    key = (K, D, lo, hi)
    if key not in null_cache:
        null_cache[key] = np.array([nn_mean(RNG.uniform(lo, hi, size=(K, D)))
                                    for _ in range(N_NULL)])
    null = null_cache[key]
    T = nn_mean(pts)
    p = min(1.0, 2.0 * min(int((null >= T).sum()), int((null <= T).sum())) / N_NULL)
    return T, float(np.median(null)), p


def loo_p(pts, lo, hi):
    K, D = pts.shape
    A, B = np.empty(K), np.empty(K)
    for i in range(K):
        known = np.delete(pts, i, axis=0)
        A[i] = cdist(pts[i:i + 1], known).min()
        u = RNG.uniform(lo, hi, size=(N_UNIF, D))
        B[i] = float(np.median(cdist(u, known).min(axis=1)))
    try:
        w = wilcoxon(A, B, method="exact")
    except Exception:
        w = wilcoxon(A, B)
    return float(np.median(A) / np.median(B)), float(w.pvalue)


def part_i(z):
    print("=== (i) power control: group A sub-sampled to K=10 ===")
    print("pid  sig_a/200  med_p_a  med_ratio_a   sig_b/200  med_p_b  med_ratio_b")
    out = []
    for pid in PIDS[:8]:
        pts, (lo, hi) = z[pid], z[f"{pid}_bounds"]
        pa, ra, pb, rb = [], [], [], []
        for _ in range(N_SUB):
            sub = pts[RNG.choice(pts.shape[0], 10, replace=False)]
            T, med, p = clark_evans_p(sub, lo, hi)
            pa.append(p); ra.append(T / med)
            r, p2 = loo_p(sub, lo, hi)
            pb.append(p2); rb.append(r)
        pa, ra, pb, rb = map(np.array, (pa, ra, pb, rb))
        row = dict(pid=pid, sig_a=int((pa < 0.05).sum()), med_p_a=float(np.median(pa)),
                   med_ratio_a=float(np.median(ra)), sig_b=int((pb < 0.05).sum()),
                   med_p_b=float(np.median(pb)), med_ratio_b=float(np.median(rb)))
        out.append(row)
        print(f"{pid}   {row['sig_a']:3d}/200   {row['med_p_a']:.4f}   {row['med_ratio_a']:.3f}"
              f"        {row['sig_b']:3d}/200   {row['med_p_b']:.4f}   {row['med_ratio_b']:.3f}",
              flush=True)
    with open(HERE / "power_control.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    return out


def read_rl(pid):
    """Restart-Lander seed-0 descents: (land_opt, best_f) pairs."""
    p = ROOT / f"analysis/mmo2024/e115/descents/{pid}-D10-PIN01_seed0.csv.gz"
    with gzip.open(p, "rt") as f:
        return [(int(r["land_opt"]), float(r["best_f"])) for r in csv.DictReader(f)]


def read_mceso(pid):
    """MC-ESO seed-0 segment dump: (land_opt, f) pairs."""
    p = ROOT / f"analysis/mmo2024/e113/hunts/{pid}-D10-PIN01_seed0_segments.csv.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt") as f:
        rows = list(csv.DictReader(f))
    key_f = "best_f" if "best_f" in rows[0] else "f"
    return [(int(r["land_opt"]), float(r[key_f])) for r in rows]


def part_c(z):
    print("\n=== (c) optima never landed on (saved dumps, seed 0) ===")
    print("pid   K  RL_touch RL@1e-5  MC_touch MC@1e-5   nn_unlanded nn_landed   MWU_p")
    out = []
    for pid in PIDS:
        pts, K = z[pid], z[pid].shape[0]
        d = cdist(pts, pts); np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        rl = read_rl(pid)
        mc = read_mceso(pid)
        t_rl = {k for k, _ in rl}
        d_rl = {k for k, v in rl if v <= 1e-5}
        t_mc = {k for k, _ in mc} if mc else set()
        d_mc = {k for k, v in mc if v <= 1e-5} if mc else set()
        un = sorted(set(range(K)) - t_rl)
        if un and len(un) < K:
            mw = mannwhitneyu(nn[un], nn[sorted(t_rl)], alternative="two-sided")
            p_mw, nn_un, nn_la = float(mw.pvalue), float(nn[un].mean()), float(nn[sorted(t_rl)].mean())
        else:
            p_mw, nn_un, nn_la = float("nan"), (float(nn[un].mean()) if un else float("nan")), float("nan")
        row = dict(pid=pid, K=K, rl_touch=len(t_rl) / K, rl_deep=len(d_rl) / K,
                   mc_touch=len(t_mc) / K, mc_deep=len(d_mc) / K,
                   n_unlanded_rl=len(un), unlanded_rl=";".join(map(str, un)),
                   nn_unlanded=nn_un, nn_landed=nn_la, mwu_p=p_mw)
        out.append(row)
        print(f"{pid}  {K:2d}   {row['rl_touch']:.2f}    {row['rl_deep']:.2f}"
              f"     {row['mc_touch']:.2f}    {row['mc_deep']:.2f}    "
              f"{nn_un:7.4f}  {nn_la:7.4f}   {p_mw:.4f}", flush=True)
    with open(HERE / "landing_d10.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    # pooled test across problems: are un-landed optima more isolated?
    allu, alll = [], []
    for pid in PIDS:
        pts, K = z[pid], z[pid].shape[0]
        d = cdist(pts, pts); np.fill_diagonal(d, np.inf); nn = d.min(axis=1)
        t = {k for k, _ in read_rl(pid)}
        un = sorted(set(range(K)) - t)
        allu += list(nn[un]); alll += list(nn[sorted(t)])
    if allu:
        mw = mannwhitneyu(allu, alll, alternative="two-sided")
        a12 = float(mw.statistic / (len(allu) * len(alll)))
        print(f"\npooled: {len(allu)} un-landed vs {len(alll)} landed, "
              f"NN mean {np.mean(allu):.4f} vs {np.mean(alll):.4f}, "
              f"MWU p={mw.pvalue:.4g}, A12={a12:.3f}")
    return out


def main():
    z = np.load(HERE / "optima_d10_pin01.npz")
    part_i(z)
    part_c(z)


if __name__ == "__main__":
    main()
