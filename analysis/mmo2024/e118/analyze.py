"""e118 (a)(b): is there structure in this suite's global-optimum placement?

Zero optimisation runs: reads the cached `optima_pos` (see extract_optima.py)
and compares it with a uniform (CSR) null. Pre-registration: prereg.md.
"""
import pathlib
import sys

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, mannwhitneyu

HERE = pathlib.Path(__file__).resolve().parent
RNG = np.random.default_rng(20260913)
N_NULL = 1000
N_UNIF = 2000
PIDS = [f"M{p:02d}" for p in range(1, 17)]


def nn_mean(pts):
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean()), d.min(axis=1)


def rank_biserial(a, b):
    """Paired rank-biserial from the signed-rank statistic (a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    rp, rn = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rn) / (rp + rn))


def main():
    z = np.load(HERE / "optima_d10_pin01.npz")
    rows = []
    for pid in PIDS:
        pts = z[pid]
        lo, hi = z[f"{pid}_bounds"]
        K, D = pts.shape

        # ---- (a) Clark-Evans against CSR -------------------------------
        T, nn_obs = nn_mean(pts)
        null = np.empty(N_NULL)
        for r in range(N_NULL):
            null[r] = nn_mean(RNG.uniform(lo, hi, size=(K, D)))[0]
        ge, le = int((null >= T).sum()), int((null <= T).sum())
        p_a = min(1.0, 2.0 * min(ge, le) / N_NULL)
        med = float(np.median(null))
        kind = "regular" if T > med else "clustered"

        # ---- (b) leave-one-out predictability --------------------------
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
        p_b, rb = float(w.pvalue), rank_biserial(A, B)

        # ---- exploratory descriptors (not decision statistics) --------
        c = pts - pts.mean(axis=0)
        sv = np.linalg.svd(c, compute_uv=False)
        eff_rank = float((sv.sum() ** 2) / (sv ** 2).sum())   # participation ratio
        distinct = [len(np.unique(np.round(pts[:, j], 6))) for j in range(D)]

        rows.append(dict(
            pid=pid, K=K, T_a=T, null_med=med,
            null_lo=float(np.percentile(null, 2.5)),
            null_hi=float(np.percentile(null, 97.5)),
            p_a=p_a, kind_a=kind, sig_a=p_a < 0.05,
            A_med=float(np.median(A)), B_med=float(np.median(B)),
            ratio=float(np.median(A) / np.median(B)),
            p_b=p_b, rb_b=rb, sig_b=p_b < 0.05,
            eff_rank=eff_rank, min_distinct=min(distinct),
            nn_min=float(nn_obs.min()), nn_max=float(nn_obs.max()),
        ))
        print(f"{pid} K={K:2d}  (a) T={T:.4f} null={med:.4f} "
              f"[{rows[-1]['null_lo']:.4f},{rows[-1]['null_hi']:.4f}] p={p_a:.4f} {kind}"
              f"   (b) A={rows[-1]['A_med']:.4f} B={rows[-1]['B_med']:.4f} "
              f"r={rows[-1]['ratio']:.3f} p={p_b:.2e} rb={rb:+.3f}"
              f"   eff_rank={eff_rank:.2f}", flush=True)

    import csv
    with open(HERE / "structure_d10.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_a = sum(r["sig_a"] for r in rows)
    n_b = sum(r["sig_b"] for r in rows)
    print(f"\n(a) significant in {n_a}/16   (b) significant in {n_b}/16")
    print("(a) not-sig:", [r["pid"] for r in rows if not r["sig_a"]])
    print("(b) not-sig:", [r["pid"] for r in rows if not r["sig_b"]])
    print("kinds:", {r["pid"]: r["kind_a"] for r in rows if r["sig_a"]})
    print("ratio A/B  min/median/max: "
          f"{min(r['ratio'] for r in rows):.3f} / "
          f"{np.median([r['ratio'] for r in rows]):.3f} / "
          f"{max(r['ratio'] for r in rows):.3f}")
    print("eff_rank (D=10):", [round(r["eff_rank"], 2) for r in rows])
    print("min distinct coord values per problem:", [r["min_distinct"] for r in rows])


if __name__ == "__main__":
    main()
