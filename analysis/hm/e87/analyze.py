"""Is N18-CF3-10D's 84% landing concentration inside the volume-proportional null?

Entry 76 measured that 2536 of 3036 MC-ESO hunts on N18-CF3-10D end nearest to
one and the same global optimum (gopt 5), and that two of the six optima
(gopt 2, gopt 3) are never the nearest one at all.  That number has never been
compared with a null.  Entry 64 asked exactly this on N09-Vincent3D against a
*volume-proportional* null (uniform draws, Voronoi assignment) and found MC-ESO
16.6 points above it -- i.e. concentrating more than the landscape forces.

Two nulls are read here, both produced by ``scripts/hunt_coverage.py --null``:

  geo   uniform draw -> nearest optimum.  The Voronoi volume share; entry 64's
        null, reproduced on this function so the two entries are comparable.
  desc  uniform draw -> isotropic descent at MC-ESO's own sigma0 and the
        observed hunt length -> nearest optimum.  This is the landing
        distribution a restart lander *without the repel rule and without the
        best-of-n_pop race* would produce, so the gap between it and the
        observation is what MC-ESO's own machinery adds.

Rejection condition, stated before the numbers were read: if the observed
per-seed concentration and the observed number of distinct optima reached both
sit inside the null's sampling range, the 84% is the landscape and the coverage
bias on F14-F20 is not a thing a method can fix -- the coverage axis closes on
this function set the way it closed on N07/N09.  If the observation sits outside
the null, entry 64's unexplained residual ("the over-concentration is not the
restart race") lands on a function set that still has headroom.

Usage: python3 analysis/hm/e87/analyze.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
K = 6
RNG = np.random.default_rng(20260907)
NSIM = 20000


def _shares(counts: np.ndarray) -> np.ndarray:
    return counts / max(1, counts.sum())


def load_observed(pattern: str, arm: str | None = None) -> pd.DataFrame:
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise SystemExit(f"no files match {pattern}")
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if arm is not None and "arm" in d.columns:
        d = d[d.arm == arm]
    return d


def per_seed_counts(d: pd.DataFrame) -> tuple[np.ndarray, list]:
    seeds = sorted(d.seed.unique())
    m = np.zeros((len(seeds), K), dtype=np.int64)
    for i, s in enumerate(seeds):
        v = d[d.seed == s].gopt.value_counts()
        for j, c in v.items():
            m[i, int(j)] = int(c)
    return m, seeds


def _find(name: str) -> Path | None:
    """Row-level dumps are stored gzipped (the repository rule); accept both."""
    for p in (HERE / name, HERE / (name + ".gz")):
        if p.exists():
            return p
    return None


def null_from_csv(path: Path) -> np.ndarray:
    t = pd.read_csv(path)
    c = np.zeros(K, dtype=np.int64)
    v = t.land_opt.value_counts()
    for j, n in v.items():
        c[int(j)] = int(n)
    return c


def simulate(p: np.ndarray, n_per_seed: np.ndarray) -> dict:
    """Null sampling distribution at the observed per-seed hunt counts.

    Each seed contributes n_i independent draws from p, which is the strongest
    form of the null: it grants the null the same number of hunts MC-ESO spent
    and asks only whether their *destinations* differ.
    """
    out = {"top_share": [], "distinct": [], "max_share": []}
    for _ in range(NSIM):
        per = []
        dis = []
        for n in n_per_seed:
            c = RNG.multinomial(n, p)
            per.append(c)
            dis.append(int((c > 0).sum()))
        per = np.array(per)
        out["top_share"].append((per[:, 5] / per.sum(axis=1)).mean())
        out["max_share"].append((per.max(axis=1) / per.sum(axis=1)).mean())
        out["distinct"].append(float(np.mean(dis)))
    return {k: np.array(v) for k, v in out.items()}


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) + 0.5 P(x == y): stochastic dominance of x over y."""
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def report(tag: str, obs: pd.DataFrame, nulls: dict) -> None:
    m, seeds = per_seed_counts(obs)
    n_per = m.sum(axis=1)
    pooled = _shares(m.sum(axis=0))
    print(f"\n{'=' * 74}\n== observed: {tag}  ({m.sum()} hunts, {len(seeds)} seeds)")
    print("  per-optimum pooled share: "
          + "  ".join(f"{j}:{pooled[j]:.4f}" for j in range(K)))
    obs_top = m[:, 5] / n_per
    obs_dis = (m > 0).sum(axis=1)
    print(f"  per-seed share on gopt 5: median {np.median(obs_top):.4f} "
          f"[{obs_top.min():.4f}, {obs_top.max():.4f}]")
    print(f"  per-seed distinct optima: median {np.median(obs_dis):.1f} "
          f"[{obs_dis.min()}, {obs_dis.max()}]  (K = {K})")
    print(f"  optima never nearest in any seed: "
          f"{[j for j in range(K) if m[:, j].sum() == 0]}")

    for nname, ncnt in nulls.items():
        p = _shares(ncnt)
        sim = simulate(p, n_per)
        o_top, o_dis = float(obs_top.mean()), float(obs_dis.mean())
        # two-sided permutation-style p: where does the observation fall in the
        # null's own sampling distribution of the same statistic?
        def pval(o, s):
            r = float((s >= o).mean())
            return 2 * min(r, 1 - r) if 0 < r < 1 else 1.0 / NSIM
        print(f"\n  -- vs null '{nname}' (n={ncnt.sum()}): "
              + "  ".join(f"{j}:{p[j]:.4f}" for j in range(K)))
        print(f"     gopt-5 share   observed {o_top:.4f}   null "
              f"{sim['top_share'].mean():.4f} "
              f"[{np.percentile(sim['top_share'], 2.5):.4f}, "
              f"{np.percentile(sim['top_share'], 97.5):.4f}]"
              f"   p = {pval(o_top, sim['top_share']):.4g}"
              f"   A12 = {a12(obs_top, sim['top_share']):.3f}")
        o_max = float((m.max(axis=1) / n_per).mean())
        print(f"     max-share      observed {o_max:.4f}   null "
              f"{sim['max_share'].mean():.4f} "
              f"[{np.percentile(sim['max_share'], 2.5):.4f}, "
              f"{np.percentile(sim['max_share'], 97.5):.4f}]"
              f"   p = {pval(o_max, sim['max_share']):.4g}")
        print(f"     distinct/seed  observed {o_dis:.3f}   null "
              f"{sim['distinct'].mean():.3f} "
              f"[{np.percentile(sim['distinct'], 2.5):.3f}, "
              f"{np.percentile(sim['distinct'], 97.5):.3f}]"
              f"   p = {pval(o_dis, sim['distinct']):.4g}")
        # per-seed G-test against the null share (df = K-1); repulsion makes
        # hunts within a seed dependent, so the seed-level agreement matters
        # more than any single p.
        gs, ps = [], []
        for i in range(len(seeds)):
            exp = p * n_per[i]
            ok = exp > 0
            g = 2 * np.sum(np.where(m[i][ok] > 0,
                                    m[i][ok] * np.log(np.maximum(m[i][ok], 1e-12)
                                                      / exp[ok]), 0.0))
            gs.append(g)
            ps.append(stats.chi2.sf(g, ok.sum() - 1))
        ps = np.array(ps)
        print(f"     per-seed G-test vs null: median G = {np.median(gs):.1f}, "
              f"seeds with p < 0.05: {(ps < 0.05).sum()}/{len(seeds)}")


def transition(path: Path, tag: str) -> None:
    """Where does a draw that *starts* nearest optimum j end up?

    The geometric null and the descent null differ only by the descent, so the
    row-normalised transition matrix is exactly the map "Voronoi cell -> basin
    of attraction".  A row that does not keep its own mass names an optimum
    whose geometric neighbourhood drains somewhere else -- which is a property
    of the landscape, not of any restart rule.
    """
    t = pd.read_csv(path)
    m = pd.crosstab(t.start_opt, t.land_opt).reindex(
        index=range(K), columns=range(K), fill_value=0)
    print(f"\n{'=' * 74}\n== start -> land, {tag} ({len(t)} draws)")
    print(m.to_string())
    print("\n  row-normalised:")
    print(m.div(m.sum(axis=1).replace(0, 1), axis=0).round(3).to_string())
    print(f"\n  draws keeping their own start optimum: "
          f"{np.trace(m.to_numpy()) / max(1, m.to_numpy().sum()):.4f}")
    q = t.groupby("land_opt").agg(n=("best_f", "size"),
                                  med_f=("best_f", "median"),
                                  med_dist=("dist", "median"),
                                  frac_le_1e1=("best_f", lambda s: (s <= 0.1).mean()))
    print("\n  endpoint quality by landed optimum:")
    print(q.round(5).to_string())


def main() -> None:
    nulls = {}
    geo = HERE / "null_geo.csv"
    if geo.exists():
        nulls["geo (Voronoi)"] = pd.read_csv(geo).cnt.to_numpy()
    for f, lab in [("null_iso_b1499.csv", "desc iso b=1499"),
                   ("null_cov_b1499.csv", "desc full-cov b=1499"),
                   ("null_iso_b15000.csv", "desc iso b=15000")]:
        p = _find(f)
        if p is not None:
            nulls[lab] = null_from_csv(p)
    if not nulls:
        raise SystemExit("no null CSVs yet")

    report("e76 base arm (MC-ESO default)",
           load_observed("analysis/hm/e76/N18_off*_hunts.csv.gz"), nulls)
    e77 = "analysis/hm/e77/N18_%s_off*_hunts.csv.gz"
    report("e77 base arm", load_observed(e77 % "base", arm="base"), nulls)
    report("e77 basin_reset arm", load_observed(e77 % "basin", arm="basin"), nulls)

    for f, lab in [("null_iso_b1499.csv", "desc iso b=1499"),
                   ("null_iso_b15000.csv", "desc iso b=15000")]:
        p = _find(f)
        if p is not None:
            transition(p, lab)


if __name__ == "__main__":
    main()
