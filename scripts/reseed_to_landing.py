#!/usr/bin/env python3
"""Does the restart *draw* decide where the hunt *lands*?

``_diversified_reseed`` returns one candidate per re-seeded population slot, and
a basin switch re-seeds every slot and resets sigma to sigma_init = 0.2 * span.
On Vincent2D that is 1.95 against optimum spacings of 0.291 to 3.595, so a
population scattered with that sigma can roll out of the basin it was dropped
in. If it does, repelling the *draw* away from basins already drilled cannot fix
coverage however well the radius is chosen — the funnel is downstream of it.

This measures the link directly. For each hunt it records every diversified
draw, maps both the draws and the hunt's endpoint onto the nearest true Vincent
optimum, and reports:

  match      the endpoint optimum was also the nearest optimum of at least one
             of that hunt's draws (the draw reached the landing)
  d(draw)    distance from the endpoint to the nearest draw of its own hunt
  width      the nearest-neighbour spacing of the optimum landed in vs. drawn
             near — the wide-basin bias, measured on the draw and on the
             landing separately

Estimator direction: `match` is generous — with n_pop draws per hunt, some draw
lands near the endpoint's optimum by chance, so a low match rate is strong
evidence and a high one is weak.

Usage:  python3 scripts/reseed_to_landing.py [--variant base|adaptive] [--seeds 3]

Paired draw-vs-landing mode (entry 65).  ``--csv`` runs the seeds in parallel and
writes per-(seed, optimum) draw/landing counts instead of the aggregate print,
so the *same* widest-quartile occupancy statistic entry 64 measured on landings
can be measured on the draws that produced them, against the same
volume-proportional null.  Draws and landings come from one run each, so the two
sides are paired within a seed and no extra optimisation is needed to compare them.

  python3 scripts/reseed_to_landing.py --func N09-Vincent3D --evals 400000 \
      --seeds 15 --procs 4 --csv analysis/hm/e65/n09_draws.csv
  python3 scripts/reseed_to_landing.py --analyze analysis/hm/e65/n09_draws.csv
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME              # noqa: E402
from core.optimizers import MultiChannelEpidemicOptimizer           # noqa: E402
from core.optimizers.mceso_adaptive_repel import AdaptiveRepelMCESO  # noqa: E402
from core.optimizers.mceso_commit_reseed import CommitReseedMCESO      # noqa: E402
from scripts.hunt_coverage import vincent_optima                    # noqa: E402


def _tracer(base_cls):
    class _Traced(base_cls):
        """Records the diversified draws of the hunt in progress, and files them
        against that hunt's endpoint when the next spillover fires."""

        def optimize(self, max_evals: int = 5000):
            self._draws: list[np.ndarray] = []
            self.records: list[dict] = []
            return super().optimize(max_evals)

        def _diversified_reseed(self, st, x_best_snap):
            cand = super()._diversified_reseed(st, x_best_snap)
            self._draws.append(np.asarray(cand, dtype=float).copy())
            return cand

        def _on_spillover_start(self, st, basin_switch: bool) -> None:
            best_i = int(np.argmin(st.pop_f))
            self.records.append({
                "end": np.asarray(st.pop_x[best_i], dtype=float).copy(),
                "f": float(st.pop_f[best_i]),
                "draws": list(self._draws),
            })
            self._draws = []
            return super()._on_spillover_start(st, basin_switch)
    return _Traced


_CLS_SPECS = {
    "base": ("core.optimizers", "MultiChannelEpidemicOptimizer", {}),
    "commit_place": ("core.optimizers.mceso_commit_reseed", "CommitReseedMCESO",
                     {"commit_sigma_mode": "place", "commit_sigma_ratio": 0.1}),
    # Entry 66's queue-1 arm: the same commitment, but the *run* sigma of the
    # restarted hunt is taken down to the local basin scale as well, not just the
    # placement of the slots. `commit_place` returns only 0.027 of base's +0.150
    # draw->landing gap, and the geometry says the rest should be the descent
    # scale (sigma_init = 0.2*span = 1.950 against 0.291-0.546 spacings), so this
    # arm is the direct test of that. Read the same caveat as the aggregate mode:
    # for both commit arms the tracer records anchor + cloud, so their "draws" are
    # a committed placement, not 20 independent draws.
    "commit_tight": ("core.optimizers.mceso_commit_reseed", "CommitReseedMCESO",
                     {"commit_sigma_mode": "run", "commit_sigma_ratio": 0.1}),
}


def _paired_one(job):
    """One traced run -> per-optimum draw/landing counts + hunt-level pairing.

    Runs in a worker process, so it re-imports and re-derives the optima rather
    than inheriting them.  Returns plain arrays: the parent writes the CSV.
    """
    func, variant, seed, evals = job
    import importlib
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.benchmarks import NICHING_BENCHMARKS_BY_NAME
    mod, cls_name, kw = _CLS_SPECS[variant]
    base_cls = getattr(importlib.import_module(mod), cls_name)
    b = NICHING_BENCHMARKS_BY_NAME[func]
    opt_pts = vincent_optima(b.dim)
    assert len(opt_pts) == b.n_global_optima, (len(opt_pts), b.n_global_optima)
    K = len(opt_pts)
    width = _widths(opt_pts)
    q4 = width >= np.quantile(width, 0.75)      # same split as hunt_coverage.py

    t0 = time.time()
    o = _tracer(base_cls)(b, seed=seed, **kw)
    o.optimize(evals)

    draws = np.zeros(K, dtype=np.int64)
    lands = np.zeros(K, dtype=np.int64)
    n_hunt = n_match = 0
    # the paired conversion: a hunt whose draws all sit in narrow cells but
    # whose endpoint is in a widest-quartile cell.  This is the funnel doing the
    # concentrating, as opposed to the draw distribution arriving concentrated.
    narrow_draw_wide_land = wide_draw_wide_land = 0
    for rec in o.records:
        if not rec["draws"]:
            continue
        n_hunt += 1
        end_i = int(np.argmin(np.linalg.norm(opt_pts - rec["end"], axis=1)))
        dr = np.asarray(rec["draws"])
        draw_i = np.argmin(np.linalg.norm(
            opt_pts[None, :, :] - dr[:, None, :], axis=-1), axis=1)
        n_match += int(end_i in set(draw_i.tolist()))
        lands[end_i] += 1
        np.add.at(draws, draw_i, 1)
        if q4[end_i]:
            if q4[draw_i].any():
                wide_draw_wide_land += 1
            else:
                narrow_draw_wide_land += 1
    return (func, variant, seed, n_hunt, n_match, narrow_draw_wide_land,
            wide_draw_wide_land, width, draws, lands, time.time() - t0)


def paired_mode(a) -> None:
    from multiprocess import Pool
    b = NICHING_BENCHMARKS_BY_NAME[a.func]
    if "Vincent" not in a.func:
        raise SystemExit(f"{a.func}: only Vincent's optima are hard-coded here")
    evals = a.evals if a.evals > 0 else b.suite_max_evals
    jobs = [(a.func, v, s * 100, evals)
            for v in a.variant.split(",") for s in range(a.seeds)]
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    summ = a.csv.with_name(a.csv.stem + "_summary.csv")
    with open(a.csv, "w", newline="") as fh, open(summ, "w", newline="") as sh:
        w = csv.writer(fh)
        w.writerow(["function", "variant", "seed", "opt", "width",
                    "draws", "lands"])
        sw = csv.writer(sh)
        sw.writerow(["function", "variant", "seed", "evals", "hunts", "match",
                     "narrow_draw_wide_land", "wide_draw_wide_land", "secs"])
        with Pool(a.procs) as pool:
            for (fn, v, sd, nh, nm, ndwl, wdwl, wid, dr, ld,
                 secs) in pool.imap_unordered(_paired_one, jobs):
                for j in range(len(wid)):
                    w.writerow([fn, v, sd, j, f"{wid[j]:.6f}", dr[j], ld[j]])
                sw.writerow([fn, v, sd, evals, nh, nm, ndwl, wdwl, f"{secs:.1f}"])
                fh.flush(); sh.flush()
                print(f"{v:<13} seed {sd:>4}  {nh:>4} hunts  {secs:6.1f}s  "
                      f"draws in {int((dr > 0).sum()):>3} cells, "
                      f"landings in {int((ld > 0).sum()):>3}", flush=True)
    print(f"rows -> {a.csv}\nsummary -> {summ}")


def paired_analyze(paths: list[str], null_q4: float = 0.814) -> None:
    """Widest-quartile occupancy of draws vs landings, paired within seed."""
    from scipy.stats import wilcoxon
    rows = []
    for p in paths:
        opener = ((lambda q: __import__("gzip").open(q, "rt", newline=""))
                  if p.endswith(".gz") else (lambda q: open(q, newline="")))
        with opener(p) as fh:
            rows += list(csv.DictReader(fh))
    per_arm: dict[str, dict[int, tuple[float, float, int, int]]] = {}
    land_vec: dict[str, dict[int, np.ndarray]] = {}
    for variant in sorted({r["variant"] for r in rows}):
        sel = [r for r in rows if r["variant"] == variant]
        seeds = sorted({int(r["seed"]) for r in sel})
        qd, ql, cd, cl = [], [], [], []
        for s in seeds:
            sr = sorted((r for r in sel if int(r["seed"]) == s),
                        key=lambda r: int(r["opt"]))
            wdt = np.array([float(r["width"]) for r in sr])
            dr = np.array([float(r["draws"]) for r in sr])
            ld = np.array([float(r["lands"]) for r in sr])
            q4 = wdt >= np.quantile(wdt, 0.75)
            qd.append(dr[q4].sum() / max(dr.sum(), 1))
            ql.append(ld[q4].sum() / max(ld.sum(), 1))
            cd.append((dr > 0).sum())
            cl.append((ld > 0).sum())
        qd, ql = np.array(qd), np.array(ql)
        print(f"\n=== {variant}  ({len(seeds)} seeds) " + "=" * 34)
        print(f"  widest-quartile share of DRAWS     {np.median(qd):.3f}"
              f"   (null {null_q4:.3f}, delta {np.median(qd) - null_q4:+.3f})")
        print(f"  widest-quartile share of LANDINGS  {np.median(ql):.3f}"
              f"   (null {null_q4:.3f}, delta {np.median(ql) - null_q4:+.3f})")
        print(f"  distinct cells drawn near / landed in   "
              f"{np.median(cd):.0f} / {np.median(cl):.0f}")
        if len(seeds) >= 5:
            st = wilcoxon(qd - null_q4)
            print(f"  draws vs null           p = {st.pvalue:.3g}   "
                  f"seed-direction above null {int((qd > null_q4).sum())}/{len(qd)}")
            st = wilcoxon(ql - qd)
            print(f"  landings vs draws (paired)  p = {st.pvalue:.3g}   "
                  f"landing>draw {int((ql > qd).sum())}/{len(qd)}   "
                  f"median gap {np.median(ql - qd):+.3f}")
        per_arm[variant] = {s: (float(qd[i]), float(ql[i]), int(cd[i]),
                                int(cl[i])) for i, s in enumerate(seeds)}
        land_vec[variant] = {s: np.array(
            [float(r["lands"]) for r in sorted(
                (q for q in sel if int(q["seed"]) == s),
                key=lambda q: int(q["opt"]))]) for s in seeds}
    _cross_arm(per_arm)
    _hunt_matched(land_vec)


def _a12(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) + 0.5 P(x == y) -- Vargha-Delaney, same convention as the rest
    of the analysis scripts."""
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def _cross_arm(per_arm: dict[str, dict[int, tuple[float, float, int, int]]]) -> None:
    """Arm-vs-arm comparison of the draw->landing gap, paired by seed number.

    Entry 65 established that the whole widest-quartile excess is made *between*
    the draw and the landing, and that `commit_place` (placement only) returns
    only 0.027 of base's +0.151. Whether the rest is the run sigma is a question
    about the *gap*, not about either occupancy alone, so this pairs the gaps.
    Seeds are the same integers across arms by construction (`seed = s * 100`),
    so the pairing is exact and no re-run is needed to compare stored arms.
    """
    from scipy.stats import wilcoxon
    arms = sorted(per_arm)
    if len(arms) < 2:
        return
    print("\n=== gap (landings - draws) across arms, paired by seed " + "=" * 12)
    for a in arms:
        g = np.array([v[1] - v[0] for v in per_arm[a].values()])
        print(f"  {a:<13} median gap {np.median(g):+.3f}  "
              f"({len(g)} seeds)")
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            common = sorted(set(per_arm[a]) & set(per_arm[b]))
            if len(common) < 5:
                print(f"  {a} vs {b}: only {len(common)} shared seeds, skipped")
                continue
            ga = np.array([per_arm[a][s][1] - per_arm[a][s][0] for s in common])
            gb = np.array([per_arm[b][s][1] - per_arm[b][s][0] for s in common])
            st = wilcoxon(ga - gb)
            print(f"  {a} - {b}: median {np.median(ga - gb):+.3f}  "
                  f"p = {st.pvalue:.3g}  "
                  f"{a}>{b} {int((ga > gb).sum())}/{len(common)}  "
                  f"A12 = {_a12(ga, gb):.2f}")
            la = np.array([per_arm[a][s][1] for s in common])
            lb = np.array([per_arm[b][s][1] for s in common])
            st = wilcoxon(la - lb)
            print(f"      landing share  {np.median(la):.3f} vs {np.median(lb):.3f}"
                  f"   p = {st.pvalue:.3g}   A12 = {_a12(la, lb):.2f}")
            ca = np.array([per_arm[a][s][3] for s in common], dtype=float)
            cb = np.array([per_arm[b][s][3] for s in common], dtype=float)
            st = wilcoxon(ca - cb)
            print(f"      cells landed in {np.median(ca):.0f} vs {np.median(cb):.0f}"
                  f"   p = {st.pvalue:.3g}   A12 = {_a12(ca, cb):.2f}")


def _hunt_matched(land_vec: dict[str, dict[int, np.ndarray]],
                  reps: int = 200) -> None:
    """Distinct cells landed in, with the number of landings matched across arms.

    An arm that shortens its hunts gets more of them (entry 25's confound), and
    "distinct cells landed in" grows with the number of landings for free. The
    widest-quartile *share* is a proportion and so is not inflated by the count,
    but the cell count is, so this re-scores the same stored landings: for each
    seed, draw the smaller arm's number of landings without replacement from the
    larger arm's landing multiset and count the cells that survive. Zero extra
    optimisation -- it is the same runs, scored at a common budget of landings.
    """
    arms = sorted(land_vec)
    if len(arms) < 2:
        return
    rng = np.random.default_rng(0)
    print("\n=== distinct cells landed in, matched on number of landings " + "=" * 6)
    for a in arms:
        tot = np.array([v.sum() for v in land_vec[a].values()])
        print(f"  {a:<13} landings/seed median {np.median(tot):.0f}")
    n_match = int(min(np.median([v.sum() for v in land_vec[a].values()])
                      for a in arms))
    print(f"  common budget = {n_match} landings/seed")
    for a in arms:
        cells = []
        for s, ld in land_vec[a].items():
            pool = np.repeat(np.arange(len(ld)), ld.astype(int))
            if len(pool) <= n_match:
                cells.append(int((ld > 0).sum()))
                continue
            cells.append(float(np.median([
                len(np.unique(rng.choice(pool, n_match, replace=False)))
                for _ in range(reps)])))
        print(f"  {a:<13} {np.median(cells):6.1f} cells "
              f"(unmatched {np.median([ (v>0).sum() for v in land_vec[a].values()]):.0f})")


def _widths(opts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(opts[:, None, :] - opts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--analyze":
        return paired_analyze(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None,
                    help="paired per-seed mode (entry 65): write per-optimum "
                         "draw/landing counts instead of the aggregate print")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--variant", default="base",
                    help="aggregate mode: base|adaptive|commit_tight. "
                         "--csv mode: comma-separated keys of _CLS_SPECS")
    ap.add_argument("--func", default="N07-Vincent2D")
    ap.add_argument("--evals", type=int, default=20000,
                    help="--csv mode: 0 means the suite's full budget")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=None,
                    help="override sigma_init/span (default 0.2). Mechanism probe: "
                         "does shrinking the restart sigma below the basin spacing "
                         "keep the population in the basin it was dropped into?")
    args = ap.parse_args()
    if args.csv is not None:
        return paired_mode(args)
    if args.variant not in ("base", "adaptive", "commit_tight"):
        raise SystemExit(f"aggregate mode: unknown --variant {args.variant}")

    b = NICHING_BENCHMARKS_BY_NAME[args.func]
    dim = b.dim
    opt_pts = vincent_optima(dim)
    D = np.linalg.norm(opt_pts[:, None, :] - opt_pts[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    width = D.min(axis=1)          # nearest-neighbour spacing per optimum
    assert len(opt_pts) == b.n_global_optima, (len(opt_pts), b.n_global_optima)

    _CLS = {"base": (MultiChannelEpidemicOptimizer, {}),
            "adaptive": (AdaptiveRepelMCESO, {"repel_mode": "adaptive"}),
            # The whole population committed to one draw, spread at 0.1x the
            # locally observed basin spacing (scripts/diagnose_niching.py's
            # `commit_tight`). Note the tracer records *every* diversified draw,
            # so for this variant the "draws" of a hunt are the anchor plus the
            # cloud placed around it -- `distinct optima drawn near` therefore
            # measures the committed placement, not an independent draw per slot.
            "commit_tight": (CommitReseedMCESO,
                             {"commit_sigma_mode": "run",
                              "commit_sigma_ratio": 0.1})}
    base_cls, kw = _CLS[args.variant]
    cls = _tracer(base_cls)
    kw = dict(kw)
    if args.sigma is not None:
        kw["sigma"] = args.sigma

    n_hunt = n_match = 0
    d_draw: list[float] = []
    land_w: list[float] = []
    draw_w: list[float] = []
    land_ct: Counter = Counter()
    draw_ct: Counter = Counter()
    span = 0.0
    for seed in range(args.seeds):
        o = cls(b, seed=seed * 100, **kw)
        o.optimize(args.evals)
        span = float(b.bounds[1] - b.bounds[0])
        for rec in o.records:
            if not rec["draws"]:
                continue
            n_hunt += 1
            end_i = int(np.argmin(np.linalg.norm(opt_pts - rec["end"], axis=1)))
            dr = np.asarray(rec["draws"])
            draw_i = np.argmin(np.linalg.norm(
                opt_pts[None, :, :] - dr[:, None, :], axis=-1), axis=1)
            n_match += int(end_i in set(draw_i.tolist()))
            d_draw.append(float(np.min(np.linalg.norm(dr - rec["end"], axis=1))))
            land_w.append(float(width[end_i]))
            land_ct[end_i] += 1
            for j in draw_i:
                draw_w.append(float(width[int(j)]))
                draw_ct[int(j)] += 1

    sig = (args.sigma if args.sigma is not None else 0.2) * span
    print(f"{args.func}  variant={args.variant}  seeds={args.seeds}  "
          f"evals={args.evals}  K={b.n_global_optima}")
    print(f"  hunts with draws            {n_hunt}")
    print(f"  endpoint optimum was drawn  {n_match}  ({n_match / max(n_hunt,1):.2f})")
    print(f"  median d(endpoint, nearest draw of its hunt)  {np.median(d_draw):.3f}"
          f"   sigma_init = {sig:.3f}")
    print(f"  basin width (NN spacing) landed in   median {np.median(land_w):.3f}")
    print(f"  basin width (NN spacing) drawn near  median {np.median(draw_w):.3f}")
    print(f"  distinct optima drawn near  {len(draw_ct)} / {b.n_global_optima}")
    print(f"  distinct optima landed in   {len(land_ct)} / {b.n_global_optima}")
    wide = width >= np.median(width)
    dw = sum(c for i, c in draw_ct.items() if wide[i]) / max(sum(draw_ct.values()), 1)
    lw = sum(c for i, c in land_ct.items() if wide[i]) / max(sum(land_ct.values()), 1)
    print(f"  share on the wide half of the optima:  draws {dw:.2f}   landings {lw:.2f}")


if __name__ == "__main__":
    main()
