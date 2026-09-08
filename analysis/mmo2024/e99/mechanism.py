"""Is the Chao1 breach real coverage or the estimator? (entry 99, mechanism check)

Entry 98's one breach (M16, `mpr_sup_chao1` 0.900) opened to nothing: both arms
had reached the same 6 optima and the whole difference was Chao1's correction
term firing on f2 = 0.  This cycle's breach is on the 16-problem mean, so the
same check has to be run before it is reported.  Two parts:

1. **Decomposition.** `mpr_sup_chao1` = S_obs/K + correction/K, averaged over the
   five accuracies.  Split the isotropic -> sigma=0.1 rise into the part that is
   optima actually landed on and the part that is the estimator, and print f1/f2
   wherever the correction fires, so an f2 = 0 blow-up is visible.

2. **Paired bootstrap over draws.** The same resampled draw indices score every
   problem and both arms, so the *difference* is a paired quantity.
   **The level is not**: resampling 200 draws with replacement leaves ~63%
   distinct, and S_obs is a richness statistic, so every bootstrap replicate of
   the level is biased DOWN.  The CI on the difference is usable; the fraction of
   replicates above 0.651 is not a test of the level and must not be read as one.

Usage: python3 analysis/mmo2024/e99/mechanism.py
"""
import csv
import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from core.benchmarks import niching_by_name          # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EPS = _load_module("e92_analyze", HERE.parent / "e92" / "analyze.py").EPS
ISO_DIRS = (HERE.parent / "e92", HERE.parent / "e94", HERE.parent / "e91")
SIG_DIRS = (HERE, HERE.parent / "e98")
NAMES = [f"M{p:02d}-D10-PIN01" for p in range(1, 17)]
PUBLISHED_D10 = 0.651
BOOT = 2000


def read(dirs, nm, suffix):
    for d in dirs:
        for ext in (".csv.gz", ".csv"):
            src = d / f"{nm}{suffix}{ext}"
            if not src.exists():
                continue
            op = gzip.open if src.suffix == ".gz" else open
            with op(src, "rt") as fh:
                rows = sorted(csv.DictReader(fh), key=lambda r: int(r["draw"]))
            return (np.array([float(r["best_f"]) for r in rows]),
                    np.array([int(r["land_opt"]) for r in rows]))
    raise FileNotFoundError(f"{nm}{suffix}")


def chao(best_f, land, K, idx):
    v = []
    for eps in EPS:
        c = np.bincount(land[idx][best_f[idx] <= eps], minlength=K)
        s = int((c > 0).sum())
        f1, f2 = int((c == 1).sum()), int((c == 2).sum())
        v.append(min(s + f1 * (f1 - 1) / (2 * (f2 + 1)), K) / K)
    return float(np.mean(v))


def main() -> None:
    data, acc = {}, {k: [] for k in ("iso_s", "iso_c", "sig_s", "sig_c")}
    print("Chao1 decomposition: observed support vs the "
          "f1(f1-1)/(2(f2+1)) correction, averaged over the five accuracies")
    print(f"{'func':<6}{'K':>4}{'iso S_obs':>11}{'iso corr':>10}"
          f"{'sig S_obs':>11}{'sig corr':>10}{'d S_obs':>9}{'d corr':>9}")
    for nm in NAMES:
        K = int(niching_by_name(nm).n_global_optima)
        bfs, ls = read(SIG_DIRS, nm, "_sig100200")
        bfi, li = read(ISO_DIRS, nm, "_desc200")
        data[nm] = (K, bfs, ls, bfi, li)
        out = {}
        for lab, bf, land in (("iso", bfi, li), ("sig", bfs, ls)):
            so, co = [], []
            for eps in EPS:
                c = np.bincount(land[bf <= eps], minlength=K)
                s = int((c > 0).sum())
                f1, f2 = int((c == 1).sum()), int((c == 2).sum())
                so.append(s)
                co.append(min(s + f1 * (f1 - 1) / (2 * (f2 + 1)), K) - s)
            out[lab] = (float(np.mean(so)), float(np.mean(co)))
        for lab in ("iso", "sig"):
            acc[f"{lab}_s"].append(out[lab][0] / K)
            acc[f"{lab}_c"].append(out[lab][1] / K)
        print(f"{nm[:3]:<6}{K:>4}{out['iso'][0]:>11.2f}{out['iso'][1]:>10.2f}"
              f"{out['sig'][0]:>11.2f}{out['sig'][1]:>10.2f}"
              f"{out['sig'][0] - out['iso'][0]:>+9.2f}"
              f"{out['sig'][1] - out['iso'][1]:>+9.2f}")

    print("\nwhere the correction fires in the sigma=0.1 arm (f1 > 1); "
          "f2 = 0 is the blow-up entry 98's breach turned out to be:")
    for nm in NAMES:
        K, bfs, ls, _, _ = data[nm]
        det = []
        for eps in EPS:
            c = np.bincount(ls[bfs <= eps], minlength=K)
            f1, f2 = int((c == 1).sum()), int((c == 2).sum())
            if f1 > 1:
                det.append(f"eps{eps:g}: f1={f1},f2={f2}")
        if det:
            print(f"  {nm[:3]}: " + "; ".join(det))

    for lab, tag in (("iso", "isotropic"), ("sig", "sigma=0.1 ")):
        s, c = np.mean(acc[f"{lab}_s"]), np.mean(acc[f"{lab}_c"])
        print(f"\n16-problem mean as a fraction of K -- {tag}: "
              f"S_obs/K {s:.4f} + correction/K {c:.4f} = {s + c:.4f}")
    print(f"  the rise: real support "
          f"{np.mean(acc['sig_s']) - np.mean(acc['iso_s']):+.4f}, "
          f"estimator {np.mean(acc['sig_c']) - np.mean(acc['iso_c']):+.4f}")

    rng = np.random.default_rng(1)
    m = len(next(iter(data.values()))[1])
    bs = {"sig": np.empty(BOOT), "iso": np.empty(BOOT)}
    for b in range(BOOT):
        ix = rng.integers(0, m, m)     # same indices for every problem and arm
        bs["sig"][b] = np.mean([chao(d[1], d[2], d[0], ix) for d in data.values()])
        bs["iso"][b] = np.mean([chao(d[3], d[4], d[0], ix) for d in data.values()])
    d = bs["sig"] - bs["iso"]
    print(f"\npaired bootstrap over draws (B={BOOT}), 16-problem mean mpr_sup_chao1:")
    for lab in ("sig", "iso"):
        print(f"  {lab:<4}: {bs[lab].mean():.4f} "
              f"[{np.percentile(bs[lab], 2.5):.4f}, {np.percentile(bs[lab], 97.5):.4f}]")
    print(f"  paired difference {d.mean():+.4f} "
          f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  "
          f"<- this is the usable CI")
    print("  NOTE: the per-arm levels above are biased DOWN (a bootstrap draw "
          "keeps ~63% distinct draws and S_obs is a richness statistic), so "
          f"they sit below the point estimates and the share of replicates over "
          f"{PUBLISHED_D10} ({(bs['sig'] > PUBLISHED_D10).mean():.3f}) is NOT a "
          "test of the level.")


if __name__ == "__main__":
    main()
