#!/usr/bin/env python3
"""e173 第 2 部: 200 → 600 抽選で `mpr_sup` が動くかを paired で見る。

`hunt_coverage.py --null` の抽選 k は k だけに依存する（`scripts/hunt_coverage.py:615-619`、
その100 が保存済みダンプと突き合わせて確認済み）ので、**600 抽選は 200 抽選の入れ子**＝
e172 の 200 本とこの回の 400 本を連結したものが 600 抽選そのものになる。

採点は **その92 の `ceiling()` を import**（新しい推定量は書かない）。MPR は 1e-1..1e-5 の 5 水準平均。
ブートストラップ 2000 回の 95% CI 併記。

使い方: PYTHONPATH=<pynmmso stub> python3 analysis/mmo2024/e173/analyze.py
"""
from __future__ import annotations
import csv, gzip, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "e92"))
from core.benchmarks import niching_by_name          # noqa: E402
from analyze import ceiling, EPS                     # noqa: E402  (e92 の推定量)
sys.path.insert(0, str(HERE.parent / "e93"))
import importlib.util as _ilu                        # noqa: E402
_spec = _ilu.spec_from_file_location("e93a", HERE.parent / "e93" / "analyze.py")
_e93 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_e93)
chao1_sup = _e93.chao1_sup                           # noqa: E402  (その93 の推定量)

BOOT = 2000
FUNCS = ("M01-D10-PIN01", "M13-D10-PIN01", "M16-D10-PIN01")
MEAS = {"M01-D10-PIN01": 0.7200, "M13-D10-PIN01": 0.8000, "M16-D10-PIN01": 0.7000}


def read_gz(p: Path) -> list[dict]:
    with gzip.open(p, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def rows_for(func: str) -> list[dict]:
    """draws 0-199（e172）＋ 200-599（この回）。draw 列で重複と欠けを検査する。"""
    out = [r for r in read_gz(HERE.parent / "e172" / "null_rl.csv.gz")
           if r["function"] == func]
    folded = HERE / "null_topup.csv.gz"
    if folded.exists():
        out += [r for r in read_gz(folded) if r["function"] == func]
    else:
        for s in sorted((HERE / "null").glob(f"{func}_s*_rl.csv.gz")):
            out += read_gz(s)
    out.sort(key=lambda r: int(r["draw"]))
    draws = [int(r["draw"]) for r in out]
    assert draws == list(range(len(draws))), f"{func}: draw 列が 0..n-1 でない"
    return out


def arrays(rows):
    return (np.array([float(r["best_f"]) for r in rows]),
            np.array([int(r["land_opt"]) for r in rows]),
            np.array([float(r["evals"]) for r in rows]))


def main() -> None:
    rng = np.random.default_rng(0)
    out = []
    for func in FUNCS:
        rows = rows_for(func)
        b = niching_by_name(func)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        rec = {"func": func, "K": K, "n": len(rows), "meas": MEAS[func]}
        for n in (200, 400, 600):
            if len(rows) < n:
                continue
            bf, ld, ev = arrays(rows[:n])
            mpr, per_eps, sup = ceiling(bf, ld, ev, K, budget)
            m = len(bf)
            boot = np.array([ceiling(bf, ld, ev, K, budget,
                                     rng.integers(0, m, m))[0] for _ in range(BOOT)])
            rec[f"mpr{n}"] = mpr
            rec[f"sup{n}"] = sup
            rec[f"chao{n}"] = chao1_sup(bf, ld, K)
            rec[f"lo{n}"] = float(np.percentile(boot, 2.5))
            rec[f"hi{n}"] = float(np.percentile(boot, 97.5))
            rec[f"cost{n}"] = float(ev.mean())
            rec[f"hit{n}"] = float((bf <= 1e-5).mean())
            rec[f"reach{n}"] = len(set(ld[bf <= 1e-5].tolist()))
            rec[f"supset{n}"] = "|".join(
                str(x) for x in sorted(set(ld[bf <= 1e-1].tolist())))
        out.append(rec)

    hdr = ["func", "K", "n", "meas", "mpr200", "mpr400", "mpr600",
           "sup200", "sup400", "sup600", "chao200", "chao400", "chao600",
           "lo600", "hi600",
           "cost200", "cost600", "hit200", "hit600", "reach200", "reach600"]
    print(" ".join(f"{h:>10}" for h in hdr))
    for r in out:
        print(" ".join(
            (f"{r[h]:>10.4f}" if isinstance(r.get(h), float) else f"{str(r.get(h,'')):>10}")
            for h in hdr))
    print()
    for r in out:
        print(f"{r['func']}: 台（f<=1e-1）  200 -> {r.get('supset200')}")
        print(f"{' ' * len(r['func'])}   600 -> {r.get('supset600')}")
        d = r.get("sup600", float("nan")) - r["sup200"]
        print(f"   chao1_sup（その93）200 {r['chao200']:.4f} -> 600 {r.get('chao600', float('nan')):.4f}"
              f"   200 の chao1 は 600 の plug-in 台 {r.get('sup600', float('nan')):.4f} を"
              f" {'覆う' if r['chao200'] + 1e-12 >= r.get('sup600', 0) else '覆わない'}")
        print(f"   mpr_sup {r['sup200']:.4f} -> {r.get('sup600', float('nan')):.4f}"
              f" ({d:+.4f})   実測 MPR {r['meas']:.4f}"
              f"   -> {'破れは残る' if r.get('sup600', 0) + 1e-12 < r['meas'] else '破れは解消'}")
    with open(HERE / "ceiling600.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in out for k in r}))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {HERE / 'ceiling600.csv'}")


if __name__ == "__main__":
    main()
