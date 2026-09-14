#!/usr/bin/env python3
"""その125 — 着地点近傍への再投入の腕（キュー 2）。

`RestartLanderOptimizer` のサブクラス。**変えるのは始点 `x0` の決め方 1 箇所だけ**で、
降下作用素（isotropic CMA-ES, sigma_ratio=0.1, descent_budget=12500）・予算の使い方・
報告集合の作り方・ダンプ形式はすべて親のまま。`core/` は 1 ビットも触っていない。

半径 `ρ` は**着地点集合の最近傍距離の中央値**で、**採点者の情報（`optima_pos`）は使わない**。
着地点が 2 個未満のあいだは `ρ` が定義できないので、**どちらの腕も一様再起動**（同一規則）。

使い方: python3 analysis/mmo2024/e125/arm.py <problem> <arm> <out-dir>
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from core.benchmarks import niching_by_name                      # noqa: E402
from core.optimizers.restart_lander import RestartLanderOptimizer  # noqa: E402


def median_nn(pts: np.ndarray) -> float:
    """着地点集合の最近傍距離の中央値。手法が自分で観測できる量。"""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


class ReseedLander(RestartLanderOptimizer):
    """`arm='reseed'`: 確率 p で既存の着地点の周り半径 ρ から引く。
    `arm='bhop'`  : 常に**直前の**着地点の周り半径 ρ から引く（素の basin-hopping、受理判定なし）。
    """

    def __init__(self, *a, arm: str = "reseed", p_reseed: float = 0.5, **kw):
        super().__init__(*a, **kw)
        self.arm = arm
        self.p_reseed = float(p_reseed)
        self._lands: list[np.ndarray] = []
        self._trace: list[tuple[str, float]] = []   # (mode, rho) per descent

    # 親の optimize() の `x0 = rng.uniform(lo, hi, size=self.dim)` に当たる分岐。
    def _x0(self, rng, lo, hi):
        if len(self._lands) < 2:                    # ρ が定義できない
            self._trace.append(("uniform", float("nan")))
            return rng.uniform(lo, hi, size=self.dim)
        pts = np.asarray(self._lands)
        rho = median_nn(pts)
        if self.arm == "bhop":
            centre = pts[-1]
        else:
            if rng.random() >= self.p_reseed:
                self._trace.append(("uniform", rho))
                return rng.uniform(lo, hi, size=self.dim)
            centre = pts[rng.integers(len(pts))]
        u = rng.normal(size=self.dim)
        u /= np.linalg.norm(u)
        r = rho * rng.random() ** (1.0 / self.dim)  # 半径 ρ の球内一様
        self._trace.append((self.arm, rho))
        return np.clip(centre + r * u, lo, hi)

    def optimize(self, max_evals: int = 5000):
        # 親の本体をそのまま写し、`x0` の 1 行と着地点の記録だけを差し替える。
        import cma
        lo, hi = self.bounds
        sigma0 = self.sigma_ratio * (hi - lo)
        rng = np.random.default_rng(self.seed)
        hx: list[np.ndarray] = []
        hf: list[float] = []
        reported: list[np.ndarray] = []
        used = 0
        k = 0
        while used < max_evals:
            x0 = self._x0(rng, lo, hi)                        # ← ここだけが違う
            cap = min(self.descent_budget, max_evals - used)
            o = {"bounds": [lo, hi], "maxfevals": cap, "seed": self.seed + k + 1,
                 "verbose": -9, "tolfun": 0, "tolfunhist": 0, "tolx": 0}
            if self.iso:
                o["CMA_on"] = 0
            es = cma.CMAEvolutionStrategy(list(x0), sigma0, o)
            best_f, best_x, spent = float("inf"), x0, 0
            while not es.stop():
                xs = es.ask()
                if used + len(xs) > max_evals:
                    break
                fs = [float(self.func(np.asarray(x))) for x in xs]
                used += len(xs)
                spent += len(xs)
                es.tell(xs, fs)
                hx.extend(np.asarray(x, dtype=float) for x in xs)
                hf.extend(fs)
                i = int(np.argmin(fs))
                if fs[i] < best_f:
                    best_f, best_x = fs[i], np.asarray(xs[i], dtype=float)
            if spent == 0:
                break
            reported.append(best_x.copy())
            self._lands.append(best_x.copy())                  # ← 腕が使う唯一の記憶
            self.descents.append({"descent": k, "evals": spent, "best_f": best_f,
                                  "stop": "|".join(sorted(es.stop())) or "budget",
                                  "x": best_x.copy()})
            k += 1
        return self._make_result(hx, hf, solutions=reported)


def dump(opt, bench, path):
    """e110/e115 と同じ列 ＋ 腕の計器 2 列（`mode` / `rho`）。read_dump は後ろの列を無視する。"""
    optima = np.asarray(bench.optima_pos, dtype=float)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["descent", "evals", "best_f", "land_opt", "dist", "stop"]
                   + [f"x{i}" for i in range(bench.dim)] + ["mode", "rho"])
        for r, (mode, rho) in zip(opt.descents, opt._trace):
            dd = np.linalg.norm(optima - r["x"], axis=1)
            j = int(np.argmin(dd))
            w.writerow([r["descent"], r["evals"], f"{r['best_f']:.12g}", j,
                        f"{dd[j]:.6g}", r["stop"]]
                       + [f"{v:.12g}" for v in r["x"]] + [mode, f"{rho:.6g}"])


def main():
    prob, arm, out = sys.argv[1], sys.argv[2], sys.argv[3]
    b = niching_by_name(prob)
    t0 = time.time()
    o = ReseedLander(b, seed=0, sigma_ratio=0.1, descent_budget=12500, arm=arm)
    o.optimize(b.suite_max_evals)
    os.makedirs(out, exist_ok=True)
    dump(o, b, os.path.join(out, f"{prob}_{arm}_seed0.csv"))
    print(f"done {prob} {arm}  descents={len(o.descents)}  "
          f"{time.time() - t0:.0f}s  {time.strftime('%H:%M:%S', time.gmtime())}")


if __name__ == "__main__":
    main()
