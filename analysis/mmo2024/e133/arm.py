#!/usr/bin/env python3
"""その133 — 重複を潰した半径での再投入の腕（キュー 3）。

**その125 の `ReseedLander` のサブクラス。変えるのは 1 行 —— `median_nn()` に渡す前に
着地点を `r = 0.5`（= 0.05 × span、合法規則 `eps_loose+dedup` と同じ半径）で貪欲に間引く。**

その125 は `ρ` が 14/16 問でちょうど 0 に潰れていた（着地点が多重集合で、降下の 62.3% が
既存の着地と距離 1e-6 未満）。**壊れたのは推定量だけ**なので、推定量だけを直す ——
**中心の選び方（`reseed` は多重集合から一様、`bhop` は直前の着地点）は 1 ビットも変えない。**

`optima_pos`（採点者の情報）は腕の中で一切参照しない。`core/` も `restart_lander.py` も触らない。

使い方: python3 analysis/mmo2024/e133/arm.py <problem> <arm> <out-dir>
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MMO, "e125"))

from arm import ReseedLander, dump, median_nn      # noqa: E402  （その125 の腕）
from core.benchmarks import niching_by_name        # noqa: E402

DEDUP_R = 0.5          # = 0.05 × span。合法規則 `eps_loose+dedup` と同じ半径


def greedy_dedup(pts: np.ndarray, r: float) -> np.ndarray:
    """到着順に走査し、既に採った点すべてから距離 > r の点だけを採る（貪欲な間引き）。

    `e125/rho_degeneracy.py` の `rho_dedup` と同じ規則。**採点者の情報は使わない。**
    """
    kept: list[np.ndarray] = []
    for p in pts:
        if not kept or np.min(np.linalg.norm(np.asarray(kept) - p, axis=1)) > r:
            kept.append(p)
    return np.asarray(kept)


class DedupReseedLander(ReseedLander):
    """`ρ` を**間引いた**着地点集合から取る以外は その125 と同一。"""

    dedup_r = DEDUP_R

    def _x0(self, rng, lo, hi):
        if len(self._lands) < 2:
            self._trace.append(("uniform", float("nan")))
            return rng.uniform(lo, hi, size=self.dim)
        pts = np.asarray(self._lands)
        kept = greedy_dedup(pts, self.dedup_r)      # ← その125 から変えた唯一の箇所
        if len(kept) < 2:                           # 間引くと 1 点 = ρ が定義できない
            self._trace.append(("uniform", float("nan")))
            return rng.uniform(lo, hi, size=self.dim)
        rho = median_nn(kept)
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


def main():
    prob, arm, out = sys.argv[1], sys.argv[2], sys.argv[3]
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # 予備確認用（0 = 正規予算）
    b = niching_by_name(prob)
    t0 = time.time()
    o = DedupReseedLander(b, seed=0, sigma_ratio=0.1, descent_budget=12500, arm=arm)
    o.optimize(budget or b.suite_max_evals)
    os.makedirs(out, exist_ok=True)
    tag = f"{prob}_{arm}_seed0" + (f"_b{budget}" if budget else "")
    dump(o, b, os.path.join(out, f"{tag}.csv"))
    rho = [r for _, r in o._trace if r == r]
    print(f"done {prob} {arm}  descents={len(o.descents)}  "
          f"rho_med={np.median(rho) if rho else float('nan'):.4f}  "
          f"{time.time() - t0:.0f}s  {time.strftime('%H:%M:%S', time.gmtime())}")


if __name__ == "__main__":
    main()
