#!/usr/bin/env python3
"""その153 (キュー 1) — RR-CMA-ES（de Nobel+ 2024）を D=5 で回す。

**その152 の `run_rrcma.py`（元は その127、その149 経由）を写し、出力先だけ `e153/dumps` に変えた。**
探索側の設定は 1 文字も変えていない ＝ 2 つのモジュールフラグだけ。

**実装は自作ではない。** GECCO'2024 MMO competition の次点 RR-CMA-ES は
de Nobel+ 2024（arXiv 2405.01226 / PPSN XVIII）の提案で、その repelling 機構は
著者自身の `modcma`（modular CMA-ES）に `c_maes.repelling` として入っている
（`modcma==1.2.0`、PyPI の manylinux wheel）。**ここで書いたのは配線だけで、
探索は 1 行も書いていない。**

構成（**既定値から動かしたのは 2 つのモジュールフラグだけ**。ad-hoc な調律をしない）:

  * `modules.restart_strategy = RESTART`  … 再起動する CMA-ES（"R"）
  * `modules.repelling_restart = True`    … taboo 点による反発（"R"）
  * `sigma0` / `lambda0` / その他はライブラリ既定（`sigma0 = 2.0 = 0.2*span`、`lambda0 = 10`）

**報告集合**: repelling の taboo archive（= 手法自身が「別の盆地の局所最適」と
判定して登録した点）＋ 最終の global best 1 点。**採点者の情報は 1 つも使っていない。**
出力はその110/115 の降下ダンプと同じ列（`best_f` / `land_opt` / `x*`）なので、
`analysis/mmo2024/e115/analyze.py` の規則と採点をそのまま import して使える。
`land_opt` / `dist` はオラクル列（診断専用）で、手法には渡らない。

使い方: python3 analysis/mmo2024/e153/run_rrcma.py <problem> [seed] [budget]
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.benchmarks import niching_by_name  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "dumps"  # e153/dumps（その152 は e152/dumps。HERE 依存なのでコード上の差はこの註だけ）


def run(prob_name: str, seed: int, budget: int | None = None):
    from modcma import c_maes

    p = niching_by_name(prob_name)
    lo, hi = p.bounds
    dim = int(p.dim)
    budget = int(budget or p.suite_max_evals)

    np.random.seed(seed)
    try:                       # 乱数の固定（C++ 側の generator）
        c_maes.utils.set_seed(seed)
    except Exception:
        pass

    mods = c_maes.Modules()
    mods.restart_strategy = c_maes.options.RestartStrategy.RESTART
    mods.repelling_restart = True

    s = c_maes.Settings(dim, modules=mods, budget=budget,
                        lb=np.full(dim, lo), ub=np.full(dim, hi))
    es = c_maes.ModularCMAES(s)

    t0 = time.time()
    es.run(lambda x: float(p.func(np.asarray(x, dtype=float))))
    elapsed = time.time() - t0

    rows = []
    for k, tp in enumerate(es.p.repelling.archive):
        rows.append((k, float(tp.solution.y),
                     np.asarray(tp.solution.x, dtype=float), "taboo"))
    gb = es.p.stats.global_best
    rows.append((len(rows), float(gb.y),
                 np.asarray(gb.x, dtype=float), "final"))

    opts = np.asarray(p.optima_pos, dtype=float)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{prob_name}_rrcma_seed{seed}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["descent", "evals", "best_f", "land_opt", "dist", "stop"]
                   + [f"x{i}" for i in range(dim)])
        for k, y, x, tag in rows:
            d = np.linalg.norm(opts - x, axis=1)
            j = int(np.argmin(d))
            w.writerow([k, "", f"{y:.12g}", j, f"{d[j]:.6g}", tag]
                       + [f"{v:.12g}" for v in x])

    print(f"{prob_name} seed{seed}: evals={es.p.stats.evaluations} "
          f"archive={len(rows) - 1} (+final) K={p.n_global_optima} "
          f"fopt={float(gb.y):.4g} {elapsed:.1f}s -> {path.name}",
          flush=True)


if __name__ == "__main__":
    prob = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    bud = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run(prob, seed, bud)
