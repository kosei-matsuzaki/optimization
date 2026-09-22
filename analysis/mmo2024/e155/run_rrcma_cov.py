#!/usr/bin/env python3
"""その155（キュー 1）— RR-CMA-ES の coverage factor c を振り、D=5 の MPR 赤字が
配線（既定 c=20）で説明できるかを見る。

**その153 の `run_rrcma.py` を写し、変えたのは 3 点だけ**:
  1. `es.p.repelling.coverage = c`（de Nobel+ 2024 §4.3 の被覆係数。**modcma 既定は 20.0**）
  2. `es.run()` を `while es.step(f)` の手回しに変え、**再起動ごとに評価回数と打ち切り理由を記録する**
     （その153 の保存物は `evals` 列が空・`stop` 列は taboo/final のタグなので、
      キュー 1 (ii) が要求した「1 再起動あたりの評価回数と打ち切り理由」は保存物からは出せない）
  3. 出力に `restarts.csv`（再起動単位）を足した。`dumps/` の列は その153 と同一。

探索側でこれ以外に動かした設定は無い（`sigma0` / `lambda0` はライブラリ既定）。

使い方: python3 analysis/mmo2024/e155/run_rrcma_cov.py <problem> <cov> [seed] [budget]
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.benchmarks import niching_by_name  # noqa: E402

HERE = Path(__file__).resolve().parent


def run(prob_name: str, cov: float, seed: int = 0, budget: int | None = None):
    from modcma import c_maes

    p = niching_by_name(prob_name)
    lo, hi = p.bounds
    dim = int(p.dim)
    budget = int(budget or p.suite_max_evals)

    np.random.seed(seed)
    try:
        c_maes.utils.set_seed(seed)
    except Exception:
        pass

    mods = c_maes.Modules()
    mods.restart_strategy = c_maes.options.RestartStrategy.RESTART
    mods.repelling_restart = True

    s = c_maes.Settings(dim, modules=mods, budget=budget,
                        lb=np.full(dim, lo), ub=np.full(dim, hi))
    es = c_maes.ModularCMAES(s)
    es.p.repelling.coverage = float(cov)

    func = lambda x: float(p.func(np.asarray(x, dtype=float)))  # noqa: E731

    t0 = time.time()
    restarts = []          # (restart_index, evals_at_restart, reason, archive_size)
    n_centers = 0
    prev_evals = 0
    while es.step(func):
        c = len(es.p.stats.centers)
        if c > n_centers:
            ev = int(es.p.stats.evaluations)
            try:
                reason = str(es.p.criteria.reason())
            except Exception:
                reason = "?"
            restarts.append((c, ev, ev - prev_evals, reason,
                             len(es.p.repelling.archive)))
            prev_evals = ev
            n_centers = c
    elapsed = time.time() - t0

    tag = f"cov{cov:g}"
    outd = HERE / "dumps"
    outd.mkdir(parents=True, exist_ok=True)
    (HERE / "restarts").mkdir(parents=True, exist_ok=True)

    rows = []
    for k, tp in enumerate(es.p.repelling.archive):
        rows.append((k, float(tp.solution.y),
                     np.asarray(tp.solution.x, dtype=float), "taboo",
                     int(tp.n_rep)))
    gb = es.p.stats.global_best
    rows.append((len(rows), float(gb.y), np.asarray(gb.x, dtype=float), "final", 0))

    opts = np.asarray(p.optima_pos, dtype=float)
    path = outd / f"{prob_name}_{tag}_seed{seed}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["descent", "evals", "best_f", "land_opt", "dist", "stop", "n_rep"]
                   + [f"x{i}" for i in range(dim)])
        for k, y, x, t, nr in rows:
            d = np.linalg.norm(opts - x, axis=1)
            j = int(np.argmin(d))
            w.writerow([k, "", f"{y:.12g}", j, f"{d[j]:.6g}", t, nr]
                       + [f"{v:.12g}" for v in x])

    rpath = HERE / "restarts" / f"{prob_name}_{tag}_seed{seed}.csv"
    with open(rpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["restart", "evals_cum", "evals_this", "reason", "archive_size"])
        w.writerows(restarts)

    print(f"{prob_name} {tag}: evals={es.p.stats.evaluations} restarts={len(restarts)} "
          f"archive={len(rows)-1} K={p.n_global_optima} {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]),
        int(sys.argv[3]) if len(sys.argv) > 3 else 0,
        int(sys.argv[4]) if len(sys.argv) > 4 else None)
