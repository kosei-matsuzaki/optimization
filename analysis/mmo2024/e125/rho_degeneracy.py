#!/usr/bin/env python3
"""その125 の事後検算 — 事前登録した `ρ`（着地点集合の最近傍距離の中央値）はなぜ 0 になるか。

**追加評価ゼロ。**保存済みの `null` 降下ダンプ（e115、seed 0）だけを読む。
出力は `rho_degeneracy.csv`（16 行）。腕の run そのものは読まない
（腕の `rho` 列は `scored.txt` の計器表にある）。

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e125/rho_degeneracy.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
# HERE を先に入れる（`analyze` は e115 の側を引きたいので e115 を後に入れて前に置く）
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(MMO, "e115"))

from analyze import read_dump                       # noqa: E402  (e115)
from arm import median_nn                           # noqa: E402
from core.benchmarks import niching_by_name         # noqa: E402

DEDUP_R = 0.5   # 合法規則 `eps_loose+dedup` と同じ半径


def main():
    out = []
    hdr = ("problem", "n_descents", "n_distinct", "rho_raw", "rho_dedup",
           "nn_optima_median", "pct_exact_duplicate")
    print(f"{hdr[0]:<8}{'desc':>7}{'distinct':>9}{'rho_raw':>9}"
          f"{'rho_dedup':>10}{'NN_opt':>8}{'dup%':>7}")
    for i in range(1, 17):
        p = f"M{i:02d}-D10-PIN01"
        path = os.path.join(MMO, "e115", "descents", f"{p}_seed0.csv.gz")
        f, _opt, xs = read_dump(path)
        raw = median_nn(xs)
        keep = []
        for j in np.argsort(f, kind="stable"):
            if all(np.linalg.norm(xs[j] - xs[k]) > DEDUP_R for k in keep):
                keep.append(int(j))
        ded = median_nn(xs[keep]) if len(keep) > 1 else float("nan")
        opts = np.asarray(niching_by_name(p).optima_pos, dtype=float)
        nn_opt = median_nn(opts)
        d = np.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        dup = float((d.min(axis=1) < 1e-6).mean()) * 100
        out.append((p, len(f), len(keep), raw, ded, nn_opt, dup))
        print(f"{p[:3]:<8}{len(f):>7}{len(keep):>9}{raw:>9.3f}{ded:>10.3f}"
              f"{nn_opt:>8.3f}{dup:>7.1f}")
    a = np.array([r[1:] for r in out], dtype=float)
    print(f"\n16 問平均: 降下 {a[:, 0].mean():.1f}  相異なる着地 {a[:, 1].mean():.1f}  "
          f"rho_raw 中央値 {np.median(a[:, 2]):.3f}  rho_dedup 平均 {np.nanmean(a[:, 3]):.3f}  "
          f"最適の NN 中央値の平均 {a[:, 4].mean():.3f}  "
          f"最近傍が同一点の降下 {a[:, 5].mean():.1f}%")
    with open(os.path.join(HERE, "rho_degeneracy.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for r in out:
            w.writerow([r[0], r[1], r[2]] + [f"{v:.4f}" for v in r[3:]])


if __name__ == "__main__":
    main()
