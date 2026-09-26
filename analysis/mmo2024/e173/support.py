#!/usr/bin/env python3
"""e173 第 1 部（追加評価ゼロ）: `mpr_sup` の台と `Restart-Lander` の検出集合を水準別に突き合わせる。

`mpr_sup`（その92 の `ceiling()`）は **着地分布の台 / K を 5 水準で平均した量**で、
その92 は「クラスの一員が再起動回数をいくら増やしても超えられない」厳密な上界と書いた。
その172 は M16 でこれが実測に破れることを見つけた（0.600 対 0.700）。

ここでは水準ごとに 2 つの集合を出して差集合を列挙する:
  * `S_null(eps)` —— `e172/null_rl.csv.gz` の 200 抽選（σ₀=0.1、降下上限 12500）で `best_f <= eps` の `land_opt`
  * `S_RL(eps)`   —— `e115/descents/<func>_seed0.csv.gz` の報告点を **`core/runner.py` と同じ経路**で採点したもの
                      （`max(100, 2K)` best-by-f 切り → `benchmark.func` で f を再評価 → `count_goptima_nn`）

差集合が空でなければ (b)（上界の定義が報告規則を覆っていない）か (a)（有限抽選の偏り）。
どちらかは第 2 部（200 → 600 抽選）が決める。`dist` 列も併記して (c)（報告半径の帰属）を見る。

使い方: PYTHONPATH=<pynmmso stub> python3 analysis/mmo2024/e173/support.py
"""
from __future__ import annotations
import csv, gzip, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from core.benchmarks import niching_by_name          # noqa: E402
from core.runner import count_goptima_nn             # noqa: E402

EPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
FUNCS = tuple(f"M{i:02d}-D10-PIN01" for i in
              (16, 1, 13, 14, 2, 3, 4, 5, 6, 7, 8, 12, 15))   # e172 が引いた 13 問


def read_gz(p: Path) -> list[dict]:
    with gzip.open(p, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def null_rows(func: str) -> list[dict]:
    rows = read_gz(HERE.parent / "e172" / "null_rl.csv.gz")
    return [r for r in rows if r["function"] == func]


def rl_reported(func: str, b):
    """`Restart-Lander` の報告集合（`core/runner._niching_counts` と同じ切り方）。"""
    rows = read_gz(HERE.parent / "e115" / "descents" / f"{func}_seed0.csv.gz")
    dim = b.dim
    X = np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rows])
    land = np.array([int(r["land_opt"]) for r in rows])
    dist = np.array([float(r["dist"]) for r in rows])
    F = np.array([float(b.func(x)) for x in X])          # runner と同じく再評価
    cap = max(100, 2 * int(b.n_global_optima))
    if len(F) > cap:
        keep = np.argsort(F)[:cap]
        X, F, land, dist = X[keep], F[keep], land[keep], dist[keep]
    return X, F, land, dist, len(rows)


def main() -> None:
    out_rows = []
    for func in FUNCS:
        b = niching_by_name(func)
        K = int(b.n_global_optima)
        opts = np.asarray(b.optima_pos, dtype=float)
        lo, hi = b.bounds
        span = hi - lo
        nd = null_rows(func)
        nbf = np.array([float(r["best_f"]) for r in nd])
        nland = np.array([int(r["land_opt"]) for r in nd])
        ndist = np.array([float(r["dist"]) for r in nd])
        X, F, land, dist, n_desc = rl_reported(func, b)

        # 最適間の最小距離（(c) の判定尺度）
        dd = np.linalg.norm(opts[:, None, :] - opts[None, :, :], axis=2)
        np.fill_diagonal(dd, np.inf)
        min_gap = float(dd.min())

        print(f"\n=== {func}  K={K}  span={span:g}  min optima gap={min_gap:.4f}")
        print(f"    null draws={len(nd)}   RL descents={n_desc} -> reported={len(F)}")
        for eps in EPS:
            s_null = sorted(set(nland[nbf <= eps].tolist()))
            s_rl = sorted(set(land[F <= eps].tolist()))
            # count_goptima_nn と一致するか（帰属の経路が同じことの確認）
            n_nn = count_goptima_nn(X, F, opts, eps)
            extra = sorted(set(s_rl) - set(s_null))
            missing = sorted(set(s_null) - set(s_rl))
            dtxt = ""
            if extra:
                sel = [(int(j), float(dist[(land == j) & (F <= eps)].min()),
                        float(F[(land == j) & (F <= eps)].min())) for j in extra]
                dtxt = "  extra(dist,f): " + " ".join(
                    f"{j}:({d:.3g},{f:.3g})" for j, d, f in sel)
            print(f"  eps={eps:g}  |S_null|={len(s_null):2d}  |S_RL|={len(s_rl):2d}"
                  f"  (nn={n_nn:2d})  extra={extra}  missing={missing}{dtxt}")
            out_rows.append({"func": func, "K": K, "eps": eps,
                             "n_null_draws": len(nd), "n_rl_reported": len(F),
                             "sup_null": len(s_null), "det_rl": len(s_rl),
                             "nn_check": n_nn,
                             "extra": "|".join(map(str, extra)),
                             "missing": "|".join(map(str, missing)),
                             "min_gap": min_gap, "span": span})
        # dist 分布（(c)）
        print(f"    dist: null median {np.median(ndist):.4g} p95 {np.percentile(ndist,95):.4g}"
              f" | RL median {np.median(dist):.4g} p95 {np.percentile(dist,95):.4g}"
              f" | 0.001*span={0.001*span:.4g}  min_gap/2={min_gap/2:.4g}")
        sup = np.mean([len(set(nland[nbf <= e].tolist())) / K for e in EPS])
        det = np.mean([len(set(land[F <= e].tolist())) / K for e in EPS])
        print(f"    mpr_sup(200 draws) = {sup:.4f}   RL measured MPR = {det:.4f}"
              f"   -> {'BREACHED' if det > sup + 1e-12 else 'tied/ok'}")
    with open(HERE / "support.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {HERE/'support.csv'}")





def breach_detail(func: str) -> None:
    """破れている 1 最適について、「盆地に入っていない」のか「入るが収束しない」のかを分ける。

    `land_opt` は f を見ない最近傍帰属なので、`land_opt == j` の抽選本数と
    そのうち `best_f <= eps` の本数を並べると 2 つを区別できる。
    前者がゼロなら台に入っていない（稀な盆地）。前者が正で後者がゼロなら
    **盆地には入るが降下が収束しない** ＝ 台の推定が「稀さ」ではなく「当たり率」で下に偏っている。
    """
    from scipy.stats import binomtest
    b = niching_by_name(func)
    K = int(b.n_global_optima)
    nd = null_rows(func)
    nbf = np.array([float(r["best_f"]) for r in nd])
    nland = np.array([int(r["land_opt"]) for r in nd])
    X, F, land, dist, n_desc = rl_reported(func, b)
    s_null = set(nland[nbf <= 1e-1].tolist())
    extra = sorted(set(land[F <= 1e-5].tolist()) - s_null)
    print(f"\n=== {func} 破れの内訳（追加評価ゼロ）")
    for j in extra:
        near_null = int((nland == j).sum())
        conv_null = int(((nland == j) & (nbf <= 1e-1)).sum())
        near_rl = int((land == j).sum())
        conv_rl = int(((land == j) & (F <= 1e-5)).sum())
        print(f"  opt {j}: null 200 抽選 —— 最近傍 {near_null} 本 / うち f<=1e-1 {conv_null} 本")
        print(f"          RL {len(F)} 報告点 —— 最近傍 {near_rl} 本 / うち f<=1e-5 {conv_rl} 本")
        if near_null and not conv_null:
            q = conv_rl / max(near_rl, 1)
            print(f"          -> 盆地には入るが降下が収束しない。RL 側の当たり率 {q:.3f} を真値とすると"
                  f" null が {near_null} 本で 0 本になる確率 = {(1-q)**near_null:.4f}")
            print(f"          -> 当たり率が両者で等しいという帰無の Fisher 的な検定（binom, RL の q）:"
                  f" p = {binomtest(conv_null, near_null, q, alternative='less').pvalue:.4f}")
        elif not near_null:
            print("          -> 台にそもそも入っていない（稀な盆地）")


if __name__ == "__main__":
    main()
    for _f in FUNCS:
        breach_detail(_f)
