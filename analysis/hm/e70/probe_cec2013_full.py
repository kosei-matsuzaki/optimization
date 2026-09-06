"""その70: CEC2013-niching F11-F20 の登録可否 probe（最適化 run はしない）。

3 つを測る:
  1. `ioh` が F11-F20（合成関数 CF1-CF4）を持っているか、メタデータが公式表と一致するか
  2. ioh の値が参照実装（mikeagn/CEC2013 python3）と一致するか
  3. 1 eval のコスト（＝ 正規予算 MaxFEs を掛けると 1 run の関数評価コストになる）

参照実装を使う分岐は `--ref-dir <dir>` を渡したときだけ動く（data/ と cec2013/ を含む
ディレクトリ。raw.githubusercontent.com/mikeagn/CEC2013/master/python3 から取れる）。
渡さなければ ioh 側だけを測る。
"""
import argparse, csv, sys, time
import numpy as np
import ioh

# 公式表（Li, Engelbrecht & Epitropakis 2013 の get_* と一致することを確認する対象）
# fid -> (dim, K, rho, MaxFEs)
OFFICIAL = {
    11: (2,  6, 0.01, 200_000), 12: (2,  8, 0.01, 200_000),
    13: (2,  6, 0.01, 200_000), 14: (3,  6, 0.01, 400_000),
    15: (3,  8, 0.01, 400_000), 16: (5,  6, 0.01, 400_000),
    17: (5,  8, 0.01, 400_000), 18: (10, 6, 0.01, 400_000),
    19: (10, 8, 0.01, 400_000), 20: (20, 8, 0.01, 400_000),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default=None)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--out", default="analysis/hm/e70/cec2013_f11_f20.csv")
    a = ap.parse_args()

    ref = None
    if a.ref_dir:
        sys.path.insert(0, a.ref_dir)
        from cec2013.cec2013 import CEC2013   # noqa: E402
        ref = CEC2013

    rows = []
    for fid, (dim, k, rho, maxfes) in OFFICIAL.items():
        p = ioh.get_problem(1100 + fid, 1, dim, ioh.ProblemClass.CEC2013)
        rng = np.random.default_rng(2)
        lb, ub = float(p.bounds.lb[0]), float(p.bounds.ub[0])
        X = lb + (ub - lb) * rng.random((a.n_points, dim))

        err = ""
        if ref is not None:
            r = ref(fid)
            err = f"{max(abs(p(list(x)) - r.evaluate(x)) for x in X):.2e}"

        t0 = time.time()
        for x in X:
            p(list(x))
        us = (time.time() - t0) / a.n_points * 1e6

        # 真の大域最適の位置が ioh から取れるか（取れれば hunt_coverage.py が転用できる）
        opt = np.array([o.x for o in p.optima])
        f_at_opt = max(abs(p(list(o))) for o in opt)

        rows.append(dict(
            fid=f"F{fid}", ioh_name=p.meta_data.name, dim=dim,
            K_official=k, K_ioh=p.n_optima, rho_official=rho, rho_ioh=p.rho,
            maxfes=maxfes, box_lo=lb, box_hi=ub, f_goptima=float(p.optimum.y),
            n_optima_listed=len(opt), max_abs_f_at_listed_optima=f"{f_at_opt:.1e}",
            max_abs_diff_vs_reference=err, us_per_eval=round(us, 1),
            eval_sec_per_run=round(us * maxfes / 1e6, 1),
        ))

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
