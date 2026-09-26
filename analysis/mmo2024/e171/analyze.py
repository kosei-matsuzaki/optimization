#!/usr/bin/env python3
"""e171: クラス上限表を `Restart-Lander` の実際の降下で引き直す（事前登録は `prereg.md`）。

その88・その89 の上限表は `hunt_coverage.py --null` の既定（sigma0 = 0.2 x span、降下上限
1499 評価 ＝ **MC-ESO の hunt**）で引かれている。`Restart-Lander` の既定は 0.1 x span / 12500
（`core/optimizers/restart_lander.py:55-56`）で、その170 §3 は CF3-2D/3D でこの 1 点が
「盆地ゼロ」と「1 run に 1.2 本」を入れ替えることを示した。本スクリプトは同じ 600 抽選を
2 設定で引き（`--null` の抽選 k は k だけに依存するので **抽選単位で対**）、

  * その88 §6 の手順で 1 run 分の相異なる最適数を出し（推定量は e92 の `ceiling()` を import。
    新しい推定量は書かない）、
  * 閾値 = 公表最良 `PR@1e-5` x K と照合し、
  * ブートストラップ 2000 回の 95% CI を併記し（上限は下向きに偏るので「超えない」側は上端で見る）、
  * CF3 の {2,3} / CF4 の {4,5} が RL の降下で開くかを抽選単位の対で読む。

使い方: PYTHONPATH=<pynmmso stub> python3 analysis/mmo2024/e171/analyze.py
"""
from __future__ import annotations
import csv, gzip, sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "e92"))
from core.benchmarks import niching_by_name          # noqa: E402
from analyze import ceiling, EPS                     # noqa: E402  (e92 の推定量)

BOOT = 2000
# 公表最良 `PR@1e-5`（その88 §1 の表。出典は acceptance_topology.md の その88 の節）
PUB = {"N14-CF3-3D": 0.793, "N15-CF4-3D": 0.750, "N16-CF3-5D": 0.677,
       "N17-CF4-5D": 0.745, "N18-CF3-10D": 0.667, "N19-CF4-10D": 0.512,
       "N20-CF4-20D": 0.465}
# その88 §1 / その89 §2 が記録した既定の降下での上限（2000 本。参照のため並べる）
E88 = {"N14-CF3-3D": 4.00, "N15-CF4-3D": 5.58, "N16-CF3-5D": 4.00,
       "N17-CF4-5D": 5.84, "N18-CF3-10D": 3.29, "N19-CF4-10D": 3.89,
       "N20-CF4-20D": 2.54}
FUNCS = list(PUB)
TAGS = ("rl", "base")
EPSF = 1e-5


_FOLD: dict[str, list[dict]] = {}


def _read(src: Path) -> list[dict]:
    op = gzip.open if src.suffix == ".gz" else open
    with op(src, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def load(func: str, tag: str):
    """畳んだ 1 本（`fold.py` の出力）を優先し、無ければ per-function のダンプを読む。

    畳む前後で本スクリプトの出力が 1 文字も変わらないことを確認してから元 14 本を消す
    （その128〜その131・その170 と同じ手順）。
    """
    folded = HERE / f"null_{tag}.csv.gz"
    if folded.exists():
        if tag not in _FOLD:
            _FOLD[tag] = _read(folded)
        rows = [r for r in _FOLD[tag] if r["function"] == func]
    else:
        p = HERE / "null" / f"{func}_{tag}.csv.gz"
        q = HERE / "null" / f"{func}_{tag}.csv"
        src = p if p.exists() else (q if q.exists() else None)
        if src is None:
            return None
        rows = _read(src)
    if not rows:
        return None
    return {"best_f": np.array([float(r["best_f"]) for r in rows]),
            "land": np.array([int(r["land_opt"]) for r in rows]),
            "evals": np.array([float(r["evals"]) for r in rows]),
            "draw": np.array([int(r["draw"]) for r in rows]) if "draw" in rows[0]
                    else np.arange(len(rows))}


def main() -> None:
    rng = np.random.default_rng(0)
    out, pairs = [], []
    for f in FUNCS:
        b = niching_by_name(f)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        row = {"func": f, "K": K, "pub": PUB[f], "thr": PUB[f] * K, "e88": E88[f]}
        for tag in TAGS:
            d = load(f, tag)
            if d is None:
                row[f"{tag}_n"] = 0
                continue
            m = len(d["best_f"])
            mpr, per_eps, _ = ceiling(d["best_f"], d["land"], d["evals"], K, budget)
            cnt = per_eps[EPS.index(EPSF)] * K
            boot = np.array([ceiling(d["best_f"], d["land"], d["evals"], K, budget,
                                     rng.integers(0, m, m))[1][EPS.index(EPSF)] * K
                             for _ in range(BOOT)])
            hit = d["best_f"] <= EPSF
            row.update({f"{tag}_n": m,
                        f"{tag}_cost": float(d["evals"].mean()),
                        f"{tag}_restarts": budget / d["evals"].mean(),
                        f"{tag}_hit": float(hit.mean()),
                        f"{tag}_reach": sorted(set(d["land"][hit].tolist())),
                        f"{tag}_cnt": float(cnt),
                        f"{tag}_lo": float(np.percentile(boot, 2.5)),
                        f"{tag}_hi": float(np.percentile(boot, 97.5))})
        out.append(row)

    # ── 表 1: 上限の引き直し ────────────────────────────────────────────────
    print("== 表 1: クラス上限（`best_f <= 1e-5` で絞ってから多項抽出、600 抽選、"
          "1 run 予算 400,000）")
    hdr = ("func", "K", "thr", "e88(2000)", "base_cnt", "base_CI",
           "RL_cnt", "RL_CI", "RL-thr", "over?")
    print(" ".join(f"{h:>12}" for h in hdr))
    for r in out:
        if not r.get("rl_n"):
            print(f"{r['func']:>12}  (RL dump なし)")
            continue
        over = "YES" if r["rl_cnt"] > r["thr"] else (
            "CI" if r["rl_hi"] > r["thr"] else "no")
        cells = [r["func"], str(r["K"]), f"{r['thr']:.2f}", f"{r['e88']:.2f}",
                 f"{r.get('base_cnt', float('nan')):.2f}",
                 f"[{r.get('base_lo', float('nan')):.2f},{r.get('base_hi', float('nan')):.2f}]",
                 f"{r['rl_cnt']:.2f}", f"[{r['rl_lo']:.2f},{r['rl_hi']:.2f}]",
                 f"{r['rl_cnt'] - r['thr']:+.2f}", over]
        print(" ".join(f"{c:>12}" for c in cells))

    # ── 表 2: 降下のコストと深さ ───────────────────────────────────────────
    print("\n== 表 2: 1 降下のコスト・再起動回数・`best_f<=1e-5` 率（対）")
    hdr2 = ("func", "base_cost", "RL_cost", "base_n_rst", "RL_n_rst",
            "base_hit", "RL_hit")
    print(" ".join(f"{h:>12}" for h in hdr2))
    for r in out:
        if not r.get("rl_n") or not r.get("base_n"):
            continue
        print(" ".join(f"{c:>12}" for c in
                       [r["func"], f"{r['base_cost']:.0f}", f"{r['rl_cost']:.0f}",
                        f"{r['base_restarts']:.0f}", f"{r['rl_restarts']:.0f}",
                        f"{r['base_hit']:.3f}", f"{r['rl_hit']:.3f}"]))

    # ── 表 3: 到達添字（その88 §4 の引き直し）─────────────────────────────
    print("\n== 表 3: `best_f<=1e-5` で到達した添字と、1 度も到達しない添字")
    for r in out:
        if not r.get("rl_n"):
            continue
        K = r["K"]
        for tag in TAGS:
            if f"{tag}_reach" not in r:
                continue
            miss = sorted(set(range(K)) - set(r[f"{tag}_reach"]))
            print(f"  {r['func']:<13} {tag:<5} reach {r[f'{tag}_reach']}"
                  f"  miss {miss}")

    # ── 対検定: 7 関数で RL 設定は上限を上げるか ───────────────────────────
    # 両設定が同じ上限に飽和した関数（N16 の 4.00 対 4.00）は差が 5e-14 ＝ 浮動小数の
    # 残差なので、**同点として落とす**（順位に混ぜると検定が見かけの n を 1 つ増やす）。
    TOL = 1e-9
    d = [(r["rl_cnt"] - r["base_cnt"]) for r in out
         if r.get("rl_n") and r.get("base_n")]
    if len(d) >= 3:
        nz = [v for v in d if abs(v) > TOL]
        pos = sum(v > TOL for v in d)
        neg = sum(v < -TOL for v in d)
        tie = len(d) - pos - neg
        w = wilcoxon(nz, alternative="two-sided") if len(nz) >= 3 else None
        pmin = 2.0 / (2 ** len(nz)) if nz else float("nan")
        print(f"\n== 対検定（同じ 600 抽選、RL 設定 − 既定）関数 n={len(d)}  "
              f"平均 {np.mean(d):+.3f}  中央 {np.median(d):+.3f}  "
              f"正 {pos} / 同点 {tie} / 負 {neg}  有効対 {len(nz)}"
              + (f"  両側 Wilcoxon exact p={w.pvalue:.5g}"
                 f"（最小達成可能 p {pmin:.5g}）" if w else ""))

    with open(HERE / "ceiling_rl.csv", "w", newline="") as fh:
        keys = sorted({k for r in out for k in r})
        w2 = csv.DictWriter(fh, fieldnames=keys)
        w2.writeheader()
        w2.writerows(out)
    print(f"\nwrote {HERE / 'ceiling_rl.csv'}")


if __name__ == "__main__":
    main()
