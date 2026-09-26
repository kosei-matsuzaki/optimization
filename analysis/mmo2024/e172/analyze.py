#!/usr/bin/env python3
"""e172: 新 suite D=10 のクラス上限を `Restart-Lander` の降下（σ₀ = 0.1×span）で引き直す。

事前登録は `prereg.md`。対照は **再走させない** —— `hunt_coverage.py --null` の抽選 k は k だけに
依存するので、凍結表 `e94/ceiling_mpr_d10.csv`（σ=0.2、同じ 200 抽選 k=0..199、同じ `ceiling()`）が
**抽選単位の対照**になる（その171 と同じ paired 構造）。

推定量は **e92 の `ceiling()` を import**（MPR = 1e-1..1e-5 の 5 水準平均）。新しい推定量は書かない。
照合する量は 3 つ:
  * **公表最良 0.651**（D=10 の RR-CMA-ES の MPR。16 問平均しか存在しない ＝ その92 §6）
  * **`Restart-Lander` の実測 MPR**（`e115/by_problem/`、seed 0、rule=current）。
    上限はクラスの一員の実測を下回れない ＝ **整合性検査**（その151 が D=20 で見つけた壊れ方）。
  * **凍結表の σ=0.2 の上限**（対の相手）。

使い方: PYTHONPATH=<pynmmso stub> python3 analysis/mmo2024/e172/analyze.py
"""
from __future__ import annotations
import csv, gzip, statistics, sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "e92"))
from core.benchmarks import niching_by_name            # noqa: E402
from analyze import ceiling, EPS                       # noqa: E402  (e92 の推定量)

BOOT = 2000
PUB_D10 = 0.651          # 公表最良（D=10、RR-CMA-ES の MPR。16 問平均のみ存在）
FUNCS = [f"M{i:02d}-D10-PIN01" for i in range(1, 17)]
EPSF = 1e-5
EPS_COLS = ("pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5")
_FOLD: dict[str, list[dict]] = {}


def _read(src: Path) -> list[dict]:
    op = gzip.open if src.suffix == ".gz" else open
    with op(src, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def load_rl(func: str):
    """畳んだ 1 本（`fold.py` の出力）を優先し、無ければ per-function のダンプを読む。"""
    folded = HERE / "null_rl.csv.gz"
    if folded.exists():
        if "rl" not in _FOLD:
            _FOLD["rl"] = _read(folded)
        rows = [r for r in _FOLD["rl"] if r["function"] == func]
    else:
        for cand in (HERE / "null" / f"{func}_rl.csv.gz", HERE / "null" / f"{func}_rl.csv"):
            if cand.exists():
                rows = _read(cand)
                break
        else:
            return None
    if not rows:
        return None
    return {"best_f": np.array([float(r["best_f"]) for r in rows]),
            "land": np.array([int(r["land_opt"]) for r in rows]),
            "evals": np.array([float(r["evals"]) for r in rows])}


def base_table() -> dict[str, dict]:
    """その92・その94 の凍結表（σ=0.2、200 抽選）。再走させない対照。"""
    p = HERE.parent / "e94" / "ceiling_mpr_d10.csv"
    return {r["func"]: r for r in csv.DictReader(open(p))}


def measured() -> dict[str, float]:
    """`Restart-Lander` の実測 MPR（e115、seed 0、rule=current）。"""
    out = {}
    for f in sorted((HERE.parent / "e115" / "by_problem").glob("*.csv")):
        for r in csv.DictReader(open(f)):
            out[r["function"]] = statistics.mean(
                float(r[c]) for c in EPS_COLS)
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    base, meas = base_table(), measured()
    out = []
    for f in FUNCS:
        d = load_rl(f)
        if d is None:
            continue
        b = niching_by_name(f)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        m = len(d["best_f"])
        mpr, per_eps, sup = ceiling(d["best_f"], d["land"], d["evals"], K, budget)
        boot = np.array([ceiling(d["best_f"], d["land"], d["evals"], K, budget,
                                 rng.integers(0, m, m))[0] for _ in range(BOOT)])
        hit = d["best_f"] <= EPSF
        reach = sorted(set(d["land"][hit].tolist()))
        bt = base[f]
        out.append({"func": f, "K": K, "n": m,
                    "rl_mpr": mpr, "rl_sup": sup,
                    "rl_lo": float(np.percentile(boot, 2.5)),
                    "rl_hi": float(np.percentile(boot, 97.5)),
                    "rl_cost": float(d["evals"].mean()),
                    "rl_restarts": budget / float(d["evals"].mean()),
                    "rl_hit": float(hit.mean()),
                    "rl_reach_n": len(reach),
                    "rl_miss": sorted(set(range(K)) - set(reach)),
                    "rl_pr1e5": per_eps[EPS.index(EPSF)],
                    "base_mpr": float(bt["mpr"]), "base_sup": float(bt["mpr_sup"]),
                    "base_draws": int(bt["draws"]),
                    "base_cost": budget / float(bt["restarts"]),
                    "base_hit": float(bt["hit_1e-5"]),
                    "base_reach_n": int(bt["reached_1e-5"]),
                    "base_pr1e5": float(bt["pr_1e-5"]),
                    "meas": meas[f]})

    if not out:
        print("no RL dumps yet")
        return

    # ── 表 1: 上限の引き直し（σ だけを揃えた） ──────────────────────────────
    print(f"== 表 1: クラス上限（固定コスト MPR、200 抽選、1 run 予算 500,000）"
          f"  完走 {len(out)}/16 問")
    hdr = ("func", "K", "base_mpr", "RL_mpr", "RL_CI", "diff",
           "base_sup", "RL_sup", "meas(RL実測)", "sup<meas?")
    print(" ".join(f"{h:>13}" for h in hdr))
    for r in out:
        bad = "INVALID" if r["rl_sup"] < r["meas"] - 1e-9 else "ok"
        print(" ".join(f"{c:>13}" for c in [
            r["func"], str(r["K"]), f"{r['base_mpr']:.4f}", f"{r['rl_mpr']:.4f}",
            f"[{r['rl_lo']:.3f},{r['rl_hi']:.3f}]",
            f"{r['rl_mpr'] - r['base_mpr']:+.4f}",
            f"{r['base_sup']:.3f}", f"{r['rl_sup']:.3f}",
            f"{r['meas']:.3f}", bad]))

    mb = statistics.mean(r["base_mpr"] for r in out)
    mr = statistics.mean(r["rl_mpr"] for r in out)
    sb = statistics.mean(r["base_sup"] for r in out)
    sr = statistics.mean(r["rl_sup"] for r in out)
    mm = statistics.mean(r["meas"] for r in out)
    print(f"\n  平均（{len(out)} 問）: 固定コスト base {mb:.4f} → RL {mr:.4f} "
          f"({mr - mb:+.4f})   無限再起動 base {sb:.4f} → RL {sr:.4f} ({sr - sb:+.4f})")
    print(f"  公表最良（D=10、16 問平均）{PUB_D10:.3f} との差: "
          f"固定コスト {mr - PUB_D10:+.4f} / 無限再起動 {sr - PUB_D10:+.4f}")
    print(f"  `Restart-Lander` の実測 MPR（同じ {len(out)} 問）{mm:.4f} "
          f"／ 上限が実測を下回る問題 "
          f"{sum(r['rl_sup'] < r['meas'] - 1e-9 for r in out)}/{len(out)}"
          f"（固定コストで見ると {sum(r['rl_mpr'] < r['meas'] - 1e-9 for r in out)}/{len(out)}）")

    # ── 表 2: 降下のコスト・深さ・被覆 ─────────────────────────────────────
    print("\n== 表 2: 1 降下のコスト・再起動回数・`best_f<=1e-5` 率・到達された最適数（対）")
    hdr2 = ("func", "base_cost", "RL_cost", "cost比", "base_n_rst", "RL_n_rst",
            "base_hit", "RL_hit", "base_reach", "RL_reach", "RL_miss")
    print(" ".join(f"{h:>11}" for h in hdr2))
    for r in out:
        print(" ".join(f"{c:>11}" for c in [
            r["func"], f"{r['base_cost']:.0f}", f"{r['rl_cost']:.0f}",
            f"{r['rl_cost'] / r['base_cost']:.2f}",
            f"{500000 / r['base_cost']:.0f}", f"{r['rl_restarts']:.0f}",
            f"{r['base_hit']:.3f}", f"{r['rl_hit']:.3f}",
            str(r["base_reach_n"]), str(r["rl_reach_n"]),
            ",".join(map(str, r["rl_miss"])) or "-"]))

    # ── 対検定（内訳。判定は問題ごとの閾値照合） ───────────────────────────
    TOL = 1e-9
    for key, lab in (("mpr", "固定コスト"), ("sup", "無限再起動")):
        d = [r[f"rl_{key}"] - r[f"base_{key}"] for r in out]
        nz = [v for v in d if abs(v) > TOL]
        pos, neg = sum(v > TOL for v in d), sum(v < -TOL for v in d)
        w = wilcoxon(nz, alternative="two-sided") if len(nz) >= 3 else None
        pmin = 2.0 / (2 ** len(nz)) if nz else float("nan")
        print(f"\n== 対検定 {lab}（同じ 200 抽選、RL 設定 − σ=0.2）問題 n={len(d)}  "
              f"平均 {statistics.mean(d):+.4f}  中央 {statistics.median(d):+.4f}  "
              f"正 {pos} / 同点 {len(d) - pos - neg} / 負 {neg}  有効対 {len(nz)}"
              + (f"  両側 Wilcoxon exact p={w.pvalue:.5g}"
                 f"（最小達成可能 p {pmin:.5g}）" if w else ""))

    with open(HERE / "ceiling_sigma_rl.csv", "w", newline="") as fh:
        keys = sorted({k for r in out for k in r})
        w2 = csv.DictWriter(fh, fieldnames=keys)
        w2.writeheader()
        for r in out:
            w2.writerow({**r, "rl_miss": ",".join(map(str, r["rl_miss"]))})
    print(f"\nwrote {HERE / 'ceiling_sigma_rl.csv'}")


if __name__ == "__main__":
    main()
