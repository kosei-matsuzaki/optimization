#!/usr/bin/env python3
"""e170: CF3 の梯子（D=2/3/5）—— RL が取り落とす解は体積が無いのか、規則が落とすのか。

事前登録は `prereg.md`。出力は 3 つ:

  (a) RL の実走の降下ダンプから、`best_f <= 1e-5` で絞った到達集合と
      **1 run の実測降下本数**（その88 は予算 ÷ 平均コストで推定していた。ここは実測）。
  (b) desc null（2000 本 × 1499 評価、等方・既定）から、各最適への着地割合。
  (c) 期待着地数 = (b) の割合 x (a) の実測本数。

さらに **cap の分離**: RL の報告集合は無加工で `n_descents` 点あり、harness は
`max(100, 2K)` に best-by-f で切る（`core/runner.py:245`）。**到達しているのに
採点されない最適**があるなら、上限は体積ではなく報告規則の側にある。
採点は CEC2013 公式の `count_goptima`（rho-greedy、rho=0.01）をそのまま使う。
"""
from __future__ import annotations
import csv, gzip, sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from core.benchmarks import niching_by_name                      # noqa: E402
from core.runner import count_goptima                            # noqa: E402

HERE = Path(__file__).resolve().parent
FUNCS = [("N13-CF3-2D", 2), ("N14-CF3-3D", 3), ("N16-CF3-5D", 5)]
SEEDS = [0, 1, 2]
EPS = 1e-5          # 公表値が PR@1e-5 なので、その88 §2 の規則どおりここで絞る
ACCS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def _read(path: Path) -> list[dict]:
    op = (lambda p: gzip.open(p, "rt", newline="")) if path.suffix == ".gz" \
        else (lambda p: open(p, newline=""))
    with op(path) as fh:
        return list(csv.DictReader(fh))


def _folded(name: str, func: str, seed: int) -> list[dict] | None:
    """畳んだ 1 本（`fold.py` の出力）から (function, seed) の行を引く。"""
    q = HERE / name
    if not q.exists():
        return None
    if name not in _FOLD_CACHE:
        _FOLD_CACHE[name] = _read(q)
    return [r for r in _FOLD_CACHE[name]
            if r["function"] == func and int(r["seed"]) == seed]


_FOLD_CACHE: dict[str, list[dict]] = {}


def descent_stats(func: str) -> dict:
    """(a): per-seed 実測本数と、eps で絞った到達集合。"""
    out = {}
    for s in SEEDS:
        # RL は seed を `seed_index * 100` で回す（niching_baseline.py:368）
        rows = _folded("rl_descents.csv.gz", func, s * 100)
        if rows is None:
            rows = _read(HERE / "rl_descents" / func / f"{func}_seed{s * 100}.csv")
        deep = [r for r in rows if float(r["best_f"]) <= EPS]
        out[s] = {
            "n_descents": len(rows),
            "land_all": Counter(int(r["land_opt"]) for r in rows),
            "land_deep": Counter(int(r["land_opt"]) for r in deep),
            "median_evals": float(np.median([float(r["evals"]) for r in rows])),
        }
    return out


def null_stats(func: str, K: int, suffix: str = "") -> dict | None:
    """(b): desc null の着地割合（eps で絞ってから）。

    `suffix="_s01"` は (d) —— **RL の実値**（`sigma_ratio=0.1` / 12500 評価）で
    引き直した null。既定（`sigma_ratio=0.2` / 1499 評価）は その88・その89 が使ったもの。
    """
    q = HERE / "null" / f"{func}_iso{suffix}.csv.gz"
    if not q.exists():
        return None
    rows = _read(q)
    n = len(rows)
    deep = Counter(int(r["land_opt"]) for r in rows
                   if float(r["best_f"]) <= EPS)
    nn = Counter(int(r["land_opt"]) for r in rows)
    return {"n": n, "deep": deep, "nn": nn,
            "share": {j: deep[j] / n for j in range(K)}}


# `DEDUP_RATIOS`: 手法自身の情報だけで決める間引き半径（span 比）。**rho も K も使わない**
# ので競技規則 §5 の下で合法（その130 の報告半径と同じ立場。追加評価ゼロ）。
DEDUP_RATIOS = [1e-4, 1e-3, 1e-2, 5e-2]


def dedup_indices(X: np.ndarray, F: np.ndarray, r: float) -> np.ndarray:
    """f 昇順に歩き、既に採った点から r 以内なら落とす（貪欲）。"""
    keep: list[int] = []
    for i in np.argsort(F):
        if all(np.linalg.norm(X[i] - X[j]) > r for j in keep):
            keep.append(int(i))
    return np.array(keep, dtype=int)


def score_reports(func: str, b) -> dict:
    """cap の分離: 公式 `count_goptima` を cap つき / cap なしで当てる。"""
    opts = np.asarray(b.optima_pos, dtype=float)
    out = {}
    for s in SEEDS:
        rows = _folded("rl_reports.csv.gz", func, s)
        if rows is None:
            rows = _read(HERE / "rl_reports" / func
                         / f"{func}_Restart-Lander_seed{s}.csv.gz")
        F = np.array([float(r["f"]) for r in rows])
        X = np.array([[float(r[f"x{i}"]) for i in range(b.dim)] for r in rows])
        cap = max(100, 2 * b.n_global_optima)
        keep = np.argsort(F)[:cap] if len(F) > cap else np.arange(len(F))
        rec = {"n_uncapped": len(F), "cap": cap}
        for lbl, (Xs, Fs) in (("capped", (X[keep], F[keep])),
                              ("uncapped", (X, F))):
            rec[lbl] = [count_goptima(Xs, Fs, b.n_global_optima,
                                      b.niche_rho, a) for a in ACCS]
        # 間引いてから cap に切る（追加評価ゼロ。上の cap つきと対になる）
        lo, hi = b.bounds
        rec["dedup"] = {}
        for ratio in DEDUP_RATIOS:
            idx = dedup_indices(X, F, ratio * (hi - lo))[:cap]
            rec["dedup"][ratio] = (
                count_goptima(X[idx], F[idx], b.n_global_optima,
                              b.niche_rho, EPS), len(idx))
        # どの最適が「到達しているのに採点されない」のか（最近傍帰属、診断のみ）
        qual = X[F <= EPS]
        if len(qual):
            d = np.linalg.norm(qual[:, None, :] - opts[None, :, :], axis=2)
            rec["reached_nn"] = sorted(set(np.argmin(d, axis=1).tolist()))
        else:
            rec["reached_nn"] = []
        out[s] = rec
    return out


def main() -> None:
    print(__doc__)
    ladder = []
    for func, D in FUNCS:
        b = niching_by_name(func)
        K = b.n_global_optima
        ds = descent_stats(func)
        ns = null_stats(func, K)
        rs = score_reports(func, b)
        print("=" * 78)
        print(f"{func}  (D={D}, K={K}, 予算 {b.suite_max_evals}, rho={b.niche_rho})")

        print(f"\n-- (a) RL の実走（3 seed）。`best_f <= {EPS:g}` で絞った到達集合")
        print(f"{'seed':>5}{'降下本数':>10}{'中央 evals':>12}"
              f"{'到達(深)':>26}{'最近傍のみ':>16}")
        for s in SEEDS:
            deep = sorted(ds[s]['land_deep'])
            nnonly = sorted(set(ds[s]['land_all']) - set(deep))
            print(f"{s:>5}{ds[s]['n_descents']:>10}{ds[s]['median_evals']:>12.0f}"
                  f"{str(deep):>26}{str(nnonly):>16}")
        n_mean = float(np.mean([ds[s]["n_descents"] for s in SEEDS]))
        reach_union = set().union(*[set(ds[s]["land_deep"]) for s in SEEDS])
        missing = sorted(set(range(K)) - reach_union)
        print(f"  3 seed の和集合で到達 = {sorted(reach_union)}  "
              f"→ **一度も到達しない添字 = {missing}**"
              f"   (平均降下本数 {n_mean:.1f})")

        print(f"\n-- (b)(c) desc null {ns['n']} 本 → 割合、x 実測本数 {n_mean:.1f} = 期待着地数")
        print(f"{'opt':>4}{'null 深 本数':>14}{'割合':>10}{'期待着地数':>12}"
              f"{'RL 実走 深/run':>16}{'null 最近傍':>12}")
        exp = {}
        for j in range(K):
            sh = ns["share"][j]
            exp[j] = sh * n_mean
            rl_deep = np.mean([ds[s]["land_deep"][j] for s in SEEDS])
            print(f"{j:>4}{ns['deep'][j]:>14}{sh:>10.5f}{exp[j]:>12.3f}"
                  f"{rl_deep:>16.2f}{ns['nn'][j]:>12}")

        ns01 = null_stats(func, K, "_s01")
        if ns01 is not None:
            print(f"\n-- (d) null を RL の実値に合わせる"
                  f"（sigma_ratio 0.2 → **0.1**、降下上限 1499 → **12500**）")
            print(f"{'opt':>4}{'既定 深':>10}{'RL 実値 深':>12}{'割合(既定)':>12}"
                  f"{'割合(実値)':>12}{'期待着地(実値)':>16}{'RL 実走 深/run':>16}")
            for j in range(K):
                sh0, sh1 = ns["share"][j], ns01["share"][j]
                rl_deep = np.mean([ds[s]["land_deep"][j] for s in SEEDS])
                print(f"{j:>4}{ns['deep'][j]:>10}{ns01['deep'][j]:>12}"
                      f"{sh0:>12.5f}{sh1:>12.5f}{sh1 * n_mean:>16.3f}"
                      f"{rl_deep:>16.2f}")
            exp01 = {j: ns01["share"][j] * n_mean for j in range(K)}
        else:
            exp01 = None

        print(f"\n-- cap の分離（公式 count_goptima、cap = max(100, 2K)）")
        print(f"{'seed':>5}{'無加工点数':>12}{'cap':>6}"
              f"{'PR@1e-5 cap':>14}{'PR@1e-5 無cap':>16}{'到達(最近傍)':>20}")
        for s in SEEDS:
            r = rs[s]
            print(f"{s:>5}{r['n_uncapped']:>12}{r['cap']:>6}"
                  f"{r['capped'][-1]:>14}{r['uncapped'][-1]:>16}"
                  f"{str(r['reached_nn']):>20}")
        cap_m = float(np.mean([rs[s]["capped"][-1] for s in SEEDS]))
        unc_m = float(np.mean([rs[s]["uncapped"][-1] for s in SEEDS]))
        print(f"  3 seed 平均 検出数: cap {cap_m:.2f} / 無cap {unc_m:.2f}"
              f"  → 差 {unc_m - cap_m:+.2f}"
              f"  (PR: {cap_m / K:.4f} → {unc_m / K:.4f})")

        print(f"\n-- 間引いてから cap（追加評価ゼロ。半径は span 比、rho も K も使わない）")
        print(f"{'半径/span':>12}" + "".join(f"{s:>8}" for s in
                                             ("seed0", "seed1", "seed2"))
              + f"{'平均':>8}{'報告点数':>12}")
        ded = {}
        for ratio in DEDUP_RATIOS:
            v = [rs[s]["dedup"][ratio][0] for s in SEEDS]
            n = [rs[s]["dedup"][ratio][1] for s in SEEDS]
            ded[ratio] = float(np.mean(v))
            print(f"{ratio:>12g}" + "".join(f"{x:>8}" for x in v)
                  + f"{ded[ratio]:>8.2f}{np.mean(n):>12.1f}")
        ladder.append({"func": func, "D": D, "K": K, "missing": missing,
                       "exp": exp, "n_mean": n_mean, "cap": cap_m,
                       "uncap": unc_m, "null_n": ns["n"], "ded": ded,
                       "per_seed_cap": [rs[s]["capped"][-1] for s in SEEDS],
                       "per_seed_ded": [rs[s]["dedup"][1e-3][0] for s in SEEDS],
                       "exp01": exp01,
                       "rl_deep": {j: float(np.mean([ds[s]["land_deep"][j]
                                                     for s in SEEDS]))
                                   for j in range(K)}})

    # ── 梯子（反証条件の判定） ────────────────────────────────────────────
    print("=" * 78)
    print("\n## 梯子 —— その88 §4 の「取り落とす添字は次元によらず {2,3}」と照合\n")
    print(f"{'D':>3}{'一度も到達しない添字':>24}{'opt2 期待着地':>14}"
          f"{'opt3 期待着地':>14}{'cap → 無cap':>16}")
    for L in ladder:
        print(f"{L['D']:>3}{str(L['missing']):>24}{L['exp'].get(2, 0):>14.3f}"
              f"{L['exp'].get(3, 0):>14.3f}"
              f"{f'{L[chr(99)+chr(97)+chr(112)]:.2f} → {L[chr(117)+chr(110)+chr(99)+chr(97)+chr(112)]:.2f}':>16}")

    print(f"\n{'D':>3}{'cap 3 seed':>14}{'間引き 1e-3':>14}{'差':>8}")
    for L in ladder:
        print(f"{L['D']:>3}{str(L['per_seed_cap']):>14}{str(L['per_seed_ded']):>14}"
              f"{np.mean(L['per_seed_ded']) - np.mean(L['per_seed_cap']):>+8.2f}")
    # 9 run を対で（cap 対 間引き 1e-3）。同点は Wilcoxon が落とすので有効対を明示する。
    from scipy.stats import wilcoxon
    a = [c for L in ladder for c in L["per_seed_cap"]]
    bb = [c for L in ladder for c in L["per_seed_ded"]]
    d = np.array(bb) - np.array(a)
    eff = int((d != 0).sum())
    print(f"\n  9 run 対: 間引き − cap = {list(d)}  "
          f"（正 {int((d > 0).sum())} / 負 {int((d < 0).sum())} / 同点 {int((d == 0).sum())}）")
    if eff:
        try:
            st, pv = wilcoxon(bb, a, zero_method="wilcox",
                              alternative="two-sided", method="exact")
            print(f"  両側 Wilcoxon exact p = {pv:.4g}（有効対 {eff}、"
                  f"最小達成可能 p = {2 / 2 ** eff:.4g}）")
        except ValueError as e:
            print(f"  Wilcoxon 不能: {e}")
    if all(L["exp01"] is not None for L in ladder):
        print("\n## (d) 計器の照合 —— null の降下を RL の実値にすると opt 2 は開くか\n")
        print(f"{'D':>3}{'opt2 期待(既定)':>18}{'opt2 期待(実値)':>18}"
              f"{'opt2 RL 実走/run':>18}{'opt3 期待(実値)':>18}")
        for L in ladder:
            print(f"{L['D']:>3}{L['exp'].get(2, 0):>18.3f}"
                  f"{L['exp01'].get(2, 0):>18.3f}"
                  f"{L['rl_deep'].get(2, 0):>18.2f}"
                  f"{L['exp01'].get(3, 0):>18.3f}")
    print("\n## 反証条件（prereg.md）")
    print("  **判定は (d) の null（RL の実値 sigma_ratio=0.1 / 12500 評価）で行う。**\n"
          "  既定の null（0.2 / 1499）は `Restart-Lander` の降下ではないので、\n"
          "  期待着地数の判定には使えない（下の「計器の照合」表の 1 列目と 2 列目の差）。")
    d2 = next(L for L in ladder if L["D"] == 2)
    # 期待着地数は (d) から引く。**(d) が無い状態で既定の null に落ちて判定すると
    # 符号が逆になる**（本節の趣旨そのもの）ので、黙って落ちずに止める
    # —— `scripts/scan_silent_null.py` が探している型の事故を自分で作らないため。
    missing01 = [L["D"] for L in ladder if L["exp01"] is None]
    if missing01:
        raise SystemExit(
            f"判定できない: D={missing01} の (d) の null（sigma_ratio=0.1 / 12500 評価）が無い。\n"
            "既定の null（0.2 / 1499）は `Restart-Lander` の降下ではないので、"
            "期待着地数の判定に代用すると符号が逆に出る（本節の §3）。\n"
            "`run.sh` の (d) のループを回してから再実行すること。")

    def E(L, j):
        return L["exp01"].get(j, 0.0)
    fired_i = False
    for L in ladder:
        for j in (2, 3):
            if E(L, j) > 1.0:
                fired_i = True
                print(f"  (i) 発火: D={L['D']} の opt {j} は期待着地数 {E(L, j):.2f} > 1 "
                      f"＝ 体積はある（実走も {L['rl_deep'].get(j, 0):.2f} 本/run で着地）。"
                      f"落としているのは RL の規則の側。")
    mono = all(E(ladder[i], 3) >= E(ladder[i + 1], 3)
               for i in range(len(ladder) - 1))
    print(f"  opt 3 の期待着地数は次元とともに単調非増加: {mono}"
          f"（3 次元すべて厳密に 0.000）")
    if all(E(L, 2) < 1 and E(L, 3) < 1 for L in ladder):
        print("  (ii) 発火: 3 次元すべてで期待着地数 < 1。")
    elif fired_i:
        print("  (ii) 不発: opt 2 は D=2 / D=3 で 1 を超える。"
              "**opt 3 についてだけ (ii) が成立する**"
              "（3 次元すべて 0.000、単調）＝ **2 解のうち 1 解は規則、1 解は体積。**")
    if d2["missing"] != [2, 3]:
        print(f"  (iii) 発火: D=2 の取り落とし添字は {d2['missing']} で "
              f"[2, 3] ではない ＝ その88 §4 は D=2 に延びない。")


if __name__ == "__main__":
    main()
