#!/usr/bin/env python3
"""その124 — キュー 1 の残り 2 問（M15 / M16）を埋め、「インスタンス効果ゼロの問題」を数える。

その123 は 6 問中 4 問しか格子を閉じられず、**その主判定量（`s_seed/s_inst` の比の中央値）は
「効果ゼロの問題」と「ある問題」を平均してしまう**という欠陥があった（実際 4 問は 2 対 2 に割れた）。
キュー 1 の本文がその欠陥を直した形を指定している ——
**主判定量は「雑音を引いた `s_inst` が 0 でない問題の数」**（比は内訳として併記）。

**採点も分散分解も検定もすべてその115 / その123 のコードを import する。新しい規則も新しい統計量も
1 つも定義しない。** 本サイクルが足したのは (i) 読み先に `e124/descents` を加えること、
(ii) 主判定量を「問題数」に変えること、の 2 点だけ。

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e124/analyze.py
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)

# e123 の analyze を読み込む（module 名は `analyze` が e115 のものと衝突するので改名）。
# 読み込むだけで e115 の `analyze` も sys.path 経由で入る。
_spec = importlib.util.spec_from_file_location(
    "e123_analyze", os.path.join(MMO, "e123", "analyze.py"))
_e123 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e123)

load, grid, sds, ss_inst, perm_test, holm = (
    _e123.load, _e123.grid, _e123.sds, _e123.ss_inst, _e123.perm_test, _e123.holm)
PROBS, INSTANCES, SEEDS = _e123.PROBS, _e123.INSTANCES, _e123.SEEDS
SEED_LABEL, ARM, ARM_R, SPAN = _e123.SEED_LABEL, _e123.ARM, _e123.ARM_R, _e123.SPAN
NULL_RATIO, N_PERM = _e123.NULL_RATIO, _e123.N_PERM

DUMP_DIRS = [os.path.join(MMO, "e115", "descents"),
             os.path.join(MMO, "e115", "s1", "descents"),
             os.path.join(MMO, "e122", "descents"),
             os.path.join(MMO, "e123", "descents"),
             os.path.join(HERE, "descents")]

# 事前登録した帯（`prereg.md`）: 主判定量 = 雑音を引いた s_inst が 0 でない問題の数。
GATE_MANY, GATE_FEW = 4, 2
from scipy import stats                                            # noqa: E402


def main() -> int:
    cells = load(DUMP_DIRS)
    print("=" * 94)
    print("その124 — 「インスタンス効果ゼロの問題」は 6 問中何問か"
          "（キュー 1、`Restart-Lander`・D=10・正規予算 50 万）")
    print("=" * 94)
    print(f"\n  規則: {ARM}  r = {ARM_R / SPAN:g} x span"
          f"（その115 の合法な最良腕。新しい規則は定義していない）")
    print(f"  読めたセル: {len(cells)} / {len(PROBS) * 9}"
          f"（6 問 × 3 インスタンス × 3 seed）")

    full = [p for p in PROBS if grid(cells, p, "mpr") is not None]
    partial = [p for p in PROBS if p not in full]
    if partial:
        have = {p: sum(1 for k in cells if k[0] == p) for p in partial}
        print("  ** 格子が欠けている問題（事前登録により SD を出さない）: "
              + ", ".join(f"{p}({have[p]}/9)" for p in partial) + " **")
    print(f"  **完全な格子で集計した問題: {len(full)} 問** "
          + ("（" + ", ".join(full) + "）" if full else ""))
    if len(full) < 2:
        print("\n  完全な格子が 2 問に満たない。集計しない。")
        return 1

    # ------------------------------------------------------------ 1. 生の格子
    for key, lab in (("mpr", "MPR"), ("score", "Score")):
        print(f"\n## 1{'ab'[key == 'score']}. {lab} の 3x3 格子"
              f"（M15 / M16 の 8 セルが本サイクル、残りは既存）\n")
        print(f"{'PID':<6}{'K':>4}{'inst':>8}"
              + "".join(f"{SEED_LABEL[s]:>9}" for s in SEEDS) + f"{'mean':>9}")
        for p in full:
            g = grid(cells, p, key)
            K = cells[(p, INSTANCES[0], 0)]["K"]
            for i, pin in enumerate(INSTANCES):
                print(f"{p if i == 0 else '':<6}{K if i == 0 else '':>4}{pin:>8}"
                      + "".join(f"{v:>9.4f}" for v in g[i]) + f"{g[i].mean():>9.4f}")

    # ------------------------------------------------------------ 2. SD の分解
    results = {}
    for key, lab in (("mpr", "MPR"), ("score", "Score")):
        print(f"\n## 2{'ab'[key == 'score']}. 分散の分解（{lab}）"
              f" —— **`s_inst` 雑音引きの列が本サイクルの主判定量**\n")
        print(f"{'PID':<6}{'s_seed':>9}{'s_inst':>9}{'比 raw':>10}"
              f"{'s_inst 雑音引き':>16}{'比 corr':>10}{'range':>9}")
        rows = []
        for p in full:
            g = grid(cells, p, key)
            s_seed, s_inst, raw, s_corr, corr = sds(g)
            rows.append((p, s_seed, s_inst, raw, s_corr, corr))
            rng_ = float(g.mean(axis=1).max() - g.mean(axis=1).min())
            print(f"{p:<6}{s_seed:>9.4f}{s_inst:>9.4f}{raw:>10.2f}"
                  f"{s_corr:>16.4f}{corr:>10.2f}{rng_:>9.4f}")
        med_raw = float(np.median([r[3] for r in rows]))
        n_nonzero = sum(1 for r in rows if r[4] > 0)
        nz_names = [r[0] for r in rows if r[4] > 0]
        z_names = [r[0] for r in rows if r[4] <= 0]
        print(f"\n  **主判定量（{lab}）: 雑音を引いた s_inst が 0 でない問題 = "
              f"{n_nonzero} / {len(full)}**"
              f"（0 でない: {', '.join(nz_names) if nz_names else 'なし'}"
              f" ／ ちょうど 0: {', '.join(z_names) if z_names else 'なし'}）")
        print(f"  内訳: 素の比の中央値 {med_raw:.2f}"
              f"（インスタンス効果ゼロが予測する sqrt(3) = {NULL_RATIO:.2f}）"
              f"、平均 s_seed {np.mean([r[1] for r in rows]):.4f} / "
              f"平均 s_inst {np.mean([r[2] for r in rows]):.4f}")
        results[key] = dict(rows=rows, med_raw=med_raw, n_nonzero=n_nonzero,
                            nz=nz_names, z=z_names)

    # ------------------------------------------------------------ 3. 検定
    print("\n## 3. インスタンス効果は run のばらつきで説明できるか\n")
    rng = np.random.default_rng(12345)
    for key, lab in (("mpr", "MPR"), ("score", "Score")):
        grids = [grid(cells, p, key) for p in full]
        obs, p_perm = perm_test(grids, rng)
        results[key]["p_perm"] = p_perm
        print(f"  並べ替え検定（{lab}、問題内でインスタンス標識を入替、{N_PERM} 回、上側）: "
              f"SS_inst 和 = {obs:.5f}  p = {p_perm:.4f}")
        ps = []
        for g in grids:
            k, n = g.shape[0], g.shape[1]
            ms_i = ss_inst(g) / (k - 1)
            ms_w = float(np.mean(np.var(g, axis=1, ddof=1)))
            ps.append(1.0 if ms_w <= 0 else
                      float(stats.f.sf(ms_i / ms_w, k - 1, k * (n - 1))))
        hp = holm(np.array(ps))
        results[key]["f_p"] = list(zip(full, ps, list(hp)))
        print("    問題ごとの一元配置 F（df 2/6）: "
              + "  ".join(f"{p}: p={pv:.3f}(Holm {h:.3f})"
                          for p, pv, h in zip(full, ps, hp)))
        print(f"    Holm 後に有意な問題: {sum(h < 0.05 for h in hp)}/{len(full)}\n")

    # ------------------------------------------------------------ 4. seed 0 の偏り
    print("## 4. その122 の問題別の表は 3 seed 平均からどれだけ外れているか"
          f"（MPR、{3 * len(full)} セル）\n")
    devs = []
    print(f"{'PID':<6}" + "".join(f"{pin:>10}" for pin in INSTANCES) + f"{'|max|':>9}")
    for p in full:
        g = grid(cells, p, "mpr")
        d = g[:, 0] - g.mean(axis=1)
        devs.extend(d.tolist())
        print(f"{p:<6}" + "".join(f"{v:>+10.4f}" for v in d) + f"{np.abs(d).max():>9.4f}")
    devs = np.array(devs)
    print(f"\n  **seed 0 の偏り: 平均 {devs.mean():+.4f}  絶対値の中央値 "
          f"{np.median(np.abs(devs)):.4f}  最大 {np.abs(devs).max():.4f}"
          f"（{len(devs)} セル）**")

    # --------------------------------------------------- 5. 事前登録の棄却条件
    print("\n## 5. 事前登録した棄却条件（`prereg.md`）\n")
    n = results["mpr"]["n_nonzero"]
    p_perm = results["mpr"]["p_perm"]
    print(f"  主判定量（MPR）= {n} / {len(full)} 問、並べ替え検定 p = {p_perm:.4f}\n")
    if n >= GATE_MANY:
        print(f"  ==> **{GATE_MANY} 問以上 ＝ インスタンス効果は例外ではなく通例。**")
        print("      問題別の結論すべてに「PIN01 の配置での結論」と明記し、")
        print("      キュー 2 の腕の判定は 16 問平均だけで行う。")
    elif n <= GATE_FEW:
        print(f"  ==> **{GATE_FEW} 問以下 ＝ インスタンス効果は少数の問題に限られる。**")
        print(f"      該当する問題: {', '.join(results['mpr']['nz']) or 'なし'}"
              " —— この問題を使う議論にだけ留保を付ける。")
    else:
        print("  ==> **3 問 ＝ 両論。** 問題ごとに列挙して記録する（§2a）。")
    # 事前登録の反証条件 1 / 2
    if n >= GATE_MANY and p_perm >= 0.05:
        print(f"\n  **反証条件 1 が発火**: 主判定量は通例側だが並べ替え検定は p = {p_perm:.4f} ≥ 0.05。")
        print("      ＝ `s_inst` の雑音引き（df=2）が 0 を外しすぎている。**検定の側を優先して書く。**")
    if n <= GATE_FEW and p_perm < 0.05:
        print(f"\n  **反証条件 2 が発火**: 該当は少数なのに並べ替え検定は p = {p_perm:.4f} < 0.05。")
        print("      ＝ 「限られる」ではなく**「少数の問題に集中する」**と書き、問題を名指しする:")
        print(f"      {', '.join(results['mpr']['nz']) or 'なし'}")
    print("\n  **選抜の留保（消えない）**: 6 問はその122 で「レンジ ≥0.10 で動いた」問題として"
          "選ばれている\n      ＝ **`s_inst` が大きい側に選抜されている。この問題数を 16 問全体に外挿してはいけない。**")

    # ------------------------------------------- 6. その123（4 問）との突き合わせ
    print("\n## 6. その123（4 問）の記録は 6 問にしても成り立つか（事前登録の反証条件 3）\n")
    sub = [p for p in full if p in ("M01", "M02", "M03", "M05")]
    if len(sub) == 4:
        rows4 = [r for r in results["mpr"]["rows"] if r[0] in sub]
        med4 = float(np.median([r[3] for r in rows4]))
        n4 = sum(1 for r in rows4 if r[4] > 0)
        print(f"  4 問だけ（その123 の集合）: 比の中央値 {med4:.2f}、"
              f"`s_inst` 雑音引きが 0 でない問題 {n4}/4")
        print(f"  6 問（本サイクル）      : 比の中央値 {results['mpr']['med_raw']:.2f}、"
              f"同 {n}/{len(full)}")
        print("  ＝ その123 の記録（中央値 2.45 / p=0.0399）は**4 問の値**であって 6 問の値ではない。")

    # ------------------------------- 7. その122 の 1 seed のレンジは予測していたか
    # 事前登録には無い探索的な検算（判定には使わない）。その122 は 6 問を
    # 「1 seed のインスタンス間レンジ ≥ 0.10」で選んだので、**その順位が
    # 3 seed 後の順位とどれだけ一致するか**は、選抜規則そのものの検証になる。
    print("\n## 7. その122 の 1 seed のレンジは「本当に動く問題」を当てていたか"
          "（探索的、判定に使わない）\n")
    r122 = {"M01": 0.19, "M02": 0.12, "M03": 0.11, "M05": 0.15, "M15": 0.34, "M16": 0.20}
    rows = {r[0]: r for r in results["mpr"]["rows"]}
    print(f"{'PID':<6}{'その122 (1 seed)':>18}{'3 seed の平均レンジ':>22}"
          f"{'s_inst 雑音引き':>16}")
    a, b = [], []
    for p in full:
        g = grid(cells, p, "mpr")
        rng3 = float(g.mean(axis=1).max() - g.mean(axis=1).min())
        a.append(r122[p])
        b.append(rng3)
        print(f"{p:<6}{r122[p]:>18.4f}{rng3:>22.4f}{rows[p][4]:>16.4f}")
    rho, pv = stats.spearmanr(a, b)
    print(f"\n  **Spearman（1 seed のレンジ 対 3 seed のレンジ）= {rho:+.3f}（p = {pv:.3f}、n=6）**")
    print(f"  1 seed のレンジは 3 seed 後に平均 {np.mean(np.array(a) - np.array(b)):+.4f} "
          f"縮む（最大 {np.max(np.array(a) - np.array(b)):.4f}）")

    out = os.path.join(HERE, "grid_d10.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "instance", "seed", "K", "n_reported", "mpr", "mean_f1", "score"])
        for p in PROBS:
            for pin in INSTANCES:
                for s in SEEDS:
                    c = cells.get((p, pin, s))
                    if c:
                        w.writerow([p, pin, s, c["K"], c["n"], f"{c['mpr']:.4f}",
                                    f"{c['f1']:.4f}", f"{c['score']:.4f}"])
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
