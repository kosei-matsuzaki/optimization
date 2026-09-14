#!/usr/bin/env python3
"""その123 — 問題別の値の留保は seed から来るのかインスタンスから来るのか（キュー 1）。

その122 は **16 問平均は 3 インスタンスで一致**（対で全部有意でない、Friedman p=0.51）と
出したが、**問題別ではレンジ ≥0.10 が 6/16** を残した。3 本とも seed 0 なので、
そのばらつきが「インスタンス差」なのか「run のばらつき」なのかが分かれていない。

**動いた 6 問（M01/M02/M03/M05/M15/M16）を 3 インスタンス × 3 seed の完全な格子で測る。**
seed 0 は既にあるので（PIN01 = e115、PIN02/03 = e122）、本サイクルで回したのは seed 1, 2。

採点規則も採点コードも**その115 / その116 / その122 のものをそのまま import する。新しい規則は
1 つも定義しない**（`rule_indices` / `score` / `read_dump`、規則 `eps_loose+dedup`、r = 0.05 × span）。
**最近傍最適への帰属（`opt`）は e122 の `attribute` をそのまま使う**（採点にだけ使うオラクル量）。

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e123/analyze.py
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MMO, "e115"))

from analyze import SPAN, read_dump, rule_indices, score          # noqa: E402

# e122 の analyze も module 名が `analyze` なので、名前を変えて読み込む（同名衝突）。
import importlib.util                                             # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "e122_analyze", os.path.join(MMO, "e122", "analyze.py"))
_e122 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e122)
attribute, bench = _e122.attribute, _e122.bench

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN
PROBS = ["M01", "M02", "M03", "M05", "M15", "M16"]
INSTANCES = ["PIN01", "PIN02", "PIN03"]
# ダンプのファイル名に載るのは `seed = seed_index * 100`（`niching_baseline.py:281`）。
# seed_index 0/1/2 -> ラベル 0/100/200。表示は seed_index で行う。
SEEDS = [0, 100, 200]
SEED_LABEL = {0: "seed0", 100: "seed1", 200: "seed2"}
# 事前登録した帯（キュー 1 本文の定義。素の比 s_seed / s_inst の 6 問中央値で判定する）
GATE_SEED, GATE_INST = 0.7, 0.4
# 「インスタンス効果ゼロ」が予測する比（Var(3 個の平均) = sigma^2/3 -> sqrt(3)）
NULL_RATIO = float(np.sqrt(3.0))
N_PERM = 20000

_FN_RE = re.compile(r"^(M(\d{2})-D10-(PIN\d{2}))_seed(\d+)\.csv(\.gz)?$")


def load(dump_dirs):
    """(problem, instance, seed) -> 採点結果 の dict を作る。"""
    cells = {}
    for d in dump_dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            m = _FN_RE.match(fn)
            if not m:
                continue
            prob_full, pid, pin, seed = m.group(1), "M" + m.group(2), m.group(3), int(m.group(4))
            if pid not in PROBS or pin not in INSTANCES or seed not in SEEDS:
                continue
            f, _opt, x = read_dump(os.path.join(d, fn))
            if x is None:          # 座標の無いダンプには合法規則を当てられない
                continue
            K, opts = bench(prob_full)
            idx = rule_indices(ARM, f, K, x, ARM_R)
            recall, _prec, f1, sc, n = score(idx, f, attribute(x, opts), K)
            cells[(pid, pin, seed)] = dict(mpr=float(recall.mean()),
                                           f1=float(f1.mean()),
                                           score=float(sc.mean()), n=int(n), K=K)
    return cells


def grid(cells, pid, key):
    """3 インスタンス × 3 seed の行列（欠けていれば None）。"""
    g = np.full((len(INSTANCES), len(SEEDS)), np.nan)
    for i, pin in enumerate(INSTANCES):
        for j, s in enumerate(SEEDS):
            c = cells.get((pid, pin, s))
            if c is not None:
                g[i, j] = c[key]
    return None if np.isnan(g).any() else g


def sds(g):
    """(s_seed プール, s_inst 平均の SD, 素の比, 雑音を引いた s_inst, 補正後の比)。"""
    s_seed = float(np.sqrt(np.mean(np.var(g, axis=1, ddof=1))))   # インスタンス内、df=2 x3
    means = g.mean(axis=1)
    s_inst = float(np.std(means, ddof=1))                         # 3 個の平均の SD、df=2
    raw = s_seed / s_inst if s_inst > 0 else float("inf")
    corr_var = max(0.0, s_inst ** 2 - s_seed ** 2 / g.shape[1])
    corr = s_seed / np.sqrt(corr_var) if corr_var > 0 else float("inf")
    return s_seed, s_inst, raw, float(np.sqrt(corr_var)), corr


def ss_inst(g):
    """インスタンス間平方和（問題内。並べ替え検定の統計量の部品）。"""
    return float(g.shape[1] * np.sum((g.mean(axis=1) - g.mean()) ** 2))


def perm_test(grids, rng):
    """問題内でインスタンスのラベルを並べ替え、SS_inst の 6 問和を上側で検定する。"""
    obs = sum(ss_inst(g) for g in grids)
    flat = [g.reshape(-1) for g in grids]
    shape = grids[0].shape
    hits = 0
    for _ in range(N_PERM):
        tot = 0.0
        for v in flat:
            tot += ss_inst(rng.permutation(v).reshape(shape))
        hits += tot >= obs
    return obs, (hits + 1) / (N_PERM + 1)


def holm(ps):
    """Holm-Bonferroni 補正後の p（順序は入力のまま返す）。"""
    order = np.argsort(ps)
    m, out, run = len(ps), np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i])
        out[i] = min(1.0, run)
    return out


def main() -> int:
    # PIN01 の seed_index 1（= ファイル名 seed100）は e115/s1 が既に持っていた。
    # 本サイクルは重複して回し直し、**バイト一致を確認したうえで e123 側の複製を消した**
    # （§6）ので、読み先に e115/s1 を足してある。
    cells = load([os.path.join(MMO, "e115", "descents"),
                  os.path.join(MMO, "e115", "s1", "descents"),
                  os.path.join(MMO, "e122", "descents"),
                  os.path.join(HERE, "descents")])
    print("=" * 94)
    print("その123 — 問題別の値の留保は seed かインスタンスか"
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
        print(f"  ** 格子が欠けている問題（事前登録により SD を出さない）: "
              + ", ".join(f"{p}({have[p]}/9)" for p in partial) + " **")
    print(f"  **完全な格子で集計した問題: {len(full)} 問** "
          + ("（" + ", ".join(full) + "）" if full else ""))
    if len(full) < 2:
        print("\n  完全な格子が 2 問に満たない。集計しない。")
        return 1

    # ------------------------------------------------------------ 1. 生の格子
    for key, lab in (("mpr", "MPR"), ("score", "Score")):
        print(f"\n## 1{'ab'[key == 'score']}. {lab} の 3x3 格子（seed 0 は既存、"
              f"seed 1-2 が本サイクル）\n")
        print(f"{'PID':<6}{'K':>4}{'inst':>8}" + "".join(f"{SEED_LABEL[s]:>9}" for s in SEEDS)
              + f"{'mean':>9}")
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
              f" —— 素の比が事前登録の判定量\n")
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
        med_corr = float(np.median([r[5] for r in rows]))
        print(f"\n  **{len(full)} 問の中央値: 素の比 {med_raw:.2f}"
              f"（雑音を引いた比 {med_corr:.2f}）**")
        print(f"  参考: インスタンス効果ちょうどゼロが予測する素の比 = sqrt(3) = {NULL_RATIO:.2f}")
        print(f"  平均 s_seed {np.mean([r[1] for r in rows]):.4f} / "
              f"平均 s_inst {np.mean([r[2] for r in rows]):.4f}")
        results[key] = dict(rows=rows, med_raw=med_raw, med_corr=med_corr)

    # ------------------------------------------------------------ 3. 検定
    print("\n## 3. インスタンス効果は run のばらつきで説明できるか\n")
    rng = np.random.default_rng(12345)
    for key, lab in (("mpr", "MPR"), ("score", "Score")):
        grids = [grid(cells, p, key) for p in full]
        obs, p_perm = perm_test(grids, rng)
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
        print(f"    問題ごとの一元配置 F（df 2/6）: "
              + "  ".join(f"{p}: p={pv:.3f}(Holm {h:.3f})"
                          for p, pv, h in zip(full, ps, hp)))
        print(f"    Holm 後に有意な問題: "
              f"{sum(h < 0.05 for h in hp)}/{len(full)}\n")

    # ------------------------------------------------------------ 4. seed 0 の偏り
    print("## 4. その122 の問題別の表は 3 seed 平均からどれだけ外れているか（MPR、18 セル）\n")
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

    # ------------------------------------------------------------ 5. 事前登録の判定
    print("\n## 5. 事前登録した棄却条件（主判定量 = 6 問の s_seed/s_inst 中央値、MPR）\n")
    med = results["mpr"]["med_raw"]
    print(f"  素の比の中央値 = {med:.2f}（{len(full)} 問）\n")
    if med >= GATE_SEED:
        print(f"  ==> **{GATE_SEED} 以上。問題別の値が動くのは主に seed であってインスタンスではない。**")
        print("      記録済みの問題別の数値には「1 seed の値」とだけ書けばよく、")
        print("      インスタンスの留保は 16 問平均の側に付けなくてよい。")
        if abs(med - NULL_RATIO) < 0.5:
            print(f"\n  **ただし事前登録の反証条件が発火**: 比が sqrt(3)={NULL_RATIO:.2f} 付近"
                  f"（|差| {abs(med - NULL_RATIO):.2f} < 0.5）")
            print("      ＝ これは「seed が主」ではなく**「インスタンス効果が検出できない」**と書くこと"
                  "（区別できないことと、無いことは違う）。")
    elif med < GATE_INST:
        print(f"  ==> **{GATE_INST} 未満。インスタンス固有の構造が実在する。**")
        print("      その118/119/121 の問題別の結論すべてに「PIN01 の配置での結論」と明記し、")
        print("      キュー 2 の腕の判定は 16 問平均だけで行う。")
    else:
        print(f"  ==> **中間（{GATE_INST}-{GATE_SEED}）。両方効いている。**"
              "比を問題ごとに列挙して記録する（§2a）。")
    print(f"\n  **選抜の留保**: 6 問はその122 で「動いた」問題として選んである"
          "（＝ s_inst が大きい側に選抜されている）。**この比を 16 問全体に外挿してはいけない。**")

    # ------------------------------------------------------------ 6. 再現チェック
    # e115/s1 が PIN01 の seed_index 1（ファイル名は seed100）を既に持っている。
    # **本サイクルはそれを知らずに同じ 6 run を回し直した**ので、**バイト比較が
    # そのままコンテナをまたいだ決定性の検算になる**（追加評価ゼロ）。
    print("\n## 6. 重複して回した run のバイト比較（e115/s1 の PIN01 seed100 対 本サイクル）\n")
    import gzip
    same = diff = 0
    for p in PROBS:
        a = os.path.join(MMO, "e115", "s1", "descents", f"{p}-D10-PIN01_seed100.csv.gz")
        b = os.path.join(HERE, "descents", f"{p}-D10-PIN01_seed100.csv")
        b2 = b + ".gz"
        if not os.path.exists(a) or not (os.path.exists(b) or os.path.exists(b2)):
            continue
        with gzip.open(a, "rb") as fa:
            ba = fa.read()
        op = gzip.open if os.path.exists(b2) else open
        with op(b2 if os.path.exists(b2) else b, "rb") as fb:
            bb = fb.read()
        ok = ba == bb
        same += ok
        diff += not ok
        print(f"  {p}-D10-PIN01 seed100: {'一致' if ok else '**不一致**'}")
    if same + diff == 0:
        print("  **本サイクルの複製は検証後に削除済み**（下の記録と scored.txt が結果）。")
    print(f"\n  **記録: 2026-09-14 の実測でバイト一致 5/5**"
          f"（M01/M02/M03/M05/M15 の PIN01 seed100）"
          f"{f'。再検証: {same}/{same + diff} 一致' if same + diff else ''}")
    print("  ＝ 駆動はコンテナをまたいで決定的で、この 5 run は回し直す必要が無かった。")

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
