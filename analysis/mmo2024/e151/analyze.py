#!/usr/bin/env python3
"""その151 — キュー 1: `Restart-Lander` を D=20 で 16 問（安いスクリーン）。

**採点・規則・統計量は `e115/analyze.py` からそのまま import する**
（`rule_indices` / `score` / `paired` / `read_dump` / `SPAN`）。**新しい統計量は 1 つも定義しない。**
1e-1 被覆と深さ保持率も `score` が返す recall 配列の第 1 / 第 5 要素を読むだけ。

入力の出自:

  * **D=20 `Restart-Lander`** -> `e151/descents/M??-D20-PIN01_seed0.csv[.gz]`（この回、新規 16 run）
  * **D=10 `Restart-Lander`** -> `e115/descents/M??-D10-PIN01_seed0.csv.gz`（保存物、**追加評価ゼロ**）
  * **公表最良 D=20 / D=10、その93 のクラス上限** -> `prereg.md` の転記表（測らない）

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e151/analyze.py
"""
from __future__ import annotations

import csv
import gzip
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MMO, "e115"))

from analyze import SPAN, paired, read_dump, rule_indices, score   # noqa: E402
from core.benchmarks import niching_by_name                        # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN

# --- 転記（prereg.md §2）。この回では 1 つも測らない ---
PUB_D20_SCORE, PUB_D20_MPR, PUB_D20_F1 = 0.4445, 0.476, 0.413
PUB_D10_SCORE, PUB_D10_MPR = 0.6080, 0.651
CEIL93_D20 = {"固定コスト": 0.3149, "早期打ち切り": 0.3273,
              "無限再起動": 0.3350, "＋Chao1": 0.3729}
GATE_D10 = {"mpr": 0.5644, "score": 0.6284}   # その116・その127・その131（PIN01 seed 0）

PROBS = [f"M{i:02d}" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]

_K: dict = {}


def K_of(prob):
    if prob not in _K:
        _K[prob] = int(niching_by_name(prob).n_global_optima)
    return _K[prob]


def find(*cands):
    for c in cands:
        for ext in ("", ".gz"):
            if os.path.exists(c + ext):
                return c + ext
    return None


def score_arrays(f, opt, xs, K):
    idx = rule_indices(ARM, f, K, x=xs, r=ARM_R)
    recall, prec, f1, sc, n = score(idx, f, opt, K)
    cov = float(recall[0])                       # 1e-1 被覆
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()), score=float(sc.mean()),
                n=int(n), ndump=len(f), cov=cov,
                keep=float(recall[4] / cov) if cov > 0 else float("nan"))


def score_dump(path, K):
    return score_arrays(*read_dump(path), K)


# 畳んだ後の退避路（その148・その149 と同じ形。畳む前後で出力が 1 文字も変わらないことを確認する）
FOLDED_RL = os.path.join(HERE, "descents.csv.gz")
_FOLD: dict = {}


def folded(prob):
    """`problem` 列つきの 1 本のダンプを problem -> (f, opt, xs) に割る（e149 の写し）。"""
    if not _FOLD:
        if not os.path.exists(FOLDED_RL):
            _FOLD["_"] = None
            return None
        op = gzip.open if FOLDED_RL.endswith(".gz") else open
        with op(FOLDED_RL, "rt") as fh:
            rows = list(csv.DictReader(fh))
        by: dict = {}
        for r in rows:
            by.setdefault(r["problem"], []).append(r)
        for pr, rs in by.items():
            dim = sum(1 for kk in rs[0] if kk.startswith("x") and kk[1:].isdigit())
            _FOLD[pr] = (np.array([float(r["best_f"]) for r in rs]),
                         np.array([int(r["land_opt"]) for r in rs]),
                         np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rs]))
    return _FOLD.get(prob)


def fmt(d):
    return (f"mean {d['mean']:+.4f}  {d['w']}/{d['t']}/{d['l']}  "
            f"p={d['p']:.5g}  rb={d['rb']:+.3f}")


def main():
    out = []
    P = out.append

    # ------------------------------------------------- 関門（D=10 の保存物を採点し直す）
    P("## 関門 —— 同じ採点器で D=10 の保存物を採点し直し、記録値と一致するか")
    d10 = {}
    for p in PROBS:
        path = find(os.path.join(MMO, "e115", "descents", f"{p}-D10-PIN01_seed0.csv"))
        if path is None:
            P(f"  **{p} の D=10 保存物が無い。関門を通せない。**")
            print("\n".join(out))
            return
        d10[p] = score_dump(path, K_of(f"{p}-D10-PIN01"))
    m_mpr = float(np.mean([d10[p]["mpr"] for p in PROBS]))
    m_sc = float(np.mean([d10[p]["score"] for p in PROBS]))
    ok = abs(m_mpr - GATE_D10["mpr"]) < 5e-5 and abs(m_sc - GATE_D10["score"]) < 5e-5
    P(f"  再計算: MPR {m_mpr:.4f} / Score {m_sc:.4f}   記録: "
      f"{GATE_D10['mpr']:.4f} / {GATE_D10['score']:.4f}  -> {'通過' if ok else '不一致'}")
    if not ok:
        P("")
        P("**関門が外れた。事前登録どおり判定は出さない。**")
        print("\n".join(out))
        return

    # ------------------------------------------------- D=20 の採点
    P("")
    P("## D=20（この回の新規 run）")
    d20, done, missing = {}, [], []
    for p in PROBS:
        prob = f"{p}-D20-PIN01"
        path = find(os.path.join(HERE, "descents", f"{prob}_seed0.csv"))
        if path is not None:
            d20[p] = score_dump(path, K_of(prob))
        elif folded(prob) is not None:
            d20[p] = score_arrays(*folded(prob), K_of(prob))
        else:
            missing.append(p)
            continue
        done.append(p)
    k = len(done)
    na = sum(1 for p in done if p in GROUP_A)
    P(f"  完走: k = {k} / 16  （群 A {na} / 群 B {k - na}）")
    P(f"  完走した問題: {' '.join(done) if done else '（なし）'}")
    P(f"  **落ちた問題: {' '.join(missing) if missing else '（なし）'}**")
    if k == 0:
        P("  **run がゼロ。判定は出せない。**")
        print("\n".join(out))
        return

    P("")
    P("  | 問題 | K | D20 MPR | D20 F1 | D20 Score | D20 報告点数 | D20 1e-1 被覆 | D20 深さ保持率 | "
      "(参考) D10 Score |")
    P("  |---|---|---|---|---|---|---|---|---|")
    for p in done:
        a, b = d20[p], d10[p]
        P(f"  | {p} | {K_of(f'{p}-D20-PIN01')} | {a['mpr']:.4f} | {a['f1']:.4f} | {a['score']:.4f} | "
          f"{a['n']} | {a['cov']:.4f} | {a['keep']:.4f} | {b['score']:.4f} |")

    def mean20(key):
        return float(np.mean([d20[p][key] for p in done]))

    def mean10(key):
        return float(np.mean([d10[p][key] for p in done]))

    P("")
    P(f"  {k} 問平均（D=20）: Score {mean20('score'):.4f} / MPR {mean20('mpr'):.4f} / "
      f"mean-F1 {mean20('f1'):.4f} / 報告点数 {mean20('n'):.2f} / "
      f"1e-1 被覆 {mean20('cov'):.4f} / 深さ保持率 {mean20('keep'):.4f} / 降下本数 {mean20('ndump'):.1f}")
    P(f"  {k} 問平均（D=10、同じ問題に制限。参考）: Score {mean10('score'):.4f} / MPR {mean10('mpr'):.4f} / "
      f"mean-F1 {mean10('f1'):.4f} / 報告点数 {mean10('n'):.2f} / "
      f"1e-1 被覆 {mean10('cov'):.4f} / 深さ保持率 {mean10('keep'):.4f} / 降下本数 {mean10('ndump'):.1f}")

    # ------------------------------------------------- 判定
    P("")
    P("## 判定（prereg.md §3）")
    sc, mpr = mean20("score"), mean20("mpr")
    P(f"  **主判定**: {k} 問平均 Score {sc:.4f}  対 公表最良 D=20 の {PUB_D20_SCORE:.4f} "
      f"＝ **{sc - PUB_D20_SCORE:+.4f}**（{'上回る' if sc > PUB_D20_SCORE else '下回る'}）")
    P(f"    内訳: MPR {mpr:.4f} 対 {PUB_D20_MPR:.3f} ＝ {mpr - PUB_D20_MPR:+.4f} / "
      f"mean-F1 {mean20('f1'):.4f} 対 {PUB_D20_F1:.3f} ＝ {mean20('f1') - PUB_D20_F1:+.4f}")
    P(f"  **反証条件 (a)**: MPR {mpr:.4f} 対 その93 のいちばん甘い上界 {CEIL93_D20['＋Chao1']:.4f} "
      f"＝ **{'発火（screen が外れている。俯瞰に上げる）' if mpr > CEIL93_D20['＋Chao1'] else '不発'}**")
    for nm, v in CEIL93_D20.items():
        P(f"    その93 {nm} {v:.4f}: 実測は {'上' if mpr > v else '下'}（{mpr - v:+.4f}）")
    P(f"  **反証条件 (b)**: Score {sc:.4f} < {PUB_D20_SCORE:.4f} "
      f"＝ **{'発火（頭の主張は D ≦ 10 の範囲つきになる）' if sc < PUB_D20_SCORE else '不発'}**")
    P(f"  外挿の見込み（prereg.md §2）は MPR 0.39 前後 / Score 0.433 前後: "
      f"実測との差は MPR {mpr - 0.39:+.4f} / Score {sc - 0.433:+.4f}")
    P(f"  D=10 で screen が当たった比（実測 ÷ 上限）は 97.2%。D=20 の同じ比は "
      f"{mpr / CEIL93_D20['＋Chao1'] * 100:.1f}%（＋Chao1 上界に対して）")

    # ------------------------------------------------- 次元間の内訳（対検定はしない）
    P("")
    P("## 次元で何が落ちたか（対検定はしない。予算も K も違うので対にならない）")
    for key, lab in (("cov", "1e-1 被覆"), ("keep", "深さ保持率"),
                     ("mpr", "MPR"), ("f1", "mean-F1"), ("n", "報告点数"),
                     ("ndump", "降下本数")):
        a, b = mean10(key), mean20(key)
        P(f"  {lab}: D=10 {a:.4f} -> D=20 {b:.4f}  （{b - a:+.4f}、比 {b / a:.3f}）"
          if a else f"  {lab}: D=10 {a:.4f} -> D=20 {b:.4f}")
    P("")
    P("  （参考、**判定には使わない**）同じ 16 問の D=10 と D=20 の Score を問題ごとに並べた符号検定:")
    aa = {p: d10[p]["score"] for p in done}
    bb = {p: d20[p]["score"] for p in done}
    P(f"    {fmt(paired(bb, aa, done))}  ＝ 次元を上げると下がるか（負なら下がる）")

    print("\n".join(out))


if __name__ == "__main__":
    main()
