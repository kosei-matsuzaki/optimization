#!/usr/bin/env python3
"""その153 — キュー 1: `Restart-Lander` と RR-CMA-ES を D=5 で 16 問ずつ（上下から挟む）。

**採点・規則・統計量は `e115/analyze.py` からそのまま import する**
（`rule_indices` / `score` / `paired` / `read_dump` / `SPAN`）。**新しい統計量は 1 つも定義しない。**

入力の出自:

  * **D=5 の 2 手法**       -> `e153/descents/`（RL）と `e153/dumps/`（RR）。この回、新規 32 run
                              （畳んだ後は `descents.csv.gz` / `dumps_rrcma.csv.gz` を読む）
  * **D=10 の 2 手法**      -> その131 §7 の表を下に転記（**測らない**）。
                              RL 側は `e115/descents/` の保存物で 1 問ずつ照合する（関門 1b）
  * **D=20 の 2 手法**      -> `e151/descents.csv.gz` と `e152/dumps_rrcma.csv.gz`（保存物、**追加評価ゼロ**）
  * **公表最良 D=5 / D=10 / D=20** -> 転記（測らない）

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e153/analyze.py
"""
from __future__ import annotations

import csv
import gzip
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MMO, "e115"))

from analyze import SPAN, paired, read_dump, rule_indices, score   # noqa: E402
from core.benchmarks import niching_by_name                        # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN

PROBS = [f"M{i:02d}" for i in range(1, 17)]
GROUP_A, GROUP_B = PROBS[:8], PROBS[8:]

# --- 転記 1: その131 §7 の表（D=10 / PIN01 / seed 0）。e152/analyze.py からそのまま写した ---
D10_SCORE_RR = dict(zip(PROBS, [0.7189, 0.3808, 0.5833, 0.4857, 0.5833, 0.5353, 0.3808, 0.7189,
                                0.6750, 0.4857, 0.6750, 0.4857, 0.7618, 0.3808, 0.2667, 0.4857]))
D10_SCORE_RL = dict(zip(PROBS, [0.7600, 0.6069, 0.5717, 0.7189, 0.7189, 0.4857, 0.5234, 0.7189,
                                0.5625, 0.6529, 0.5175, 0.8416, 0.8444, 0.3808, 0.3886, 0.7618]))
D10_MPR_RR = dict(zip(PROBS, [0.65, 0.30, 0.50, 0.40, 0.50, 0.45, 0.30, 0.65,
                              0.60, 0.40, 0.60, 0.40, 0.70, 0.30, 0.20, 0.40]))
D10_MPR_RL = dict(zip(PROBS, [0.72, 0.53, 0.49, 0.65, 0.65, 0.40, 0.44, 0.65,
                              0.50, 0.60, 0.46, 0.82, 0.80, 0.30, 0.32, 0.70]))
GATE_D10 = {"rr_score": 0.5377, "rl_score": 0.6284, "rr_mpr": 0.4594, "rl_mpr": 0.5644}

# --- 転記 2: その151 / その152 の 16 問平均（関門 2 の照合先） ---
GATE_D20 = {"rl_score": 0.4504, "rl_mpr": 0.3831, "rr_score": 0.4575}

# --- 転記 3: 公表最良（競技資料、出典 1 本）。D=5 の最良手法は RR-CMA-ES 自身 ---
PUB = {5: dict(score=0.7145, mpr=0.844, f1=0.585),
       10: dict(score=0.6080, mpr=0.651, f1=0.565),
       20: dict(score=0.4445, mpr=0.476, f1=0.413)}
SD_RL, SD_RR = 0.0066, 0.0238          # その123 の n=3 seed ばらつき
R_INSTANCE = 0.4899                    # その149 の instance 間相関（反証条件 (b) の閾値）
R_D10_D20 = 0.6238                     # その152 の D 間相関
D10_PAIR_MEAN, D20_PAIR_MEAN = 0.0907, -0.0071   # 既知の対差（Score、RL − RR）

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
    cov = float(recall[0])
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()), score=float(sc.mean()),
                n=int(n), ndump=len(f), cov=cov,
                keep=float(recall[4] / cov) if cov > 0 else float("nan"))


def score_dump(path, K):
    return score_arrays(*read_dump(path), K)


# --- 畳んだダンプ（`problem` 列つき 1 本）を problem -> (f, opt, xs) に割る ---
_FOLD: dict = {}


def _load_folded(path, cache):
    if cache in _FOLD:
        return
    _FOLD[cache] = {}
    if not os.path.exists(path):
        return
    with gzip.open(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    by: dict = {}
    for r in rows:
        by.setdefault(r["problem"], []).append(r)
    for pr, rs in by.items():
        dim = sum(1 for kk in rs[0] if kk.startswith("x") and kk[1:].isdigit())
        _FOLD[cache][pr] = (np.array([float(r["best_f"]) for r in rs]),
                            np.array([int(r["land_opt"]) for r in rs]),
                            np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rs]))


def folded(path, cache, prob):
    _load_folded(path, cache)
    return _FOLD[cache].get(prob)


def fmt(d):
    return (f"mean {d['mean']:+.4f}  {d['w']}/{d['t']}/{d['l']}  "
            f"p={d['p']:.5g}  rb={d['rb']:+.3f}")


def main():
    out = []
    P = out.append

    # ---------------------------------------------- 関門 1a: 転記表（その131 §7）の平均
    P("## 関門 1a —— 転記した その131 §7 の表（D=10）が記録値を再現するか（追加評価ゼロ）")
    g1 = {"rr_score": float(np.mean(list(D10_SCORE_RR.values()))),
          "rl_score": float(np.mean(list(D10_SCORE_RL.values()))),
          "rr_mpr": float(np.mean(list(D10_MPR_RR.values()))),
          "rl_mpr": float(np.mean(list(D10_MPR_RL.values())))}
    ok1a = all(abs(g1[k] - GATE_D10[k]) < 5e-5 for k in GATE_D10)
    for k in ("rr_score", "rl_score", "rr_mpr", "rl_mpr"):
        P(f"  {k}: 転記から {g1[k]:.4f}   記録 {GATE_D10[k]:.4f}   差 {g1[k] - GATE_D10[k]:+.5f}")
    P(f"  -> {'通過' if ok1a else '**不一致**'}")

    # ---------------------------------------------- 関門 1b: D=10 RL は保存物で 1 問ずつ照合できる
    P("")
    P("## 関門 1b —— D=10 の RL は保存物が残っているので<u>問題ごとに</u>転記と突き合わせる")
    worst, miss1b = 0.0, []
    for p in PROBS:
        path = find(os.path.join(MMO, "e115", "descents", f"{p}-D10-PIN01_seed0.csv"))
        if path is None:
            miss1b.append(p)
            continue
        s = score_dump(path, K_of(f"{p}-D10-PIN01"))["score"]
        worst = max(worst, abs(s - D10_SCORE_RL[p]))
    ok1b = (not miss1b) and worst < 1e-4
    P(f"  16 問中 {16 - len(miss1b)} 問を再計算。転記との最大差 {worst:.2e}"
      f"{'  （欠: ' + ' '.join(miss1b) + '）' if miss1b else ''}  -> {'通過' if ok1b else '**不一致**'}")

    # ---------------------------------------------- 関門 2: D=20 の保存物を読み直す
    P("")
    P("## 関門 2 —— その151 / その152 の D=20 保存物を<u>この回の採点コード</u>で読み直す（追加評価ゼロ）")
    rl20, rr20, miss20 = {}, {}, []
    for p in PROBS:
        prob = f"{p}-D20-PIN01"
        a = folded(os.path.join(MMO, "e151", "descents.csv.gz"), "rl20", prob)
        b = folded(os.path.join(MMO, "e152", "dumps_rrcma.csv.gz"), "rr20", prob)
        if a is None or b is None:
            miss20.append(p)
            continue
        rl20[p] = score_arrays(*a, K_of(prob))
        rr20[p] = score_arrays(*b, K_of(prob))
    if miss20:
        P(f"  **D=20 の保存物が無い問題: {' '.join(miss20)}。関門を通せない。**")
        ok2 = False
    else:
        m = {"rl_score": float(np.mean([rl20[p]["score"] for p in PROBS])),
             "rl_mpr": float(np.mean([rl20[p]["mpr"] for p in PROBS])),
             "rr_score": float(np.mean([rr20[p]["score"] for p in PROBS]))}
        ok2 = all(abs(m[k] - GATE_D20[k]) < 5e-5 for k in GATE_D20)
        for k in ("rl_score", "rl_mpr", "rr_score"):
            P(f"  {k}: 再計算 {m[k]:.4f}   記録 {GATE_D20[k]:.4f}   差 {m[k] - GATE_D20[k]:+.5f}")
        P(f"  -> {'通過' if ok2 else '**不一致**'}")

    if not (ok1a and ok1b and ok2):
        P("")
        P("**関門が外れた。事前登録（§4）どおり判定は出さない。**")
        print("\n".join(out))
        return

    # ---------------------------------------------- D=5 の採点
    P("")
    P("## D=5 の 16 問（この回の新規 32 run）")
    rl5, rr5, done, missing = {}, {}, [], []
    for p in PROBS:
        prob = f"{p}-D05-PIN01"
        pa = find(os.path.join(HERE, "descents", f"{prob}_seed0.csv"))
        a = score_dump(pa, K_of(prob)) if pa else (
            (lambda t: score_arrays(*t, K_of(prob)) if t else None)(
                folded(os.path.join(HERE, "descents.csv.gz"), "rl5", prob)))
        pb = find(os.path.join(HERE, "dumps", f"{prob}_rrcma_seed0.csv"))
        b = score_dump(pb, K_of(prob)) if pb else (
            (lambda t: score_arrays(*t, K_of(prob)) if t else None)(
                folded(os.path.join(HERE, "dumps_rrcma.csv.gz"), "rr5", prob)))
        if a is None or b is None:
            missing.append(f"{p}({'RL' if a is None else ''}{'RR' if b is None else ''})")
            continue
        rl5[p], rr5[p] = a, b
        done.append(p)
    k = len(done)
    na = sum(1 for p in done if p in GROUP_A)
    P(f"  完走した対: k = {k} / 16  （群 A {na} / 群 B {k - na}）")
    P(f"  **対が欠けた問題: {' '.join(missing) if missing else '（なし）'}**")
    if k == 0:
        P("  **対がゼロ。判定は出せない。**")
        print("\n".join(out))
        return

    P("")
    P("  | 問題 | K | RL MPR | RL F1 | RL Score | RL 報告 | RR MPR | RR F1 | RR Score | RR 報告 | "
      "差 RL−RR | (参考) D10 差 | (参考) D20 差 |")
    P("  |---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in done:
        a, b = rl5[p], rr5[p]
        P(f"  | {p} | {K_of(f'{p}-D05-PIN01')} | {a['mpr']:.4f} | {a['f1']:.4f} | {a['score']:.4f} | "
          f"{a['n']} | {b['mpr']:.4f} | {b['f1']:.4f} | {b['score']:.4f} | {b['n']} | "
          f"{a['score'] - b['score']:+.4f} | {D10_SCORE_RL[p] - D10_SCORE_RR[p]:+.4f} | "
          f"{rl20[p]['score'] - rr20[p]['score']:+.4f} |")

    def mean_of(d, key):
        return float(np.mean([d[p][key] for p in done]))

    P("")
    P(f"  {k} 問平均 `Restart-Lander`（D=5）: Score {mean_of(rl5,'score'):.4f} / "
      f"MPR {mean_of(rl5,'mpr'):.4f} / mean-F1 {mean_of(rl5,'f1'):.4f} / 報告点数 {mean_of(rl5,'n'):.2f}")
    P(f"  {k} 問平均 RR-CMA-ES（D=5）:        Score {mean_of(rr5,'score'):.4f} / "
      f"MPR {mean_of(rr5,'mpr'):.4f} / mean-F1 {mean_of(rr5,'f1'):.4f} / 報告点数 {mean_of(rr5,'n'):.2f}")

    # ---------------------------------------------- 判定 (i)
    P("")
    P("## 判定 (i) —— D=5 の 16 問対検定（両側 Wilcoxon exact、RL − RR）")
    res5 = {}
    for key, lab in (("score", "Score（主）"), ("mpr", "MPR"), ("f1", "mean-F1")):
        res5[key] = paired({p: rl5[p][key] for p in done}, {p: rr5[p][key] for p in done}, done)
        P(f"  {lab:<12}: {fmt(res5[key])}")
    sign5 = res5["score"]["mean"]
    P(f"  **反証条件 (a)**: D=5 の Score 差の符号 = "
      f"{'正（RL > RR、D=10 と同じ）' if sign5 > 0 else '**負（RR > RL ＝ 反転）**'} ＝ "
      f"**{'不発' if sign5 > 0 else '発火（頭の主張は D=10 の 16 問に限定される。俯瞰に上げる）'}**")

    # ---------------------------------------------- 判定 (ii)
    P("")
    P("## 判定 (ii) —— 公表最良 D=5（Score 0.7145 / MPR 0.844 / mean-F1 0.585、手法は RR-CMA-ES）との差")
    for lab, d, sd in (("`Restart-Lander`", rl5, SD_RL), ("RR-CMA-ES   ", rr5, SD_RR)):
        for key in ("score", "mpr", "f1"):
            pass
        P(f"  {lab}: Score {mean_of(d,'score'):.4f}（{mean_of(d,'score') - PUB[5]['score']:+.4f}）/ "
          f"MPR {mean_of(d,'mpr'):.4f}（{mean_of(d,'mpr') - PUB[5]['mpr']:+.4f}）/ "
          f"mean-F1 {mean_of(d,'f1'):.4f}（{mean_of(d,'f1') - PUB[5]['f1']:+.4f}）"
          f"  [1 seed の SD {sd:.4f}]")
    gap_rl = mean_of(rl5, "score") - PUB[5]["score"]
    fired_c = gap_rl < -SD_RL
    P(f"  **反証条件 (c)**: RL の公表最良との差 {gap_rl:+.4f} は SD {SD_RL:.4f} より"
      f"{'大きく下回る' if fired_c else '下回らない'} ＝ "
      f"**{'発火（公表値に勝つ／並ぶは D=10 限定）' if fired_c else '不発'}**")
    P(f"  （RR 側の照合）**公表最良 D=5 は RR-CMA-ES 自身**なので、"
      f"RR の実測 {mean_of(rr5,'score'):.4f} と公表 {PUB[5]['score']:.4f} の差 "
      f"{mean_of(rr5,'score') - PUB[5]['score']:+.4f} は**この harness の再現度**として読む"
      f"（その149 の PIN01/PIN02 は D=10 で −0.0353 / +0.0210）。")

    # ---------------------------------------------- 判定 (iii)
    P("")
    P("## 判定 (iii) —— D 間の差の相関（反証条件 (b)）")
    d5v = np.array([rl5[p]["score"] - rr5[p]["score"] for p in done])
    d10v = np.array([D10_SCORE_RL[p] - D10_SCORE_RR[p] for p in done])
    d20v = np.array([rl20[p]["score"] - rr20[p]["score"] for p in done])
    rs = {}
    for lab, u, v in (("D=5 ↔ D=10", d5v, d10v), ("D=5 ↔ D=20", d5v, d20v),
                      ("D=10 ↔ D=20（その152 の再掲）", d10v, d20v)):
        r, pr = stats.pearsonr(u, v)
        rho, prho = stats.spearmanr(u, v)
        ag = int(np.sum(np.sign(u) == np.sign(v)))
        rs[lab] = r
        P(f"  {lab:<28}: Pearson r = {r:+.4f}（p={pr:.4g}） / Spearman rho = {rho:+.4f}"
          f"（p={prho:.4g}） / 符号一致 {ag}/{k}")
    r510 = rs["D=5 ↔ D=10"]
    fired_b = r510 >= R_INSTANCE
    P(f"  閾値は その149 の instance 間 r = {R_INSTANCE:+.4f}")
    P(f"  **反証条件 (b)**: r(D=5, D=10) {r510:+.4f} {'≥' if fired_b else '<'} {R_INSTANCE:.4f} ＝ "
      f"**{'発火（D=5 も疑似反復。次元で n は増やせない ＝ その152 の結論が 2 本目でも成立）' if fired_b else '不発（D=5 は D=20 より独立に近い。プールの採否は俯瞰の判断）'}**")

    # ---------------------------------------------- 判定 (iv)
    P("")
    P("## 判定 (iv) —— 3 次元の並び（対差は Score、RL − RR。予算も K も次元で違うので対検定はしない）")
    P("")
    P("  | D | 正規予算 | RL Score | RR Score | 対差 RL−RR | p | 公表最良 Score | RL − 公表 | RR − 公表 |")
    P("  |---|---|---|---|---|---|---|---|---|")
    P(f"  | 5 | 250,000 | {mean_of(rl5,'score'):.4f} | {mean_of(rr5,'score'):.4f} | "
      f"{res5['score']['mean']:+.4f} | {res5['score']['p']:.4g} | {PUB[5]['score']:.4f} | "
      f"{mean_of(rl5,'score') - PUB[5]['score']:+.4f} | {mean_of(rr5,'score') - PUB[5]['score']:+.4f} |")
    P(f"  | 10 | 500,000 | {GATE_D10['rl_score']:.4f} | {GATE_D10['rr_score']:.4f} | "
      f"{D10_PAIR_MEAN:+.4f} | 0.029541 | {PUB[10]['score']:.4f} | "
      f"{GATE_D10['rl_score'] - PUB[10]['score']:+.4f} | {GATE_D10['rr_score'] - PUB[10]['score']:+.4f} |")
    P(f"  | 20 | 1,000,000 | {GATE_D20['rl_score']:.4f} | {GATE_D20['rr_score']:.4f} | "
      f"{D20_PAIR_MEAN:+.4f} | 0.6698 | {PUB[20]['score']:.4f} | "
      f"{GATE_D20['rl_score'] - PUB[20]['score']:+.4f} | {GATE_D20['rr_score'] - PUB[20]['score']:+.4f} |")
    P("")
    P("  （次元耐性、D=5 を 1 とした比）")
    for lab, d5d, d10v_, d20v_ in (
            ("RL Score", mean_of(rl5, "score"), GATE_D10["rl_score"], GATE_D20["rl_score"]),
            ("RR Score", mean_of(rr5, "score"), GATE_D10["rr_score"], GATE_D20["rr_score"]),
            ("RL MPR  ", mean_of(rl5, "mpr"), GATE_D10["rl_mpr"], GATE_D20["rl_mpr"]),
            ("RR MPR  ", mean_of(rr5, "mpr"), GATE_D10["rr_mpr"], 0.0)):
        if d5d <= 0:
            continue
        tail = "" if d20v_ <= 0 else f" / D=20 {d20v_ / d5d:.3f}"
        P(f"  {lab}: D=5 1.000 / D=10 {d10v_ / d5d:.3f}{tail}")

    # ---------------------------------------------- 広さと深さの内訳（参考）
    P("")
    P("## 広さ（1e-1 被覆）と深さ（保持率 @1e-5 ÷ @1e-1）の内訳（参考。その151 の読みの続き）")
    for lab, d in (("`Restart-Lander` D=5", rl5), ("RR-CMA-ES D=5       ", rr5)):
        P(f"  {lab}: 1e-1 被覆 {mean_of(d,'cov'):.4f} / 深さ保持率 {mean_of(d,'keep'):.4f}")
    P(f"  `Restart-Lander` D=20（保存物の再計算）: 1e-1 被覆 "
      f"{float(np.mean([rl20[p]['cov'] for p in done])):.4f} / 深さ保持率 "
      f"{float(np.mean([rl20[p]['keep'] for p in done])):.4f}")

    # ---------------------------------------------- 群別（参考）
    P("")
    P("## 群別（参考。K=20 の群 A と K=10 の群 B）")
    for lab, grp in (("群 A (K=20)", [p for p in done if p in GROUP_A]),
                     ("群 B (K=10)", [p for p in done if p in GROUP_B])):
        if not grp:
            continue
        a = float(np.mean([rl5[p]["score"] for p in grp]))
        b = float(np.mean([rr5[p]["score"] for p in grp]))
        P(f"  {lab} n={len(grp)}: RL {a:.4f} / RR {b:.4f} / 差 {a - b:+.4f}")

    print("\n".join(out))
    with open(os.path.join(HERE, "scored.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")

    with open(os.path.join(HERE, "by_problem_d5.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "K", "rl_d5_mpr", "rl_d5_f1", "rl_d5_score", "rl_d5_n",
                    "rr_d5_mpr", "rr_d5_f1", "rr_d5_score", "rr_d5_n",
                    "rl_d10_score", "rr_d10_score", "rl_d20_score", "rr_d20_score"])
        for p in done:
            a, b = rl5[p], rr5[p]
            w.writerow([p, K_of(f"{p}-D05-PIN01"),
                        f"{a['mpr']:.4f}", f"{a['f1']:.4f}", f"{a['score']:.4f}", a["n"],
                        f"{b['mpr']:.4f}", f"{b['f1']:.4f}", f"{b['score']:.4f}", b["n"],
                        f"{D10_SCORE_RL[p]:.4f}", f"{D10_SCORE_RR[p]:.4f}",
                        f"{rl20[p]['score']:.4f}", f"{rr20[p]['score']:.4f}"])


if __name__ == "__main__":
    main()
