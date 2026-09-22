#!/usr/bin/env python3
"""その154 — キュー 1: 報告点数を競技の上限側に寄せたときの MPR / mean-F1 / Score。

**採点・規則・統計量は `e115/analyze.py` からそのまま import する**
（`rule_indices` / `score` / `paired` / `read_dump` / `SPAN` / `LEVELS`）。
**新しい統計量は 1 つも定義しない。追加評価ゼロ（保存済みダンプの再採点だけ）。**

入力（事前登録 §2 の表）:
  D=5  RL `e153/descents.csv.gz`      RR `e153/dumps_rrcma.csv.gz`
  D=10 RL `e115/descents/`（16 本）    RR 無し（その150 が e149 のダンプを削除）
  D=20 RL `e151/descents.csv.gz`      RR `e152/dumps_rrcma.csv.gz`

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e154/analyze.py
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

from analyze import LEVELS, SPAN, paired, read_dump, rule_indices, score  # noqa: E402
from core.benchmarks import niching_by_name                               # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN
PROBS = [f"M{i:02d}" for i in range(1, 17)]
CAPS = ["core", 13, 20, 26, 38, 50, 75, 100, "all"]
RADII = [1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]

# --- 転記: 公表最良（競技資料、出典 1 本）。D=5/10/20 とも最良は RR-CMA-ES 自身 ---
PUB = {5: dict(mpr=0.844, f1=0.585, score=0.7145),
       10: dict(mpr=0.651, f1=0.565, score=0.5730),
       20: dict(mpr=0.476, f1=0.413, score=0.4445)}
# --- 転記: その153 §5 / その151 / その152 の 16 問平均（関門の照合先、現行規則 = core） ---
GATE = {(5, "RL"): dict(score=0.7770, mpr=0.7306, f1=0.8235),
        (5, "RR"): dict(score=0.7220, mpr=0.6619, f1=0.7822),
        (10, "RL"): dict(score=0.6284, mpr=0.5644),
        (20, "RL"): dict(score=0.4504, mpr=0.3831),
        (20, "RR"): dict(score=0.4575, mpr=0.3844)}

_K: dict = {}


def K_of(prob):
    if prob not in _K:
        _K[prob] = int(niching_by_name(prob).n_global_optima)
    return _K[prob]


# ------------------------------------------------------------------ 入力
_FOLD: dict = {}


def _load_folded(path, cache):
    """畳んだダンプ（`problem` 列つき 1 本）を problem -> (f, opt, xs) に割る。"""
    if cache in _FOLD:
        return
    if not os.path.exists(path):
        raise SystemExit(f"入力のダンプが無い: {path}\n  -> 削除済みならこの経路は再現できない。"
                         " 黙って飛ばさない（その150 §3 の教訓）。")
    with gzip.open(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    by: dict = {}
    for r in rows:
        by.setdefault(r["problem"], []).append(r)
    out = {}
    for pr, rs in by.items():
        dim = sum(1 for kk in rs[0] if kk.startswith("x") and kk[1:].isdigit())
        out[pr] = (np.array([float(r["best_f"]) for r in rs]),
                   np.array([int(r["land_opt"]) for r in rs]),
                   np.array([[float(r[f"x{i}"]) for i in range(dim)] for r in rs]))
    _FOLD[cache] = out


def load_cell(D, meth):
    """(D, 手法) -> {problem: (f, opt, xs)}。入力が無ければ理由を印字して exit 1。"""
    if (D, meth) == (10, "RL"):
        dd = os.path.join(MMO, "e115", "descents")
        if not os.path.isdir(dd):
            raise SystemExit(f"入力の降下ダンプが無い: {dd}")
        out = {}
        for p in PROBS:
            path = None
            for ext in (".csv", ".csv.gz"):
                c = os.path.join(dd, f"{p}-D10-PIN01_seed0{ext}")
                if os.path.exists(c):
                    path = c
            if path is None:
                raise SystemExit(f"入力の降下ダンプが無い: {p}-D10-PIN01_seed0 in {dd}")
            out[f"{p}-D10-PIN01"] = read_dump(path)
        return out
    src = {(5, "RL"): ("e153", "descents.csv.gz"), (5, "RR"): ("e153", "dumps_rrcma.csv.gz"),
           (20, "RL"): ("e151", "descents.csv.gz"), (20, "RR"): ("e152", "dumps_rrcma.csv.gz")}
    e, fn = src[(D, meth)]
    _load_folded(os.path.join(MMO, e, fn), f"{e}:{fn}")
    tab = _FOLD[f"{e}:{fn}"]
    out = {}
    for p in PROBS:
        prob = f"{p}-D{D:02d}-PIN01"
        if prob not in tab:
            raise SystemExit(f"入力のダンプに問題が無い: {prob} in {e}/{fn}")
        out[prob] = tab[prob]
    return out


# ------------------------------------------------------------------ 報告集合
def report_indices(f, opt, xs, K, cap):
    """核（現行 `eps_loose+dedup`）に残りを `best_f` 昇順で足して cap 点にする。

    cap == "core" は足さない（= その153 の現行規則）。"all" は全降下。
    """
    core = rule_indices(ARM, f, K, x=xs, r=ARM_R)
    if cap == "core":
        return core
    n = len(f) if cap == "all" else int(cap)
    if len(core) >= n:
        return core[:n]
    used = set(int(i) for i in core)
    pad = [int(i) for i in np.argsort(f, kind="stable") if int(i) not in used]
    return np.array(list(core) + pad[: n - len(core)], dtype=int)


def measure(f, opt, xs, K, cap):
    idx = report_indices(f, opt, xs, K, cap)
    recall, prec, f1, sc, n = score(idx, f, opt, K)
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()), score=float(sc.mean()),
                n=int(n), prec=float(prec.mean()), cov=float(recall[0]), deep=float(recall[4]))


def sweep_cell(cell, D):
    """{cap: {problem: measure}} を返す。"""
    out = {}
    for cap in CAPS:
        out[cap] = {p: measure(*cell[f"{p}-D{D:02d}-PIN01"], K_of(f"{p}-D{D:02d}-PIN01"), cap)
                    for p in PROBS}
    return out


def avg(tab, key):
    return float(np.mean([tab[p][key] for p in PROBS]))


# ------------------------------------------------------------------ main
def main():
    out = []
    P = out.append
    P("=" * 96)
    P("その154 — 報告点数を競技の上限側（100 点）に寄せたときの MPR / mean-F1 / Score（追加評価ゼロ）")
    P("=" * 96)

    cells, sweeps = {}, {}
    for D, meth in [(5, "RL"), (5, "RR"), (10, "RL"), (20, "RL"), (20, "RR")]:
        cells[(D, meth)] = load_cell(D, meth)
        sweeps[(D, meth)] = sweep_cell(cells[(D, meth)], D)

    # ------------------------------------------------ 関門: core が記録値を再現するか
    P("")
    P("## 関門 —— `cap=core`（現行規則）が その151・その152・その153 の 16 問平均を再現するか（閾 5e-5）")
    ok = True
    for (D, meth), g in sorted(GATE.items()):
        t = sweeps[(D, meth)]["core"]
        for k, v in g.items():
            d = avg(t, k) - v
            ok &= abs(d) < 5e-5
            P(f"  D={D:<3}{meth:<3}{k:<7} 再計算 {avg(t, k):.4f}   記録 {v:.4f}   差 {d:+.5f}")
    P(f"  -> {'通過' if ok else '**不一致 — 事前登録 §5 どおり判定は出さない**'}")
    if not ok:
        print("\n".join(out))
        return 1

    # ------------------------------------------------ 判定 (i): 掃引表
    P("")
    P("## 判定 (i) —— 報告点数 N を上げたときの 16 問平均（核 = 現行 `eps_loose+dedup` に best_f 昇順で padding）")
    for D, meth in [(5, "RL"), (5, "RR"), (10, "RL"), (20, "RL"), (20, "RR")]:
        pub = PUB[D]
        P("")
        P(f"### D={D} {meth}   （公表最良 = RR-CMA-ES: MPR {pub['mpr']:.3f} / F1 {pub['f1']:.3f} / Score {pub['score']:.4f}）")
        P("")
        P("  | cap N | 実報告点数 | MPR | mean-F1 | Score | precision | 1e-1 被覆 | 1e-5 到達 | MPR−公表 | F1−公表 | Score−公表 |")
        P("  |---|---|---|---|---|---|---|---|---|---|---|")
        for cap in CAPS:
            t = sweeps[(D, meth)][cap]
            m = {k: avg(t, k) for k in ("mpr", "f1", "score", "n", "prec", "cov", "deep")}
            P(f"  | {cap} | {m['n']:.2f} | {m['mpr']:.4f} | {m['f1']:.4f} | **{m['score']:.4f}** | "
              f"{m['prec']:.4f} | {m['cov']:.4f} | {m['deep']:.4f} | "
              f"{m['mpr'] - pub['mpr']:+.4f} | {m['f1'] - pub['f1']:+.4f} | {m['score'] - pub['score']:+.4f} |")
        sc = [(avg(sweeps[(D, meth)][c], "score"), c) for c in CAPS]
        best = max(sc)
        P("")
        P(f"  **Score 最大は cap={best[1]}（{best[0]:.4f}）**、"
          f"core との差 {best[0] - avg(sweeps[(D, meth)]['core'], 'score'):+.4f}")

    # ------------------------------------------------ 判定 (ii): 公表値との同時一致
    P("")
    P("## 判定 (ii) —— `|MPR−公表| < 0.05` かつ `|F1−公表| < 0.05` を**同時に**満たす N はあるか")
    P("")
    P("  照合できるのは公表最良と同じ手法 = RR-CMA-ES（D=5 / D=20）。RL は別手法なので参考。")
    P("")
    P("  | セル | 両方 0.05 内の N | MPR だけ 0.05 内の N | F1 だけ 0.05 内の N | 最小の max(|ΔMPR|,|ΔF1|) |")
    P("  |---|---|---|---|---|")
    hit_any = False
    for D, meth in [(5, "RR"), (20, "RR"), (5, "RL"), (20, "RL"), (10, "RL")]:
        pub = PUB[D]
        both, only_m, only_f, bestpair = [], [], [], (9.9, None)
        for cap in CAPS:
            t = sweeps[(D, meth)][cap]
            dm, df = avg(t, "mpr") - pub["mpr"], avg(t, "f1") - pub["f1"]
            worst = max(abs(dm), abs(df))
            if worst < bestpair[0]:
                bestpair = (worst, cap)
            if abs(dm) < 0.05 and abs(df) < 0.05:
                both.append(str(cap))
            elif abs(dm) < 0.05:
                only_m.append(str(cap))
            elif abs(df) < 0.05:
                only_f.append(str(cap))
        if meth == "RR" and both:
            hit_any = True
        P(f"  | D={D} {meth} | **{' '.join(both) if both else '（無し）'}** | "
          f"{' '.join(only_m) if only_m else '—'} | {' '.join(only_f) if only_f else '—'} | "
          f"{bestpair[0]:.4f} (cap={bestpair[1]}) |")
    P("")
    P(f"  **反証条件（事前登録 §5）**: RR のどのセルでも両方 0.05 内の N が無ければ「原因は報告点数ではない」。"
      f"-> {'**不発**（説明できる N がある）' if hit_any else '**発火**（報告点数では説明できない）'}")

    # ------------------------------------------------ 天井の計器（P4）
    P("")
    P("## 天井の計器 —— 全降下を報告したときの MPR（報告規則では動かせない recall の上限）")
    P("")
    P("  | セル | 全降下 MPR | 公表 MPR | 差 | 全降下を報告しても公表に届かない問題 |")
    P("  |---|---|---|---|---|")
    for D, meth in [(5, "RL"), (5, "RR"), (10, "RL"), (20, "RL"), (20, "RR")]:
        t = sweeps[(D, meth)]["all"]
        ceil = avg(t, "mpr")
        short = [p for p in PROBS if t[p]["mpr"] < PUB[D]["mpr"] - 1e-12]
        P(f"  | D={D} {meth} | {ceil:.4f} | {PUB[D]['mpr']:.3f} | {ceil - PUB[D]['mpr']:+.4f} | "
          f"{len(short)}/16{('  ' + ' '.join(short)) if short else ''} |")

    # ------------------------------------------------ 判定 (iii): 順位の反転
    P("")
    P("## 判定 (iii) —— RL − RR の対差（Score）が N で反転するか（両側 Wilcoxon exact、16 問対）")
    for D in (5, 20):
        P("")
        P(f"### D={D}")
        P("")
        P("  | cap N | 対差 Score | 勝/分/敗 | p | rb | 対差 MPR | 対差 mean-F1 |")
        P("  |---|---|---|---|---|---|---|")
        for cap in CAPS:
            a, b = sweeps[(D, "RL")][cap], sweeps[(D, "RR")][cap]
            r = paired({p: a[p]["score"] for p in PROBS}, {p: b[p]["score"] for p in PROBS}, PROBS)
            rm = paired({p: a[p]["mpr"] for p in PROBS}, {p: b[p]["mpr"] for p in PROBS}, PROBS)
            rf = paired({p: a[p]["f1"] for p in PROBS}, {p: b[p]["f1"] for p in PROBS}, PROBS)
            P(f"  | {cap} | **{r['mean']:+.4f}** | {r['w']}/{r['t']}/{r['l']} | {r['p']:.5g} | "
              f"{r['rb']:+.3f} | {rm['mean']:+.4f} | {rf['mean']:+.4f} |")
        signs = {np.sign(paired({p: sweeps[(D, 'RL')][c][p]['score'] for p in PROBS},
                                {p: sweeps[(D, 'RR')][c][p]['score'] for p in PROBS}, PROBS)["mean"])
                 for c in CAPS}
        P("")
        P(f"  **符号の集合: {sorted(signs)} -> {'反転あり' if len(signs) > 1 else '全 N で同符号（反転なし）'}**")

    # ------------------------------------------------ 判定 (iii) の切り分け: 報告点数を揃える
    P("")
    P("## 判定 (iii) の切り分け —— 対ごとに**実報告点数を揃えた**ときの RL − RR")
    P("")
    P("  `cap=N` は本数の多い側だけを縛るので実報告点数が揃わない（D=5 cap=100 は RL 76.4 対 RR 26.3）。")
    P("  ここでは**問題ごとに両手法の小さいほう**に揃える。`min_core` は核の点数、`min_all` は降下本数。")
    for D in (5, 20):
        P("")
        P(f"### D={D}")
        P("")
        P("  | 揃え方 | RL 報告 | RR 報告 | RL Score | RR Score | 対差 Score | 勝/分/敗 | p | 対差 MPR | 対差 F1 |")
        P("  |---|---|---|---|---|---|---|---|---|---|")
        for mode in ("min_core", "min_all", "cap100 (対照)"):
            A, B = {}, {}
            for p in PROBS:
                prob = f"{p}-D{D:02d}-PIN01"
                fa, oa, xa = cells[(D, "RL")][prob]
                fb, ob, xb = cells[(D, "RR")][prob]
                K = K_of(prob)
                if mode == "min_core":
                    n = min(len(report_indices(fa, oa, xa, K, "core")),
                            len(report_indices(fb, ob, xb, K, "core")))
                elif mode == "min_all":
                    n = min(len(fa), len(fb))
                else:
                    n = 100
                for tab, (f, o, x) in ((A, (fa, oa, xa)), (B, (fb, ob, xb))):
                    tab[p] = measure(f, o, x, K, n)
            g = lambda t, k: float(np.mean([t[p][k] for p in PROBS]))
            r = paired({p: A[p]["score"] for p in PROBS}, {p: B[p]["score"] for p in PROBS}, PROBS)
            rm = paired({p: A[p]["mpr"] for p in PROBS}, {p: B[p]["mpr"] for p in PROBS}, PROBS)
            rf = paired({p: A[p]["f1"] for p in PROBS}, {p: B[p]["f1"] for p in PROBS}, PROBS)
            P(f"  | {mode} | {g(A, 'n'):.2f} | {g(B, 'n'):.2f} | {g(A, 'score'):.4f} | "
              f"{g(B, 'score'):.4f} | **{r['mean']:+.4f}** | {r['w']}/{r['t']}/{r['l']} | "
              f"{r['p']:.5g} | {rm['mean']:+.4f} | {rf['mean']:+.4f} |")
    P("")
    P("  **降下本数（16 問平均）**: "
      + "  ".join(f"D={D} RL {np.mean([len(cells[(D, 'RL')][f'{p}-D{D:02d}-PIN01'][0]) for p in PROBS]):.1f} / "
                  f"RR {np.mean([len(cells[(D, 'RR')][f'{p}-D{D:02d}-PIN01'][0]) for p in PROBS]):.1f}"
                  for D in (5, 20)))

    # ------------------------------------------------ 副次: 半径を Score / F1 の上で掃く
    P("")
    P("## 副次 —— 間引き半径を **Score / mean-F1 の上で** 掃く（その115・その130 は PR の上だけで掃いた）")
    for D, meth in [(5, "RL"), (5, "RR"), (20, "RL"), (20, "RR")]:
        P("")
        P(f"### D={D} {meth}")
        P("")
        P("  | r/span | 報告点数 | MPR | mean-F1 | Score |")
        P("  |---|---|---|---|---|")
        for frac in RADII:
            tab = {}
            for p in PROBS:
                prob = f"{p}-D{D:02d}-PIN01"
                f, opt, xs = cells[(D, meth)][prob]
                K = K_of(prob)
                idx = rule_indices(ARM, f, K, x=xs, r=frac * SPAN)
                recall, prec, f1, sc, n = score(idx, f, opt, K)
                tab[p] = dict(mpr=float(recall.mean()), f1=float(f1.mean()),
                              score=float(sc.mean()), n=int(n))
            P(f"  | {frac:g} | {avg(tab, 'n'):.2f} | {avg(tab, 'mpr'):.4f} | "
              f"{avg(tab, 'f1'):.4f} | **{avg(tab, 'score'):.4f}** |")

    # ------------------------------------------------ CSV（集計だけ。数百行なので生のまま置く）
    csv_path = os.path.join(HERE, "caps_by_problem.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["D", "method", "problem", "K", "cap", "n_reported", "mpr", "mean_f1",
                    "score", "precision", "cov_1e-1", "det_1e-5"])
        for (D, meth), sw in sorted(sweeps.items()):
            for cap in CAPS:
                for p in PROBS:
                    m = sw[cap][p]
                    w.writerow([D, meth, p, K_of(f"{p}-D{D:02d}-PIN01"), cap, m["n"],
                                f"{m['mpr']:.4f}", f"{m['f1']:.4f}", f"{m['score']:.4f}",
                                f"{m['prec']:.4f}", f"{m['cov']:.4f}", f"{m['deep']:.4f}"])
    P("")
    P(f"  -> {csv_path}（{len(sweeps) * len(CAPS) * 16} 行 ＝ 集計。行単位のダンプは作っていない）")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
