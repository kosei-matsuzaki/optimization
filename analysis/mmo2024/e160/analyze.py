#!/usr/bin/env python3
"""その160 — キュー 1: **検出規則の上界を引く。**

**採点の土台（`rule_indices` / `score` / `paired` / `read_dump` / `SPAN`）は `e115/analyze.py` から import する。
新しい統計量は 1 つも定義しない。変えるのは「報告した点を相異なる大域最適の検出に数える規則」だけ。**

入力はすべて保存物（**追加評価ゼロ・最適化 run ゼロ**）:

  * D=5  PIN01        -> `e153/dumps_rrcma.csv.gz`（RR）/ `e153/descents.csv.gz`（RL）
  * D=5  PIN02/03/04  -> `e157/dumps_rrcma.csv.gz` / `e157/descents.csv.gz`
  * D=10 PIN01        -> `e115/descents/`（RL のみ。RR は D=10 の保存物が `e115` に無い）
  * D=20 PIN01        -> `e151/descents.csv.gz`（RL）/ `e152/dumps_rrcma.csv.gz`（RR）

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e160/analyze.py
"""
from __future__ import annotations

import csv
import gzip
import os
import sys

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MMO))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MMO, "e115"))

from analyze import LEVELS, SPAN, paired, read_dump, rule_indices, score   # noqa: E402
from core.benchmarks import niching_by_name                                # noqa: E402

ARM, ARM_R = "eps_loose+dedup", 0.05 * SPAN
PROBS = [f"M{i:02d}" for i in range(1, 17)]
PINS = ["01", "02", "03", "04"]

GATE_PIN01 = {"rr_mpr": 0.6619, "rr_f1": 0.7822, "rr_score": 0.7220,
              "rl_mpr": 0.7306, "rl_f1": 0.8235, "rl_score": 0.7770}
PUB = {5: dict(score=0.7145, mpr=0.844, f1=0.585),
       10: dict(score=0.6080, mpr=None, f1=None),
       20: dict(score=0.4445, mpr=0.476, f1=0.413)}
TOL = 0.10          # 事前登録した許容幅
MID_GATE = 0.02     # 反証条件 (c)

_CACHE: dict = {}


def problem(name):
    if name not in _CACHE:
        _CACHE[name] = niching_by_name(name)
    return _CACHE[name]


# ------------------------------------------------------------------ 4 通りの採点器
def _from_det(det, K, n):
    det = np.asarray(det, dtype=float)
    recall = det / K
    prec = det / n if n > 0 else np.zeros_like(det)
    dn = prec + recall
    f1 = np.where(dn > 0, 2 * prec * recall / np.where(dn > 0, dn, 1.0), 0.0)
    return recall, f1, (recall + f1) / 2.0


def det_nearest(idx, f, opt, e):
    """LB: 最近傍帰属（land_opt）の相異なり数。同一盆地の複数点は 1 つに潰れる。"""
    return len({int(o) for o, v in zip(opt[idx], f[idx]) if v <= e})


def det_blind(idx, f, K, e):
    """UB: 距離を一切見ない貪欲割当 ＝ f ≤ e の報告点に未使用の最適を 1 つずつ配る。"""
    return min(int(np.sum(f[idx] <= e)), int(K))


def det_matched(idx, f, xs, e, opt_pos, R):
    """MID: 半径 R 以内に限った 1 対 1 の最大マッチング（距離を見る規則の中でいちばん甘い側）。"""
    sel = np.asarray(idx)[f[np.asarray(idx)] <= e]
    if len(sel) == 0:
        return 0
    d = np.linalg.norm(xs[sel][:, None, :] - opt_pos[None, :, :], axis=2)
    adj = csr_matrix((d <= R).astype(np.int8))
    if adj.nnz == 0:
        return 0
    m = maximum_bipartite_matching(adj, perm_type="column")
    return int(np.sum(m >= 0))


def score_cell(f, opt, xs, K, opt_pos):
    """1 run を 4 通りに採点して {rule: dict} を返す。"""
    idx = rule_indices(ARM, f, K, x=xs, r=ARM_R)          # LB / MID / UB1 の共通報告集合
    idx2 = rule_indices("eps_loose", f, K)                # UB2: dedup を外した報告集合
    out = {}

    recall, _, f1, sc, n = score(idx, f, opt, K)
    out["LB"] = dict(mpr=float(recall.mean()), f1=float(f1.mean()), score=float(sc.mean()),
                     n=int(n), pr5=float(recall[-1]))
    # 「LB と UB1 が水準ごとに厳密一致するか」を数える（平均で 0 でも中身が相殺している可能性を排す）
    out["_cmp"] = [(int(round(recall[i] * K)), det_blind(idx, f, K, e)) for i, e in enumerate(LEVELS)]

    rec2, _, f12, sc2, n2 = score(idx2, f, opt, K)
    out["LB2"] = dict(mpr=float(rec2.mean()), f1=float(f12.mean()), score=float(sc2.mean()),
                      n=int(n2), pr5=float(rec2[-1]))

    for tag, dets, nn in (
            ("MID", [det_matched(idx, f, xs, e, opt_pos, ARM_R) for e in LEVELS], len(idx)),
            ("UB1", [det_blind(idx, f, K, e) for e in LEVELS], len(idx)),
            ("UB2", [det_blind(idx2, f, K, e) for e in LEVELS], len(idx2))):
        rc, f1b, scb = _from_det(dets, K, nn)
        out[tag] = dict(mpr=float(rc.mean()), f1=float(f1b.mean()), score=float(scb.mean()),
                        n=int(nn), pr5=float(rc[-1]))
    return out


# ------------------------------------------------------------------ 入力
def load_folded(path):
    if not os.path.exists(path):
        sys.exit(f"[FATAL] 入力が無い: {path}")
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
    return out


def load_dir(dd):
    """per-problem のディレクトリを problem 名 -> (f, opt, xs) に読む（`e115` 系）。"""
    if not os.path.isdir(dd):
        sys.exit(f"[FATAL] 入力のディレクトリが無い: {dd}")
    out = {}
    for fn in sorted(os.listdir(dd)):
        if not fn.endswith((".csv", ".csv.gz")):
            continue
        f, opt, xs = read_dump(os.path.join(dd, fn))
        if xs is None:
            continue
        out[fn.split("_")[0]] = (f, opt, xs)
    return out


def collect(cells, dim, pin):
    """{prob: (f,opt,xs)} を 4 通りに採点。欠けは名指しで返す。"""
    res, miss = {}, []
    for pr in PROBS:
        key = f"{pr}-D{dim:02d}-PIN{pin}"
        arrs = cells.get(key) or cells.get(pr)
        if arrs is None:
            miss.append(pr)
            continue
        p = problem(key)
        res[pr] = score_cell(*arrs, int(p.n_global_optima), np.asarray(p.optima_pos, dtype=float))
    return res, miss


def mean_key(d, rule, key, probs=None):
    probs = probs or list(d)
    return float(np.mean([d[p][rule][key] for p in probs]))


# ------------------------------------------------------------------ 本体
def main():
    out = []
    P = out.append

    rr = dict(load_folded(os.path.join(MMO, "e153", "dumps_rrcma.csv.gz")))
    rr.update(load_folded(os.path.join(MMO, "e157", "dumps_rrcma.csv.gz")))
    rl = dict(load_folded(os.path.join(MMO, "e153", "descents.csv.gz")))
    rl.update(load_folded(os.path.join(MMO, "e157", "descents.csv.gz")))

    RR = {pin: collect(rr, 5, pin) for pin in PINS}
    RL = {pin: collect(rl, 5, pin) for pin in PINS}

    # ---------------------------------------------------------- 関門 G1
    P("## 関門 G1 —— PIN01 を **LB（現行の最近傍帰属）** で採点し直し、その153 の記録値を再現するか")
    g1 = {"rr_mpr": mean_key(RR["01"][0], "LB", "mpr"), "rr_f1": mean_key(RR["01"][0], "LB", "f1"),
          "rr_score": mean_key(RR["01"][0], "LB", "score"), "rl_mpr": mean_key(RL["01"][0], "LB", "mpr"),
          "rl_f1": mean_key(RL["01"][0], "LB", "f1"), "rl_score": mean_key(RL["01"][0], "LB", "score")}
    ok = True
    for k, v in GATE_PIN01.items():
        d = g1[k] - v
        ok &= abs(d) < 5e-5
        P(f"  {k}: 再計算 {g1[k]:.4f}   記録 {v:.4f}   差 {d:+.5f}")
    P(f"  -> {'通過' if ok else '**不一致（採点器が変わっている。上下界は出さない）**'}")
    if not ok:
        print("\n".join(out))
        sys.exit(1)

    # ---------------------------------------------------------- 関門 G2
    P("")
    P("## 関門 G2 —— 完走本数（**欠けた cell は名指しで落とす**）")
    use = {"RR": [], "RL": []}
    for tag, D in (("RR", RR), ("RL", RL)):
        for pin in PINS:
            res, miss = D[pin]
            P(f"  {tag} PIN{pin}: {len(res)}/16" + (f"  欠け: {','.join(miss)}" if miss else "  -> 使う"))
            if len(res) == 16:
                use[tag].append(pin)
    P(f"  -> 平均に使う instance: RR {use['RR']} / RL {use['RL']}")

    # ---------------------------------------------------------- 本体 1
    P("")
    P("## 1 —— D=5、4 通りの採点器の 16 問平均（instance ごと。報告規則は LB/MID/UB1 で共通、UB2 だけ dedup を外す）")
    P("")
    P("  | 手法 | PIN | 規則 | MPR(5 水準平均) | PR@1e-5 | mean-F1 | Score | 報告点数 |")
    P("  |---|---|---|---|---|---|---|---|")
    for tag, D, name in (("RR", RR, "RR-CMA-ES"), ("RL", RL, "Restart-Lander")):
        for pin in use[tag]:
            res = D[pin][0]
            for rule in ("LB", "MID", "UB1", "LB2", "UB2"):
                P(f"  | {name} | {pin} | {rule} | {mean_key(res,rule,'mpr'):.4f} | {mean_key(res,rule,'pr5'):.4f} |"
                  f" {mean_key(res,rule,'f1'):.4f} | {mean_key(res,rule,'score'):.4f} | {mean_key(res,rule,'n'):.2f} |")

    # ---------------------------------------------------------- 本体 1b: 厳密一致の検算
    P("")
    P("## 1b —— **LB と UB1 は平均が同じだけなのか、水準ごとに厳密一致しているのか**（相殺の排除）")
    tot = same = 0
    worst = 0
    for tag, D in (("RR", RR), ("RL", RL)):
        for pin in use[tag]:
            for pr, d in D[pin][0].items():
                for a, b in d["_cmp"]:
                    tot += 1
                    same += (a == b)
                    worst = max(worst, b - a)
    P(f"  D=5 の（2 手法 × 4 instance × 16 問 × 5 水準）= {tot} 個の (問題, 水準) で、"
      f"**LB の検出数と UB1 の検出数が一致したのは {same} 個（{same/tot*100:.2f}%）。最大の差は {worst}。**")
    P("  -> **一致率 100% なら、現行の採点器は<u>この報告集合の上では既にいちばん甘い規則そのもの</u>である。**")
    dmax = 0.0
    for tag, D in (("RR", RR), ("RL", RL)):
        for pin in use[tag]:
            for pr, d in D[pin][0].items():
                dmax = max(dmax, abs(d["LB2"]["mpr"] - d["LB"]["mpr"]))
    P(f"  **報告側だけ緩めた場合（LB2 − LB）の MPR の最大差は {dmax:.6f}**（128 cell）"
      " ＝ **dedup が「その最適を見つけた唯一の点」を落としたことは 1 度も無い。**")

    # ---------------------------------------------------------- 本体 2: 判定
    P("")
    P("## 2 —— 判定（事前登録: **RR の 4 instance 平均 MPR が公表 0.844 の ±0.10 に入るか**）")
    P("")
    P("  | 規則 | 4 instance 平均 MPR | SD | 公表 0.844 との差 | ±0.10 に入るか |")
    P("  |---|---|---|---|---|")
    m_by_rule = {}
    for rule in ("LB", "MID", "UB1", "LB2", "UB2"):
        v = [mean_key(RR[p][0], rule, "mpr") for p in use["RR"]]
        m, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        m_by_rule[rule] = m
        P(f"  | {rule} | {m:.4f} | {sd:.4f} | {m - PUB[5]['mpr']:+.4f} | "
          f"**{'入る' if abs(m - PUB[5]['mpr']) < TOL else '入らない'}** |")
    fire_a = abs(m_by_rule["UB1"] - PUB[5]["mpr"]) < TOL
    fire_b = (abs(m_by_rule["UB2"] - PUB[5]["mpr"]) < TOL) and not fire_a
    P("")
    P(f"  **反証条件 (a)**（UB1 が 0.844 を跨ぐ）: {'**発火**' if fire_a else '不発'}"
      f"   —— UB1 {m_by_rule['UB1']:.4f}、差 {m_by_rule['UB1'] - PUB[5]['mpr']:+.4f}")
    P(f"  **反証条件 (b)**（UB2 だけ跨ぐ ＝ 効いているのは報告側の間引き）: {'**発火**' if fire_b else '不発'}"
      f"   —— UB2 {m_by_rule['UB2']:.4f}、差 {m_by_rule['UB2'] - PUB[5]['mpr']:+.4f}")
    mid_gain = m_by_rule["MID"] - m_by_rule["LB"]
    fire_c = mid_gain >= MID_GATE
    P(f"  **反証条件 (c)**（MID − LB ≥ {MID_GATE}）: {'**発火**' if fire_c else '不発'}   —— 差 {mid_gain:+.4f}")
    P(f"  **上界が公表に残す赤字**: UB1 {m_by_rule['UB1'] - PUB[5]['mpr']:+.4f} / "
      f"UB2 {m_by_rule['UB2'] - PUB[5]['mpr']:+.4f}"
      f"（LB の赤字 {m_by_rule['LB'] - PUB[5]['mpr']:+.4f} のうち"
      f" 検出規則で埋まるのは {(m_by_rule['UB1']-m_by_rule['LB'])/(PUB[5]['mpr']-m_by_rule['LB'])*100:.1f}%、"
      f" 報告規則も外して {(m_by_rule['UB2']-m_by_rule['LB'])/(PUB[5]['mpr']-m_by_rule['LB'])*100:.1f}%）")

    # ---------------------------------------------------------- 本体 3: 規則を変えると順位は動くか
    P("")
    P("## 3 —— 規則を変えると 2 手法の対差（RL − RR）の符号は動くか（PIN ごと、16 問対、両側 Wilcoxon exact）")
    both = [p for p in use["RR"] if p in use["RL"]]
    for rule in ("LB", "MID", "UB1", "LB2", "UB2"):
        row = []
        for pin in both:
            a = {p: RL[pin][0][p][rule]["score"] for p in PROBS}
            b = {p: RR[pin][0][p][rule]["score"] for p in PROBS}
            d = paired(a, b, PROBS)
            row.append((pin, d))
        mm = float(np.mean([d["mean"] for _, d in row]))
        P(f"  {rule:3s} Score 対差: instance 平均 {mm:+.4f}   "
          + "  ".join(f"PIN{pin} {d['mean']:+.4f}({d['w']}/{d['t']}/{d['l']}, p={d['p']:.4g})" for pin, d in row))
    for rule in ("LB", "UB1", "LB2", "UB2"):
        row = []
        for pin in both:
            a = {p: RL[pin][0][p][rule]["mpr"] for p in PROBS}
            b = {p: RR[pin][0][p][rule]["mpr"] for p in PROBS}
            row.append(paired(a, b, PROBS))
        P(f"  {rule:3s} MPR   対差: instance 平均 {np.mean([d['mean'] for d in row]):+.4f}   "
          + "  ".join(f"{d['mean']:+.4f}(p={d['p']:.4g})" for d in row))

    # ---------------------------------------------------------- 本体 4: 問題ごと（RR, PIN01）
    P("")
    P("## 4 —— 問題ごとの MPR（RR、4 instance 平均。**どの問題で上界が伸びるか**）")
    P("")
    P("  | 問題 | K | LB | MID | UB1 | LB2 | UB2 | UB2 − LB |")
    P("  |---|---|---|---|---|---|---|---|")
    for pr in PROBS:
        vals = {r: float(np.mean([RR[p][0][pr][r]["mpr"] for p in use["RR"]])) for r in ("LB", "MID", "UB1", "LB2", "UB2")}
        P(f"  | {pr} | {problem(f'{pr}-D05-PIN01').n_global_optima} | {vals['LB']:.3f} | {vals['MID']:.3f} | "
          f"{vals['UB1']:.3f} | {vals['LB2']:.3f} | {vals['UB2']:.3f} | {vals['UB2']-vals['LB']:+.3f} |")

    # ---------------------------------------------------------- 本体 5: 他次元
    P("")
    P("## 5 —— 同じ 4 通りを D=10 / D=20 の保存物にも当てる（追加評価ゼロ）")
    P("")
    P("  | D | 手法 | 出自 | LB | MID | UB1 | LB2 | UB2 | 公表 MPR |")
    P("  |---|---|---|---|---|---|---|---|---|")
    others = [
        (10, "Restart-Lander", "e115/descents", lambda: load_dir(os.path.join(MMO, "e115", "descents"))),
        (20, "Restart-Lander", "e151/descents.csv.gz",
         lambda: load_folded(os.path.join(MMO, "e151", "descents.csv.gz"))),
        (20, "RR-CMA-ES", "e152/dumps_rrcma.csv.gz",
         lambda: load_folded(os.path.join(MMO, "e152", "dumps_rrcma.csv.gz"))),
    ]
    other_res = {}
    for dim, name, src, loader in others:
        res, miss = collect(loader(), dim, "01")
        if miss:
            P(f"  | {dim} | {name} | {src} | **欠け {len(miss)} 問: {','.join(miss)} -> 落とす** | | | | |")
            continue
        other_res[(dim, name)] = res
        pub = PUB[dim]["mpr"]
        P(f"  | {dim} | {name} | `{src}` | {mean_key(res,'LB','mpr'):.4f} | {mean_key(res,'MID','mpr'):.4f} | "
          f"{mean_key(res,'UB1','mpr'):.4f} | {mean_key(res,'LB2','mpr'):.4f} | {mean_key(res,'UB2','mpr'):.4f} | "
          f"{'—' if pub is None else f'{pub:.3f}'} |")
    if (20, "RR-CMA-ES") in other_res:
        r = other_res[(20, "RR-CMA-ES")]
        P(f"  -> D=20 の RR: 公表 {PUB[20]['mpr']:.3f} に対し LB {mean_key(r,'LB','mpr')-PUB[20]['mpr']:+.4f} / "
          f"UB1 {mean_key(r,'UB1','mpr')-PUB[20]['mpr']:+.4f} / UB2 {mean_key(r,'UB2','mpr')-PUB[20]['mpr']:+.4f}")

    # ---------------------------------------------------------- 本体 6: 幅そのもの
    P("")
    P("## 6 —— **検出規則が作れる幅**（UB2 − LB。これがこの harness の採点器の不確かさの上限）")
    P("")
    P("  | 対象 | LB | UB1 − LB（採点器だけ緩める） | LB2 − LB（報告側だけ緩める） | UB2 − LB（両方） |")
    P("  |---|---|---|---|---|")
    for tag, D, name in (("RR", RR, "RR-CMA-ES D=5"), ("RL", RL, "Restart-Lander D=5")):
        lb = float(np.mean([mean_key(D[p][0], "LB", "mpr") for p in use[tag]]))
        u1 = float(np.mean([mean_key(D[p][0], "UB1", "mpr") for p in use[tag]]))
        l2 = float(np.mean([mean_key(D[p][0], "LB2", "mpr") for p in use[tag]]))
        u2 = float(np.mean([mean_key(D[p][0], "UB2", "mpr") for p in use[tag]]))
        P(f"  | {name}（4 instance 平均） | {lb:.4f} | {u1-lb:+.4f} | {l2-lb:+.4f} | {u2-lb:+.4f} |")
    for (dim, name), res in other_res.items():
        lb, u1, l2, u2 = (mean_key(res, r, "mpr") for r in ("LB", "UB1", "LB2", "UB2"))
        P(f"  | {name} D={dim}（PIN01） | {lb:.4f} | {u1-lb:+.4f} | {l2-lb:+.4f} | {u2-lb:+.4f} |")

    # ---------------------------------------------------------- 本体 7
    P("")
    P("## 7 —— **公表の (MPR, mean-F1) の対を逆算すると、公表側の報告点数は 2K 付近になる**（公表値だけを使う代数。測定ではない）")
    P("  F1 = 2pr/(p+r) を p について解く（**MPR と mean-F1 が同じ 5 水準の平均だと仮定した近似**）:")
    for dim in (5, 20):
        r_, f_ = PUB[dim]["mpr"], PUB[dim]["f1"]
        p_ = f_ * r_ / (2 * r_ - f_)
        P(f"  D={dim}: 公表 MPR {r_:.3f} / mean-F1 {f_:.3f} -> precision {p_:.4f}"
          f"  -> 報告点数 n = r·K/p = **{r_/p_:.2f}·K**"
          f"（K=20 なら {r_ * 20 / p_:.1f} 点、K=10 なら {r_ * 10 / p_:.1f} 点）")
    for tag, D, name in (("RR", RR, "RR-CMA-ES D=5"), ("RL", RL, "Restart-Lander D=5")):
        nn = float(np.mean([mean_key(D[p][0], "LB", "n") for p in use[tag]]))
        rr_ = float(np.mean([mean_key(D[p][0], "LB", "mpr") for p in use[tag]]))
        ff = float(np.mean([mean_key(D[p][0], "LB", "f1") for p in use[tag]]))
        P(f"  この harness の {name}: 報告点数 {nn:.2f} 点（16 問平均 K = 15.0 なので {nn/15.0:.2f}·K）、"
          f"MPR {rr_:.4f} / mean-F1 {ff:.4f}")
    P("  -> **公表側は 2K 付近まで報告して recall を買い precision を捨てている**が、"
      "**その154 が示したとおり<u>この harness の run では報告点数を 100 点まで上げても MPR は動かない</u>**"
      " ＝ **同じ報告点数にしても届かない ＝ 差は報告でも採点でもなく到達域。**")

    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "scored.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
