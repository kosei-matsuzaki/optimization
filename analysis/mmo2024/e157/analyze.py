#!/usr/bin/env python3
"""その157 — キュー 1: **公表値との −0.18 の食い違いは instance を 1 本で代表させたせいか。**

**採点・規則・統計量は `e115/analyze.py` からそのまま import する**（`rule_indices` / `score` / `paired` / `read_dump` / `SPAN`）。
**新しい統計量は 1 つも定義しない。**

入力の出自:

  * **PIN01 の 2 手法（D=5）** -> `e153/dumps_rrcma.csv.gz`（RR）と `e153/descents.csv.gz`（RL）。**追加評価ゼロ**
  * **PIN02/03/04 の RR**     -> `e157/dumps/`（この回の新規 48 run）
  * **PIN02/03/04 の RL**     -> `e157/descents/`（枠が余った場合のみ。**無ければ「未測定」と印字して落とす。黙って飛ばさない**）
  * **公表最良 D=5**          -> 転記（測らない）

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e157/analyze.py
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
PINS = ["01", "02", "03", "04"]

# --- 転記 1: その153 の PIN01 実測（関門 G1 の照合先） ---
GATE_PIN01 = {"rr_mpr": 0.6619, "rr_f1": 0.7822, "rr_score": 0.7220,
              "rl_mpr": 0.7306, "rl_f1": 0.8235, "rl_score": 0.7770}
# --- 転記 2: 公表最良（競技資料、出典 1 本）。D=5 の最良手法は RR-CMA-ES 自身 ---
PUB = dict(score=0.7145, mpr=0.844, f1=0.585)
SD_SEED_RR = 0.0238        # その123 の n=3 seed ばらつき（反証条件 (b) の閾値）
R_INSTANCE_D10 = 0.4899    # その149 の D=10 instance 間相関
TOL = 0.10                 # 判定の許容幅（事前登録）

_K: dict = {}


def K_of(prob):
    if prob not in _K:
        _K[prob] = int(niching_by_name(prob).n_global_optima)
    return _K[prob]


def score_arrays(f, opt, xs, K):
    idx = rule_indices(ARM, f, K, x=xs, r=ARM_R)
    recall, prec, f1, sc, n = score(idx, f, opt, K)
    return dict(mpr=float(recall.mean()), f1=float(f1.mean()), score=float(sc.mean()),
                n=int(n), ndump=len(f), cov=float(recall[0]))


def load_folded(path):
    """`problem` 列つきの畳んだダンプを problem -> (f, opt, xs) に割る。"""
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


def cell_paths(meth, prob, pin):
    """その cell の生ダンプの path（無ければ None）。**存在しない入力は黙って飛ばさない。**"""
    name = f"{prob}-D05-PIN{pin}"
    if meth == "RR":
        p = os.path.join(HERE, "dumps", f"{name}_cov20_seed0.csv")
    else:
        p = os.path.join(HERE, "descents", f"{name}_seed0.csv")
    for cand in (p, p + ".gz"):
        if os.path.exists(cand):
            return cand
    return None


# 畳んだ後は `problem` 列つきの 1 本を読む（畳む前の per-problem と**出力が 1 文字も変わらない**ことを確認済み）
_FOLDED: dict = {}


def folded_cell(meth, prob, pin):
    f = "dumps_rrcma.csv.gz" if meth == "RR" else "descents.csv.gz"
    if meth not in _FOLDED:
        path = os.path.join(HERE, f)
        if not os.path.exists(path):
            sys.exit(f"[FATAL] 入力が無い: {path}\n"
                     "  その162 の統合で削除した（同じディレクトリの DUMPS_REMOVED.md を読むこと）。\n"
                     "  この回の数値は acceptance_topology.md の その157 の節と scored.txt にある。")
        _FOLDED[meth] = load_folded(path)
    return _FOLDED[meth].get(f"{prob}-D05-PIN{pin}")


def collect(meth, folded_pin01):
    """meth の (pin -> prob -> 採点) と、欠けた cell の一覧を返す。"""
    res, missing = {}, {}
    for pin in PINS:
        res[pin], miss = {}, []
        for pr in PROBS:
            K = K_of(f"{pr}-D05-PIN{pin}")
            if pin == "01":
                arrs = folded_pin01.get(f"{pr}-D05-PIN01")
                if arrs is None:
                    miss.append(pr)
                    continue
                res[pin][pr] = score_arrays(*arrs, K)
            else:
                arrs = folded_cell(meth, pr, pin)
                if arrs is None:
                    path = cell_paths(meth, pr, pin)
                    if path is None:
                        miss.append(pr)
                        continue
                    arrs = read_dump(path)
                res[pin][pr] = score_arrays(*arrs, K)
        missing[pin] = miss
    return res, missing


def mean_key(d, key, probs=None):
    probs = probs or list(d)
    return float(np.mean([d[p][key] for p in probs]))


def main():
    out = []
    P = out.append

    rr01 = load_folded(os.path.join(MMO, "e153", "dumps_rrcma.csv.gz"))
    rl01 = load_folded(os.path.join(MMO, "e153", "descents.csv.gz"))
    RR, RR_miss = collect("RR", rr01)
    RL, RL_miss = collect("RL", rl01)

    # ------------------------------------------------ 関門 G1
    P("## 関門 G1 —— PIN01 の保存物を同じ採点器で読み直し、その153 の記録値を再現するか（追加評価ゼロ）")
    g1 = {"rr_mpr": mean_key(RR["01"], "mpr"), "rr_f1": mean_key(RR["01"], "f1"),
          "rr_score": mean_key(RR["01"], "score"), "rl_mpr": mean_key(RL["01"], "mpr"),
          "rl_f1": mean_key(RL["01"], "f1"), "rl_score": mean_key(RL["01"], "score")}
    okg1 = True
    for k, v in GATE_PIN01.items():
        d = g1[k] - v
        okg1 &= abs(d) < 5e-5
        P(f"  {k}: 再計算 {g1[k]:.4f}   記録 {v:.4f}   差 {d:+.5f}")
    P(f"  -> {'通過' if okg1 else '**不一致（採点器が変わっている。instance の比較はしない）**'}")
    if not okg1:
        print("\n".join(out))
        sys.exit(1)

    # ------------------------------------------------ 関門 G2
    P("")
    P("## 関門 G2 —— 完走本数（**欠けた cell は名指しで落とす。黙って飛ばさない**）")
    usable = {"RR": [], "RL": []}
    for meth, res, miss in (("RR", RR, RR_miss), ("RL", RL, RL_miss)):
        for pin in PINS:
            n = len(res[pin])
            tag = "使う" if n == 16 else ("**落とす**" if n < 16 else "?")
            if n == 16:
                usable[meth].append(pin)
            P(f"  {meth} PIN{pin}: {n}/16 完走 -> {tag}"
              + (f"  欠け: {','.join(miss[pin])}" if miss[pin] else ""))
    P(f"  -> 平均に使う instance: RR {usable['RR']} / RL {usable['RL'] or '（未測定）'}")

    # ------------------------------------------------ 本体 1: instance ごとの 16 問平均
    P("")
    P("## 1 —— instance ごとの 16 問平均（D=5、seed 0、正規予算 25 万、報告規則 eps_loose+dedup）")
    P("")
    P("  | 手法 | PIN | MPR | mean-F1 | Score | 報告点数 | 1e-1 被覆 |")
    P("  |---|---|---|---|---|---|---|")
    for meth, res in (("RR-CMA-ES", RR), ("Restart-Lander", RL)):
        key = "RR" if meth.startswith("RR") else "RL"
        for pin in usable[key]:
            d = res[pin]
            P(f"  | {meth} | {pin} | {mean_key(d,'mpr'):.4f} | {mean_key(d,'f1'):.4f} | "
              f"{mean_key(d,'score'):.4f} | {mean_key(d,'n'):.2f} | {mean_key(d,'cov'):.4f} |")

    # ------------------------------------------------ 本体 2: 判定
    P("")
    P("## 2 —— 判定（事前登録: **RR の instance 平均 MPR が公表 0.844 の ±0.10 に入るか**）")
    rr_mprs = [mean_key(RR[p], "mpr") for p in usable["RR"]]
    rr_scores = [mean_key(RR[p], "score") for p in usable["RR"]]
    rr_f1s = [mean_key(RR[p], "f1") for p in usable["RR"]]
    m = float(np.mean(rr_mprs))
    sd = float(np.std(rr_mprs, ddof=1)) if len(rr_mprs) > 1 else float("nan")
    P(f"  RR の instance 平均 MPR（n={len(rr_mprs)} instance）: {m:.4f}   SD {sd:.4f}"
      f"   [{min(rr_mprs):.4f} .. {max(rr_mprs):.4f}]")
    P(f"  公表 MPR {PUB['mpr']:.3f} との差: {m - PUB['mpr']:+.4f}   （PIN01 単独では {rr_mprs[0] - PUB['mpr']:+.4f}）")
    P(f"  Score : instance 平均 {np.mean(rr_scores):.4f}  SD {np.std(rr_scores, ddof=1):.4f}  公表 {PUB['score']:.4f} との差 {np.mean(rr_scores)-PUB['score']:+.4f}")
    P(f"  meanF1: instance 平均 {np.mean(rr_f1s):.4f}  SD {np.std(rr_f1s, ddof=1):.4f}  公表 {PUB['f1']:.4f} との差 {np.mean(rr_f1s)-PUB['f1']:+.4f}")
    fire_a = abs(m - PUB["mpr"]) >= TOL
    P(f"  **判定**: {'公表の ±0.10 に入らない' if fire_a else '公表の ±0.10 に入る'}"
      f" -> **反証条件 (a) は {'発火' if fire_a else '不発'}**")
    fire_b = sd == sd and sd <= 10 * SD_SEED_RR and sd < abs(m - PUB["mpr"]) / 2
    P(f"  **反証条件 (b)**: instance 間 SD {sd:.4f} 対 seed ばらつき {SD_SEED_RR:.4f}"
      f"（比 {sd/SD_SEED_RR:.2f} 倍） -> {'**発火（instance は seed と同じ疑似反復）**' if fire_b else '不発'}")

    # ------------------------------------------------ 本体 3: instance 間相関（疑似反復の直接測定）
    P("")
    P("## 3 —— 問題ごとの値の instance 間相関（n=16。その149 の D=10 instance 間 r=+0.4899 と比べる）")
    for key in ("score", "mpr"):
        for i, a in enumerate(usable["RR"]):
            for b in usable["RR"][i + 1:]:
                va = np.array([RR[a][p][key] for p in PROBS])
                vb = np.array([RR[b][p][key] for p in PROBS])
                r, pr = stats.pearsonr(va, vb)
                rho, prho = stats.spearmanr(va, vb)
                P(f"  RR {key:5s} PIN{a} vs PIN{b}: Pearson r={r:+.4f}（p={pr:.4g}） / Spearman rho={rho:+.4f}（p={prho:.4g}）")

    # ------------------------------------------------ 本体 4: RL − RR の対差（副次）
    P("")
    P("## 4 —— 副次: RL − RR の対差（両手法が揃った instance だけ。両側 Wilcoxon exact）")
    both = [p for p in usable["RR"] if p in usable["RL"]]
    if not both:
        P("  **`Restart-Lander` は枠に入らなかった ＝ 未測定。事前登録どおり落とした側。**")
    for pin in both:
        for key in ("score", "mpr", "f1"):
            a = {p: RL[pin][p][key] for p in PROBS}
            b = {p: RR[pin][p][key] for p in PROBS}
            d = paired(a, b, PROBS)
            P(f"  PIN{pin} {key:5s}: mean {d['mean']:+.4f}  {d['w']}/{d['t']}/{d['l']}  p={d['p']:.5g}  rb={d['rb']:+.3f}")
    if len(both) > 1:
        for key in ("score", "mpr"):
            ds = [paired({p: RL[pin][p][key] for p in PROBS},
                         {p: RR[pin][p][key] for p in PROBS}, PROBS)["mean"] for pin in both]
            P(f"  **instance 平均の対差（{key}）: {np.mean(ds):+.4f}**（SD {np.std(ds, ddof=1):.4f}、"
              f"4 本: {' / '.join(f'{d:+.4f}' for d in ds)}）")
    if both:
        rl_m = [mean_key(RL[p], "mpr") for p in usable["RL"]]
        rl_s = [mean_key(RL[p], "score") for p in usable["RL"]]
        sd_m = f"{np.std(rl_m, ddof=1):.4f}" if len(rl_m) > 1 else "n=1 なので無し"
        sd_s = f"{np.std(rl_s, ddof=1):.4f}" if len(rl_s) > 1 else "n=1 なので無し"
        P(f"  RL の instance 平均（n={len(rl_m)}）: MPR {np.mean(rl_m):.4f}（SD {sd_m}） / "
          f"Score {np.mean(rl_s):.4f}（SD {sd_s}、公表最良 {PUB['score']:.4f} との差 {np.mean(rl_s)-PUB['score']:+.4f}）")

    # ------------------------------------------------ 本体 5: 問題ごとの表（RR）
    P("")
    P("## 5 —— 問題ごとの RR の MPR（instance 別）")
    P("")
    P("  | 問題 | K | " + " | ".join(f"PIN{p}" for p in usable["RR"]) + " | 4 本の SD |")
    P("  |---|---|" + "---|" * (len(usable["RR"]) + 1))
    for pr in PROBS:
        vals = [RR[p][pr]["mpr"] for p in usable["RR"]]
        P(f"  | {pr} | {K_of(f'{pr}-D05-PIN01')} | " + " | ".join(f"{v:.3f}" for v in vals)
          + f" | {np.std(vals, ddof=1):.3f} |")

    # ------------------------------------------------ 本体 6: PIN01 はどれだけ難しい側か / 2 つの直しを足しても届くか
    P("")
    P("## 6 —— PIN01 は平均より難しい側か（RR、16 問対検定）と、**その155 の c=2 と足し合わせた上界**")
    for pin in usable["RR"][1:]:
        a = {p: RR[pin][p]["mpr"] for p in PROBS}
        b = {p: RR["01"][p]["mpr"] for p in PROBS}
        d = paired(a, b, PROBS)
        P(f"  PIN{pin} − PIN01 (MPR): mean {d['mean']:+.4f}  {d['w']}/{d['t']}/{d['l']}  p={d['p']:.5g}  rb={d['rb']:+.3f}")
    C2_GAIN = 0.0487        # その155 §3 の c=2 − c=20（PIN01、MPR、非有意）
    ub = m + C2_GAIN
    P(f"  instance 平均 {m:.4f} ＋ その155 の c=2 の効き {C2_GAIN:+.4f} = **{ub:.4f}**"
      f"   公表 0.844 との差 {ub - PUB['mpr']:+.4f}"
      f" -> **2 つの直しを足しても ±0.10 に{'入る' if abs(ub-PUB['mpr'])<TOL else '入らない'}**"
      "（c=2 の効きは PIN01 でしか測っておらず、しかも非有意なので、これは上界として読む）")
    P(f"  PIN01 が説明する赤字の割合: {(m - rr_mprs[0]) / (PUB['mpr'] - rr_mprs[0]) * 100:.1f}%"
      f"（PIN01 の赤字 {PUB['mpr'] - rr_mprs[0]:.4f} のうち {m - rr_mprs[0]:.4f} 分）")

    # ------------------------------------------------ 本体 7: 再起動の計器（instance 別。追加評価ゼロ）
    rp = os.path.join(HERE, "restarts.csv.gz")
    if not os.path.exists(rp):
        sys.exit(f"[FATAL] 入力が無い: {rp}\n"
                 "  その162 の統合で削除した（同じディレクトリの DUMPS_REMOVED.md を読むこと）。\n"
                 "  §4 の表（再起動本数 125.2-125.8 / 1 本あたり評価 2,031-2,045 / archive 21.94-25.38）は\n"
                 "  acceptance_topology.md の その157 の節 §4 と scored.txt にある。")
    if True:
        P("")
        P("## 7 —— RR の再起動の計器（instance 別、16 問平均。その155 の PIN01 実測 125.1 本 / 1 本 2,044 評価と比べる）")
        with gzip.open(rp, "rt") as fh:
            rows = list(csv.DictReader(fh))
        per: dict = {}
        for r in rows:
            pin = r["problem"][-2:]
            per.setdefault(pin, {}).setdefault(r["problem"], []).append(r)
        P("")
        P("  | PIN | 再起動本数（16 問平均） | 1 本あたり評価 | archive（16 問平均） |")
        P("  |---|---|---|---|")
        for pin in sorted(per):
            ns = [len(v) for v in per[pin].values()]
            ev = [float(v[-1]["evals_cum"]) / len(v) for v in per[pin].values()]
            ar = [float(v[-1]["archive_size"]) for v in per[pin].values()]
            P(f"  | {pin} | {np.mean(ns):.1f} | {np.mean(ev):,.0f} | {np.mean(ar):.2f} |")

    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "scored.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
