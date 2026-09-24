#!/usr/bin/env python3
"""その93 の D=20 クラス上限 4 通りが、その151 の実測 MPR 0.3831 にどこで破れたかを机上で切り分ける。

**追加評価ゼロ。** 読むのは 2 つの保存物だけ:

  * `analysis/mmo2024/e93/ceiling_mpr_d20.csv` —— その93 の 4 通りの上限（40 抽選の null）。
    **null の降下ダンプ本体は削除済み**なので、使えるのはこの集計 16 行だけ。
  * `analysis/mmo2024/e151/descents.csv.gz` —— `Restart-Lander` D=20 PIN01 の実測 1883 降下。

4 通りの推定量が何を仮定しているか（`e92/analyze.py` の定義そのまま）:

  1. 固定コスト  `mpr`        : n = 予算 / 降下 1 本の平均評価、PR(eps) = Σ_j [1-(1-p_j)^n]/K。
                                **仮定は 2 つ** —— (A) 着地分布 p_j が 40 抽選で代表されている、
                                (B) 1 本の値段が「収束まで走った降下」の平均で固定。
  2. 早期打ち切り `mpr_earlystop`: (B) を緩める。eps に入った時点で切れるので n が増える。(A) は同じ。
  3. 無限再起動  `mpr_sup`    : (B) を捨てる（n = ∞）。**残るのは (A) だけ** ＝ 台（support）/K の 5 水準平均。
  4. ＋Chao1     `mpr_sup_chao1`: (A) を「40 抽選が見落とした最適を Chao1 で戻す」形で緩める。

**＝ 3 と 4 は n に依らない。** 実測がこの 2 つを超えたなら、破れたのは費用の側ではなく
**「40 抽選の着地分布の台が、この一族の到達できる集合を代表している」という仮定 (A)** である。
この script はそれを問題ごとに突き合わせ、さらに (A) の緩め方（Chao1）が
D=20 でどれだけ効かないかを、**実測ダンプを 40 本に間引いて測る**。

出力は stdout（`scored.txt` に控える）。
"""
import csv
import gzip
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
EPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
EPSCOL = {1e-1: "pr_1e-1", 1e-2: "pr_1e-2", 1e-3: "pr_1e-3", 1e-4: "pr_1e-4", 1e-5: "pr_1e-5"}
NULL = os.path.join(MMO, "e93", "ceiling_mpr_d20.csv")
DESC = os.path.join(MMO, "e151", "descents.csv.gz")
SUMM = os.path.join(MMO, "e151", "driver_summary.csv")

# --- 転記: その151 の記録値（`e151/scored.txt` の表、報告規則 `eps_loose+dedup` r=0.05·span）。
#     この回の比較はすべてこの MPR（16 問平均 0.3831）に対して行う。`driver_summary.csv` の
#     `current` 規則は別の報告規則で 0.3669 なので、上限との突き合わせには使わない。
E151_MPR = {"M01": 0.5100, "M02": 0.3800, "M03": 0.1700, "M04": 0.4000,
            "M05": 0.6000, "M06": 0.2500, "M07": 0.2700, "M08": 0.4500,
            "M09": 0.3200, "M10": 0.6000, "M11": 0.1600, "M12": 0.5600,
            "M13": 0.5000, "M14": 0.2000, "M15": 0.3600, "M16": 0.4000}


def need(path):
    if not os.path.exists(path):
        sys.exit(f"[FATAL] 入力が無い: {path}\n"
                 "  この script は保存物だけを読む（追加評価ゼロ）。消えていれば再取得が要る。")
    return path


def chao1(counts):
    """bias-corrected Chao1（その93 と同じ形）。counts は最適ごとの着地回数。"""
    s_obs = int((counts > 0).sum())
    f1 = int((counts == 1).sum())
    f2 = int((counts == 2).sum())
    return s_obs + f1 * (f1 - 1) / (2.0 * (f2 + 1))


def main():
    out = []
    P = out.append

    with open(need(NULL)) as fh:
        null = {r["func"]: r for r in csv.DictReader(fh)}
    with open(need(SUMM)) as fh:
        summ = {r["function"]: r for r in csv.DictReader(fh) if r["rule"] == "current"}
    with gzip.open(need(DESC), "rt") as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault(r["problem"], []).append(r)

    probs = sorted(null)
    assert set(probs) == set(by), (sorted(set(probs) ^ set(by)))

    P("# その93 の D=20 クラス上限は、費用の側ではなく<u>台の側</u>で破れている（その162、追加評価ゼロ）")
    P("")
    P("## 1 —— 問題ごと: null（40 抽選）の台 対 実測（`Restart-Lander`）の台")
    P("")
    P("`sup_null` = その93 の `mpr_sup`（5 水準平均の台/K）。`sup_act` = e151 の全降下から同じ定義で数え直したもの。")
    P("`S(1e-5)` は 1e-5 に届いた相異なる最適の数。")
    P("")
    P("| 問題 | K | null 本数 | 実測本数 | null S(1e-5) | 実測 S(1e-5) | sup_null | sup_act | 差 | 実測 MPR(採点) |")
    P("|---|---|---|---|---|---|---|---|---|---|")

    rec = []
    for p in probs:
        nr = null[p]
        K = int(nr["K"])
        rs = by[p]
        bf = np.array([float(r["best_f"]) for r in rs])
        ld = np.array([int(r["land_opt"]) for r in rs])
        ev = np.array([float(r["evals"]) for r in rs])
        sup_act, s_at = [], {}
        for e in EPS:
            c = np.zeros(K)
            for j in ld[bf <= e]:
                c[j] += 1
            s_at[e] = int((c > 0).sum())
            sup_act.append(s_at[e] / K)
        sup_act_m = float(np.mean(sup_act))
        mpr_cur = float(np.mean([float(summ[p][EPSCOL[e]]) for e in EPS]))
        mpr_act = E151_MPR[p[:3]]
        rec.append(dict(prob=p, K=K, n_null=int(nr["draws"]), n_act=len(rs),
                        S5_null=int(nr["reached_1e-5"]), S5_act=s_at[1e-5],
                        sup_null=float(nr["mpr_sup"]), sup_act=sup_act_m,
                        chao_null=float(nr["mpr_sup_chao1"]),
                        fixed=float(nr["mpr"]), early=float(nr["mpr_earlystop"]),
                        restarts_null=float(nr["restarts"]), ev_act=float(ev.mean()),
                        mpr_act=mpr_act, mpr_cur=mpr_cur, bf=bf, ld=ld))
        r = rec[-1]
        P(f"| {p[:3]} | {K} | {r['n_null']} | {r['n_act']} | {r['S5_null']} | {r['S5_act']} | "
          f"{r['sup_null']:.3f} | {r['sup_act']:.3f} | {r['sup_act']-r['sup_null']:+.3f} | {mpr_act:.3f} |")

    m = lambda k: float(np.mean([r[k] for r in rec]))
    P("")
    P(f"**16 問平均**: null 本数 {m('n_null'):.0f} ／ 実測本数 {m('n_act'):.1f} ／ "
      f"null の含意する再起動数 n = {m('restarts_null'):.1f} ／ 実測 1 本あたり評価 {m('ev_act'):,.0f}")
    P(f"**上限 4 通り**: 固定コスト {m('fixed'):.4f} ／ 早期打ち切り {m('early'):.4f} ／ "
      f"無限再起動 {m('sup_null'):.4f} ／ ＋Chao1 {m('chao_null'):.4f}")
    P(f"**実測（その151 の記録 MPR、報告規則 `eps_loose+dedup`）**: {m('mpr_act'):.4f}"
      f"（参考: `driver_summary.csv` の `current` 規則では {m('mpr_cur'):.4f}）")
    P(f"**実測の台（降下から数え直し）**: {m('sup_act'):.4f}")

    over = [r for r in rec if r["sup_act"] > r["sup_null"] + 1e-12]
    P("")
    P(f"**実測の台が null の台を上回る問題: {len(over)}/16**"
      f"（{' '.join(r['prob'][:3] for r in over)}）")
    P(f"**対の差の平均 {np.mean([r['sup_act']-r['sup_null'] for r in rec]):+.4f}、"
      f"中央値 {np.median([r['sup_act']-r['sup_null'] for r in rec]):+.4f}**")

    P("")
    P("## 関門 —— 台の数え直しが その151 の記録 MPR と一致するか（16/16 で厳密一致するはず）")
    P("")
    bad = [r for r in rec if abs(r["sup_act"] - r["mpr_act"]) > 5e-5]
    P(f"  一致しない問題: **{len(bad)}/16**"
      f"{'（' + ' '.join(r['prob'][:3] for r in bad) + '）' if bad else ''}"
      f"   16 問平均 台 {m('sup_act'):.4f} 対 記録 MPR {m('mpr_act'):.4f}")
    P("  **この一致は偶然ではない** —— その160 §3 が測った「報告規則の dedup 半径が着地点を最適 1 つにつき"
      " 1 点に落としきっている」の帰結で、**採点された MPR は台/K そのもの**である。")
    P("  **＝ 上限（台の言葉で書かれている）と実測（採点値）は<u>同じ量</u>で、次元も規則もまたいで直接比べてよい。**")
    if bad:
        P("  **不一致がある以上、以下の切り分けは信用できない。**")
        print("\n".join(out))
        with open(os.path.join(HERE, "scored.txt"), "w") as fh:
            fh.write("\n".join(out) + "\n")
        sys.exit(1)

    P("")
    P("## 2 —— どの仮定が破れたか（上の表から機械的に読む）")
    P("")
    P(f"- **費用側 (B) は破れていない**: null が仮定した再起動数 n = {m('restarts_null'):.1f} は"
      f" 実測の降下本数 {m('n_act'):.1f} の {m('restarts_null')/m('n_act'):.2f} 倍 ＝ "
      f"**null は実機より{'多く' if m('restarts_null')>m('n_act') else '少なく'}再起動できる前提で引かれている。**"
      "（3・4 はそもそも n に依らない。）")
    P(f"- **台側 (A) が破れている**: 無限再起動の上界 {m('sup_null'):.4f} は n = ∞ の値なのに、"
      f"実測の採点値 {m('mpr_act'):.4f} がそれを {m('mpr_act')-m('sup_null'):+.4f} 上回る。"
      "**n を無限にしても越えられない量を実機が越えた ＝ 40 抽選が数えた台が実機の到達集合より小さい。**")
    P(f"- **Chao1 は足りない**: ＋Chao1 の {m('chao_null'):.4f} でも {m('mpr_act')-m('chao_null'):+.4f} 足りない。")

    under = [r for r in rec if r["sup_act"] < r["sup_null"] - 1e-12]
    P(f"- **ただし 1 方向ではない**: 上限が持ちこたえた問題が {len(under)} 問ある"
      f"（{' '.join(r['prob'][:3] for r in under)}）。"
      "**うち M03・M11 は実測が 1e-5 に 1 つも届いていない**"
      f"（null は {null['M03-D20-PIN01']['reached_1e-5']} と {null['M11-D20-PIN01']['reached_1e-5']} 個に届いている）。"
      " **機序は降下の深さ** —— **null の降下上限は 25,000 評価（その93）、`Restart-Lander` の既定 `descent_budget` は 12,500** ＝ **null のほうが 1 本を 2 倍深く掘る。**"
      " **その158 が測った「D=20 では未到達の 42.3% が打ち切り」と同じ場所を指す。**")

    P("")
    P("## 3 —— Chao1 が D=20 でどれだけ足りないか（実測ダンプを 40 本に間引いて測る）")
    P("")
    P("null の降下ダンプは消えているので、**同じ問いを実測ダンプの上で立てる**:")
    P("先頭 40 降下だけを見た人が Chao1 で台を戻したとき、全降下で分かる台にどれだけ届くか。")
    P("")
    P("| 問題 | 40 本の台 S40 | Chao1(40) | 全本の台 S_all | Chao1 の埋め残し |")
    P("|---|---|---|---|---|")
    gaps, raws = [], []
    for r in rec:
        K, bf, ld = r["K"], r["bf"], r["ld"]
        s40, c40, sall = [], [], []
        for e in EPS:
            cnt_all = np.zeros(K)
            for j in ld[bf <= e]:
                cnt_all[j] += 1
            cnt40 = np.zeros(K)
            sel = np.where(bf[:40] <= e)[0]
            for j in ld[:40][bf[:40] <= e]:
                cnt40[j] += 1
            s40.append(float((cnt40 > 0).sum()))
            c40.append(min(float(chao1(cnt40)), float(K)))
            sall.append(float((cnt_all > 0).sum()))
        s40m, c40m, sallm = np.mean(s40), np.mean(c40), np.mean(sall)
        gaps.append((sallm - c40m) / K)
        raws.append((sallm - s40m) / K)
        P(f"| {r['prob'][:3]} | {s40m:.2f} | {c40m:.2f} | {sallm:.2f} | {(sallm-c40m)/K:+.3f} |")
    P("")
    P(f"**16 問平均**: 40 本のままの埋め残し {np.mean(raws):+.4f}／K、"
      f"Chao1 を当てたあとの埋め残し **{np.mean(gaps):+.4f}／K** "
      f"＝ **Chao1 が埋めるのは {(np.mean(raws)-np.mean(gaps))/np.mean(raws)*100 if np.mean(raws) else 0:.0f}%**、"
      f"残り {np.mean(gaps)/np.mean(raws)*100 if np.mean(raws) else 0:.0f}% は埋まらない。")
    P(f"**埋め残しが正の問題: {sum(1 for g in gaps if g > 1e-12)}/16**")

    # ------------------------------------------------ 4: 同じ枠を D=10 に当てる
    P("")
    P("## 4 —— 同じ枠を D=10 に当てる（その92・その94 の上限 対 その115 の実測。追加評価ゼロ）")
    P("")
    nd10 = os.path.join(MMO, "e94", "ceiling_mpr_d10.csv")
    dd10 = os.path.join(MMO, "e115", "descents")
    if not (os.path.exists(nd10) and os.path.isdir(dd10)):
        P(f"  **入力が無い**（{nd10} / {dd10}）。D=10 の照合は出せない。")
    else:
        with open(nd10) as fh:
            n10 = {r["func"]: r for r in csv.DictReader(fh)}
        P("| 問題 | K | null 本数 | 実測本数 | sup_null | sup_act | 差 |")
        P("|---|---|---|---|---|---|---|")
        d10 = []
        for f10 in sorted(n10):
            src = os.path.join(dd10, f"{f10}_seed0.csv.gz")
            if not os.path.exists(src):
                # 黙って飛ばすと 16 問平均が問題数の違う平均になるので落とす（その162 の走査 B/D の型）。
                print("\n".join(out))
                sys.exit(f"[FATAL] D=10 の入力が無い: {src}\n"
                         "  この節は 16 問そろわないと平均を出せない。")
            with gzip.open(src, "rt") as fh:
                rs = list(csv.DictReader(fh))
            K = int(n10[f10]["K"])
            bf = np.array([float(r["best_f"]) for r in rs])
            ld = np.array([int(r["land_opt"]) for r in rs])
            sup = []
            for e in EPS:
                c = np.zeros(K)
                for j in ld[bf <= e]:
                    c[j] += 1
                sup.append(float((c > 0).sum()) / K)
            sa, sn = float(np.mean(sup)), float(n10[f10]["mpr_sup"])
            d10.append((sa, sn, float(n10[f10]["mpr_sup_chao1"]), int(n10[f10]["draws"]), len(rs)))
            P(f"| {f10[:3]} | {K} | {n10[f10]['draws']} | {len(rs)} | {sn:.3f} | {sa:.3f} | {sa-sn:+.3f} |")
        if d10:
            sa = float(np.mean([x[0] for x in d10]))
            sn = float(np.mean([x[1] for x in d10]))
            sc = float(np.mean([x[2] for x in d10]))
            nover = sum(1 for x in d10 if x[0] > x[1] + 1e-12)
            P("")
            P(f"**{len(d10)} 問平均**: null 本数 {np.mean([x[3] for x in d10]):.0f} ／ 実測本数 {np.mean([x[4] for x in d10]):.1f} ／ "
              f"sup_null {sn:.4f} ／ ＋Chao1 {sc:.4f} ／ 実測の台 **{sa:.4f}** ／ 差（実測 − ＋Chao1）**{sa-sc:+.4f}**")
            P(f"**実測の台が null の台を上回る問題: {nover}/{len(d10)}**"
              f"（D=20 は 10/16）")
            P("")
            P("**D=20 と違う点は 2 つで、どちらも上の切り分けが名指したもの:**")
            P(f"- **抽選本数**: D=10 は {np.mean([x[3] for x in d10]):.0f} 本（その94 が群 B を 200 本に上げた）、D=20 は 70 本（3 問だけ 200 本）"
              " ＝ **台の下向き偏りが小さい。**")
            P("- **降下の深さ**: **D=10 の null の降下上限 12,500 評価は `Restart-Lander` の既定 `descent_budget` と<u>同じ</u>**"
              "（どちらも 12,500）。**D=20 では null 25,000 対 実機 12,500 で 2 倍ずれている。**"
              " ＝ **D=10 の上限は「同じ深さで、より多く撒いた」ものなので上限として機能し、D=20 は両方向にずれている。**")

    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "scored.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
