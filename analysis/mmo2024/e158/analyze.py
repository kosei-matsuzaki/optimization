#!/usr/bin/env python3
"""その158 — 降下 1 本あたりの当たり率を「初出 / 重複 / 未到達」に分解する（キュー 1）。

保存済みの降下ダンプだけを読む。**探索は 1 ビットも変えない。追加評価ゼロ。**
事前登録は `prereg.md`。しきい値は `e115/analyze.py` の採点器のものを import する
（距離の固定値は置かない ＝ その76 の ρ 誤設定と同じ罠を避ける）。

使い方: python3 analysis/mmo2024/e158/analyze.py
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)


def _load_e115():
    """採点器（LEVELS / EPS_* / score）を e115 から読む。main() は走らせない。"""
    path = os.path.join(MMO, "e115", "analyze.py")
    if not os.path.isfile(path):
        raise SystemExit(f"採点器が無い: {path}\n  -> しきい値を自前で置かない規則なので続行しない。")
    spec = importlib.util.spec_from_file_location("e115_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E115 = _load_e115()
LEVELS, LEVEL_NAMES = E115.LEVELS, E115.LEVEL_NAMES
EPS_TIGHT, EPS_LOOSE = E115.EPS_TIGHT, E115.EPS_LOOSE
PROBS = [f"M{i:02d}" for i in range(1, 17)]


# ------------------------------------------------------------------ 読み込み
def _need(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"入力が無い: {path}\n"
            "  -> その150 §3 の経路（欠けた入力を黙って飛ばして nan の表を刷る）は踏まない。")
    return path


def _rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(_need(path), "rt") as fh:
        return list(csv.DictReader(fh))


def load_runs():
    """1 run = (D, PIN, problem) 単位で降下列を返す。`descent` 昇順 ＝ 発生順。"""
    runs = defaultdict(list)          # (D, pin, prob) -> [row, ...]
    K = {}                            # (D, pin, prob) -> K

    # D=5 PIN01（e153）
    for r in _rows(os.path.join(MMO, "e153", "descents.csv.gz")):
        p, d, pin = r["problem"].split("-")          # M01-D05-PIN01
        runs[(int(d[1:]), pin, p)].append(r)
    for r in _rows(os.path.join(MMO, "e153", "by_problem_d5.csv")):
        K[(5, "PIN01", r["problem"])] = int(r["K"])

    # D=5 PIN02/03/04（e157）
    for r in _rows(os.path.join(MMO, "e157", "descents.csv.gz")):
        p, d, pin = r["problem"].split("-")
        runs[(int(d[1:]), pin, p)].append(r)
    for r in _rows(os.path.join(MMO, "e157", "by_problem_rl.csv")):
        p, d, pin = r["function"].split("-")
        K[(int(d[1:]), pin, p)] = int(r["n_optima"])

    # D=10 PIN01（e115、per-problem）
    dd = os.path.join(MMO, "e115", "descents")
    if not os.path.isdir(dd):
        raise SystemExit(f"入力が無い: {dd}")
    for fn in sorted(os.listdir(dd)):
        if not fn.endswith(".csv.gz"):
            continue
        p, d, pin = fn.split("_")[0].split("-")
        runs[(int(d[1:]), pin, p)].extend(_rows(os.path.join(dd, fn)))
    bp = os.path.join(MMO, "e115", "by_problem")
    for fn in sorted(os.listdir(bp)):
        if not fn.endswith(".csv"):
            continue
        p, d, pin = fn[:-4].split("-")
        K[(int(d[1:]), pin, p)] = int(_rows(os.path.join(bp, fn))[0]["n_optima"])

    # D=20 PIN01（e151）
    for r in _rows(os.path.join(MMO, "e151", "descents.csv.gz")):
        p, d, pin = r["problem"].split("-")
        runs[(int(d[1:]), pin, p)].append(r)
    for r in _rows(os.path.join(MMO, "e151", "driver_summary.csv")):
        p, d, pin = r["function"].split("-")
        K[(int(d[1:]), pin, p)] = int(r["n_optima"])

    out = {}
    for key, rs in runs.items():
        if key not in K:
            raise SystemExit(f"K が引けない run: {key}")
        rs = sorted(rs, key=lambda r: int(r["descent"]))
        out[key] = dict(K=K[key],
                        f=np.array([float(r["best_f"]) for r in rs]),
                        opt=np.array([int(r["land_opt"]) for r in rs]),
                        stop=[r["stop"] for r in rs])
    return out


# ------------------------------------------------------------------ 3 分類
def classify(f, opt, eps):
    """発生順に (i) 初出ヒット / (ii) 重複ヒット / (iii) 未到達 を数える。"""
    seen, new, dup, miss = set(), 0, 0, 0
    miss_idx = []
    for i, (v, o) in enumerate(zip(f, opt)):
        if v <= eps:
            if o in seen:
                dup += 1
            else:
                seen.add(o)
                new += 1
        else:
            miss += 1
            miss_idx.append(i)
    return new, dup, miss, len(seen), miss_idx


def coupon_null(H, K):
    """ヒット H 本が K 個の最適に一様独立に落ちるときの期待相異なり数。"""
    if H == 0 or K == 0:
        return 0.0
    return K * (1.0 - (1.0 - 1.0 / K) ** H)


# ------------------------------------------------------------------ 検定
def paired(a, b):
    """b - a の対検定（両側 Wilcoxon exact）と rank-biserial。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    nz = d[d != 0]
    if len(nz) == 0:
        return float(d.mean()), 1.0, 0.0, (0, len(d), 0)
    try:
        p = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                                 mode="exact").pvalue)
    except TypeError:
        p = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                                 method="exact").pvalue)
    r = stats.rankdata(np.abs(nz))
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return float(d.mean()), p, float(rb), (int((d > 0).sum()), int((d == 0).sum()),
                                           int((d < 0).sum()))


def main():
    runs = load_runs()
    cells = sorted({(k[0], k[1]) for k in runs})
    print(f"読んだ run: {len(runs)} 本 / セル {len(cells)}: "
          + ", ".join(f"D{d}-{p}" for d, p in cells))
    print(f"採点器の水準（e115 から import）: {LEVEL_NAMES}  主水準 ϵ={EPS_TIGHT:g} / 感度 ϵ={EPS_LOOSE:g}\n")

    # ---- 問題 × セル × 水準の 3 分類
    rec = {}          # (eps_name, D, pin, prob) -> dict
    for (D, pin, prob), r in runs.items():
        for en, eps in zip(LEVEL_NAMES, LEVELS):
            new, dup, miss, distinct, midx = classify(r["f"], r["opt"], eps)
            n = new + dup + miss
            exp = coupon_null(new + dup, r["K"])
            rec[(en, D, pin, prob)] = dict(
                n=n, new=new, dup=dup, miss=miss, K=r["K"], distinct=distinct,
                p_new=new / n, p_dup=dup / n, p_miss=miss / n,
                hit=(new + dup) / n,
                dup_of_hit=(dup / (new + dup)) if (new + dup) else float("nan"),
                conc=(distinct / exp) if exp > 0 else float("nan"),
                cover=distinct / r["K"],
                miss_stop=Counter(r["stop"][i] for i in midx))

    out_csv = os.path.join(HERE, "by_problem.csv.gz")   # 481 行 > 300 なので gz（CLAUDE.md の規約）
    with gzip.open(out_csv, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["eps", "D", "pin", "problem", "K", "n_descent", "new", "dup", "miss",
                    "p_new", "p_dup", "p_miss", "dup_of_hit", "distinct", "conc", "cover"])
        for key in sorted(rec, key=lambda k: (LEVEL_NAMES.index(k[0]), k[1], k[2], k[3])):
            e = rec[key]
            w.writerow([key[0], key[1], key[2], key[3], e["K"], e["n"], e["new"], e["dup"],
                        e["miss"], f"{e['p_new']:.4f}", f"{e['p_dup']:.4f}",
                        f"{e['p_miss']:.4f}",
                        "" if np.isnan(e["dup_of_hit"]) else f"{e['dup_of_hit']:.4f}",
                        e["distinct"], "" if np.isnan(e["conc"]) else f"{e['conc']:.4f}",
                        f"{e['cover']:.4f}"])

    def col(en, D, pin, field):
        return [rec[(en, D, pin, p)][field] for p in PROBS]

    # ---- 主表: PIN01 の 3 次元
    for en in (LEVEL_NAMES[-1], LEVEL_NAMES[0]):        # 1e-5（主）, 1e-1（感度）
        tag = "主水準" if en == LEVEL_NAMES[-1] else "感度水準"
        print("=" * 78)
        print(f"■ {tag} ϵ={en} — PIN01 の 3 次元（16 問平均、括弧は問題ごとの SD）")
        print(f"{'D':>4} {'降下':>7} {'初出(i)':>16} {'重複(ii)':>16} {'未到達(iii)':>16} "
              f"{'重複/ヒット':>11} {'集中度':>8} {'被覆':>7}")
        for D in (5, 10, 20):
            n = np.mean(col(en, D, "PIN01", "n"))
            vs = {k: np.array(col(en, D, "PIN01", k), float)
                  for k in ("p_new", "p_dup", "p_miss", "dup_of_hit", "conc", "cover")}
            print(f"{D:>4} {n:>7.1f} "
                  f"{vs['p_new'].mean():>8.4f}({vs['p_new'].std(ddof=1):.3f}) "
                  f"{vs['p_dup'].mean():>8.4f}({vs['p_dup'].std(ddof=1):.3f}) "
                  f"{vs['p_miss'].mean():>8.4f}({vs['p_miss'].std(ddof=1):.3f}) "
                  f"{np.nanmean(vs['dup_of_hit']):>11.4f} "
                  f"{np.nanmean(vs['conc']):>8.4f} {vs['cover'].mean():>7.4f}")

        print(f"\n  対検定（16 問対、両側 Wilcoxon exact、α=0.05）  ϵ={en}")
        print(f"  {'対':>10} {'量':>12} {'差':>10} {'w/t/l':>10} {'p':>12} {'rb':>8}")
        fired = []
        for (a, b) in ((5, 10), (10, 20), (5, 20)):
            for field, name in (("p_new", "初出(i)"), ("p_dup", "重複(ii)"),
                                ("p_miss", "未到達(iii)"), ("dup_of_hit", "重複/ヒット"),
                                ("conc", "集中度")):
                x, y = col(en, a, "PIN01", field), col(en, b, "PIN01", field)
                if np.isnan(x).any() or np.isnan(y).any():
                    ok = [i for i in range(16) if not (np.isnan(x[i]) or np.isnan(y[i]))]
                    if len(ok) < 6:
                        print(f"  {f'D{a}->D{b}':>10} {name:>12} "
                              f"{'(有効対 %d 本で検定せず)' % len(ok)}")
                        continue
                    x, y = [x[i] for i in ok], [y[i] for i in ok]
                d, p, rb, wtl = paired(x, y)
                mark = " *" if p < 0.05 else ""
                print(f"  {f'D{a}->D{b}':>10} {name:>12} {d:>+10.4f} "
                      f"{'%d/%d/%d' % wtl:>10} {p:>12.6g} {rb:>+8.3f}{mark}")
                if (a, b) == (5, 20) and field in ("p_new", "p_dup", "p_miss"):
                    fired.append((name, abs(d) >= 0.05, p < 0.05))
        print()
        if en == LEVEL_NAMES[-1]:
            big = [f"{n}(|Δ|≥0.05={m}, p<0.05={s})" for n, m, s in fired]
            refuted = all((not m) and (not s) for _, m, s in fired)
            print("  反証条件 (a)（3 分類とも |D5-D20 差| < 0.05 かつ 3 本とも非有意）: "
                  + ("**発火**" if refuted else "不発") + " — " + " / ".join(big))
            print()

    # ---- (b) 水準で支配が入れ替わるか
    print("=" * 78)
    print("■ 反証条件 (b) — 主水準と感度水準で「D とともに増える分類」が入れ替わるか")
    for en in (LEVEL_NAMES[-1], LEVEL_NAMES[0]):
        d5 = {k: np.mean(col(en, 5, "PIN01", k)) for k in ("p_new", "p_dup", "p_miss")}
        d20 = {k: np.mean(col(en, 20, "PIN01", k)) for k in ("p_new", "p_dup", "p_miss")}
        grow = max(("p_dup", "p_miss"), key=lambda k: d20[k] - d5[k])
        print(f"  ϵ={en:>4}: D5->D20 の増分 重複 {d20['p_dup']-d5['p_dup']:+.4f} / "
              f"未到達 {d20['p_miss']-d5['p_miss']:+.4f} -> 支配は "
              f"{'重複(ii)' if grow=='p_dup' else '未到達(iii)'}")
    print()

    # ---- 問題ごと（平均を出すだけでは潰れる、という俯瞰の指摘）
    print("=" * 78)
    print(f"■ 問題ごと（ϵ={LEVEL_NAMES[-1]}、PIN01）— 初出 / 重複 / 未到達 の割合と被覆")
    print(f"  {'prob':>5} {'K':>3} | " + " | ".join(
        f"{'D=%d  i/ii/iii  cov' % D:>26}" for D in (5, 10, 20)))
    for p in PROBS:
        cells_txt = []
        for D in (5, 10, 20):
            e = rec[(LEVEL_NAMES[-1], D, "PIN01", p)]
            cells_txt.append(f"{e['p_new']:.3f}/{e['p_dup']:.3f}/{e['p_miss']:.3f} "
                             f"{e['cover']:.2f} (n={e['n']:>3d})")
        print(f"  {p:>5} {rec[(LEVEL_NAMES[-1], 5, 'PIN01', p)]['K']:>3} | "
              + " | ".join(f"{c:>26}" for c in cells_txt))
    print()

    # ---- 体制の割れ（俯瞰が見た 2 体制）
    print("=" * 78)
    print(f"■ 体制の割れ（ϵ={LEVEL_NAMES[-1]}、PIN01）— ヒット率 0 の問題と 0.5 超の問題")
    for D in (5, 10, 20):
        hit = {p: rec[(LEVEL_NAMES[-1], D, "PIN01", p)]["hit"] for p in PROBS}
        zero = [p for p in PROBS if hit[p] == 0.0]
        high = [p for p in PROBS if hit[p] > 0.5]
        print(f"  D={D:>2}: ヒット率 0 の問題 {len(zero):>2} 本 {zero}")
        print(f"        ヒット率 >0.5 の問題 {len(high):>2} 本 {high}")
    print()

    # ---- 未到達の stop 内訳
    print("=" * 78)
    print(f"■ 未到達(iii) の打ち切り理由（ϵ={LEVEL_NAMES[-1]}、PIN01、16 問合算）")
    for D in (5, 10, 20):
        c = Counter()
        for p in PROBS:
            c.update(rec[(LEVEL_NAMES[-1], D, "PIN01", p)]["miss_stop"])
        tot = sum(c.values())
        print(f"  D={D:>2} 未到達 {tot:>5} 本: " + ", ".join(
            f"{k} {v} ({v/tot:.1%})" for k, v in c.most_common()) if tot else f"  D={D}: 0")
    print()

    # ---- 乗法分解: p_new = ヒット率 x (1 - 重複/ヒット)
    print("=" * 78)
    print(f"■ 当たり率の乗法分解（ϵ={LEVEL_NAMES[-1]}、PIN01）— p_new = ヒット率 x 相異なり率")
    print("  相異なり率 = 1 - 重複/ヒット（ヒットした 1 本が新しい最適である確率）")
    print(f"  {'D':>4} {'p_new':>9} {'ヒット率':>9} {'相異なり率':>11}")
    en = LEVEL_NAMES[-1]
    for D in (5, 10, 20):
        pn = np.mean(col(en, D, "PIN01", "p_new"))
        hr = np.mean(col(en, D, "PIN01", "hit"))
        dr = np.nanmean(1 - np.array(col(en, D, "PIN01", "dup_of_hit"), float))
        print(f"  {D:>4} {pn:>9.4f} {hr:>9.4f} {dr:>11.4f}")
    # 問題ごとの log 分解（両方が正の問題だけ）
    print("\n  問題ごとの log 分解（D=5 -> D=20、ヒット率も相異なり率も両次元で正の問題のみ）")
    rows = []
    for p_ in PROBS:
        a, b = rec[(en, 5, "PIN01", p_)], rec[(en, 20, "PIN01", p_)]
        da = 1 - a["dup_of_hit"] if a["hit"] > 0 else 0.0
        db = 1 - b["dup_of_hit"] if b["hit"] > 0 else 0.0
        if a["hit"] > 0 and b["hit"] > 0 and da > 0 and db > 0:
            rows.append((p_, np.log(a["hit"] / b["hit"]), np.log(da / db)))
    lh = np.array([r[1] for r in rows]); ld = np.array([r[2] for r in rows])
    tot = lh + ld
    print(f"  有効 {len(rows)} 問 / 16（除外: ヒットか相異なりが 0 の問題）")
    print(f"  log 低下の合計 {tot.mean():+.4f} = ヒット率 {lh.mean():+.4f} "
          f"({lh.mean()/tot.mean():.1%}) + 相異なり率 {ld.mean():+.4f} ({ld.mean()/tot.mean():.1%})")
    d_, p_v, rb_, wtl_ = paired(ld, lh)
    print(f"  どちらが大きいか（問題対検定、ヒット率 - 相異なり率）: {d_:+.4f} "
          f"w/t/l {wtl_[0]}/{wtl_[1]}/{wtl_[2]}  p={p_v:.6g}  rb={rb_:+.3f}"
          f"{'  *' if p_v < 0.05 else ''}")
    print()

    # ---- 打ち切り（maxfevals）の比重
    print("=" * 78)
    print("■ 降下の打ち切り（`maxfevals` ＝ descent_budget 12500 を使い切った降下）の比重")
    print(f"  {'D':>4} {'全降下':>7} {'maxfevals':>10} {'全体比':>8} {'未到達に占める比':>16}")
    for D in (5, 10, 20):
        allstop = Counter()
        for p_ in PROBS:
            allstop.update(runs[(D, "PIN01", p_)]["stop"])
        n_all = sum(allstop.values())
        mf = sum(v for k, v in allstop.items() if "maxfevals" in k)
        miss = Counter()
        for p_ in PROBS:
            miss.update(rec[(en, D, "PIN01", p_)]["miss_stop"])
        n_miss = sum(miss.values())
        mfm = sum(v for k, v in miss.items() if "maxfevals" in k)
        print(f"  {D:>4} {n_all:>7} {mf:>10} {mf/n_all:>8.1%} {mfm/n_miss:>16.1%}")
    print()

    # ---- D=5 の 4 instance（頑健性）
    print("=" * 78)
    print(f"■ D=5 の 4 instance（ϵ={LEVEL_NAMES[-1]}、16 問平均）— PIN01 は他の 3 本と同じ体制か")
    print(f"  {'PIN':>6} {'降下':>7} {'初出':>8} {'重複':>8} {'未到達':>8} {'集中度':>8} {'被覆':>7}")
    for pin in ("PIN01", "PIN02", "PIN03", "PIN04"):
        n = np.mean(col(LEVEL_NAMES[-1], 5, pin, "n"))
        vs = {k: np.array(col(LEVEL_NAMES[-1], 5, pin, k), float)
              for k in ("p_new", "p_dup", "p_miss", "conc", "cover")}
        print(f"  {pin:>6} {n:>7.1f} {vs['p_new'].mean():>8.4f} {vs['p_dup'].mean():>8.4f} "
              f"{vs['p_miss'].mean():>8.4f} {np.nanmean(vs['conc']):>8.4f} "
              f"{vs['cover'].mean():>7.4f}")
    print()

    print(f"  -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
