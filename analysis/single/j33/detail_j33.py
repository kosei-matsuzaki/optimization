#!/usr/bin/env python3
"""j33: per-function SR@1e-10 deltas, evals_succ_mean pairs, Wilcoxon counts."""
import csv, sys, math
from pathlib import Path

REF = "MC-ESO"
ARMS = ["v2_ir3_noFrz", "v2_rsT2s"]
SR = ["sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10"]

def pct(s):
    s = (s or "").strip().rstrip("%")
    try: return float(s)
    except ValueError: return float("nan")

def num(s):
    try: v = float(s)
    except (ValueError, TypeError): return float("nan")
    return v if math.isfinite(v) else float("nan")

def load(p):
    rows = list(csv.DictReader(open(p)))
    d = {}
    for r in rows:
        d[(r["function"], r["method"])] = r
    funcs = []
    for r in rows:
        if r["function"] not in funcs: funcs.append(r["function"])
    return d, funcs, rows

def report(run_dir, dim, out):
    S, funcs, srows = load(Path(run_dir) / f"dim{dim}" / "summary.csv")
    W = list(csv.DictReader(open(Path(run_dir) / f"dim{dim}" / "wilcoxon.csv")))
    methods = [REF] + ARMS
    w = out.write
    w(f"===== dim{dim}  run={Path(run_dir).name} =====\n")
    w(f"functions={len(funcs)}  summary_rows={len(srows)}  wilcoxon_rows={len(W)}\n")
    # integrity
    missing = [(f, m) for f in funcs for m in methods if (f, m) not in S]
    dup = len(srows) - len(S)
    w(f"missing cells: {missing if missing else 'none'}   duplicate rows: {dup}\n\n")

    # [1] overall
    w("[1] OVERALL (mean over %d functions), SR%% and evals\n" % len(funcs))
    w("  method            " + "  ".join(f"{k:>9}" for k in SR) + "   own_evals(nf)   pair_evals ref->arm (nf)\n")
    for m in methods:
        means = [sum(pct(S[(f, m)][k]) for f in funcs) / len(funcs) for k in SR]
        own = [num(S[(f, m)]["evals_succ_mean"]) for f in funcs]
        own = [v for v in own if not math.isnan(v)]
        line = f"  {m:<16}" + "  ".join(f"{v:9.2f}" for v in means)
        line += f"   {sum(own)/len(own):8.1f}({len(own)})"
        if m != REF:
            pair = [(num(S[(f, REF)]["evals_succ_mean"]), num(S[(f, m)]["evals_succ_mean"])) for f in funcs]
            pair = [(a, b) for a, b in pair if not math.isnan(a) and not math.isnan(b)]
            if pair:
                line += f"   {sum(a for a,_ in pair)/len(pair):8.1f} -> {sum(b for _,b in pair)/len(pair):8.1f} ({len(pair)})"
        w(line + "\n")
    w("\n")

    # [2] per-function SR@1e-10 deltas vs REF
    for m in ARMS:
        w(f"[2] per-function SR@1e-10 change  {REF} -> {m}  (only functions that moved)\n")
        n = 0
        for f in funcs:
            a, b = pct(S[(f, REF)]["sr_1e-10"]), pct(S[(f, m)]["sr_1e-10"])
            if a != b:
                n += 1
                w(f"    {f:<24} {a:5.0f} -> {b:5.0f}  ({b-a:+.0f}pt)\n")
        w(f"    moved: {n} / {len(funcs)}\n\n")

    # [2b] functions where all three are 0
    zero = [f for f in funcs if all(pct(S[(f, m)]["sr_1e-10"]) == 0 for m in methods)]
    w(f"[2b] SR@1e-10 == 0 for all three arms: {len(zero)} -> {', '.join(zero) if zero else 'none'}\n\n")

    # [3] named functions
    named = ["F03-RastriginSep", "F04-BucheRastrigin", "F15-RastriginRot",
             "F21-Gallagher101", "F22-Gallagher21"]
    w("[3] named functions: SR@1e-10 %% and evals_succ_mean, order = " + " / ".join(methods) + "\n")
    for f in named:
        if (f, REF) not in S:
            w(f"    {f:<22} (absent)\n"); continue
        srs = " / ".join(f"{pct(S[(f,m)]['sr_1e-10']):.0f}" for m in methods)
        ev = []
        for m in methods:
            v = num(S[(f, m)]["evals_succ_mean"])
            ev.append("-" if math.isnan(v) else f"{v:.0f}")
        w(f"    {f:<22} SR {srs:>16} %   evals {' / '.join(ev)}\n")
    w("\n")

    # [4] Wilcoxon
    w("[4] WILCOXON (ref=%s, alpha=0.05 two-sided, A12 direction; p=nan treated as n.s.)\n" % REF)
    for m in ARMS:
        wins, losses = [], []
        for r in W:
            if r["method"] != m: continue
            p = num(r["p_value_two_sided"]); a12 = num(r["a12"])
            if math.isnan(p) or p >= 0.05: continue
            tag = f"{r['function']} A12={a12:.2f} {r['a12_magnitude']}"
            (wins if a12 > 0.5 else losses).append(tag)
        w(f"    {m}: ref-better {len(wins)} / arm-better {len(losses)}\n")
        w(f"      ref better : {'; '.join(wins) if wins else 'none'}\n")
        w(f"      arm better : {'; '.join(losses) if losses else 'none'}\n")
    w("\n")

    # [5] full-cell equality check vs REF (all result columns except timing)
    skip = {"function", "category", "tags", "method", "mean_time_s"}
    cols = [c for c in srows[0].keys() if c not in skip]
    for m in ARMS:
        diffs = []
        for f in funcs:
            for c in cols:
                if S[(f, REF)][c] != S[(f, m)][c]:
                    diffs.append((f, c, S[(f, REF)][c], S[(f, m)][c]))
        w(f"[5] cell-identity {REF} vs {m}: {len(funcs)}x{len(cols)} = {len(funcs)*len(cols)} cells, "
          f"{len(diffs)} differ\n")
        for d in diffs[:12]:
            w(f"      {d[0]:<24} {d[1]:<18} {d[2]} vs {d[3]}\n")
        if len(diffs) > 12: w(f"      ... {len(diffs)-12} more\n")
    w("\n")

if __name__ == "__main__":
    run_dir, dim, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(outp, "w") as fh:
        report(run_dir, dim, fh)
    print(open(outp).read())
