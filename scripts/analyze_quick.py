#!/usr/bin/env python3
"""Aggregate a quick-check run into the project's 3-metric evaluation report.

Reads ``<run_dir>/dim<dim>/{summary.csv,wilcoxon.csv}`` and prints a compact,
rule-compliant report so an evaluator does not have to hand-parse the ~280-row
summary.csv. Mirrors the rules in docs/experiments.md (評価方法論):

  - 3 metrics always reported together: SR (multi-level) / evals_succ_mean /
    Wilcoxon. Single-metric judgement is not allowed.
  - Primary metric is SR@1e-10; SR@1e-2/1e-4/1e-7 are auxiliary.
  - Wilcoxon: MC-ESO is the reference. ``p_value_ref_better`` is the one-sided
    p that MC-ESO is better; ``a12`` > 0.5 means MC-ESO is better.
  - Every function is aggregated and per-function regressions are listed.
  - Hard rule: a config that lowers overall SR@1e-10 is rejected. With
    --baseline, MC-ESO SR@1e-10 is diffed function-by-function vs the prior run.

Usage:
    python scripts/analyze_quick.py results/<...>_quick
    python scripts/analyze_quick.py results/<new>_quick --baseline results/<prev>_quick
    python scripts/analyze_quick.py results/<...>_quick --dim 2 --ref MC-ESO
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SR_KEYS = ["sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10"]  # primary = last
PRIMARY = "sr_1e-10"


def _pct(s: str) -> float:
    """Parse a '100%' / '85%' cell to a 0..1 float; '' / bad -> nan."""
    s = (s or "").strip().rstrip("%")
    try:
        return float(s) / 100.0
    except ValueError:
        return float("nan")


def _num(s: str) -> float:
    s = (s or "").strip()
    if s in ("", "inf", "nan", "---"):
        return float("inf") if s == "inf" else float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _load_summary(run_dir: Path, dim: int) -> dict[str, dict[str, dict]]:
    """Return {function: {method: row}} for the given dimension."""
    path = run_dir / f"dim{dim}" / "summary.csv"
    if not path.exists():
        raise SystemExit(f"summary.csv not found: {path}")
    out: dict[str, dict[str, dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["function"], {})[row["method"]] = row
    return out


def _load_wilcoxon(run_dir: Path, dim: int,
                   name: str = "wilcoxon.csv") -> dict[tuple[str, str], dict]:
    path = run_dir / f"dim{dim}" / name
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[(row["function"], row["method"])] = row
    return out


def _evals(row: dict) -> float:
    """Successful-run mean evals to target (doc-preferred metric).

    Falls back to the median column for results produced before
    evals_succ_mean was written to summary.csv."""
    if row.get("evals_succ_mean") not in (None, ""):
        return _num(row["evals_succ_mean"])
    return _num(row.get("evals_succ_med", ""))


def _meta(run_dir: Path) -> dict:
    p = run_dir / "result.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="quick run dir (results/..._quick)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="prior run dir to diff the proposed method against "
                         "(cross-run, same method name as --ref)")
    ap.add_argument("--baseline-method", default=None,
                    help="another method IN THIS SAME run to diff --ref against "
                         "(within-run; e.g. the original MC-ESO registered under "
                         "a different name). Use this for two-method-only runs.")
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--ref", default="MC-ESO",
                    help="reference (proposed) method name (default MC-ESO)")
    args = ap.parse_args()

    ref = args.ref
    summ = _load_summary(args.run_dir, args.dim)
    wil = _load_wilcoxon(args.run_dir, args.dim)
    # Multi-solution significance lives in its own file: the best_f test says
    # nothing about how many optima a method reported.
    wil_pr = _load_wilcoxon(args.run_dir, args.dim, "wilcoxon_pr.csv")
    meta = _meta(args.run_dir)

    funcs = sorted(summ)
    methods: list[str] = []
    for fn in funcs:
        for m in summ[fn]:
            if m not in methods:
                methods.append(m)
    if ref not in methods:
        raise SystemExit(f"reference method {ref!r} not in run "
                         f"(methods: {methods})")

    print("=" * 78)
    print(f"QUICK EVALUATION REPORT   dim={args.dim}   ref={ref}")
    print(f"run: {args.run_dir.name}")
    if meta:
        print(f"meta: n_runs={meta.get('n_runs','?')} "
              f"max_evals={meta.get('max_evals','?')} "
              f"set={meta.get('set','?')} commit={meta.get('commit','?')}")
    print(f"functions: {len(funcs)}   methods: {', '.join(methods)}")
    print("=" * 78)

    # ── 1. Overall 3-metric standings (mean over all functions) ──────────────
    print("\n[1] OVERALL (mean over all functions)")
    print(f"  {'method':<14} {'SR@1e-10':>9} {'SR@1e-7':>8} {'SR@1e-4':>8} "
          f"{'SR@1e-2':>8} {'evals_mean':>11} {'Wins/Loss*':>11}")
    print("  " + "-" * 74)
    # Wilcoxon win/loss for each method = funcs where MC-ESO sig better / worse.
    sig_better: dict[str, int] = {m: 0 for m in methods if m != ref}
    sig_worse: dict[str, int] = {m: 0 for m in methods if m != ref}
    for (fn, m), w in wil.items():
        p = _num(w.get("p_value_two_sided", ""))
        a12 = _num(w.get("a12", ""))
        if not math.isnan(p) and p < 0.05:
            if a12 > 0.5:
                sig_better[m] = sig_better.get(m, 0) + 1
            elif a12 < 0.5:
                sig_worse[m] = sig_worse.get(m, 0) + 1
    for m in methods:
        srs = {k: _mean([_pct(summ[fn][m][k]) for fn in funcs if m in summ[fn]])
               for k in SR_KEYS}
        ev = _mean([_evals(summ[fn][m]) for fn in funcs
                    if m in summ[fn] and not math.isinf(_evals(summ[fn][m]))])
        ev_str = f"{ev:>11.0f}" if not math.isnan(ev) else f"{'---':>11}"
        wl = "" if m == ref else f"{sig_better.get(m,0)}/{sig_worse.get(m,0)}"
        print(f"  {m:<14} {srs['sr_1e-10']:>8.1%} {srs['sr_1e-7']:>7.1%} "
              f"{srs['sr_1e-4']:>7.1%} {srs['sr_1e-2']:>7.1%} {ev_str} {wl:>11}")
    print(f"  * Wins/Loss = #functions where {ref} is Wilcoxon-significant "
          f"better/worse (p<0.05, two-sided, by A12 direction).")

    # ── 1b. Niching suite: peak ratio over the reported solution set ─────────
    # Only fires for --suite niching runs (every other suite writes N/A here).
    nfuncs = [fn for fn in funcs
              if any(_num(r.get("cec_k", "")) > 0 for r in summ[fn].values())]
    if nfuncs:
        print("\n[1b] NICHING - PR over the reported solution set (CEC2013 rules)")
        # Wilcoxon on per-run peak counts (reference vs each method).
        pr_better: dict[str, int] = {}
        pr_worse: dict[str, int] = {}
        for (fn, m), w in wil_pr.items():
            if fn not in nfuncs:
                continue
            p = _num(w.get("p_value_two_sided", ""))
            a12 = _num(w.get("a12", ""))
            if not math.isnan(p) and p < 0.05:
                if a12 > 0.5:
                    pr_better[m] = pr_better.get(m, 0) + 1
                elif a12 < 0.5:
                    pr_worse[m] = pr_worse.get(m, 0) + 1
        print(f"  {'method':<14} {'PRmean':>8} {'PR@1e-2':>8} {'PR@1e-4':>8} "
              f"{'SRall':>7} {'#reported':>10} {'Wins/Loss*':>11}")
        print("  " + "-" * 72)
        for m in methods:
            rows = [summ[fn][m] for fn in nfuncs if m in summ[fn]]
            if not rows:
                continue
            wl = "" if m == ref else f"{pr_better.get(m, 0)}/{pr_worse.get(m, 0)}"
            print(f"  {m:<14} "
                  f"{_mean([_num(r['cec_pr_mean']) for r in rows]):>8.2f} "
                  f"{_mean([_num(r['cec_pr_1e-2']) for r in rows]):>8.2f} "
                  f"{_mean([_num(r['cec_pr_1e-4']) for r in rows]):>8.2f} "
                  f"{_mean([_pct(r['cec_sr_mean']) for r in rows]):>6.0%} "
                  f"{_mean([_num(r['n_reported']) for r in rows]):>10.0f} "
                  f"{wl:>11}")
        if not wil_pr:
            print(f"  * no wilcoxon_pr.csv in this run (peak-count test added "
                  f"2026-08-30; older runs only have the best_f test)")
        else:
            print(f"  * Wins/Loss = #functions where {ref} reports significantly "
                  f"more/fewer peaks (paired Wilcoxon on per-run peak counts, p<0.05)")
        print("\n  per function (PRmean; K = number of global optima)")
        print("  " + f"{'function':<26}" + "".join(f"{m[:11]:>12}" for m in methods))
        for fn in nfuncs:
            k = int(max(_num(r.get("cec_k", "0")) for r in summ[fn].values()))
            cells = "".join(
                f"{_num(summ[fn][m]['cec_pr_mean']):>12.2f}" if m in summ[fn]
                else f"{'---':>12}" for m in methods)
            print(f"  {fn + f' (K={k})':<26}{cells}")

    # ── 2. SR@1e-10 non-regression check (the hard rule) ────────────────────
    # Two modes:
    #   --baseline <dir>       : cross-run, same method name (new code vs old run)
    #   --baseline-method NAME : within-run, ref vs another method in THIS run
    #                            (e.g. two-method-only run: new vs original MC-ESO)
    base_summ: dict[str, dict[str, dict]] | None = None
    base_method = args.baseline_method or ref
    if args.baseline:
        base_summ = _load_summary(args.baseline, args.dim)
        src_label = f"baseline run: {args.baseline.name} (method {base_method})"
    elif args.baseline_method:
        base_summ = summ  # within-run: read the other method from this run
        src_label = f"within-run baseline method: {base_method}"
    if base_summ is not None:
        print(f"\n[2] {PRIMARY} NON-REGRESSION CHECK  ({ref} vs baseline)")
        print(f"  {src_label}")
        improved, regressed, unchanged = [], [], []
        new_vals, base_vals = [], []
        for fn in funcs:
            if ref not in summ.get(fn, {}) or base_method not in base_summ.get(fn, {}):
                continue
            new = _pct(summ[fn][ref][PRIMARY])
            old = _pct(base_summ[fn][base_method][PRIMARY])
            if math.isnan(new) or math.isnan(old):
                continue
            new_vals.append(new)
            base_vals.append(old)
            d = new - old
            if d > 1e-9:
                improved.append((fn, old, new))
            elif d < -1e-9:
                regressed.append((fn, old, new))
            else:
                unchanged.append(fn)
        ov_new, ov_old = _mean(new_vals), _mean(base_vals)
        print(f"  overall {PRIMARY}: {ov_old:.1%} -> {ov_new:.1%} "
              f"({ov_new - ov_old:+.1%})")
        if regressed:
            print(f"  REGRESSED ({len(regressed)}):")
            for fn, o, n in regressed:
                print(f"    - {fn:<22} {o:.0%} -> {n:.0%}  ({n - o:+.0%})")
        if improved:
            print(f"  IMPROVED ({len(improved)}):")
            for fn, o, n in improved:
                print(f"    + {fn:<22} {o:.0%} -> {n:.0%}  ({n - o:+.0%})")
        print(f"  unchanged: {len(unchanged)}")
        verdict = "REJECT" if (ov_new < ov_old - 1e-9 or regressed) else "OK"
        why = []
        if ov_new < ov_old - 1e-9:
            why.append("overall SR@1e-10 dropped")
        if regressed:
            why.append(f"{len(regressed)} function(s) regressed on SR@1e-10")
        print(f"  >>> SR@1e-10 VERDICT: {verdict}"
              + (f"  ({'; '.join(why)})" if why else ""))
        print("      Rule: a config that lowers SR@1e-10 is not adopted "
              "(docs/experiments.md). Per-function regressions must be justified.")

    # ── 3. Per-function: ref vs best baseline on the primary metric ──────────
    print(f"\n[3] PER-FUNCTION  {ref} vs best baseline on {PRIMARY}")
    print(f"  {'function':<22} {ref+' SR@1e-10':>14} {'best base':>20} "
          f"{'evals(ref/base)':>20}")
    print("  " + "-" * 74)
    for fn in funcs:
        if ref not in summ.get(fn, {}):
            continue
        ref_sr = _pct(summ[fn][ref][PRIMARY])
        ref_ev = _evals(summ[fn][ref])
        bases = {m: summ[fn][m] for m in summ[fn] if m != ref}
        if bases:
            bm = max(bases, key=lambda m: (_pct(bases[m][PRIMARY])
                                           if not math.isnan(_pct(bases[m][PRIMARY]))
                                           else -1))
            b_sr = _pct(bases[bm][PRIMARY])
            b_ev = _evals(bases[bm])
            base_str = f"{bm} {b_sr:.0%}"
            ev_str = (f"{ref_ev:.0f}/{b_ev:.0f}"
                      if not math.isinf(ref_ev) and not math.isinf(b_ev)
                      else "-")
        else:
            base_str, ev_str = "-", "-"
        flag = "  <-- ref behind" if (bases and ref_sr < b_sr - 1e-9) else ""
        print(f"  {fn:<22} {ref_sr:>13.0%} {base_str:>20} {ev_str:>20}{flag}")

    # ── 4. Wilcoxon detail: functions where a baseline beats ref ─────────────
    print(f"\n[4] WILCOXON: functions where a baseline significantly beats {ref}")
    print("  (p_two<0.05 and A12<0.5 => baseline better)")
    any_loss = False
    for (fn, m), w in sorted(wil.items()):
        p = _num(w.get("p_value_two_sided", ""))
        a12 = _num(w.get("a12", ""))
        if not math.isnan(p) and p < 0.05 and not math.isnan(a12) and a12 < 0.5:
            any_loss = True
            print(f"  {fn:<22} vs {m:<14} p={p:.3g} A12={a12:.2f} "
                  f"({w.get('a12_magnitude','')})")
    if not any_loss:
        print(f"  none — {ref} is never significantly beaten.")
    print()


if __name__ == "__main__":
    main()
