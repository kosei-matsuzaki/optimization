#!/usr/bin/env python3
"""その131 の事故復旧 —— `e129/by_problem_current.csv` の欠けた 5 行を保存ダンプから作り直す。

**追加評価ゼロ。** その131 の片付け中に、`e129/by_problem/` の per-problem `current` 規則 CSV を
**`by_problem_current.csv` に足す前に消してしまった**（fold script が `problem` 列の有無で落ちた直後に
`rm` が走った）。行の中身は `e129/descents_seed200.csv.gz`（畳み済み・削除していない）から
**`current` 規則（= e115 の `cur` ＝ `f` 昇順の上位 max(100, 2K) 点）を当て直せば完全に再現できる**ので、
run はやり直さない。

**関門**: 既存 11 問の行を同じ経路で作り直し、`by_problem_current.csv` の記録と全列一致すること。
一致して初めて欠けた 5 行を足す。

使い方: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e131/rebuild_by_problem_current.py
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location(
    "e129_analyze", os.path.join(MMO, "e129", "analyze.py"))
a129 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a129)
a115 = sys.modules["analyze"]

TARGET = os.path.join(MMO, "e129", "by_problem_current.csv")
DUMP = os.path.join(MMO, "e129", "descents_seed200.csv.gz")
HEADER = ["problem", "function", "method", "rule", "seed", "evals", "n_optima",
          "n_reported", "pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]


def row_for(prob, f, opt, xs):
    K = a129.K_of(prob)
    idx = a115.rule_indices("cur", f, K)
    recall, _prec, _f1, _sc, n = a115.score(idx, f, opt, K)
    return [prob, prob, "Restart-Lander", "current", "2", "500000", str(K), str(int(n))] + \
           [f"{v:.4f}" for v in recall]


def main():
    dump = a129.read_combined(DUMP)
    have = {r[0]: r for r in list(csv.reader(open(TARGET)))[1:]}
    bad = 0
    for prob, row in sorted(have.items()):
        rebuilt = row_for(prob, *dump[prob])
        if rebuilt != row:
            bad += 1
            print(f"  **不一致** {prob}\n    記録 {row}\n    再構成 {rebuilt}")
    print(f"  関門: 既存 {len(have)} 問を作り直して不一致 {bad} 件")
    if bad:
        raise SystemExit("再構成が記録に一致しないので足さない")
    added = [row_for(p, *dump[p]) for p in sorted(dump) if p not in have]
    rows = sorted(list(have.values()) + added, key=lambda r: r[0])
    with open(TARGET, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"  +{len(added)} 行 -> {os.path.basename(TARGET)}（{len(rows)} 問）")
    for r in added:
        print("   ", ",".join(r))


if __name__ == "__main__":
    main()
