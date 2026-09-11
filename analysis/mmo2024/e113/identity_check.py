#!/usr/bin/env python3
"""その113 — 計器が探索を 1 ビットも変えていないことの確認。

`TracedMCESO` は 2 つの hook を override するだけで、どちらも `super()` を呼び、
RNG も評価も消費しない。それでも「計器つきの数字は base の数字である」と書く前に
機械的に確かめる —— 同じ seed で評価履歴と報告集合がバイト一致すること。

その111 の教訓（保存ダンプと run の恒等検査を毎回やる）をこの回に当てたもの。

使い方: PYTHONPATH=/tmp/pystub python3 identity_check.py [problem] [budget]
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from core.benchmarks import niching_by_name                      # noqa: E402
from core.optimizers import MultiChannelEpidemicOptimizer as Base  # noqa: E402
from core.optimizers.mceso_traced import TracedMCESO             # noqa: E402


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "M09-D10-PIN01"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    b = niching_by_name(name)
    r1 = Base(b, seed=0).optimize(budget)
    r2 = TracedMCESO(b, seed=0).optimize(budget)
    h1, h2 = np.asarray(r1.history_f), np.asarray(r2.history_f)
    s1 = np.asarray([np.asarray(x) for x in r1.final_solutions])
    s2 = np.asarray([np.asarray(x) for x in r2.final_solutions])
    ok_h = h1.shape == h2.shape and np.array_equal(h1, h2)
    ok_s = s1.shape == s2.shape and np.array_equal(s1, s2)
    print(f"{name}  budget={budget}  K={b.n_global_optima}")
    print(f"  evaluation history identical: {ok_h}  ({len(h1)} / {len(h2)} evals)")
    print(f"  reported set identical:       {ok_s}  (|rep| {len(s1)} / {len(s2)})")
    raise SystemExit(0 if (ok_h and ok_s) else 1)


if __name__ == "__main__":
    main()
