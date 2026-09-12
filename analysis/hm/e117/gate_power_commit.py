#!/usr/bin/env python3
"""Does the BBOB-24 dim2 gate have any power over `commit_place_r010`?

Entry 29's lesson, restated: an arm that is numerically identical to base on the
gate has two very different causes, and the gate cannot tell them apart.

  (i)  the committed restart fires and the run ends up in the same place anyway
       -> the variant is genuinely safe here (entry 33's type (c) if the
          divergence is always downstream of the banked answer);
  (ii) the committed restart never gets to decide anything -> the gate measured
       nothing, and a tie by construction is not evidence of safety.

`CommitReseedMCESO` only differs from base inside `_maybe_spillover` on an
*exhausted* basin switch, so this probe counts that event directly on the gate's
own cells. Output per (function, seed, budget):

  commits    how many times the committed placement actually replaced base's
             twenty independent repelled draws.
  first_ev   evaluation count at the first commit (-1 if it never fired).
  bank_ev    evaluation count at which best_f reached its final value.
  post       True when first_ev < bank_ev, i.e. the arm diverged *before* the
             run banked its answer -- the only cells where the commitment could
             have moved `best_f` at all.

Usage:
  python3 analysis/hm/e117/gate_power_commit.py --seeds 5 --evals 5000 \
      --csv analysis/hm/e117/gatepower_commit_5k.csv
"""
from __future__ import annotations
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.benchmarks import BENCHMARKS_BY_NAME                        # noqa: E402
from core.optimizers.mceso_commit_reseed import CommitReseedMCESO     # noqa: E402


class _CountingCommit(CommitReseedMCESO):
    """Read-only tap on the committed-restart decision.

    The variant already counts the two outcomes itself (`n_commit` = the
    commitment was taken, i.e. the arm deviated from base; `n_commit_fallback` =
    the arm was reached but fewer than three distinct basins had been drilled,
    so it ran base's own path). All this subclass adds is *when* the first
    deviation happened, which is what decides whether the tie is entry 33's
    type (c) or entry 29's type (ii).
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.first_commit_ev = -1

    def _diversified_reseed(self, st, x_best_snap):
        before = self.n_commit
        out = super()._diversified_reseed(st, x_best_snap)
        if self.n_commit > before and self.first_commit_ev < 0:
            hec = list(getattr(st, "history_eval_count", []) or [])
            self.first_commit_ev = int(hec[-1]) if hec else -1
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="shard offset; seed = 100 * (seed_start + i), "
                         "the same seeds the gate itself uses")
    ap.add_argument("--evals", type=int, default=5000)
    ap.add_argument("--csv", type=str, required=True)
    args = ap.parse_args()

    names = sorted(n for n in BENCHMARKS_BY_NAME if n.startswith("F"))
    rows = []
    for name in names:
        bench = BENCHMARKS_BY_NAME[name]
        for s in range(args.seed_start, args.seed_start + args.seeds):
            seed = s * 100
            opt = _CountingCommit(bench, seed=seed, commit_mode="on",
                                  commit_sigma_mode="place",
                                  commit_sigma_ratio=0.1)
            res = opt.optimize(args.evals)
            hist = np.asarray(getattr(res, "history_best", []) or [], dtype=float)
            best = float(res.best_f)
            if hist.size:
                bank = int(np.argmax(hist <= best) + 1)
            else:
                bank = -1
            rows.append({
                "function": name, "seed": seed, "evals": args.evals,
                "commits": opt.n_commit, "fallbacks": opt.n_commit_fallback,
                "first_ev": opt.first_commit_ev,
                "bank_ev": bank, "best_f": repr(best),
                "post": bool(0 <= opt.first_commit_ev < bank),
            })
            print(f"{name:<22} seed={seed:<4} commits={opt.n_commit:<4} "
                  f"fb={opt.n_commit_fallback:<4} "
                  f"first_ev={opt.first_commit_ev:<6} bank_ev={bank:<6} "
                  f"best_f={best:.6e}", flush=True)

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_fire = sum(1 for r in rows if r["commits"] > 0)
    n_post = sum(1 for r in rows if r["post"])
    print(f"\ncells {len(rows)}  fired {n_fire}  fired-before-banking {n_post}")
    print("fired == 0            -> the gate measured nothing (entry 29 type (ii))")
    print("fired > 0, post == 0  -> divergence is always downstream of the answer "
          "(entry 33 type (c))")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
