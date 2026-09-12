#!/usr/bin/env python3
"""Check that every work-log entry has a cost row in the plan table.

The hole this closes (acceptance_topology.md, 8th form of the consolidation
lesson): each cycle is supposed to add one row to
`acceptance_topology.md` -> "1 run の実測コスト" / "1 サイクルの実績" giving the
configuration it ran and the wall clock it took, because the work log gets
folded and a cost that is only in the log disappears at the next
consolidation.  Writing that into the procedure cut the miss rate from 80%
to 20% but did not reach zero, and it went back to 100% for the five cycles
after entry 108 -- it is a "copy one line across" step, which is exactly the
kind of step a person drops and a script does not.

Usage:
    python3 scripts/check_cost_rows.py          # exit 1 if a cycle is missing
    python3 scripts/check_cost_rows.py --list   # also print what was found

Run it before committing a cycle (procedure step 7) and during a
consolidation cycle (procedure step 2), the same way check_doc_links.py is
run.  It reads only the entries still present in the log, so a cycle whose
entry has already been folded is out of its reach: the check has to happen
before the fold, which is why step 7 is where it belongs.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "docs" / "research_loop.md"
COSTS = ROOT / "docs" / "acceptance_topology.md"

COST_TABLE_START = "### 1 run の実測コスト"
CYCLE = re.compile(r"その(\d+)")


def log_cycles(text: str) -> list[int]:
    """Cycle numbers that have a `### <date> そのN` entry in the work log."""
    body = text.split("\n## 作業ログ", 1)
    if len(body) == 1:
        raise SystemExit("docs/research_loop.md: '## 作業ログ' が見つからない")
    out = []
    for line in body[1].split("\n"):
        # `### 畳んだログ（その15〜その104, ...）` is the fold summary, not an entry
        if line.startswith("### ") and not line.startswith("### 畳んだログ"):
            m = CYCLE.search(line)
            if m:
                out.append(int(m.group(1)))
    return out


def cost_cycles(text: str) -> set[int]:
    """Cycle numbers named in the cost / per-cycle tables.

    A row may name several cycles at once (the consolidation row lists all of
    them), so every `そのN` inside the table region counts.
    """
    i = text.find(COST_TABLE_START)
    if i < 0:
        raise SystemExit(f"docs/acceptance_topology.md: '{COST_TABLE_START}' が見つからない")
    region = text[i:]
    # the region runs to the next top-level heading
    j = region.find("\n## ")
    if j > 0:
        region = region[:j]
    return {int(m) for line in region.split("\n") if line.startswith("|")
            for m in CYCLE.findall(line)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="見つかった回番号も表示する")
    args = ap.parse_args()

    entries = log_cycles(LOG.read_text())
    covered = cost_cycles(COSTS.read_text())
    missing = [n for n in entries if n not in covered]

    if args.list:
        print(f"作業ログの回: {', '.join('その%d' % n for n in entries) or '(なし)'}")
        print(f"コスト表の回: {len(covered)} 件")

    if missing:
        print(f"コスト行が無い回が {len(missing)} 件: "
              + ", ".join(f"その{n}" for n in missing))
        print("→ acceptance_topology.md の「1 サイクルの実績」に 1 行足すこと"
              " (構成・並列数・壁時計。測定ゼロの回も「run ゼロで何分か」を書く)")
        return 1

    print(f"コスト行の穴なし ({len(entries)} 件の作業ログを照合)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
