#!/usr/bin/env python3
"""Fail when a document outgrows its line budget.

The record grew to 23,000 lines of docs (acceptance_topology.md alone reached
15,037) because every cycle appended and nothing ever had to be folded. A cap
that is checked mechanically is the only rule that held here: a rule written in
prose was skipped until it became a check (see check_cost_rows.py for the same
story). This table is the single source for the caps; docs refer to it.

Usage:
  python3 scripts/check_doc_size.py          # exit 1 if any file is over
"""
from __future__ import annotations
import sys
from pathlib import Path

# path (or glob) → max lines. First match wins, so specific entries go first.
CAPS: list[tuple[str, int]] = [
    ("docs/status.md", 120),          # rewritten whole by the review routine
    ("docs/research_loop.md", 300),   # hand-over between routine and sessions
    ("docs/findings.md", 400),        # current theme's established results
    ("docs/archive/*.md", 500),       # one summary per paused route
    ("docs/report.html", 2500),       # self-contained page (CSS + inline SVG)
    ("docs/*.md", 1000),
    ("CLAUDE.md", 150),
    ("README.md", 100),
]

# Temporarily over the cap, with the date the exemption was granted. Remove the
# entry once the file is folded back under its cap.
EXEMPT: dict[str, str] = {
    "docs/history.md": "2026-09-29 — 1555 lines; fold to 1000 in an interactive session",
}


def cap_for(path: Path) -> int | None:
    for pattern, cap in CAPS:
        if path.match(pattern) or path.as_posix() == pattern:
            return cap
    return None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    files = sorted({p for pattern, _ in CAPS for p in root.glob(pattern)})
    over = []
    for p in files:
        rel = p.relative_to(root)
        cap = cap_for(rel)
        if cap is None:
            continue
        n = sum(1 for _ in p.open(encoding="utf-8"))
        flag = ""
        if n > cap:
            flag = "EXEMPT" if rel.as_posix() in EXEMPT else "OVER"
            if flag == "OVER":
                over.append(rel)
        print(f"{n:6d} / {cap:<5d} {rel}  {flag}")
    for rel, why in EXEMPT.items():
        print(f"  exempt: {rel} ({why})")
    if over:
        print(f"\n{len(over)} file(s) over the cap: fold them before committing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
