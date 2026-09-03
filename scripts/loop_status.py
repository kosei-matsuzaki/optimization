#!/usr/bin/env python3
"""One screen of what the research loop is doing, for a human deciding direction.

The loop runs hourly in the cloud and writes everything to docs/research_loop.md,
which is long. This prints the parts a person needs to steer: the standing
instruction, the goal, which questions are open and who is on them, what has been
settled or ruled out, and what the last cycles actually concluded.

Run it after `git pull` on the research-loop branch.

Usage:
  python3 scripts/loop_status.py [--cycles 8] [--full]
"""
from __future__ import annotations
import argparse
import io
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DOC = Path("docs/research_loop.md")


def _section(text: str, head: str, level: str = "## ") -> str:
    i = text.find(level + head)
    if i < 0:
        return ""
    j = text.find("\n" + level, i + 1)
    return text[i:j if j > 0 else len(text)]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                  # pragma: no cover
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=8,
                    help="how many recent log entries to list")
    ap.add_argument("--full", action="store_true",
                    help="print each listed entry in full, not just its heading")
    args = ap.parse_args()

    text = io.open(DOC, encoding="utf-8").read().replace("\r\n", "\n")
    now = datetime.now(timezone.utc)

    st = Path("docs/status.md")
    if st.exists():
        snap = io.open(st, encoding="utf-8").read().replace("
", "
")
        print("=" * 78)
        print("OVERVIEW  (docs/status.md — rewritten daily by the review cycle)")
        print("=" * 78)
        for head in ("## ユーザーの判断を仰ぐこと", "## いま論文に書ける主張",
                     "## ゴールとの距離", "## いちばん弱い環", "## 次に効く一手"):
            sec = _section(snap, head[3:])
            if sec:
                print(sec.rstrip())
                print()
        print("(full document: docs/status.md)")
        print()

    steer = _section(text, "方針（ユーザーが書く欄）")
    if steer:
        body = "\n".join(l for l in steer.split("\n")[1:] if l.strip())
        print("=" * 78)
        print("STANDING INSTRUCTION  (docs/research_loop.md, edit this to steer)")
        print("=" * 78)
        print(body or "  (empty)")
        print()

    goal = _section(text, "この研究のゴール（毎回、着手する問いをここに照らすこと）")
    for line in goal.split("\n"):
        if line.startswith("**") and line.endswith("**"):
            print("GOAL:", line.strip("*"))
            break

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    behind = _git("rev-list", "--count", "HEAD..origin/research-loop")
    n_today = len([l for l in _git("log", "--since=24 hours ago",
                                   "--format=%h").split("\n") if l])
    print(f"branch {branch}, {n_today} commits in the last 24h"
          + (f", {behind} not pulled (run: git pull)" if behind not in ("", "0")
             else ", up to date"))
    print()

    print("-" * 78)
    print("OPEN QUESTIONS   (the loop takes the top unclaimed one each hour)")
    print("-" * 78)
    qs = _section(text, "未解決の問い（上から着手する）")
    for line in qs.split("\n"):
        m = re.match(r"^(\d+)\. \*\*(.+?)\*\*(.*)$", line)
        if not m:
            continue
        n, title, rest = m.groups()
        claim = re.search(r"claimed (\d{4}-\d\d-\d\d \d\d:\d\d) UTC", rest)
        tag = ""
        if claim:
            age = (now - datetime.strptime(claim.group(1), "%Y-%m-%d %H:%M")
                   .replace(tzinfo=timezone.utc)).total_seconds() / 3600
            tag = (f"  [claimed {age:.1f}h ago]" if age < 2
                   else f"  [claim stale, {age:.1f}h — next run may take it]")
        print(f"  {n}. {title}{tag}")
    print()

    folded = _section(text, "畳んだ問い", level="**")
    if not folded:
        i = text.find("**畳んだ問い**")
        folded = text[i:text.find("\n---", i)] if i > 0 else ""
    if folded:
        print("-" * 78)
        print("RULED OUT   (do not propose these again)")
        print("-" * 78)
        print("\n".join(l for l in folded.split("\n")[:12] if l.strip()))
        print()

    print("-" * 78)
    print(f"LAST {args.cycles} CYCLES   (newest first)")
    print("-" * 78)
    heads = [(m.start(), m.group(0))
             for m in re.finditer(r"^### \d{4}-\d\d-\d\d.*$", text, re.M)]
    for k, (pos, head) in enumerate(heads[:args.cycles]):
        print(f"  {head[4:]}")
        if args.full:
            end = heads[k + 1][0] if k + 1 < len(heads) else len(text)
            print("\n".join("      " + l for l in text[pos:end].split("\n")[1:]))
        print()

    print("-" * 78)
    print("TO STEER: edit docs/research_loop.md and push to research-loop.")
    print("  - reorder or rewrite '未解決の問い' to change what gets worked on")
    print("  - write under '方針（ユーザーが書く欄）' to give a standing instruction")
    print("  - every cycle reads both before it claims anything")


if __name__ == "__main__":
    main()
