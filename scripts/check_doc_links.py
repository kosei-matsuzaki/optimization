#!/usr/bin/env python3
"""docs 間の相互参照アンカーが実在する見出しを指しているかを検査する。

統合の回（`docs/research_loop.md` の手順 2）で使う。その88 が壊れたアンカーを 2 本
目で見つけて 1 サイクル放置され、その90 がこの検査で残りゼロを確認した ——
リンクは押すまで壊れて見えないので、畳む前に 1 回走らせる。

    python3 scripts/check_doc_links.py        # 壊れていれば非ゼロで終了

GitHub の見出しアンカー規則に合わせている: 小文字化 → 単語文字・空白・ハイフン
以外を除去（`@` や括弧は落ちる。`—` も落ちるので前後の空白がそのままハイフンになる）
→ 空白をハイフンに。h1 も見出しアンカーになる。
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

DOCS = ["README.md", *sorted(glob.glob("docs/*.md"))]


def slug(heading: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # [表示](url) -> 表示
    text = text.replace("`", "").replace("*", "").strip()
    return re.sub(r"[^\w\- ]", "", text.lower(), flags=re.UNICODE).replace(" ", "-")


def headings(path: str) -> set[str]:
    out = set()
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            out.add(slug(m.group(2)))
    return out


def main() -> int:
    tables = {p: headings(p) for p in DOCS if Path(p).exists()}
    broken, checked = [], 0
    for src in tables:
        text = Path(src).read_text()
        for m in re.finditer(r"\]\((?:([A-Za-z0-9_./-]+\.md))?#([^)\s]+)\)", text):
            target = m.group(1)
            if target:  # docs/ 相対と同ディレクトリ相対の両方を試す
                cands = [target, str(Path(src).parent / Path(target).name)]
                table = next((tables[c] for c in cands if c in tables), None)
                if table is None:
                    continue  # docs 外へのリンクは対象外
            else:
                table = tables[src]
            checked += 1
            if m.group(2) not in table:
                broken.append((src, target or src, m.group(2)))

    print(f"checked {checked} anchors across {len(tables)} files")
    for src, dst, anchor in broken:
        print(f"BROKEN {src} -> {dst}#{anchor}")
    if broken:
        print(f"{len(broken)} broken anchor(s)")
        return 1
    print("no broken anchors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
