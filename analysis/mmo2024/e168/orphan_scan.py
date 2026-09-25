#!/usr/bin/env python3
"""その168（統合の回）— 走査 E: `os.path.join` の**途中の要素が変数**であるパス。

**なぜ要るか。** `scripts/scan_silent_null.py` の走査 A / C / D は、どれもパスの
**segment がリテラルであること**を前提にしている ——
A は 1 個の文字列の中に `/` を要求し、C は先頭要素が `eNNN` のリテラルであることを要求し、
D は `os.path.join(HERE, "file")` の 2 引数形だけを見る。
**`os.path.join(MMO, entry, "descents.csv.gz")`（`entry` は for の変数）はこの 3 本のどれにも掛からない。**

**実害（その168 が見つけた）**: `e165/analyze.py` がまさにこの形で
`e163` / `e164` / `e151` の 3 本を読む。その166 が `e163` / `e164` を消した時点でこの script は
走れなくなったが、**走査 A / C / D はその166 の点検でも通過したので、誰も気づかなかった。**
その結果 **`e165/descents.csv.gz`（152 KB）が読み手を失ったまま 1 サイクル残った。**
同型は その118 でも起きている（`e110/descents` / `e111/descents` の削除で `e115/analyze.py` の
`main()` が死に、**`e115/s1/` 32 ファイルが読み手を失った**）。

**この走査が言えること／言えないこと。** 言えるのは「このパスは静的走査の死角にある」まで。
**そのデータに読み手が残っているかは、この走査では決まらない**（変数の値が分からないため）。
判定は列挙された script を 1 本ずつ叩いて確かめる（その161 の規則: 叩く前後で `git status` を見る）。

**この script は読むだけで、何も書かない。**

使い方: python3 analysis/mmo2024/e168/orphan_scan.py
"""
from __future__ import annotations
import ast
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCAN_ROOTS = ("analysis", "scripts", "core", "web")


def joins_with_variable_segment(path: str) -> list[tuple[int, str]]:
    """`os.path.join(...)` の引数に「変数 → その後にリテラル」が並ぶ呼び出しを返す。"""
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "join"):
            continue
        args = node.args
        if len(args) < 3:
            continue                                  # 2 引数形は走査 D が見ている
        mid = args[1:-1]
        last = args[-1]
        if not (isinstance(last, ast.Constant) and isinstance(last.value, str)):
            continue                                  # 末尾がリテラルでないものは対象外
        varnames = [a.id for a in mid if isinstance(a, ast.Name)]
        if varnames:
            hits.append((node.lineno, f"join(..., {'/'.join(varnames)}, {last.value!r})"))
    return hits


def main() -> None:
    print("## 走査 E —— os.path.join の途中の要素が変数であるパス（走査 A / C / D の死角）\n")
    total = 0
    files = 0
    for base in SCAN_ROOTS:
        for root, dirs, names in os.walk(os.path.join(ROOT, base)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for n in sorted(names):
                if not n.endswith(".py"):
                    continue
                p = os.path.join(root, n)
                hits = joins_with_variable_segment(p)
                if not hits:
                    continue
                files += 1
                print(f"  {os.path.relpath(p, ROOT)}")
                for lineno, what in hits:
                    total += 1
                    print(f"      :{lineno}  {what}")
    print(f"\n  → {files} 本 / {total} 箇所")
    print("\n  判定はこの一覧を 1 本ずつ叩いて行う（この走査は死角を指すだけで、実害は言わない）。")


if __name__ == "__main__":
    main()
