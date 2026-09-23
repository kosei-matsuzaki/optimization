#!/usr/bin/env python3
"""その161 (iii) — 「記録が黙って帰無を作れる」経路を analysis/ 全体で機械的に数える。

その150 §5 の 2 通りの走査を両方かける:
  走査 A: script が文字列で組むパスのうち、いま実在しないもの（＝ 既に壊れている）
  走査 B: 実在チェックのあと return / continue / pass する loader
          （＝ いまは動くが、その入力を誰かが消した瞬間に黙って壊れる）

さらに走査 B の各 script について「欠けたときに落ちるか、黙って通るか」を区別する。
追加評価ゼロ。使い方: python3 scan_silent_null.py
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))            # analysis/
REPO = os.path.dirname(ROOT)

# 文字列リテラルの中で analysis 配下を指していそうな断片。
# 散文（docstring の日本語・空白入りの文）を拾わないよう、パスらしい形だけに絞る。
PATH_LIT = re.compile(r'["\']([A-Za-z0-9_./*{}-]*?(?:e\d{2,3}|hm)/[A-Za-z0-9_./*{}-]*)["\']')
# f-string の {} は実在確認できないので、プレフィックスだけを見る
GUARD = re.compile(r'if\s+not\s+os\.path\.(exists|isdir|isfile)\s*\(|'
                   r'if\s+not\s+\w+\.exists\s*\(\)')
# 黙って抜ける文。`return out` のように「空のまま組み立て中の入れ物」を返す形も
# 含める（その156 が名指しした e142/analyze.py:132 がこれ。裸の return だけを
# 見ていると漏れる）。
SILENT = re.compile(r'^\s*(continue|pass|return(\s+([A-Za-z_]\w*|None|\[\]|'
                    r'\{\}|set\(\)|0))?)\s*$')


def candidates(text):
    """文字列リテラルから、実在を確認できるパス断片を拾う。"""
    for m in PATH_LIT.finditer(text):
        s = m.group(1)
        if "{" in s:                      # f-string の可変部は先頭の固定部だけ見る
            s = s.split("{")[0]
        s = s.strip("/")
        if not s or "/" not in s:
            continue
        yield s


def resolve(frag, script_dir):
    """断片を 4 通り（analysis/ 相対、script のディレクトリ相対、その親相対、
    リポジトリ相対）で解く。実験ディレクトリ同士の相互参照は親相対で当たる。"""
    return [os.path.join(ROOT, frag), os.path.join(script_dir, frag),
            os.path.join(os.path.dirname(script_dir), frag),
            os.path.join(REPO, frag)]


def scan_a():
    hits = []
    for dirpath, _, files in os.walk(ROOT):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            text = open(p, encoding="utf-8", errors="replace").read()
            missing = []
            for frag in sorted(set(candidates(text))):
                # ディレクトリ名だけの断片（e113/hunts など）も見る
                if any(os.path.exists(c) for c in resolve(frag, dirpath)):
                    continue
                # 接頭辞で実在するなら（glob の前置きなど）除く
                base = os.path.dirname(frag)
                if base and any(os.path.exists(c) for c in resolve(base, dirpath)):
                    missing.append((frag, "親のみ実在"))
                else:
                    missing.append((frag, "親も不在"))
            if missing:
                hits.append((os.path.relpath(p, ROOT), missing))
    return hits


JOIN = re.compile(r'os\.path\.join\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*'
                  r'((?:["\'][A-Za-z0-9_.-]+["\']\s*,\s*)+)')


def scan_c():
    """走査 A の穴埋め: os.path.join(BASE, "eNNN", "descents", ...) の形。
    走査 A は文字列リテラル 1 個の中に `/` がある場合しか拾えないので、
    分割して join する書き方（その150 の一覧の e113 / e110 がこれ）が漏れる。"""
    hits = []
    for dirpath, _, files in os.walk(ROOT):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            text = open(p, encoding="utf-8", errors="replace").read()
            missing = []
            for m in JOIN.finditer(text):
                parts = re.findall(r'["\']([A-Za-z0-9_.-]+)["\']', m.group(2))
                if not parts or not re.match(r'^(e\d{2,3}|hm)$', parts[0]):
                    continue
                frag = "/".join(parts)
                if any(os.path.exists(c) for c in resolve(frag, dirpath)):
                    continue
                missing.append(frag)
            if missing:
                hits.append((os.path.relpath(p, ROOT), sorted(set(missing))))
    return hits


def scan_b():
    hits = []
    for dirpath, _, files in os.walk(ROOT):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
            marks = []
            for i, ln in enumerate(lines):
                if not GUARD.search(ln):
                    continue
                # ガード直後の 1-2 行に黙って抜ける文があるか
                for j in (i + 1, i + 2):
                    if j < len(lines) and SILENT.match(lines[j]):
                        marks.append((i + 1, lines[i].strip()[:72],
                                      lines[j].strip()))
                        break
            if marks:
                hits.append((os.path.relpath(p, ROOT), marks))
    return hits


def main():
    print("=== 走査 A: 文字列で組むパスが実在しない script ===")
    a = scan_a()
    if not a:
        print("  （該当なし）")
    for rel, miss in a:
        print(f"  {rel}")
        for frag, how in miss:
            print(f"      {frag}   [{how}]")
    print(f"  → {len(a)} 本")

    print("\n=== 走査 C: os.path.join で組むパスが実在しない script（走査 A の穴埋め）===")
    c = scan_c()
    if not c:
        print("  （該当なし）")
    for rel, miss in c:
        print(f"  {rel}")
        for frag in miss:
            print(f"      {frag}")
    print(f"  → {len(c)} 本")

    print("\n=== 走査 B: 実在チェックのあと黙って抜ける loader ===")
    b = scan_b()
    if not b:
        print("  （該当なし）")
    for rel, marks in b:
        print(f"  {rel}")
        for lno, guard, act in marks:
            print(f"      L{lno}: {guard}  ->  {act}")
    print(f"  → {len(b)} 本 / 箇所 {sum(len(m) for _, m in b)}")

    print("\n=== まとめ ===")
    broken = {r for r, _ in a} | {r for r, _ in c}
    guards = {r for r, _ in b}
    both = sorted(broken & guards)
    print(f"いま入力が壊れている（A または C）{len(broken)} 本 / "
          f"黙って抜ける loader を持つ（B）{len(guards)} 本 / "
          f"両方 {len(both)} 本 ＝ <u>いま黙って帰無を出しうる</u>")
    for r in both:
        print(f"  両方: {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
