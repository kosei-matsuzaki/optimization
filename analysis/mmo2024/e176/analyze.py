#!/usr/bin/env python3
"""その176 — キュー 1 の残り: NMMSO × 新 suite **D=20** の 16 問を採点する。

**採点・規則・統計量・出力書式はすべて `e175/analyze.py` をそのまま実行する**
（`e175/analyze.py` 自身が `e115/analyze.py` から `SPAN` / `rule_indices` / `score` / `paired` を
import している）。**この回で新しく定義した統計量・規則は 1 つも無い。**
この shim がしていることは 1 つだけ —— **ダンプの読み先を `e176/dumps/` に差し替えること。**

入力:
  * NMMSO D=20 -> `e176/dumps/D20/M??-D20-PIN01_NMMSO_seed0.csv.gz`
    （14 本はこの回の新規 run、**M09 / M10 の 2 本は その175 の保存物**
     `e175/report_sets_D20.csv.gz` から per-problem に戻したもの ＝ 追加評価ゼロ）
  * `Restart-Lander` D=20 -> `e151/descents.csv.gz`（保存物、PIN01 seed 0。**追加評価ゼロ**）

使い方: PYTHONPATH=<patched pynmmso> python3 analysis/mmo2024/e176/analyze.py [D20]
"""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location(
    "e175_analyze", os.path.join(MMO, "e175", "analyze.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

a.HERE = HERE           # ダンプと by_problem.csv の置き場だけを e176 に向ける

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + (sys.argv[1:] or ["D20"])
    a.main()
