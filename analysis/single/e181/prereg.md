# e181 事前登録 — pin 93.54/798 対 base 92.08/677.7 の決着（キュー 1）

- 日付: 2026-09-28（サイクル その181、実行役）
- 問い: `CLAUDE.md` が pin する **SR@1e-10 93.5% / evals 798** が、この環境で 4 回連続 **92.08% / 677.7** になる。
  原因は **(A) 環境差** か **(B) `core/` の drift** か。
- 前提（俯瞰が確定させた）: pin の値は [history.md](../../../docs/history.md):489 の `softmax_beta=5` 行（93.54 / 798）と一致し、
  `softmax_beta` の既定は今も 5.0（`core/optimizers/mceso.py:362`）。

## 設計

1. `git fetch --unshallow` —— **済（65 → 666 commit）。**
2. pin の値を記録した commit を同定 —— **`25418d9`（2026-08-24 19:02, "Fix two dimension/scale invariance defects found by a full parameter audit"）。**
   `git log -S'20260824_125218_dimf_softmax_d2_quick' -- docs/history.md` が 1 件だけ返す。
   この commit は同時に `core/optimizers/mceso.py` を変更しており、`softmax_beta=5` の採用コミットである。
3. **同一環境・同一コマンドで 2 本回す（対照は同じセッション内で取る）。**
   - **arm OLD**: `25418d9` の worktree で `./run.sh quick --all --methods MC-ESO --n-runs 20 --label pin_old`
   - **arm HEAD**: 現 HEAD で `./run.sh quick --all --methods MC-ESO --n-runs 20 --label pin_head`
   どちらも 2D BBOB-24 F01-F24 / n=20 / 5000 評価（quick の既定）。
4. 主指標 **SR@1e-10**、補助 SR@1e-2/1e-4/1e-7 と `evals_succ_mean`。
   **関数別 SR@1e-10 を 24 関数すべて列挙**して差分の所在を出す。
   2 arm は同じ seed 系列（`quick_check.py` の seed 規則が両 commit で同じなら run 対応がつく）なので、
   **関数 × run の成否を対で並べる**（対応のある比較）。
5. drift が原因と出た場合、**`mceso.py` を触った 5 commit**（`7391f7f` `bc891be` `88fdc6b` `43178d9` `e2f262a`）が容疑者。
   40 分に収まる範囲で最も疑わしい 1〜2 点を追加で回す。

## 反証条件（事前登録）

- **(a) arm OLD がこの環境で 92.08 前後を出す → 環境差で確定。** この 1.46pt は永久に諦め、弱い環 4 を降ろし、
  `acceptance_topology.md`:14401 の「原因未特定」を書き換える。
- **(b) arm OLD が 93.54 前後を出す → 既定が drift している。** **この時点で止めて俯瞰に上げる**
  （既定を戻すかは判断であって実行ではない。実行役は MC-ESO の既定を変えない）。
- **(c) arm OLD が 93.54 でも 92.08 でもない第 3 の値を出す** → seed / ライブラリ版の効果が両者より大きい、
  すなわち **pin も base も再現性の単位として使えない**。この場合は「1.46pt を赤字の一部として数えるのをやめる」と書く。
- **(d) arm HEAD が 92.08 を再現しない** → この環境の base 自体が不安定で、対照が成立しない（5 度目の再現が崩れる）。

## やらないこと

- `core/` を 1 行も触らない。MC-ESO の既定を変えない。
- 腕を実装しない。
