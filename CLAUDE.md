# CLAUDE.md

MC-ESO（提案手法）と既存最適化手法を BBOB 等のベンチマークで比較する研究プロジェクト。
本ファイルは作業時に遵守するルールをまとめる。コード・構成の詳細は領域別 README を参照。

## ドキュメント（領域別 README）

作業対象の領域に入る**前に該当 README を参照**し、変更後は**該当 README を更新**すること。
記述とコードが常に一致した状態を保つ。

| README | 範囲 |
|---|---|
| [README.md](README.md) | プロジェクト全体（概要・実験フロー・結果の見方・コマンド一覧） |
| [core/README.md](core/README.md) | 最適化手法（`core/optimizers/`）とベンチマーク関数（`core/benchmarks.py`） |
| [web/README.md](web/README.md) | Results UI（Flask アプリの構成・アーキテクチャ・ルート/API） |

- 更新対象の振り分け: web 配下の変更 → `web/README.md`、core 配下の変更 → `core/README.md`、横断的・全体に関わる変更 → ルート `README.md`。

## 実行ルール

### main.py（本番実験）

- `main.py` の実行はこのローカル PC 上では行わない。
- 実行は必ず **GitHub Actions workflow** 経由で行う。
- コードの変更後は、リモートのワークフローをトリガーして結果を確認する。

### quick_check.py / run.sh（ローカル確認・実験管理）

- `quick_check.py` はローカルでの軽量動作確認専用スクリプト。
- **`./run.sh quick` はユーザーから「検証して」「比較して」「分析して」などの明示的な指示があった場合のみ実行する。コード変更後の自動動作確認目的では実行しない。**
- `./run.sh quick` を使うこと。`python3 quick_check.py` を直接呼ばない。
- 実験管理は `run.sh` で行う（trigger / download / quick / list / status / ui）。
- 結果はすべて `results/YYYYMMDD_HHMMSS_<commit>/` にバージョン管理される。

## Git

- **`git push` はユーザーから明示的に指示されるまで行わない。**
- コミットまでは行ってよいが、push は必ず指示を待つこと。

## 研究上の考慮事項

### 提案手法の価値・意義の評価

- 実装・変更を行う前に、その手法が**研究として価値・意義があるか**を検討する。
- 以下の観点から評価する:
  - 既存手法との差別化（新規性）
  - 実験的な裏付けの有無（再現性・公平な比較）
  - 対象問題における理論的・実用的な意義
  - 関連研究との位置づけ
- 意義が不明確な場合は実装前にユーザーに確認する。

### 手法比較・評価の基準（必ずこの3点で評価する）

手法を比較・評価する際は、必ず以下の **3 指標** を揃えて報告する。単一指標（SR のみ等）での判定は不可。

1. **SR（Success Rate）**
   - BBOB は `f - f_opt` 正規化により最適値が 0。**最高精度 (SR@1e-10) を主指標**として、0 への到達率で評価する。
   - 補助的に SR@1e-2 / SR@1e-4 / SR@1e-7 も併記し、精度階層全体での挙動を確認する。
   - 全関数で集計し、関数別の改善・悪化を必ず列挙する（regression の見落としを防ぐ）。

2. **平均評価回数（Evals to target）**
   - 成功 run のみでの中央値 / 平均評価回数 (`evals_succ_med`)。
   - SR が同等なら少ない評価回数の方が優位。SR と速さのトレードオフを明示する。

3. **統計的優位性（Wilcoxon signed-rank test）**
   - V1 (`MC-ESO`) を reference とし、各 variant の `p_value_ref_better` / `p_value_two_sided` を確認。
   - 有意水準 α=0.05。p < 0.05 で「有意差あり」と判定し、A12 で効果量（negligible/small/medium/large）も併記。
   - **n=10 (quick) は signal-to-noise が低い** ことを明記し、本実験 (n=100) との差を考慮する。

### 評価範囲・実行コマンドの基準

- 手法評価は**必ず全ベンチマーク関数で実施**する（quick-12 サブセットでの判定は不可）。
  - ローカル quick: `./run.sh quick --all`（BBOB 24 + Custom 11）
  - 本実験: `./run.sh trigger`（n=100, BBOB 24 + Custom + CEC2022 hold-out）
- 結果は `results/YYYYMMDD_HHMMSS_<label>_quick/dim{N}/{summary,wilcoxon}.csv` から上記 3 指標を集計して報告する。
