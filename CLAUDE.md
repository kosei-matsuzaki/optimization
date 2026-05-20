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
