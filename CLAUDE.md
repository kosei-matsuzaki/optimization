# CLAUDE.md

MC-ESO（提案手法）と既存最適化手法を BBOB 等のベンチマークで比較する研究プロジェクト。
本ファイルは作業時に遵守するルールをまとめる。コード・構成の詳細は領域別ドキュメント（`docs/`）を参照。

## ドキュメント（領域別 docs/）

詳細は `docs/` 配下に領域別で分割している（`README.md` はリンクのみ）。作業対象の領域に入る**前に該当ドキュメントを参照**し、変更後は**該当ドキュメントを更新**すること。記述とコードが常に一致した状態を保つ。

| ドキュメント | 範囲 |
|---|---|
| [README.md](README.md) | プロジェクト概要 ＋ 各 docs へのリンクのみ |
| [docs/mceso.md](docs/mceso.md) | 提案手法 MC-ESO（`core/optimizers/mceso.py`）のコンセプト・アーキテクチャ・パラメータ・新規性 |
| [docs/baselines.md](docs/baselines.md) | 比較対象の既存手法（`core/optimizers/` の CMA-ES / IPOP・BIPOP / PSO / DE / L-SHADE / SaVOA） |
| [docs/experiments.md](docs/experiments.md) | ディレクトリ構造・実行方法（run.sh）・実験条件・ベンチマーク関数（`core/benchmarks.py`）・評価基準・結果の見方 |
| [docs/related_work.md](docs/related_work.md) | 多峰（niching）分野の関連研究・標準ベンチマーク・指標・先行例の整理 |
| [docs/history.md](docs/history.md) | 試した工夫・フラグ・ablation 記録（採用 / 不採用とその理由） |
| [docs/web.md](docs/web.md) | Results UI（Flask アプリの構成・アーキテクチャ・ルート/API） |

- 更新対象の振り分け: 提案手法 MC-ESO の変更 → `docs/mceso.md`、ベースライン手法の変更 → `docs/baselines.md`、ベンチマーク関数・実行・条件・評価基準の変更 → `docs/experiments.md`、試行錯誤・フラグの追加削除 → `docs/history.md`、関連研究の調査結果 → `docs/related_work.md`、web 配下の変更 → `docs/web.md`、プロジェクト概要・docs リンク構成の変更 → ルート `README.md`。

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
- 手法の検証・比較・分析は「比較手法設定 → `./run.sh quick` 実行 → monitor → `scripts/analyze_quick.py` で 3 指標分析 → 判定」の順で行う。比較手法は `--methods` で必要分だけに絞る（不要なベースラインを回さない）。詳細は [docs/experiments.md の評価の分析・自動化](docs/experiments.md#評価の分析自動化) を参照。

## Claude の構成（`.claude/`）

`.claude/agents/` と `.claude/commands/` は**版管理下にある**（`.claude/` 配下の他はローカル状態なので ignore）。エージェントやコマンドを足したら commit すること。

| コマンド | 中身 | 使いどき |
|---|---|---|
| `/status` | `./run.sh loop` の要約 | 現況を知りたいとき。判断待ち事項が最初に出る |
| `/professor` | `professor` エージェント | 方針を決める前、路線に踏み込む前、手が動いているのに前進感がないとき |
| `/summary` | `librarian` エージェント | 記録が実態からずれた疑いがあるとき。**主張せず条件つきで記述するだけ** |
| `/tidy` | `curator` エージェント | `scripts/` や `analysis/` が散らかったとき、路線を畳んだ直後、マージ前 |
| `/docs-check` | `docs-keeper` エージェント | コード変更後、マージ前、記述が古い疑いがあるとき |
| `/merge-loop` | `research-loop` → `main` | 1 日 1 回を目安 |

エージェントの役割は意図的に分けてある。`professor` は**判断する**（意義・新規性・弱点）、`librarian` は**記述するだけ**（推奨も順位づけもしない）、`curator` は**片付ける**、`docs-keeper` は**記述とコードを一致させる**。判断と記述を同じ担当に混ぜると、記録が主張に寄って読めなくなる。

## 研究ループの役割分担

`research-loop` ブランチ上で自動サイクルが回っている。詳細は [docs/research_loop.md](docs/research_loop.md)、現況は [docs/status.md](docs/status.md)（毎回上書き）。

| 役割 | 担当 | 頻度 | 権限 |
|---|---|---|---|
| 決定 | ユーザー | 随時 | `docs/research_loop.md` の「方針」欄。**すべてに優先** |
| 俯瞰（研究者） | クラウド review ルーチン | 1 日 1 回 | `status.md` を書き直し、問いのキューを書き換える。**実験はしない。ゴールは変えず提案する** |
| 実行 | クラウド execute ルーチン | 2 時間ごと | キュー先頭を測る。**並べ替え・ゴール変更・status.md 編集は禁止** |
| 対話 | この Claude Code セッション | 随時 | 報告と方針修正の補助 |

## ファイルを増やすときの規約

自動サイクルが 2 時間ごとに書き込むので、放置すると生データがリポジトリを埋める（実際に差分の 99.7% が機械の吐いた CSV になった）。

- **行単位の生 CSV は `.csv.gz`。** 生のまま置いてよいのは数百行までの集計結果。
- **数値は必ず `docs/` に書き出してから commit する。** CSV は再解析用の控えであって、結論の置き場ではない。
- **路線を畳んだら、その生データも消す。** 結論は docs にあり、git 履歴からも取り出せる。
- `analysis/` はテーマごとにサブディレクトリを切る。**トップレベルには置かない。**
- 閉じたテーマのスクリプトは `scripts/<theme>/` へ移す。`scripts/` 直下は現行のキューが呼ぶものだけ。
- `research-loop` は 1 日 1 回を目安に `main` へマージする（`/merge-loop`）。溜めると差分が読めなくなる。

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
- MC-ESO の既存手法との差別化・新規性の現状整理は [docs/mceso.md](docs/mceso.md)、過去に試した工夫の採否は [docs/history.md](docs/history.md) を参照（同じ検証の再実施を避ける）。

### 手法比較・評価の基準（遵守ルール）

指標の定義・読み方の詳細は [docs/experiments.md の評価方法論](docs/experiments.md#評価方法論) を参照。作業時に必ず守るルールは以下:

- 手法を比較・評価する際は、必ず **3 指標**（SR / 平均評価回数 `evals_succ_mean` / Wilcoxon 検定）を揃えて報告する。単一指標（SR のみ等）での判定は不可。
- **主指標は SR@1e-10**（最高精度）。補助的に SR@1e-2/1e-4/1e-7 も併記する。
- **SR@1e-10 を下げる構成は採用しない**（多解探索の改善でも例外なし。最低 1 解は深精度到達を保証）。
- **評価は原則 2 次元 BBOB-24（F01-F24）のみで集計・判定する**。**関数別の改善・悪化を必ず列挙**する（regression の見落としを防ぐ）。
- **Custom ベンチ（C01-C11）は評価の主対象ではない**。多峰・多解性能など特定の目的を重視して確認したいときにのみ `--custom` で追加参照する（判定の採否は原則 BBOB-24 で決める）。
- Wilcoxon は `MC-ESO` を reference とし α=0.05、A12 効果量も併記。

### 評価範囲・実行コマンドの基準（判定は quick n=20 / eval=5000 で統一）

- **手法の検証・評価はすべて `./run.sh quick --all --n-runs 20 --max-evals 5000` で行う**（quick のデフォルトが n_runs=20 / max_evals=5000 なので `--all` だけで可）。`--all` は **2 次元 BBOB-24（F01-F24）のみ**を回す。quick-12 サブセットでの判定は不可。
- **Custom ベンチ（C01-C11）を確認したい場合のみ `--all --custom` を使う**（多峰・多解など特定目的の参照用。判定の採否は原則 BBOB-24 で決める）。Custom 単独は `--funcs C01,C02,...` で選択できる。
- **GitHub Actions の `./run.sh trigger`（n=100）は裏で補助的に回す実験であり、手法の検証・評価では参照しない。** 評価の根拠は常に上記 quick n=20 の結果とする。
- 結果は `results/YYYYMMDD_HHMMSS_<label>_quick/dim{N}/{summary,wilcoxon}.csv` から 3 指標を集計して報告する。
- ベンチマーク関数・指標カラムの詳細は [docs/experiments.md](docs/experiments.md) を参照。
