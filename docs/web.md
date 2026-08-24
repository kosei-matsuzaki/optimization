# Results UI（Flask）

実験の起動・結果閲覧・可視化をブラウザから行う Flask アプリ。`results/` 配下に保存された実験結果を読み取り、Quick Run / GitHub Actions のトリガー・ダウンロードまでを一画面で管理する。プロジェクト全体の概要は [../README.md](../README.md)、最適化手法は [mceso.md](mceso.md) / [baselines.md](baselines.md)、ベンチマーク・実行は [experiments.md](experiments.md) を参照。

実装は `web/`（`app.py` + `app_lib/` + `static/` + `templates/`）。

---

## 起動

```bash
./run.sh ui          # → http://localhost:8080
# または
python3 web/app.py
```

開発サーバ（`debug=True`）で動作する。ホットリロード時にメモリ上のジョブ状態（実行中の Quick Run / ダウンロード）はリセットされるため、リロードをまたぐ進捗復元はブラウザ側の `localStorage` ＋ `.quick.pid` で補完している。

> 結果データは `results/YYYYMMDD_HHMMSS_<commit>/` を直接読む。UI 自体は結果を生成せず、Quick Run は `quick_check.py` をサブプロセスとして起動するだけ。`main.py`（本番実験）はローカルでは実行しない（リポジトリ全体のルール）。

---

## 主な機能

| 機能 | 説明 |
|---|---|
| Quick Run | `quick_check.py` をバックグラウンド実行。手法・関数セット・次元（BBOB dim 2/3/5/10/20）をモーダルで指定し、ライブターミナル出力を表示。`max_evals` は 50000 まで指定可（dim20 の 2500×d=50000 予算に対応）。次元タブは数値順（dim2→dim20）で並ぶ |
| GitHub Actions Trigger | `gh` CLI 経由でワークフローをトリガー |
| Remote Runs | 最新 10 件のワークフロー実行を一覧表示。完了済みは進捗バー付きでダウンロード可能 |
| Local Results | `results/` 配下の結果一覧。名前変更・削除・実行中ジョブの停止に対応 |
| 結果詳細 | 次元タブ・関数タブで切替え。Landscape / Convergence / Evals / Population 等の図を表示 |
| Summary テーブル | 手法別の成績を色分け表示（best=緑、worst=赤）。ヘッダークリックでソート可能。**SR@target** 列は 1e⁻⁴ / 1e⁻⁷ / 1e⁻¹⁰ の 3 目標を色付きヒートマップ（濃い緑=高 SR、数値=正確な%）で並べる（1e⁻¹⁰が主指標）。行を展開すると各 seed が目標ごとに ✓（到達）/✗（未到達）で表示される（旧「ECDF profile」ミニバーを置換） |
| 全体評価ナビ（左タブ 3 分割） | 左サイドバーの「全体評価」を **ランキング / 成績詳細 / 統計的優位差** の 3 エントリに分割（`#overall-nav`）。各エントリが 1 カードに対応し、選択中のカードのみ表示（評価範囲セレクタは 3 ビュー共通で常時表示）。URL ハッシュは `#dim2/__overall__/<view>` で永続化しリロードでも同じサブビューに戻る |
| 評価範囲セレクタ（suite scope） | 全体評価の先頭に **BBOB / Custom /（あれば CEC2022）/ 全体（混在）** の切替バーを表示。選択スイートに応じて**ランキング・SR mean・カテゴリ別/形状タグ別・関数別・Wilcoxon をすべて再集計**する（BBOB と Custom を混ぜた平均を出さない）。バックエンドが関数名プレフィックス（F/C/G）でスイート分割し `by_suite` として全ペイロードを返すため、Friedman χ²_F / Nemenyi CD もスイート内の関数数で正しく再計算される。単一スイートのみの次元（dim3/4 は BBOB のみ等）ではバー非表示 |
| Overall ランキング | 全関数横断の Friedman 平均順位を **best_f（全 run 平均 mean_best_f）/ Evals（succ-only mean）** の 2 列で表示（並べ替えは Evals→SR）＋ Nemenyi 臨界差。SR は **SR@1e-10（主指標）/ SR@1e-4（補助）/ PR@1e-4（多解の最適点発見率）** の 3 列を併記（ECDF ランクは成績詳細ビューに残置せず非表示） |
| 成績詳細（統合ビュー） | **カテゴリ別（BBOB 公式グループ）・形状タグ別・関数別**の内訳を **指標セレクタ**（SR@1e-10 / SR@1e-4 / PR@1e-4＝% ヒートマップ、best_f / Evals＝順位チップ）で切替。**形状タグ別は関数×タグ対応マトリクスと手法集計を 1 つに統合**: タグ列を軸（modality / separability / …）でグループ化し、上段=各手法のそのタグを持つ関数群での集計値、下段=どの関数がそのタグを持つか（● の対応表、`/benchmarks` と同じ配色）。これにより「各手法がどの形状に強い/弱いか」と「そのタグを構成する関数」を同一の列上で読める。手法は選択指標の平均で並べ替え。**関数のタグ対応（下段）の関数名は F01 等のスイート番号を前置し、クリックでその関数の関数別ビュー（`selectFunc`）へ遷移する** |
| Per-run Stats | 各 run の詳細統計（成功 / 失敗を色分け） |

### ビューモード（結果詳細画面）

右上の `[Function] [Method] [Compare]` タブでビューを切り替える。

| モード | 説明 |
|---|---|
| **Function** | 関数を選択 → 選択した可視化タイプを全手法グリッドで表示 |
| **Method** | 手法を選択 → 選択した可視化タイプを全関数グリッドで表示 |
| **Compare** | 関数・手法をマルチセレクト → 関数×手法のマトリクスグリッドで比較 |

---

## ディレクトリ構成

```
web/
├── app.py                 # Flask アプリ（ルーティングのみの薄い層）
├── app_lib/               # バックエンドロジック
│   ├── __init__.py
│   ├── config.py          # パス・GitHub 定数（sys.path に project root を追加）
│   ├── results.py         # results/ のデータ層（一覧・メディア索引・集計・ランキング）
│   └── jobs.py            # Quick Run / アーティファクトDL のバックグラウンドジョブ＋状態
├── static/                # 静的アセット（CSS / JS）
│   ├── style.css          # 共通スタイル（全ページ共有）
│   ├── modal.js           # 共通ダイアログ（alert / confirm / prompt の置換）
│   ├── index.css  / index.js     # トップ画面
│   ├── result.css / result.js    # 結果詳細画面
│   └── methods.css / methods.js  # 手法解説画面
└── templates/             # Jinja2 テンプレート
    ├── base.html          # 共通レイアウト（<head> / <header> を集約）
    ├── index.html         # トップ画面（Quick Run / GH Actions / 結果一覧）
    ├── result.html        # 結果詳細画面（可視化・テーブル・ランキング）
    └── methods.html       # 手法解説画面（MC-ESO）
```

---

## アーキテクチャ

責務ごとに 3 層へ分離している。`app.py` はリクエストを受けて `app_lib` に委譲するだけの薄いルーティング層に保つ。

| 層 | 役割 | 依存 |
|---|---|---|
| `app.py` | ルート定義・フォーム検証・レスポンス整形 | `app_lib.*` |
| `app_lib/results.py` | `results/` の読み取り専用データ層（Flask 非依存・純粋関数中心） | `config` |
| `app_lib/jobs.py` | バックグラウンドジョブ（Quick Run / DL）とインメモリ状態・PID ファイル管理 | `config`, `results` |
| `app_lib/config.py` | パス・GitHub 定数。import 時に project root を `sys.path` へ追加し `core` / `quick_check` を可搬に保つ | — |

### テンプレート（Jinja 継承）

全ページが `base.html` を `{% extends %}` し、重複していた `<head>` と `<header>` を一箇所へ集約。各ページはブロックの上書きのみで差分を表現する。

| ブロック | 用途 |
|---|---|
| `title` | `<title>` |
| `head` | ページ固有の CSS / JS / 外部 CDN（KaTeX・フォント等） |
| `header_back` / `header_title` / `header_nav` | ヘッダーの戻るボタン・見出し・ナビ（methods はナビ無し） |
| `content` | 本文 |
| `scripts` | `#page-data`（Jinja → JSON 埋め込み）＋ ページ固有 JS |

### CSS / JS の分離

各ページのスタイル・スクリプトはインライン記述をやめ `static/` へ分離。サーバ側のデータは `<script id="page-data" type="application/json">` に `{{ ... | tojson }}` で埋め込み、JS 側は `JSON.parse` で読み出す（テンプレートとロジックを疎結合に保つ）。`style.css` / `modal.js` のみ全ページ共有。

---

## ルート / API 一覧

### ページ

| メソッド | パス | 説明 |
|---|---|---|
| GET | `/` | ダッシュボード（結果一覧・Quick Run・GH Actions） |
| GET | `/benchmarks` | ベンチマーク関数 × 形状タグ 対応マトリクス（run 非依存の静的リファレンス。`SHAPE_TAGS` / `TAG_AXES` 由来。関数を行・タグを列とし、軸ごとに色分け。ヘッダ nav からアクセス） |
| GET | `/methods` | MC-ESO 手法解説ページ |
| GET | `/results/<run_id>` | 結果詳細ページ |
| GET | `/media/<path>` | `results/` 配下の図・ファイル配信 |

### API

| メソッド | パス | 説明 |
|---|---|---|
| GET | `/api/methods` | 利用可能な最適化手法名 |
| GET | `/api/functions` | ベンチマーク関数（カテゴリ別）＋ quick-12 プリセット |
| POST | `/api/run` | Quick Run を開始 → `job_id` |
| GET | `/api/status/<job_id>` | Quick Run の状態・出力 |
| POST | `/api/stop/<job_id>` | Quick Run を停止 |
| GET | `/api/shell-job` | `run.sh quick`（シェル起動ジョブ）の検出 |
| POST | `/api/shell-stop` | シェル起動ジョブの停止 |
| POST | `/api/gh-trigger` | GitHub Actions ワークフローをトリガー |
| GET | `/api/gh-runs` | 最新のワークフロー実行履歴 |
| POST | `/api/download` | アーティファクトのダウンロードを開始 → `job_id` |
| GET | `/api/dl-status/<job_id>` | ダウンロード進捗 |
| GET | `/api/results` | 結果一覧＋メタ＋実行中ディレクトリ |
| POST | `/api/results/<run_id>/rename` | 結果ディレクトリ名の変更 |
| DELETE | `/api/results/<run_id>` | 結果ディレクトリの削除 |
| GET | `/api/stats/<run_id>/<dim>/<func>` | per-run 詳細統計 CSV |
| GET | `/api/media-index/<run_id>/<dim>` | 可視化ファイルの索引 |
| GET | `/api/result-data/<run_id>` | 次元・関数・summary・wilcoxon |
| GET | `/api/overall/<run_id>/<dim>` | 全関数横断の Friedman ランキング。`scopes`（例 `["bbob","custom","all"]`）と `by_suite`（各スイートの完全ペイロード: leaderboard / friedman / func_categories / func_tags / func_scores …）を返す。トップレベルは後方互換のため既定スコープ（混在時は `all`）のペイロードを併載 |

---

## 開発メモ

- ルートは `app.py` に集約し、ロジックは `app_lib` に置く（`app.py` を肥大化させない）。
- `results.py` は副作用の無い読み取り中心に保ち、ジョブ実行・ファイル移動などの副作用は `jobs.py` に閉じ込める。
- 新しいページを追加するときは `base.html` を継承し、CSS / JS を `static/<page>.css`・`static/<page>.js` に分け、テンプレートにインラインで書かない。
- フロントへ渡すサーバデータは `#page-data` 経由で受け渡す（テンプレート内に Jinja 式を含む `<script>` を増やさない）。
- 結果詳細画面の選択状態は URL ハッシュ `#dim<N>/<func>` で永続化する。全体評価ビューは `<func>` に `__overall__` を割り当てており（`selectOverall()` で `_updateHash()` を呼ぶ）、15 秒ごとの auto-sync ポーリングやリロードでも関数別ビューへ勝手に遷移しない。
