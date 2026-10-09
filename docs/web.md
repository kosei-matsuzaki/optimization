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

開発サーバ（`debug=True`）で動作する。ホットリロード時にメモリ上のジョブ状態（実行中の Quick Run / ダウンロード）はリセットされるため、リロードをまたぐ進捗復元はブラウザ側の `localStorage` ＋ `.quick.pid` で補完している。Windows（Git Bash）では `.quick.pid` が MSYS の PID で Python から扱えないため、`run.sh quick` は同じジョブのネイティブ PID を `.quick.winpid` にも書き、web 側は Windows ではこちらだけを読んで `OpenProcess` で生存確認する（Windows の `os.kill(pid, 0)` は生存確認ではなく TerminateProcess になるので使わない）。

> 結果データは `results/YYYYMMDD_HHMMSS_<commit>/`（この PC だけ、図あり）と `runs/<run>/`（git で共有、数値のみ・CSV.gz）の両方を読む（`app_lib/results.py` の `run_path` / `_open_csv`）。同名なら `results/` を優先。`runs/` だけにある run は一覧で「共有」と表示し、名前変更・削除の API は拒否する（git で扱う）。UI 自体は結果を生成せず、Quick Run は `quick_check.py` をサブプロセスとして起動するだけ。`main.py`（本番実験）はローカルでは実行しない（リポジトリ全体のルール）。

---

## 主な機能

| 機能 | 説明 |
|---|---|
| Quick Run | `quick_check.py` をバックグラウンド実行。手法・関数セット・次元（BBOB dim 2/3/5/10/20）をモーダルで指定し、ライブターミナル出力を表示。`max_evals` は 50000 まで指定可（dim20 の 2500×d=50000 予算に対応）。次元タブは数値順（dim2→dim20）で並ぶ |
| GitHub Actions で実行 | `gh` CLI 経由でワークフローをトリガー（`gh` が無い環境では JSON のエラーを返し、画面に理由を出す） |
| GitHub Actions 履歴 | ダッシュボード右列に最新 10 件のワークフロー実行を一覧表示。成功したものは「取り込む」で進捗バー付きダウンロード |
| Local Results | `results/` 配下の結果一覧。名前変更・削除・実行中ジョブの停止に対応 |
| 結果詳細 | 次元タブ・関数タブで切替え。Landscape / Convergence / Evals / Population 等の図を表示 |
| Summary テーブル | 手法別の成績を色分け表示（best=緑、worst=赤）。ヘッダークリックでソート可能。**SR@target** 列は 1e⁻⁴ / 1e⁻⁷ / 1e⁻¹⁰ の 3 目標を発散型ヒートマップ（赤=低 SR → 中立 → 緑=高 SR、数値=正確な%。`result.js` の `heatColor`）で並べる（1e⁻¹⁰が主指標）。行を展開すると各 seed が目標ごとに ✓（到達）/✗（未到達）で表示される（旧「ECDF profile」ミニバーを置換） |
| 全体評価ナビ（左タブ 3 分割） | 左サイドバーの「全体評価」を **ランキング / 成績詳細 / 統計的優位差** の 3 エントリに分割（`#overall-nav`）。各エントリが 1 カードに対応し、選択中のカードのみ表示（評価範囲セレクタは 3 ビュー共通で常時表示）。URL ハッシュは `#dim2/__overall__/<view>` で永続化しリロードでも同じサブビューに戻る |
| 評価範囲セレクタ（suite scope） | 全体評価の先頭に **BBOB / Custom /（あれば CEC2022）/ 全体（混在）** の切替バーを表示。選択スイートに応じて**ランキング・SR mean・カテゴリ別/形状タグ別・関数別・Wilcoxon をすべて再集計**する（BBOB と Custom を混ぜた平均を出さない）。バックエンドが関数名プレフィックス（F/C/G）でスイート分割し `by_suite` として全ペイロードを返すため、Friedman χ²_F / Nemenyi CD もスイート内の関数数で正しく再計算される。単一スイートのみの次元（dim3/4 は BBOB のみ等）ではバー非表示 |
| Overall ランキング | 全関数横断の Friedman 平均順位を **best_f（全 run 平均 mean_best_f）/ Evals（succ-only mean）** の 2 列で表示（並べ替えは Evals→SR）＋ Nemenyi 臨界差。SR は **SR@1e-10（主指標）/ SR@1e-4（補助）/ PR@1e-4（履歴）** の 3 列を併記（ECDF ランクは成績詳細ビューに残置せず非表示）。**PR@1e-4（履歴）は `summary.csv` の `pr_1e-4`**＝既知の大域最適 K 点のうち評価履歴に半径内かつ f ≤ 1e-4 の点がある割合（`core/runner.py:peak_metrics`、[experiments.md の多解（MMO）報告](experiments.md#多解mmo報告)）。BBOB は K=1 なので SR@1e-4 とほぼ同じ値になり、K>1 なのは Custom の C01-C03 だけ。一時停止中の多解路線が使う CEC2013 報告集合の `cec_pr_*` とは別物で、UI はそちらを表示しない |
| 成績詳細（統合ビュー） | **カテゴリ別（BBOB 公式グループ）・形状タグ別・関数別**の内訳を **指標セレクタ**（SR@1e-10 / SR@1e-4 / PR@1e-4（履歴）＝% ヒートマップ、best_f / Evals＝順位チップ）で切替。**形状タグ別は関数×タグ対応マトリクスと手法集計を 1 つに統合**: タグ列を軸（modality / separability / …）でグループ化し、上段=各手法のそのタグを持つ関数群での集計値、下段=どの関数がそのタグを持つか（● の対応表、`/benchmarks` と同じ配色）。これにより「各手法がどの形状に強い/弱いか」と「そのタグを構成する関数」を同一の列上で読める。手法は選択指標の平均で並べ替え。**関数のタグ対応（下段）の関数名は F01 等のスイート番号を前置し、クリックでその関数の関数別ビュー（`selectFunc`）へ遷移する** |
| Per-run Stats | 各 run の詳細統計（成功 / 失敗を色分け） |

### ビューモード（結果詳細画面）

左サイドバーの「関数・手法ごとに見る」の `[関数] [手法] [比較]` でビューを切り替える。

| モード | 説明 |
|---|---|
| **Function** | 関数を選択 → 下の「図（関数ビュー）」の 3 つの対話的な図と、`--viz` の run なら画像タブ |
| **Method** | 手法を選択 → 選択した可視化タイプを全関数グリッドで表示 |
| **Compare** | 関数・手法をマルチセレクト → 関数×手法のマトリクスグリッドで比較 |

### 図（関数ビュー、2026-10-09 から）

画像を run の最後に焼く方式から、**データを保存して UI が描く**方式に移した（`static/viz.js`）。図はすべての run で見られる（画像の無い run、共有 `runs/` の run でも）。

| タブ | 中身 | データの出どころ |
|---|---|---|
| **収束** | 手法ごとの f − f* の中央値（全 run）、MC-ESO の四分位の帯、1e-10 の線。凡例にカーソルで手法を強調、図の上にカーソルでその評価回数の値を一覧 | run が書く `dim{N}/curves/{Func}.json.gz`（`core/run_data.save_curves`） |
| **探索の様子** | 手法と run（全 seed、到達 / 未到達つき）を選び、評価回数のスライダーと再生で動かす。地図: 評価点（MC-ESO は子を作った経路で色分け）・集団（2D は宿主ごとの σ の円）・最良点の軌跡・☆ 大域最適。横に、評価ごとの f − f* と最適解までの距離（どちらも全評価、次元に依らない）、座標ごとの集団の広がりのヒートマップ | **その run を再実行**（`/api/replay`）。背景は `/api/landscape` |
| **MC-ESO の内部** | 全 run の表（ルート、確定時点、スピルオーバー・盆地乗換えの回数、経路の割合、集団サイズ）。行を押すとその run の内部状態: 最良値、σ（drilling を灰色）、集団サイズ、系統の数、子の作り方の割合、ルーター信号 3 つ（固有値比・軸への揃い・座標方向の隙間）と閾値、学習共分散の条件数（3 次元以上）、停滞カウンタ。縦線がスピルオーバー / 盆地乗換え / 最初に掘り切った時点 | 表は run が書く `dim{N}/mceso_runs.csv`（`core/run_data.append_mceso_runs`）、内部状態は再実行の `trace` |

- **3 次元以上**: 地図は横軸・縦軸の座標を選んだ 2 座標への投影で、背景は**最適解を通るその 2 座標の断面**（他の座標は最適解の値に固定）。f − f* と距離の散布図、座標ごとの広がりは次元に依らず読める。
- **再実行**: run は seed = run 番号 × 100 で決まるので、同じ seed で回し直せばその run になる（2D で 1〜2 秒、10D で数秒、結果はサーバーのメモリに 24 件までキャッシュ）。**記録された最終値と比べ、一致すれば「再実行・記録と一致」、違えば警告を出す**。違うときは run の `env.json` とこの PC の環境を比べ、別の環境なら「別の環境の run・表示はこの PC での再実行」（環境が違うと同じ seed でも経路が分かれる）、同じ環境なら「コードが変わったか seed で再現しない手法」と出し分ける。サイドバーの run 情報にも計算環境（OS / CPU / BLAS）を出す。
- 関数地形は run ごとに作らない。`/api/landscape` がその場で格子を計算し（120×120、2D で 0.1 秒以下）、サーバーのメモリにキャッシュする。
- URL の `?view=viz_search` などで開くタブを指定できる。手法・比較ビューは従来どおり画像（`--viz` の run のみ）。

---

## 見た目の方針

`static/style.css` の `:root` にトークンを集約している（地色 `--bg`、文字 `--text`、罫線 `--border`、提案手法・成功・主操作の深緑 `--accent`、有意に劣る・失敗の煉瓦色 `--danger`）。書体は IBM Plex Sans JP（数値は tabular-nums）、run 名・commit・ターミナルだけ IBM Plex Mono（Google Fonts、オフライン時はシステムフォントにフォールバック）。

- **提案手法の行を強調する**: ランキング・成績詳細・関数別テーブルでは Wilcoxon の reference 手法（既定 `MC-ESO`、`result.js` の `refMethod()`）の行に深緑の帯を付ける。順位の金銀銅は使わない。
- **成功率の色は失敗側に色を載せる**: 100% は淡い緑、低いほど煉瓦色へ寄る発散型（`heatColor`）。赤緑の色相回転は使わない。
- ヘッダーは全ページ共通で `base.html` が持ち、ナビの現在地は `request.path` から決める（各ページは `header_title` だけを上書きする）。

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
│   ├── benchmarks.css            # ベンチマーク一覧画面
│   ├── methods.css / methods.js  # 手法解説画面（成績表・ヒートマップ・系譜図は JS が描く）
│   └── methods_data.json         # 手法解説の成績データ（scripts/web/methods_data.py が生成）
└── templates/             # Jinja2 テンプレート
    ├── base.html          # 共通レイアウト（<head> / <header> を集約）
    ├── index.html         # トップ画面（Quick Run / GH Actions / 結果一覧）
    ├── result.html        # 結果詳細画面（可視化・テーブル・ランキング）
    ├── benchmarks.html    # ベンチマーク関数 × 形状タグ一覧
    └── methods.html       # 手法解説画面（MC-ESO の仕組み・成績・比較手法の系譜と解説）
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
| GET | `/methods` | 手法解説ページ。MC-ESO の考え方・1 世代の流れ・仕組み、quick の成績（次元タブ・関数別ヒートマップ）、比較手法の系譜図（発表年 × 系統、JS が `methods.js` の `NODES` / `EDGES` から描く）と系統別の解説カード |
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
| GET | `/api/curves/<run_id>/<dim>/<func>` | 収束の中央値・四分位（`curves/{Func}.json.gz`） |
| GET | `/api/mceso-runs/<run_id>/<dim>` | MC-ESO の全 run の要約（`mceso_runs.csv`） |
| GET | `/api/replay/<run_id>/<dim>/<func>/<method>/<seed>` | その run を再実行して全評価点（float32 base64）、f − f*、最適解までの距離、経路、集団（最大 240 コマ）、MC-ESO の内部状態、記録との一致を返す（`app_lib/replay.py`） |
| GET | `/api/landscape/<dim>/<func>?a=&b=` | 最適解を通る x_a–x_b 平面の f の格子（log10、120×120） |
| GET | `/api/media-index/<run_id>/<dim>` | 可視化ファイルの索引 |
| GET | `/api/result-data/<run_id>` | 次元・関数・summary・wilcoxon |
| GET | `/api/overall/<run_id>/<dim>` | 全関数横断の Friedman ランキング。`scopes`（例 `["bbob","custom","all"]`）と `by_suite`（各スイートの完全ペイロード: leaderboard / friedman / func_categories / func_tags / func_scores …）を返す。トップレベルは後方互換のため既定スコープ（混在時は `all`）のペイロードを併載 |

---

## 手法解説ページの成績データ

`/methods` の数値は手で書かず、quick の CSV から `scripts/web/methods_data.py` で `web/static/methods_data.json` を作って読む（集計は `scripts/analyze_quick.py` と同じ）。新しい quick を載せるときは:

```bash
python scripts/web/methods_data.py results/<2D の run> results/<5D の run> results/<10D の run>
```

本文の「読みどころ」（既存手法との比較、旧既定からの変化）も JSON から JS が組み立てるので、数値と文章がずれない。比較手法を足したら `methods.html` に解説カード（`id="m-<登録名>"`、`data-method="<登録名>"`）を、`methods.js` の `NODES` / `EDGES` に系譜を足す。

## 開発メモ

- ルートは `app.py` に集約し、ロジックは `app_lib` に置く（`app.py` を肥大化させない）。
- `results.py` は副作用の無い読み取り中心に保ち、ジョブ実行・ファイル移動などの副作用は `jobs.py` に閉じ込める。
- 新しいページを追加するときは `base.html` を継承し、CSS / JS を `static/<page>.css`・`static/<page>.js` に分け、テンプレートにインラインで書かない。
- フロントへ渡すサーバデータは `#page-data` 経由で受け渡す（テンプレート内に Jinja 式を含む `<script>` を増やさない）。
- 結果詳細画面の選択状態は URL ハッシュ `#dim<N>/<func>` で永続化する。全体評価ビューは `<func>` に `__overall__` を割り当てており（`selectOverall()` で `_updateHash()` を呼ぶ）、15 秒ごとの auto-sync ポーリングやリロードでも関数別ビューへ勝手に遷移しない。
