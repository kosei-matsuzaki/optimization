# 研究ループ（2026-09-29 から 運用 A）

このファイルは、クラウドの測定ルーチンと対話セッションの間の引き継ぎ書である。**上限 300 行**（`scripts/check_doc_size.py` が検査する）。

- 現在のテーマは **MC-ESO の単一解性能（1 つの最適解をどれだけ精度よく速く見つけるか）の向上**。
- 確定した測定結果は [findings.md](findings.md)、手法の工夫の採否は [history.md](history.md)、いま何が言えるかは [status.md](status.md)。
- 一時停止中の多解路線（peak ratio）は [archive/multisolution.md](archive/multisolution.md) に要約がある。全文は git タグ `archive/multisolution-2026-09-29`。

---

## 運用（2026-09-29 のユーザー決定 ＝ 案 A）

| 役割 | 担当 | 頻度 | やること | やらないこと |
|---|---|---|---|---|
| **決定** | ユーザー | 随時 | 下の「方針」欄に書く。**すべてに優先する**。既定値の採用を承認する | — |
| **設計・判定** | 対話セッション（Claude Code） | 随時 | 診断・腕の実装・判定・既定値の変更案。**下の「測定ジョブ」を書く**。記録の統合と docs の上限の維持 | ユーザーの承認なしに既定値を変えない |
| **測定** | クラウドの測定ルーチン | 1 時間ごと（キューが空ならすぐ終了する。作業中のジョブがあれば何もせず終了する。2026-10-13 のユーザー指示） | 「測定ジョブ」の先頭を、書かれたとおりに回して集計し、結果を記録する。**正準環境**（ローカルの macOS では base が再現しない。原因は未特定） | 問いを選ぶ・並べ替える・腕を作る・既定値を変える・採否を判断する |
| **現況まとめ** | クラウドの俯瞰ルーチン | 週 1 回 | [status.md](status.md) と [report.html](report.html) を書き直す。次に測るべきものを status.md に**提案**する | 実験。キューの編集 |

**なぜこの形にしたか**: 旧運用（6 時間ごとに 1 問を自律で選んで測る）は、判断の絡む手法開発に向かなかった。
テーマ切替後の 6 サイクル（その177〜その182）のうち、手法の実験は 1 回だけだった。
しかも両ルーチンは既定値の変更を禁じられていたので、改良を採用する経路がそもそも無かった。
いまは判断を対話で速く回し、重い測定と正準環境での再現だけをクラウドに任せる。

---

## 方針（ユーザーが書く欄）

**この欄はユーザーが書く。すべてに優先する。**
09-08 / 09-11 の決定（多解路線の指示）は路線の一時停止に伴いアーカイブへ移した（[archive/multisolution.md](archive/multisolution.md)）。

### 2026-10-30（ユーザー決定）— `v4_k50fR_r50` を既定に／v5 として改善を続ける

1. **`v4_k50fR_r50`（隔離 50%・予報の子のルート限定・乗換え後の延命・回復 50）を既定にしてよい。** 資料（web・README など）もすべて更新する。旧既定は `MC-ESO-v3`、以後の腕は `v5_*`（新既定の上）。
2. **その後 v5 としてさらに改善に取り組む。** 10-13・10-14 の決定（CEC 順位を主に、合わせ込まない、既存手法へ寄せていかない）は引き続き有効。

### 2026-10-14（ユーザー決定）— 既存手法は「考え方」を参考にし、探索を丸投げしない

1. **既存手法（CMA-ES・DE など）の考え方を MC-ESO の機構に取り込むのはよい。切り替えの判断だけ MC-ESO が行い、残りの探索を既存手法に任せる（終盤を丸投げする）方向は取らない。** 予算を分け与える分不利で、大きな性能改善につながらないため。
2. **（2026-10-15 補足、ユーザー）pycma の波などを使うなということではない。** CMA-ES が得意な場面で引用するのは構わない。問題は研究の進め方で、**改良を重ねるごとに既存手法へ寄せていく（予算や局面を既存手法に渡す範囲を広げていく）ことは避ける。** 改良は MC-ESO 自身の機構（感染のチャネル・学習共分散・スピルオーバーなど）の側で行う。
   - これに従い、外部委譲を広げる v3 の腕（CMA 株の並走 `strain_*`、試し打ち `probe_*`、引き渡しの拡張）は畳む。既定の停滞時の波・封じ込めは残す（外す必要はない）。

### 2026-10-13（ユーザー決定）— `v2_scan2d` を既定に／v3 は CEC 順位を主に／特定のベンチマークに合わせ込まない

1. **`v2_scan2d` を既定にしてよい**（10D の一般化の検証の結果が出てから実装し、資料を更新する）。
2. **v3 としての改善は CEC 順位（run ごとの誤差、1e-8 未満は 0、の平均による順位）の改善を主に意識する。** SR@1e-10 は深すぎ、特定の関数で数値が良くなりがちなため。
3. **特定のベンチマークに特化しない**: 複数 instance での検証、20D より上での検証などを行う。

（対話セッションの運用案。上の決定の実装）v3 の腕は `--instances` で調整用 instance（1〜5）に run を散らして測り、CEC 順位は既定・腕と比較手法（ELSHADE-SPACMA・LSHADE-SPACMA・EBOwithCMAR・L-SHADE・IPOP-CMA-ES）を同じ run に並べて出す。5D / 10D / 20D を見て、採用の前に調整に使っていない instance（6〜10）と別 seed で確かめる。3 指標の報告は続ける。

### 2026-10-05（ユーザー決定）— 論文は手法の改良で書く／ablation と分析は進めてよい／ルーチンの間隔を一時的に短くする

1. **論文の主題は「診断」にしない。** MC-ESO の改良（性能の向上）を主題にする。負ける機序の測定は、改良の根拠として使う。
2. **ablation などの分析の実験は進めてよい。** professor の指摘（2D の勝ちがどの部品から来ているかが未測定）を受けて、2D の部品ごとの ablation をジョブ 1 に置いた。
3. **測定ルーチンの間隔を一時的に 2 時間ごとにする**（キューの 1〜3 が済むまで。済んだら 6 時間ごとに戻す）。

### 2026-09-29（ユーザー決定）— 運用を A に切り替える／docs の巨大化を止める

1. **運用を上の表のとおりに切り替える。** 手法の設計・判定は対話セッションで行い、クラウドは測定の代行と週 1 回の現況まとめに絞る。
2. **docs に行数の上限を設ける**（`scripts/check_doc_size.py`。上限は同スクリプトの表が正）。上限を超えたら、対話セッションが統合してから commit する。
3. **一時停止中の多解路線の記録（約 14,000 行）は、要約 [archive/multisolution.md](archive/multisolution.md) ＋ git タグ `archive/multisolution-2026-09-29` に置き換えた。** 09-27 の決定「後で論文化・再開できる形で消さない」は、タグで全文を固定することで守っている。

### 2026-09-27（ユーザー決定・最優先）— テーマを単一解性能の向上に切り替える。多解の路線は一時停止
**この項目は下の 09-08 / 09-11 の決定すべてに優先する。矛盾する箇所はこちらが勝つ。**
[status.md](status.md) の「ユーザーの判断を仰ぐこと」(1)(2) への回答を兼ねる。
**決定は 3 点:**
1. **主テーマを「MC-ESO が 1 つの最適解をどれだけ精度よく速く見つけるか」の向上に切り替える。** 実験と手法提案はこれを中心に行う。
   ゴール文は下の「この研究のゴール」に書き直した。
2. **09-08 の案 (C)「MC-ESO の変種を新しく作らない」は撤回する。** MC-ESO 本体の改良・新しい機構の提案を解禁する。
   採否は CLAUDE.md の規則どおり、**3 指標（SR / `evals_succ_mean` / Wilcoxon ＋ A12）を揃え、SR@1e-10 を下げる構成は採用しない。**
   09-11 の決定 (3) の「採用すると比較手法が変わる」という代償は、MC-ESO が再び提案手法になったので適用しない。
3. **多解（niching / peak ratio）の路線は一時停止する。** キューから外し、新しい測定も論文化の作業もしない。
   結論は [acceptance_topology.md](acceptance_topology.md) と [status.md](status.md) に残っているので消さない（後で論文化・再開できる形で置いておく）。
   **09-08 / 09-11 の方針のうち、新 suite・CEC2013・Score・peak ratio に関する指示は、この路線を再開するまで効力を止める。**
**評価の条件:**
- **判定の主対象は従来どおり 2 次元 BBOB-24**（`./run.sh quick --all`、n=20、5000 評価）。
- **5 次元・10 次元の BBOB-24 も必ず併記する**（`--dim 5 --max-evals 12500` / `--dim 10 --max-evals 25000`。予算は [experiments.md](experiments.md) の目安 `2500×D`）。
  高次元は伸びしろが最も大きい（10D の SR@1e-10 は 35% 程度という記録がある）ので、**改善の対象に含める**。
  **2D で SR@1e-10 を下げる変更は、高次元で上がっても採用しない。**
- **比較対象は CMA-ES / IPOP-CMA-ES / BIPOP-CMA-ES / DE / L-SHADE**（`--methods` で絞る）。高次元ではこれらと同じ条件で並べた比較がまだ無い。
- **関数別の改善・悪化を必ず列挙する。**
- **pin（CLAUDE.md の SR@1e-10 93.5% ／ evals 798）はこの環境で再現しない**（92.08% ／ 677.7）。
  **対照は同じ環境で測り直した base にすること。** pin の数値自体は書き換えない。
**進め方:**
- **最初は現在地の測定から入る**（2D の基準の測り直し、5D/10D での各手法との比較）。どの関数・次元で負けているかが分かるまで、新しい機構を作らない。
- **次に失敗の診断をする。** 負けている関数ごとに、失敗の型（精度が足りない／別の盆地に収束する／評価回数が多い）を分ける。
- **改善案は [history.md](history.md) を照合してから出す。** 同じ工夫の再検証を避けるため（V2a の UCB チャネル配分、チャネル比のスケジューラ、`migratory` などは不採用の記録がある）。
- **先行研究の調査は引き続き実装前に行う**（手順 3）。新規性の基準は 09-08 の決定 (1)（組み合わせで見る）をそのまま使う。
- 1 回の計算が 40 分に収まらない場合は、次元・関数・手法で分割してよい（分割したと書く）。

---

## この研究のゴール（毎回、着手する問いをここに照らすこと）

**MC-ESO が 1 つの最適解を見つける精度と速さを、既存手法（CMA-ES 系・DE 系）より高める。**

- **主指標**: v3（2026-10-13〜）から CEC 式の平均誤差の順位（ユーザー決定）。SR@1e-2 / 1e-4 / 1e-7 / 1e-10、`evals_succ_mean`（比べる手法どうしで同じ関数集合を平均する）、Wilcoxon（MC-ESO を reference、α=0.05）＋ A12 を併記する。2026-10-12 までの主指標は SR@1e-10。
- **判定**: 調整は `--instances 1-5`、採用の前に instance 6-10 と別 seed（`--seed-offset`）で確かめる。2D / 5D / 10D（予算 `2500×D`）はローカル、20D 以上はクラウド。**2D の CEC 順位を下げる変更は採用しない**（2D の SR@1e-10 の大きな悪化も併せて報告する）。
- **比較対象**: 標準の 16 手法パネル（[baselines.md](baselines.md)）。CEC 順位は 10 手法パネル（`scripts/single/cec_panel.py`）で出す。
- **現在地**: 2026-10-13 から既定は `v2_scan2d`（[findings.md](findings.md) の 7・8 節）。18 手法の中で SR@1e-10 は 2D/5D/10D とも 1 位、CEC 順位は 2D 3 位・5D 1 位・10D 2 位、20D（8 手法）は最下位、CEC2022 は下位。v4 の候補 `switch_pace` は既定化を保留中（[history.md](history.md) 末尾）。
- **判定の注意**: `best_f` の Wilcoxon で large が出ても、SR が 100% の関数では主指標の何桁も下の差であることが多い（F06 がその例）。それだけを改善対象の根拠にしない。

**占有されている領域（単一解テーマ）**
- MC-ESO の部品は個別にはどれも既存（学習 C の rank-μ ＝ CMA-ES、リスタートでの C リセット ＝ IPOP/BIPOP、複数分布から子を作って選択に任せる ＝ DE/EDA（Sun et al. 2005））。主張できるのは組み合わせ（瞬時 EMNA 共分散と累積 CMA 共分散の同一世代並走）と次元ゲートだけ。
- 代理モデル支援の多目的 CMA-ES（Appl. Soft Comput. 2025）が、演算子のアンサンブルから候補を作り代理モデルで 1 つを選ぶ。MC-ESO の主張は「判別器を置かず、実評価の選択に裁定させる」側に置くこと。
- 2026 年の CMA-ES 変種・ハイブリッド総説（`10.1007/s11831-026-10557-z`）は未確認（クラウド環境から読めない）。通るまで「新規」「初めて」と書かない。
- **複数の手法の生成器を 1 つの共有集団で併用する骨格は占有されている**（2026-10-06 調査）。AMALGAM-SO（Vrugt, Robinson & Hyman, IEEE TEC 2009: CMA・GA・PSO・DE が同じ集団から子を作る。子の数は成功の度合いでリスタート間に再配分、選択は親子 2N からの上位選択）、HMHH（Grobler, Engelbrecht, Kendall ら, CEC 2010 / Inf. Sci. 2015: GA・PSO・DE・CMA-ES が共通の親集団を使い、新解が対応する親を置き換える）、MOS（LaTorre ら 2011）、LSHADE-SPACMA（2017）、UMOEAs（2014）、PAP（2010）。勢いのチャネル（PSO 型）を足すと「CMA＋DE＋PSO」と読まれ、HMHH との距離が縮む。この調査で見つからなかったのは、配分を成功率で学ばず地形の特徴で決める点、最悪 25% の枠に限り元の宿主とだけ比べる入れ替え、2 系統の共分散の同一世代並走。主張するなら AMALGAM-SO / HMHH との比較が要る。 **最も近いのは EA4eig（CEC 2022 優勝）**: 共有集団で CoBiDE・IDEbd・CMA-ES・jSO を併用し、枠ごとに貪欲に置換する。MC-ESO との差は配分の決め方（成功率のルーレット 対 地形ルーター）にほぼ限られる（[related_work.md](related_work.md)）。
- BBOB f24 の欺瞞ファネルは探索空間の約 70% を占める。再起動系はこの関数を解いていない（唯一 45% を出すのは再起動を持たない DE）。F24 の対策として再起動の腕を作らない。

---

## 測定ジョブ（上から実行する）

**書くのは対話セッション。** 各ジョブは、コマンド・集計・記録先まで書き切る。測定ルーチンが判断を足さなくて済むようにするため。
ジョブを取ったら行末に `(claimed YYYY-MM-DD HH:MM UTC)` を書いて即 push する。終えたらジョブを消し、下の「作業ログ」に結果を書く。
**標準の集計**: `scripts/analyze_quick.py` の [1] 全体表（SR@1e-2 / 1e-4 / 1e-7 / 1e-10、`evals_succ_mean` は比べる手法の両方で成功のある関数だけで平均）、関数別の SR@1e-10 の変化（全関数）、Wilcoxon の有意な関数（両方向、A12 つき）、参考として [5] CEC 式の平均誤差の順位（2026-10-13 追加）。

**2026-10-30 から `MC-ESO` は新しい既定（旧 `v4_k50fR_r50`）。** 旧既定は `MC-ESO-v3`。`v3_*` / `v4_*` の腕は旧既定の上で走る（記録した数値が再現する）。`v5_*` と `ctl_*` は新しい既定の上。j47 / j49 / j57 のパネルの `MC-ESO` は**旧既定**なので、新しい既定のパネル順位は同じ run に入れた `v4_k50fR_r50`（新しい既定と bit 一致）で読む。

1. **【j62 (a) v5 の 2 本目（5D/10D）】** 手法は上と同じ 6 つ、j50 と同じ 4 設定（label `j62_a5` / `j62_a5ho` / `j62_a10` / `j62_a10ho`、`--jobs 4`）。1 回の起動に収まらなければ設定ごとに書き戻してよい。**集計**: j61 (a) と同じ（新しい既定のパネル順位は 5D 4.12/4.15・10D 4.25/4.46）。2D は v5 の 2 本目の腕がすべて 3 次元以上でしか動かないので回さない。**記録**: `analysis/single/j62/`。採否の判断はしない。

**対話セッションでやること（測定ルーチンは取らない）**: いまは無し。

---

## 規則

**測定**
- 単一 seed で結論を出さない。集計から落ちた関数を必ず報告する（24/24 そろったかを書く）。
- 3 指標をそろえる。`evals_succ_mean` は、比べる手法の両方で成功のある関数だけで平均する。
- 比較手法（`cma` ライブラリ系）は run 間で決定的でない。MC-ESO は決定的。比較手法の 1 関数 5pt の揺れをコード差の証拠にしない。

**リポジトリに残すもの**
- 行単位の生 CSV は `.csv.gz`。生のまま置いてよいのは数百行までの集計結果。出力は `analysis/<テーマ>/<ジョブ名>/`。
- 数値は必ず docs に書き出してから commit する。`analysis/` はファイル数 約 400 を上限とする。
- **docs の行数の上限**は `scripts/check_doc_size.py` の表が正。commit 前に実行し、超えていたら作業ログに 1 行書く（統合は対話セッションが行う）。

**MC-ESO を壊さない**
- `SR@1e-10` を下げる構成は採用しない。既定値を変えるのはユーザーの承認を得た対話セッションだけ。
- 診断用の変種は既定値を変えない引数として足し、`quick_check.py` に腕として登録する。

**ブランチ**
- 測定ルーチンは `research-loop` に積み、各回の最初に `git merge --no-edit origin/main` で main を取り込む。衝突したら解決せずに止める。
- 対話セッションの作業は main から切ったブランチで行い、ユーザーがマージする。

---

## 作業ログ

### 2026-10-30 j62 (a20) 20D を完走 — `v5_wlam`（波の λ を CMA-ES 既定へ）がパネル CEC 順位 4.25・SR@1e-10 45.0% で、新しい既定（4.73 / 37.1%）を大きく上回る。F08 0→85%・F09 5→95% が効いた。`v5_cpl`（wlam ＋ chainR ＋ pmax3）4.35 がこれに次ぐ。`v5_wnoC` 4.77 と `v5_wsig` 4.94 は基準より上（悪い）
- **ランした分**: ジョブの 6 手法・`--all --dim 20 --max-evals 50000 --n-runs 20 --instances 1-5 --jobs 4` を**関数 3 shard に分けて逐次**（`j62_a20_a` F01-F08 15:41:07-15:57:22 **16.3 分**・`_b` F09-F16 -16:15:55 **18.5 分**・`_c` F17-F24 -16:30:57 **15.0 分**、合計 **49.8 分**）。3 本とも `result.json` が `status=done` / `set=all-d20`。連結後 **24/24 関数**（summary 144 行 = 24×6・wilcoxon 120 行 = 24×5・stats 24 ファイル・`mceso_runs.csv` 2880 行 = 6×24×20、(関数, 手法) の重複・欠落ゼロ、wilcoxon の reference はすべて `MC-ESO`）。連結後の run は `results/20261030_154107_j62_a20m_quick`
- **環境**: numpy 2.4.6 / cma 4.5.0、python 3.11.17（`run.sh` の `PY` ＝ `/usr/local/bin/python3`。`pip` は 3.13 向けなので `python3 -m pip` で入れ直した）。CPU Intel Xeon @2.80GHz（family 6 / **model 85 / stepping 7**、4 CPUs）、avx512f あり、BLAS は openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1`。**容器仕様は j61 (a20)・j59 (a20)・j47 と同じ**。`pynmmso` は空 stub を `PYTHONPATH` に置いた
- **再現の検算**: 基準 `MC-ESO` は j61 (a20) の同名の行と 24 関数 × 49 列（`mean_time_s` 以外）で **1176/1176 セル一致**。`v5_cp`（＝ `v5_chainR` ＋ `v5_pmax3`）は j61 の `v5_chainR` と **1150/1152 一致**で、差は F18 の `mean_best_f` / `median_best_f` の 2 セルだけ＝ j61 の「`v5_pmax3` の差は F18 の 2 セルだけ」と整合。パネル基準（旧既定）5.81 も j47・j59・j61 と同値
- **全体表（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals_mean`＝`analyze_quick` [1]）**: 基準 `MC-ESO` **37.1 / 40.8 / 43.1 / 46.9、26100** ｜ **`v5_wlam` 45.0 / 47.1 / 50.0 / 52.1、23132** ｜ `v5_wnoC` 36.7 / 40.8 / 43.1 / 46.9、26174 ｜ `v5_wsig` 37.5 / 41.2 / 43.3 / 46.0、24585 ｜ `v5_cp` 37.3 / 40.8 / 42.9 / 46.9、25819 ｜ **`v5_cpl` 44.8 / 46.9 / 50.0 / 52.1、24134**
- **`evals_succ_mean`（基準と腕の両方で成功のある関数だけで平均）**: `v5_wlam` 25795.6→22605.6（14 関数、**-3190**）｜ `v5_wnoC` 26099.8→26173.8（15、+74）｜ `v5_wsig` 25032.4→24585.4（14、-447）｜ `v5_cp` 26099.8→25819.1（15、-281）｜ `v5_cpl` 25795.6→23678.9（14、**-2117**）
- **関数別 SR@1e-10 の変化（基準から動いた全関数。動かなかった関数は省略）**: `v5_wlam` F02 95→100・F04 0→5・F06 30→**55**・F07 10→**0**・F08 0→**85**・F09 5→**95**・F10 95→100・F12 80→90・F13 25→**0** ｜ `v5_wnoC` **F02 95→85 のみ** ｜ `v5_wsig` F02 95→85・F07 10→**30** ｜ `v5_cp` **F04 0→5 のみ** ｜ `v5_cpl` F02 95→100・F04 0→5・F06 30→**55**・F07 10→**0**・F08 0→**85**・F09 5→**95**・F10 95→100・F11 100→95・F12 80→90・F13 25→**0**
- **平均誤差（ジョブ指定の関数。`cec_mean_errors`、1e-8 未満は 0。基準 / 腕）**: F07 1.356 / wsig **1.247**・wlam 1.365 ｜ **F08 7.198 / wlam・cpl 0.2565（大改善）/ wsig 11.59（悪化）** ｜ **F09 2.146 / wlam・cpl 0.1993 / wsig 5.257（悪化）** ｜ F10 8.76e-3 / wlam・cpl **0**・cp 7.26e-3 / wnoC 0.4503・wsig 0.7783（悪化）｜ F13 1.48e-4 / wlam・cpl 2.75e-3・wsig 8.61e-4（悪化）｜ F15 22.22（6 手法すべて同値）｜ F17 0.0239 / wnoC 0.02387 / wlam・cpl 0.0308・wsig 0.0265（悪化）｜ F18 0.1278 / cp **0.1222**・wsig 0.1313・cpl 0.1334・wlam 0.1363 ｜ **F21 0.6133・F22 1.290・F24 59.68 は 6 手法すべて同値**（j61 で残った F21/F22 の代償はこの 5 腕のどれでも動かなかった）
- **パネル CEC 順位（`cec_panel.py --dim 20 --panel <j47 の 20D>`、8 手法。低いほど良い。かっこ内は平均誤差が基準より下 / 上の関数数）**: **`v5_wlam` 4.25（17/3、最良）** ＜ `v5_cpl` 4.35（17/3）＜ `v5_cp` 4.69（16/4）＜ 新しい既定 **4.73**（15/5、再現）＜ `v5_wnoC` 4.77（15/5）＜ `v5_wsig` 4.94（15/5）。パネル基準（旧既定 `MC-ESO`）5.81。**j61 の `ctl_plainwave` は 4.23** で `v5_wlam` 4.25 とほぼ並ぶ
- **run 内 CEC 順位（`analyze_quick` [5]、6 手法）**: `v5_cp` 3.10 ＜ `v5_wnoC` 3.27 ＜ 基準 `MC-ESO` 3.40 ＜ `v5_cpl` 3.58 ＜ `v5_wlam` 3.67 ＜ `v5_wsig` 3.98
- **Wilcoxon（reference `MC-ESO`、α=0.05 両側、基準が勝ち / 腕が勝ち）**: `v5_wlam` 5/3（基準勝ち F01 A12=0.73・F10 0.82・F14 0.84・F17 0.63・F23 0.53、腕勝ち **F06 0.28・F08 0.01・F09 0.07**）｜ `v5_wnoC` **0/0** ｜ `v5_wsig` **4/0**（基準勝ち F06 0.53・F08 0.94・F09 0.86・F17 0.56）｜ `v5_cp` 0/1（腕勝ち F23 0.32）｜ `v5_cpl` 4/3（基準勝ち F01 0.73・F10 0.82・F14 0.84・F17 0.63、腕勝ち F06 0.28・F08 0.01・F09 0.07）
- **記録**: summary / wilcoxon は `analysis/single/j62/{summary,wilcoxon}_j62_a20_dim20_all24.csv`、`analyze_quick` 全文は `aq_j62_a20.txt`、腕ごとの表と Wilcoxon 明細は `arms_j62_a20.txt`、パネル順位は `cecpanel_j62_a20.txt`、j61 との照合は `check_j62_a20.txt`、per-run stats は `analysis/single/j62/a20/dim20/stats/*.csv.gz`（24、`cec_panel.py --panel` で読めることを確認）
- **書かれたとおりに行かなかったこと**: (1) `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順どおり空 stub を `PYTHONPATH` に置き、`pip` が python3.13 を指していたため `python3 -m pip --ignore-installed ...` で入れ直した。(2) `--funcs` は `F01` 等の短縮名では 1 関数も一致せず（`No matching functions`）、`F01-Sphere` の完全名で指定し直した（空振りの 3 ディレクトリは削除）。(3) `scripts/check_doc_size.py` はこの記録を足したあとも全ファイル上限内（`docs/research_loop.md` 259/300）。(4) `find analysis -type f` は **1679 で上限の約 400 を大きく超えている**（この run で +30。削除はしていない）。採否の判断はしていない

### 2026-10-30 j61 (d2) 2D の 2 設定を完走 — 2D で動く v5 の腕は `v5_paceR` だけ（`v5_pmax3` は既定と bit 一致）。パネル CEC 順位は `v4_k50fR_r50` 4.44 で j60 (d2) を再現、`v5_paceR` は 4.52 で新既定より上（悪い）。SR@1e-10 は調整で基準 94.8% に対し `v5_paceR` 94.4%、hold-out は基準 94.4% に対し 94.6%。Wilcoxon は 2 設定とも 3 腕すべて 0勝0敗
- **ランした分**: ジョブ (d2) の 2 設定。`./run.sh quick --all --max-evals 5000 --n-runs 20 --instances 1-5 [--instances 6-10 --seed-offset 7] --methods "MC-ESO,v4_k50fR_r50,v5_paceR,v5_pmax3"` を **2 本同時に `--jobs 2` ずつ**（ジョブは `--jobs 4`。`--jobs` は結果を変えない）。`j61_d2` 02:36:55-02:50:14 = **13.3 分**、`j61_d2ho` 02:37:50-02:51:01 = **13.2 分**。shard 分割なし、合計 wall **14.1 分**。2 本とも `result.json` が `status=done`、**各 24/24 関数**（summary 96 行 = 24×4・wilcoxon 72 行 = 24×3、reference はすべて `MC-ESO`）
- **環境**: numpy 2.4.6 / cma 4.5.0 / scipy 1.17.1、python 3.11.17（`run.sh` の `PY` ＝ `/usr/local/bin/python3`）。CPU Intel Xeon @**2.10GHz**（family 6 / **model 207 / stepping 2**、4 CPUs）、avx512f あり、BLAS は openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1`。容器仕様は j60 (c20)・j61 (m2) と同じで、**j60 (d2) の容器（@2.80GHz / model 85 / stepping 7）とは別仕様。だが下の bit 一致のとおり 2D では 1 セルも動かなかった**
- **容器差は 2D では出なかった（パネル比較の妥当性）**: この run の `v4_k50fR_r50` は j60 (d2) の同名手法と per-run の `best_f` / `n_evals` / `success` が **1440/1440 一致**（hold-out も新既定 `MC-ESO` 対 j60 の `v4_k50fR_r50` で 1440/1440）。j60 の `MC-ESO`（旧既定）対 j57 パネルの `MC-ESO` も 1440/1440
- **名前の付け替えの検算**: 新既定 `MC-ESO` 対 `v4_k50fR_r50` が 2 設定とも per-run 1440/1440 一致・[1] の全セル一致 ＝ 付け替えは bit 一致
- **2D で `v5_pmax3` は既定と区別できない**: `v5_pmax3` 対 `MC-ESO` も 2 設定とも **1440/1440 一致**。ジョブは「2D で動くのは延命の 2 腕」と書いていたが、**実際に動いたのは `v5_paceR` だけ**（`v5_paceR` 対基準は調整 1419/1440・hold-out 1416/1440、差は F04・F19・F20・F23・F24 に限る）
- **全体表 2D 調整 instance 1-5（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals_mean` ＝ `analyze_quick` [1]）**: 基準 `MC-ESO` **94.8 / 95.4 / 96.0 / 96.2、776** ｜ `v4_k50fR_r50` 94.8 / 95.4 / 96.0 / 96.2、776（基準と同値）｜ `v5_paceR` **94.4 / 95.2 / 95.8 / 95.8、772** ｜ `v5_pmax3` 94.8 / 95.4 / 96.0 / 96.2、776（基準と同値）
- **全体表 2D hold-out instance 6-10 (+seed-offset 7)**: 基準 `MC-ESO` **94.4 / 95.2 / 95.6 / 95.6、874** ｜ `v4_k50fR_r50` 同値 ｜ `v5_paceR` **94.6 / 95.4 / 95.8 / 95.8、876** ｜ `v5_pmax3` 同値 ＝ **`v5_paceR` は調整で 0.4pt 下、hold-out で 0.2pt 上**（符号が逆）
- **`evals_succ_mean`（基準と腕の両方で成功のある関数だけの対応平均。腕 対 基準、共通関数数）**: 調整 `v5_paceR` 772.2 対 775.5（24、−3.3）｜ `v4_k50fR_r50` ＝ `v5_pmax3` 775.5 対 775.5（24）。hold-out `v5_paceR` 876.3 対 873.5（24、+2.8）｜ 他 2 腕は 873.5 対 873.5（24）
- **パネル CEC 順位（低いほど良い。かっこ内は平均誤差が基準より下 / 上の関数数）**: 調整 instance の 10 手法パネル（`cec_panel.py --dim 2 --panel analysis/single/j57/j57_d2 --drop` で旧腕 8 手法を外す。**パネルの基準 ＝ 旧既定 4.52 ＝ j57 を再現**）**`v4_k50fR_r50` 4.44（1/2）＝ `v5_pmax3` 4.44（1/2）＝ 新既定の値で j60 (d2) の 4.44 を再現**（動いた関数も同じ F19 2→4・F20 7→3）｜ **`v5_paceR` 4.52（1/0）** ＝ 新既定 4.44 より上（悪い）で旧既定と同値。hold-out は instance が違うのでパネルを当てていない
- **run 内の CEC 順位（`analyze_quick` [5]、参考。4 手法）**: 調整 `v5_paceR` **2.44**（24 関数中 23 で最良）＜ 基準 ＝ `v4_k50fR_r50` ＝ `v5_pmax3` 2.52（22）｜ hold-out `v5_paceR` **2.38**（23）＜ 他 3 つ 2.54（21）＝ **run 内順位では `v5_paceR` が 2 設定とも上。パネル順位とは向きが逆**
- **関数別 SR@1e-10 の変化（基準 → 腕、動いた関数のみ）**: 調整 `v5_paceR` up 1 [F19 95→100] down 2 [F20 100→90, F23 100→95]、`v4_k50fR_r50` と `v5_pmax3` は **変化なし**。F01-F11・F14-F17・F21・F22 の 17 関数は 4 手法すべて 100%、0% の関数は無し ｜ hold-out `v5_paceR` up 2 [F19 95→100, F20 80→85] down 1 [F23 100→95]、他 2 腕は **変化なし**。19 関数が全手法 100%、0% の関数は無し（F24 は全手法 10%）
- **Wilcoxon（`best_f`、reference `MC-ESO`、α=0.05 両側、基準の 勝/敗 ＋ A12）**: **調整・hold-out とも 3 腕すべて 0勝0敗**（有意差のある関数が無い）。`analyze_quick` [4] も「MC-ESO is never significantly beaten」
- **記録**: summary / wilcoxon は `analysis/single/j61/{summary,wilcoxon}_j61_{d2,d2ho}.csv`、`analyze_quick` 全文は `aq_j61_{d2,d2ho}.txt`、腕ごとの関数別表と Wilcoxon 明細は `arms_j61_{d2,d2ho}.txt`、パネル順位は `cecpanel_j61_d2.txt`、per-run stats は `analysis/single/j61/j61_{d2,d2ho}/dim2/stats/*.csv.gz`
- **書かれたとおりに行かなかったこと**: (1) `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順どおり空 stub を `PYTHONPATH` に置いた。さらに**この容器では `pip` が python 3.13 を指していて `run.sh` の `python3`（3.11）に何も入っていなかった**ので、`/usr/local/bin/python3 -m pip` で 3.11 側に入れ直し、過去 12 回の記録と同じ numpy 2.4.6 に固定した（素の `pip install numpy` は 2.5.4 になる）。`minionpy` / `nevergrad` は今回の 4 手法に不要なので入れていない。(2) 2 設定を同時に `--jobs 2` ずつで回した（ジョブは `--jobs 4`）。(3) `arms_*.txt` を出すスクリプトはリポジトリに無いので、記録用に同じ体裁のものを書いて使った（判断は足していない）。(4) `find analysis -type f` は **1649 で上限の約 400 を大きく超えている**（この run で +57。削除はしていない）。(5) `check_doc_size.py` が **`docs/research_loop.md` 346 / 300 行で OVER**（この追記を含む）。統合は対話セッションの担当なのでそのまま commit した

### 2026-10-30 j61 (a10ho) 10D hold-out を完走 — パネル CEC 順位は `v4_k50fR_r50` 4.46 で j59 の値を再現。腕で基準より下（良い）は `v5_pmax3` 4.38 と `v5_slow5` 4.42 の 2 つだけ、`v5_paceR` 4.54 ＜ `ctl_plainwave` ＝ `v5_herd` 4.81 は上。SR@1e-10 は基準 63.8% に対し上回る腕は無く、`ctl_plainwave` と `v5_herd` が 54.8% で大きく下。腕が Wilcoxon で基準に勝った関数はゼロ
- **ランした分**: ジョブのとおり 1 本（shard 分割なし）。`./run.sh quick --all --dim 10 --max-evals 25000 --n-runs 20 --instances 6-10 --seed-offset 7 --methods "MC-ESO,v4_k50fR_r50,v5_paceR,v5_pmax3,v5_herd,v5_slow5,v5_chainR,ctl_plainwave" --jobs 4`（label `j61_a10ho`）。01:36:21-02:05:35 = **29.2 分**。`result.json` は `status=done` / `set=all-d10`、**24/24 関数**（summary 192 行 = 24×8・wilcoxon 168 行 = 24×7・stats 24・`mceso_runs.csv` 3840 行 = 8×24×20、(関数, 手法) の重複・欠落ゼロ、wilcoxon の reference はすべて `MC-ESO`）
- **環境**: numpy 2.4.6 / cma 4.5.0 / scipy 1.17.1、python 3.11.17（`run.sh` の `PY` ＝ `/usr/local/bin/python3`。`pip` は 3.13 向けなので `python3 -m pip` で入れ直した）。CPU Intel Xeon @2.10GHz（family 6 / **model 207 / stepping 2**、4 CPUs）、avx512f あり、BLAS は openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1`。**容器仕様は j61 (a) の 5D・10D 調整の回と同じ**
- **確かめること（どちらも厳密に一致）**: `MC-ESO` と `v4_k50fR_r50` は 24 関数 × 47 列（`mean_time_s` 以外）で **1128/1128 セル一致**＝既定の付け替えどおり。`v4_k50fR_r50` は j59 の同設定（`summary_j59_j59_a10ho.csv`）とも **1128/1128 一致**し、パネル順位 **4.46 を再現**
- **全体表（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals_mean`＝`analyze_quick` [1]）**: 基準 `MC-ESO` **63.8 / 65.0 / 68.5 / 69.8、10328** ｜ `v4_k50fR_r50` ・`v5_pmax3` 同値（bit 一致）｜ `ctl_plainwave` 54.8 / 59.0 / 59.8 / 66.2、9361 ｜ `v5_paceR` 63.8 / 64.8 / 68.3 / 70.0、10053 ｜ `v5_herd` 54.8 / 56.9 / 60.0 / 60.8、10249 ｜ `v5_slow5` 62.7 / 64.0 / 69.2 / 70.4、10713 ｜ `v5_chainR` 63.5 / 64.8 / 68.3 / 69.6、10357
- **`evals_succ_mean`（基準と腕の両方で成功のある関数だけの対応平均。腕 対 基準、共通関数数）**: `ctl_plainwave` 9360.7 対 9845.5（共通 18）｜ ほかの 6 腕は対 基準 10328.4（共通 20）で `v4_k50fR_r50` ・`v5_pmax3` 10328.4（±0）・`v5_paceR` 10053.0（**-275**）・`v5_herd` 10248.9（-80）・`v5_slow5` 10713.2（**+385**）・`v5_chainR` 10357.1（+29）
- **パネル CEC 順位（`cec_panel.py --dim 10 --panel analysis/single/j49/j49_p10ho`、10 手法。低いほど良い。パネル基準 `MC-ESO` ＝ 旧既定 4.75。かっこ内は平均誤差が基準より下 / 上の関数数）**: **`v5_pmax3` 4.38（9/4、最良）** ＜ `v5_slow5` 4.42（10/3）＜ `v4_k50fR_r50` ＝ `v5_chainR` 4.46（9/4）＜ `v5_paceR` 4.54（9/4）＜ `ctl_plainwave` 4.81（7/8）＝ `v5_herd` 4.81（6/9）
- **run 内の 8 手法だけでの CEC 順位（`analyze_quick` [5]、参考）**: `v5_pmax3` 4.04 ＜ 基準 `MC-ESO` ＝ `v4_k50fR_r50` 4.10 ＜ `v5_chainR` 4.17 ＜ `v5_slow5` 4.42 ＜ `v5_paceR` 4.58 ＜ `ctl_plainwave` 5.10 ＜ `v5_herd` 5.48
- **関数別 SR@1e-10 の変化（基準 → 腕、動いた関数のみ。全 24 関数を照合）**: `v4_k50fR_r50` ・`v5_pmax3` 変化ゼロ ｜ `ctl_plainwave` up 0 down 7 [F04 50→45, F07 95→85, F09 100→90, F13 100→5, F16 25→0, F17 55→0, F23 15→0] ｜ `v5_paceR` up 1 [F21 55→65] down 1 [F16 25→15] ｜ `v5_herd` up 1 [F21 55→70] down 9 [F04 50→40, F07 95→60, F09 100→90, F12 95→50, F13 100→60, F16 25→5, F17 55→15, F22 40→15, F23 15→10] ｜ `v5_slow5` up 1 [F21 55→70] down 3 [F16 25→15, F17 55→30, F23 15→10] ｜ `v5_chainR` up 0 down 1 [F23 15→10]
- **指定の 6 関数の SR@1e-10（基準 / `v4_k50fR_r50` / plainwave / paceR / pmax3 / herd / slow5 / chainR）**: F15 全手法 0、F17 **55/55/0/55/55/15/30/55**、F18 全手法 0、F21 **55/55/55/65/55/70/70/55**、F22 **40/40/40/40/40/15/40/40**、F24 全手法 0。平均誤差では F21 基準 0.551 に対し paceR 0.311・slow5 0.353・herd 0.394 が改善、F24 基準 15.48 に対し herd 13.56 が改善・paceR 18.16・slow5 17.45 が悪化
- **Wilcoxon（`best_f`、reference `MC-ESO`、α=0.05 両側、基準の 勝/敗 ＋ A12）**: `v4_k50fR_r50` ・`v5_pmax3` ・`v5_chainR` **0勝0敗** ｜ `ctl_plainwave` **8勝0敗**（F13 1.00 / F14 1.00 / F17 0.91 / F18 0.76 / F23 0.77 large、F06 0.63 / F16 0.58 / F19 0.59 small）｜ `v5_herd` **6勝0敗**（F12 0.71 / F13 0.76 / F17 0.82 / F18 0.82 large、F07 0.68 / F16 0.69 medium）｜ `v5_paceR` **3勝0敗**（F15 0.77 / F24 0.73 large、F19 0.71 medium）｜ `v5_slow5` **2勝0敗**（F24 0.72 large、F19 0.69 medium）。**どの腕も基準に有意に勝った関数は無い（全腕 0敗）**
- **記録**: per-run stats は `analysis/single/j61/j61_a10ho/dim10/stats/*.csv.gz`（24、`cec_panel.py --panel` で読めることを確認）、summary / wilcoxon は `analysis/single/j61/{summary,wilcoxon}_j61_j61_a10ho.csv`、`analyze_quick` 全文は `aq_j61_a10ho.txt`、腕ごとの表と Wilcoxon 明細は `arms_j61_a10ho.txt`、パネル順位は `cecpanel_j61_a10ho.txt`。これで j61 (a) の 4 設定すべてが揃った
- **書かれたとおりに行かなかったこと**: (1) `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順どおり空 stub を `PYTHONPATH` に置き、`--ignore-installed blinker ...` を `python3 -m pip` で入れ直した（`pip` だけだと python3.13 側に入って `ioh` が見えない）。(2) `scripts/check_doc_size.py` は **`docs/research_loop.md` が 332/300 行で超過**（この記録を足してさらに増える。統合は対話セッションの担当）。(3) `find analysis -type f` は **1592 で上限の約 400 を大きく超えている**（この run で +29。削除はしていない）。採否の判断はしていない

### 2026-10-30 j61 (a) 5D の 2 設定と 10D の調整 instance を完走（10D hold-out はキューへ書き戻し）— パネル CEC 順位は `v4_k50fR_r50` が 5D 4.12 / 4.15・10D 4.25 で j59 の値を再現。腕で基準より下（良い）になったのは 5D 調整の `v5_slow5` 3.88 と 10D の `v5_slow5` 4.38 だけで、ほかは基準と同値か上。SR@1e-10 は `v5_slow5` が 3 設定すべてで基準以上（77.1 / 76.0 / 64.2%）、`v5_paceR` は 5D hold-out と 10D で上（75.6 / 64.2%）、`v5_herd` と `ctl_plainwave` は 3 設定すべてで大きく下
- **ランした分**: ジョブの 4 設定のうち **3 設定**（`./run.sh quick --all --dim D --max-evals B --n-runs 20 --instances I [--seed-offset 7] --methods "MC-ESO,v4_k50fR_r50,v5_paceR,v5_pmax3,v5_herd,v5_slow5,v5_chainR,ctl_plainwave" --jobs 4`、shard 分割なしの 1 本ずつ逐次）。`j61_a5`(D5/B12500/1-5) 23:32:39-23:54:16 = **21.6 分**、`j61_a5ho`(D5/6-10 +offset 7) 00:02:22-00:23:59 = **21.6 分**、`j61_a10`(D10/B25000/1-5) 00:24:26-00:52:34 = **28.1 分**、合計 **71.3 分**。3 本とも `result.json` が `status=done` / `set=all-d5` か `all-d10`、**各 24/24 関数**（summary 192 行 = 24×8・wilcoxon 168 行 = 24×7・stats 各 24・`mceso_runs.csv` 各 3840 行 = 8×24×20、(関数, 手法) の重複・欠落ゼロ、wilcoxon の reference はすべて `MC-ESO`）。**10D hold-out（`j61_a10ho`）は 90 分に収まらないのでキューへ書き戻した**
- **環境**: numpy 2.4.6 / cma 4.5.0 / scipy 1.17.1、python 3.11.17（`run.sh` の `PY` ＝ `/usr/local/bin/python3`。`pip` は 3.13 向けなので `python3 -m pip` で入れ直した）。CPU Intel Xeon @2.10GHz（family 6 / **model 207 / stepping 2**、4 CPUs）、avx512f あり、BLAS は openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1`。**容器仕様は j59 (a10/a10ho)・j58 (a10/a5ho) と同じで、昨日の j61 (a20)（model 85 / stepping 7）とは別**
- **確かめること**: 3 run すべてで `MC-ESO` と `v4_k50fR_r50` が 24 関数 × 45 列（`mean_time_s` 以外）で **各 1080/1080 セル一致**＝既定の付け替えどおり。`v4_k50fR_r50` は j59 の同名の行（`summary_j59_j59_a5[ho].csv` / `summary_j59_j59_a10.csv`）とも **各 1080/1080 一致**し、パネル順位 **5D 4.12 / 4.15・10D 4.25 をいずれも再現**（10D hold-out の 4.46 は未測定）
- **全体表 5D 調整 `j61_a5`（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals_mean`）**: 基準 `MC-ESO` **76.9 / 78.3 / 80.4 / 81.2、3841** ｜ `v4_k50fR_r50` 同値（bit 一致）｜ `v5_pmax3` 同値（bit 一致）｜ `ctl_plainwave` 68.5 / 71.9 / 78.5 / 81.0、3874 ｜ `v5_paceR` 76.9 / 77.9 / 79.8 / 80.6、3692 ｜ `v5_herd` 72.7 / 74.2 / 75.2 / 76.5、3671 ｜ **`v5_slow5` 77.1 / 77.9 / 80.8 / 82.3、3929** ｜ `v5_chainR` 76.7 / 77.9 / 80.2 / 81.0、3837
- **全体表 5D hold-out `j61_a5ho`**: 基準 **75.4 / 76.7 / 77.7 / 78.1、3916** ｜ `v4_k50fR_r50` ・`v5_pmax3` 同値（bit 一致）｜ `ctl_plainwave` 66.2 / 70.2 / 75.6 / 77.9、3976 ｜ **`v5_paceR` 75.6 / 76.5 / 77.7 / 78.1、3953** ｜ `v5_herd` 70.0 / 71.5 / 72.5 / 74.4、3663 ｜ **`v5_slow5` 76.0 / 77.5 / 78.1 / 78.3、4117** ｜ `v5_chainR` 75.2 / 76.5 / 77.5 / 77.9、3898
- **全体表 10D 調整 `j61_a10`**: 基準 **63.5 / 64.8 / 69.6 / 71.0、10541** ｜ `v4_k50fR_r50` ・`v5_pmax3` 同値（bit 一致）｜ `ctl_plainwave` 56.9 / 60.8 / 61.5 / 67.1、9325 ｜ **`v5_paceR` 64.2 / 65.2 / 70.0 / 71.5、10460** ｜ `v5_herd` 60.4 / 61.5 / 64.4 / 65.8、10022 ｜ **`v5_slow5` 64.2 / 65.0 / 70.0 / 70.8、10549** ｜ `v5_chainR` 63.5 / 64.6 / 69.4 / 70.8、10512
- **`evals_succ_mean`（基準と腕の両方で成功のある関数だけの対応平均。腕 対 基準、共通関数数）**: 5D 調整 `ctl_plainwave` 3874.0 / `v5_paceR` 3692.2 / `v5_herd` 3671.2（対 基準 3713.3、共通 21）/ `v5_slow5` 3928.7 / `v5_chainR` 3836.9 対 基準 3840.6（共通 22）｜ 5D hold-out 3976.1 / 3953.0 / 3662.7 / 4116.7 / 3898.5 対 基準 3916.1（共通 22）｜ 10D 調整 `ctl_plainwave` 9324.9（対 9370.2、共通 17）・`v5_paceR` 10459.9 / `v5_herd` 10022.1 / `v5_slow5` 10549.4 / `v5_chainR` 10511.5 対 基準 10540.8（共通 20）
- **パネル CEC 順位（`cec_panel.py --dim D --panel analysis/single/j49/j49_p5[ho] / j49_p10`、10 手法。低いほど良い。基準 `MC-ESO` はパネル側＝旧既定。かっこ内は平均誤差が基準より下 / 上の関数数）**: 5D 調整（基準 **4.06**）**`v5_slow5` 3.88（8/3、最良）** ＜ `v4_k50fR_r50` ＝ `v5_chainR` ＝ `v5_pmax3` 4.12（8/3）＝ `v5_paceR` 4.12（6/4）＜ `ctl_plainwave` 4.25（6/5）＜ `v5_herd` 5.15（3/11）｜ 5D hold-out（基準 **4.10**）`v4_k50fR_r50` ＝ `v5_chainR` ＝ `v5_pmax3` 4.15（3/7）＝ `v5_paceR` 4.15（4/6）＜ `v5_slow5` 4.19（6/4）＜ `ctl_plainwave` 4.23（2/9）＜ `v5_herd` 4.81（3/10）｜ 10D 調整（基準 **4.62**）**`v4_k50fR_r50` ＝ `v5_chainR` ＝ `v5_pmax3` 4.25（7/4）** ＜ `v5_slow5` 4.38（7/4）＜ `v5_paceR` ＝ `ctl_plainwave` 4.46 ＜ `v5_herd` 4.85（6/9）＜ 基準 4.62 より上は `v5_herd` だけ
- **run 内の 8 手法だけでの CEC 順位（`analyze_quick` [5]、参考）**: 5D 調整 基準 `MC-ESO` ＝ `v4_k50fR_r50` ＝ `v5_pmax3` 4.17 ＜ `v5_slow5` 4.25 ＜ `v5_chainR` 4.35 ＜ `ctl_plainwave` 4.48 ＜ `v5_paceR` 4.77 ＜ `v5_herd` 5.65 ｜ 5D hold-out `v5_slow5` 3.96 ＜ `v5_paceR` 4.06 ＜ `v5_chainR` 4.42 ＜ 基準 ＝ `ctl_plainwave` ＝ `v4_k50fR_r50` ＝ `v5_pmax3` 4.52 ＜ `v5_herd` 5.48 ｜ 10D 調整 `v5_pmax3` 4.02 ＜ `v5_chainR` 4.12 ＜ 基準 ＝ `v4_k50fR_r50` 4.15 ＜ `v5_paceR` 4.69 ＜ `v5_herd` 4.90 ＜ `v5_slow5` 4.92 ＜ `ctl_plainwave` 5.06
- **関数別 SR@1e-10 の変化（基準 → 腕、動いた関数のみ）5D 調整**: `v4_k50fR_r50` ・`v5_pmax3` 変化ゼロ ｜ `ctl_plainwave` up 0 down 5 [F13 100→55, F16 65→55, F17 80→10, F18 70→5, F23 50→40] ｜ `v5_paceR` up 2 [F21 75→95, F22 90→95] down 4 [F15 10→5, F16 65→55, F20 15→10, F23 50→45] ｜ `v5_herd` up 3 [F17 80→85, F21 75→80, F22 90→95] down 8 [F04 90→80, F07 100→95, F09 100→90, F13 100→90, F15 10→0, F18 70→45, F20 15→10, F23 50→10] ｜ `v5_slow5` up 3 [F16 65→70, F21 75→95, F23 50→60] down 3 [F17 80→70, F18 70→55, F20 15→10] ｜ `v5_chainR` up 1 [F18 70→75] down 2 [F15 10→5, F16 65→60]
- **関数別 SR@1e-10 の変化 5D hold-out**: `v4_k50fR_r50` ・`v5_pmax3` 変化ゼロ ｜ `ctl_plainwave` up 1 [F20 5→10] down 6 [F04 75→70, F13 100→25, F16 65→45, F17 90→25, F18 55→5, F23 40→30] ｜ `v5_paceR` up 1 [F21 80→95] down 1 [F23 40→30] ｜ `v5_herd` up 0 down 9 [F04 75→60, F08 100→90, F12 100→90, F15 10→5, F16 65→60, F17 90→70, F18 55→25, F22 90→75, F23 40→20] ｜ `v5_slow5` up 2 [F18 55→75, F21 80→90] down 3 [F15 10→5, F22 90→85, F23 40→35] ｜ `v5_chainR` up 0 down 1 [F16 65→60]
- **関数別 SR@1e-10 の変化 10D 調整**: `v4_k50fR_r50` ・`v5_pmax3` ・`v5_chainR` 変化ゼロ ｜ `ctl_plainwave` up 0 down 6 [F07 80→70, F13 95→5, F16 10→0, F17 30→0, F22 55→50, F23 15→0] ｜ `v5_paceR` up 1 [F21 75→95] down 1 [F16 10→5] ｜ `v5_herd` up 2 [F16 10→15, F18 0→5] down 7 [F04 65→60, F07 80→65, F08 100→95, F12 100→70, F13 95→80, F17 30→20, F23 15→10] ｜ `v5_slow5` up 3 [F13 95→100, F18 0→5, F21 75→90] down 2 [F22 55→50, F23 15→10]
- **指定の 6 関数の SR@1e-10（基準 / plainwave / paceR / pmax3 / herd / slow5 / chainR、`v4_k50fR_r50` は基準と同値）**: 5D 調整 F15 10/10/5/10/0/10/5、F17 80/10/80/80/85/70/80、F18 70/5/70/70/45/55/75、F21 75/75/95/75/80/95/75、F22 90/90/95/90/95/90/90、F24 全手法 0 ｜ 5D hold-out F15 10/10/10/10/5/5/10、F17 90/25/90/90/70/90/90、F18 55/5/55/55/25/75/55、F21 80/80/95/80/80/90/80、F22 90/90/90/90/75/85/90、F24 全手法 0 ｜ 10D F15 全手法 0、F17 30/0/30/30/20/30/30、F18 0/0/0/0/5/5/0、F21 75/75/95/75/75/90/75、F22 55/50/55/55/55/50/55、F24 全手法 0
- **Wilcoxon（`best_f`、reference `MC-ESO`、α=0.05 両側、基準の 勝/敗 ＋ A12）**: 5D 調整 `ctl_plainwave` 5勝0敗（F13 / F14 A12=1.00 large、F17 0.87・F18 0.88 large、F23 small）｜ `v5_herd` 3勝0敗（F15 0.84 / F18 0.77 / F23 0.80 large）｜ `v5_slow5` 0勝1敗（F21 0.40 small）｜ ほかの 4 腕 0勝0敗 ｜ 5D hold-out `ctl_plainwave` 6勝0敗 ｜ `v5_herd` 3勝0敗（F17 0.84 / F18 0.75 large）｜ `v5_paceR` 1勝0敗（F23 small）｜ ほか 0勝0敗 ｜ 10D `ctl_plainwave` 7勝0敗（F13 0.99 / F14 1.00 / F18 0.90 / F23 0.71 large）｜ `v5_paceR` 3勝0敗（F15 0.71 large、F19 / F24 medium）｜ `v5_slow5` 3勝0敗（F19 0.81 large、F04 / F20 negligible）｜ `v5_herd` 2勝1敗（勝 F12 / F18、敗 F15 0.34 medium）｜ `v4_k50fR_r50` ・`v5_pmax3` ・`v5_chainR` 0勝0敗
- **記録**: per-run stats は `analysis/single/j61/j61_a5[ho]/dim5/stats/*.csv.gz` と `j61_a10/dim10/stats/*.csv.gz`（3 label × 24、`cec_panel.py --panel` で読めることを確認）、summary / wilcoxon は `analysis/single/j61/{summary,wilcoxon}_j61_j61_a5[ho].csv` / `_j61_a10.csv`、`analyze_quick` 全文は `aq_j61_a5[ho].txt` / `aq_j61_a10.txt`、腕ごとの表と Wilcoxon 明細は `arms_j61_a5[ho].txt` / `arms_j61_a10.txt`、パネル順位は `cecpanel_j61_a5[ho].txt` / `cecpanel_j61_a10.txt`
- **書かれたとおりに行かなかったこと**: (1) `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順どおり空 stub を `PYTHONPATH` に置き、`--ignore-installed blinker ...` で入れ直した（集計スクリプトも `core.optimizers` 経由で `pynmmso` を import するので stub が必要）。(2) 4 設定は 90 分に収まらないので 3 設定だけ回し、`j61_a10ho` をキューへ書き戻した。(3) `scripts/check_doc_size.py` は **`docs/research_loop.md` が 302/300 行で既に超過**（この記録を足してさらに増える。統合は対話セッションの担当）。(4) `find analysis -type f` は **1563 で上限の約 400 を大きく超えている**（この run で +75。削除はしていない）

### 2026-10-29 j61 (a20) 20D を完走 — パネル CEC 順位は新しい既定 4.73 に対し `ctl_plainwave` 4.23 ＜ `v5_herd` 4.62 ＜ `v5_chainR` 4.69 ＜ `v5_pmax3` 4.73（並び）＜ `v5_paceR` / `v5_slow5` 5.10。SR@1e-10 は基準 37.1% に対し `v5_herd` 38.3% ・`v5_chainR` 37.3% が上、`v5_slow5` 35.6% ・`ctl_plainwave` 36.5% が下。F21/F22 の代償は paceR / pmax3 のどちらでも消えなかった
- **ランした分**: ジョブの 8 手法・`--all --dim 20 --max-evals 50000 --n-runs 20 --instances 1-5 --jobs 4` を**関数 3 shard に分けて逐次**（`j61_a20_a` F01-F08 21:41:26-22:07:45 **26.3 分**・`_b` F09-F16 22:07:45-22:37:14 **29.5 分**・`_c` F17-F24 22:37:14-23:01:54 **24.7 分**、合計 **80.5 分**）。3 本とも `result.json` が `status=done` / `set=all-d20`。連結後 **24/24 関数**（summary 192 行 = 24×8・wilcoxon 168 行 = 24×7・stats 24 ファイル・`mceso_runs.csv` 3840 行 = 8×24×20、(関数, 手法) の重複・欠落ゼロ、wilcoxon の reference はすべて `MC-ESO`）。連結後の run は `results/20261029_214126_j61_a20m_quick`
- **環境**: numpy 2.4.6 / cma 4.5.0、python 3.11.17（`run.sh` の `PY` ＝ `/usr/local/bin/python3`。`pip` は 3.13 向けなので `python3 -m pip` で入れ直した）。CPU Intel Xeon @2.80GHz（family 6 / **model 85 / stepping 7**、4 CPUs）、avx512f あり、BLAS は openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1`。**容器仕様は j59 (a20)・j58 (a20)・j47 と同じ**。`pynmmso` はビルドできないので findings.md の空 stub を `PYTHONPATH` に置いた（BBOB のみなので影響なし）
- **確かめること (i)(ii) はどちらも厳密に一致**: `MC-ESO` と `v4_k50fR_r50` は 24 関数 × 48 列（`mean_time_s` 以外）で **1152/1152 セル一致**＝既定の付け替えどおり。`v4_k50fR_r50` は j59 の 20D（`summary_j59_a20_dim20_all24.csv`）の同名の行とも **1152/1152 一致**で、パネル順位 **4.73 を再現**（パネル基準 `MC-ESO`＝旧既定 5.81 も j47・j59 と同値）
- **全体表（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals_mean`＝`analyze_quick` [1]）**: 基準 `MC-ESO`（＝新しい既定）**37.1 / 40.8 / 43.1 / 46.9、26100** ｜ `v4_k50fR_r50` 37.1 / 40.8 / 43.1 / 46.9、26100（基準と bit 一致）｜ `ctl_plainwave` 36.5 / 39.2 / 41.0 / 46.0、25521 ｜ `v5_paceR` 36.9 / 40.6 / 42.9 / 46.9、26188 ｜ `v5_pmax3` 37.1 / 40.8 / 43.1 / 46.9、26100 ｜ **`v5_herd` 38.3** / 40.8 / 41.7 / 44.8、25840 ｜ `v5_slow5` 35.6 / 41.0 / 42.7 / 47.3、27368 ｜ **`v5_chainR` 37.3** / 40.8 / 42.9 / 46.9、25819
- **`evals_succ_mean`（基準と腕の両方で成功のある関数だけで平均）**: `v4_k50fR_r50` 26099.8→26099.8（15 関数、±0）｜ `ctl_plainwave` 22796.3→23551.4（12 関数、**+755**）｜ `v5_paceR` →26188.4（15、+89）｜ `v5_pmax3` →26099.8（15、±0）｜ `v5_herd` 25795.6→24947.9（14、**-848**）｜ `v5_slow5` →26030.4（15、-69）｜ `v5_chainR` →25819.1（15、**-281**）
- **関数別 SR@1e-10 の変化（基準から動いた全関数。動かなかった関数は省略）**: `ctl_plainwave` F02 95→100・F06 30→55・F07 10→0・F09 5→0・F10 95→100・F12 80→70・F13 25→0 ｜ `v5_paceR` **F21 45→40 のみ** ｜ `v5_pmax3` **変化なし**（差は F18 の 2 セルだけ）｜ `v5_herd` F02 95→100・F04 0→5・F06 30→**80**・F07 10→0・F08 0→**20**・F09 5→10・F10 95→100・F12 80→**55**・F13 25→**0** ｜ `v5_slow5` F06 30→**5**・F12 80→70・F13 25→30・F21 45→40 ｜ `v5_chainR` **F04 0→5 のみ**
- **平均誤差（ジョブ指定の関数。`cec_mean_errors`、1e-8 未満は 0）**: F07 基準 1.356 / herd 1.890 / slow5 **1.288** ｜ F08 基準 7.198 / plainwave **0.471** / herd 4.825 ｜ F09 基準 2.146 / plainwave **0.147** / herd 1.626 / slow5 3.559 ｜ F10 基準 8.76e-3 / plainwave **0** / herd **0** / chainR 7.26e-3 ｜ F15 基準 22.22 / paceR **38.22** / slow5 31.35 ｜ F17 基準 0.0239 / slow5 **0.0153** / herd 0.0859 ｜ F18 基準 0.1278 / slow5 **0.0734** / pmax3 0.1222 / paceR 0.5473 ｜ **F21 基準 0.6133 / herd 0.5559（唯一の改善）/ paceR 0.7737・slow5 0.8011（悪化）** ｜ **F22 基準 1.290 / paceR 1.034・herd 1.160（改善）/ slow5 1.655** ｜ F24 基準 59.68 / herd **56.32** / paceR 73.91 / slow5 81.72
- **パネル CEC 順位（`cec_panel.py --dim 20 --panel <j47 の 20D>`、8 手法。低いほど良い。かっこ内は平均誤差が基準より下 / 上の関数数）**: **`ctl_plainwave` 4.23（16/4、最良）** ＜ `v5_herd` 4.62（15/5）＜ `v5_chainR` 4.69（16/4）＜ `v5_pmax3` 4.73（15/5）＝ `v4_k50fR_r50` 4.73（15/5）＜ `v5_paceR` 5.10（13/7）＝ `v5_slow5` 5.10（14/6）。パネル基準（旧既定 `MC-ESO`）5.81
- **run 内 CEC 順位（`analyze_quick` [5]、8 手法）**: `ctl_plainwave` 3.69 ＜ `v5_chainR` 4.15 ＜ `v5_pmax3` 4.25 ＜ `v5_herd` 4.27 ＜ 基準 `MC-ESO` 4.38 ＝ `v4_k50fR_r50` 4.38 ＜ `v5_paceR` 5.42 ＜ `v5_slow5` 5.48
- **Wilcoxon（reference `MC-ESO`、α=0.05 両側、基準が勝ち / 腕が勝ち）**: `v4_k50fR_r50` **0/0** ｜ `ctl_plainwave` 6/4（腕勝ち F06 A12=0.28・F08 0.01・F09 0.18・F21 0.44）｜ `v5_paceR` **6/0**（基準勝ち F15 0.83・F16 0.70・F18 0.71・F19 0.83・F23 0.73・F24 0.70）｜ `v5_pmax3` **0/0** ｜ `v5_herd` 4/3（基準勝ち F07 0.68・F13 0.77・F17 0.86・F18 0.81、腕勝ち F04 0.19・F06 0.17・F08 0.32）｜ `v5_slow5` 9/1（腕勝ち F18 0.37）｜ `v5_chainR` 0/1（腕勝ち F23 0.32）
- **書いたとおりに行かなかった点**: `pip install -r requirements.txt` は `pynmmso` のビルドで落ち、`pip` が python3.13 を指していたため `python3 -m pip` で入れ直した（findings.md のセットアップ手順どおり）。`analysis` のファイル数は **1476 で上限 約 400 を大きく超えている**（今回の追加は 30 ファイル。削除はしていない）。`scripts/check_doc_size.py` は **`docs/research_loop.md` が 302/300 行で OVER**（統合は対話セッションの担当）。採否の判断はしていない

### 2026-10-29 j61 (m2) CEC2022 の 2 設定を完走 — 新しい既定 `MC-ESO` は 18 手法中 CEC 順位 10D 10 位（9.67）・20D 11 位（9.62）で比較手法の上位に届かず、旧既定 `MC-ESO-v3` には 10D 4勝0敗・20D 9勝0敗。SR@1e-10 は 10D 30.8%（18 手法中 3 位）・20D 16.2%（同 7 位）。j60 (c10)/(c20) の `MC-ESO`＝`MC-ESO-v3`、`v4_k50fR_r50`＝新 `MC-ESO` が全セル一致（名前の付け替えどおり）
- **ランした分**: ジョブの残り 2 設定とも。`./run.sh quick --suite cec2022 --all --dim {10|20} --max-evals {25000|50000} --n-runs 20 --jobs 4 --methods <18 手法>`（instance 既定 1）。`j61_mc10` 19:34:23-19:51:06 = **16.7 分**（shard 分割なし、12/12 関数）、`j61_mc20` は 20:00:16 に 1 本で始めたが **G10 の途中で親プロセスが OOM で殺された**（anon-rss 4.8 GB、cgroup 制限。G01-G09 の 9 関数は完成）。残り 3 関数を `--funcs "G10-Composition2,G11-Composition3,G12-Composition4"` の shard（`j61_mc20b` 20:25:52-20:35:16 = **9.4 分**）で取り、**関数 shard を連結**して `20261029_200016_j61_mc20m_quick` にした（summary 216 行 = 12×18、wilcoxon 204 行 = 12×17、reference はすべて `MC-ESO`、(関数, 手法) の重複・欠落ゼロ、**12/12 関数**）。合計 wall **51.1 分**（setup 約 3 分を含む）
- **環境**: numpy 2.4.6 / cma 4.5.0 / scipy 1.17.1、python 3.11.17（`run.sh` の `PY` ＝ `/usr/local/bin/python3`）。CPU Intel Xeon @**2.10GHz**（family 6 / **model 207 / stepping 2**、4 CPUs）、avx512f あり、BLAS は openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1`。**j61 (m) の BBOB 3 設定の容器（@2.80GHz / model 85 / stepping 7）とは別仕様**なので BBOB との横並びは参考にとどめる。j60 (c20) と同じ仕様
- **一致の確認（名前の付け替えの検算）**: CEC2022 10D も 20D も、j60 の同設定の `MC-ESO`（旧既定）＝ この run の `MC-ESO-v3`、j60 の `v4_k50fR_r50` ＝ この run の `MC-ESO` が [1] の全セルで一致（10D 30.8/37.1/37.5/37.9・8649 と 30.8/37.5/37.9/38.7・8238、20D 12.9/14.2/20.8/21.7・20390 と 16.2/17.1/24.2/26.2・22729）。Wilcoxon も同じ 9 関数・同じ p 値で A12 が反転（例 G01 0.25↔0.75）＝ **shard 連結と既定の入れ替えはどちらも正しい**
- **全体表 CEC2022 10D（12 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals_mean`）**: RCMAES 33.8 ＞ ARRDE 32.9 ＞ **`MC-ESO` 30.8 / 37.5 / 37.9 / 38.7、8238** ＝ `MC-ESO-v3` 30.8 / 37.1 / 37.5 / 37.9、8649 ＞ L-SRTDE 30.4 ＞ LSHADE-SPACMA 29.6 ＞ jSO 27.5 ＞ AMALGAM-SO 24.6 ＞ EBOwithCMAR 24.2 ＞ BIPOP-aCMA-ES / ELSHADE-SPACMA 21.7 ＞ EA4eig / L-SHADE 20.4 ＞ BIPOP-CMA-ES 19.2 ＞ IPOP-CMA-ES 17.9 ＞ SPS-L-SHADE-EIG 16.7 ＞ IMODE 12.1 ＞ NL-SHADE-RSP 8.3
- **全体表 CEC2022 20D**: RCMAES 22.1 ＞ AMALGAM-SO 21.7 ＞ ARRDE / BIPOP-aCMA-ES / BIPOP-CMA-ES 17.5 ＞ EBOwithCMAR / IPOP-CMA-ES 16.7 ＞ **`MC-ESO` 16.2 / 17.1 / 24.2 / 26.2、22729** ＞ LSHADE-SPACMA 15.8 ＞ **`MC-ESO-v3` 12.9 / 14.2 / 20.8 / 21.7、20390** ＞ EA4eig 10.0 ＞ IMODE 8.8 ＞ ELSHADE-SPACMA / jSO / L-SHADE / L-SRTDE 8.3 ＞ SPS-L-SHADE-EIG 0.4 ＞ NL-SHADE-RSP 0.0
- **run 内の CEC 式の平均誤差の順位（18 手法、低いほど良い）**: 10D ARRDE 5.83 ＜ LSHADE-SPACMA 6.75 ＜ EA4eig / SPS-L-SHADE-EIG 6.83 ＜ L-SHADE 7.25 ＜ ELSHADE-SPACMA / L-SRTDE 7.58 ＜ jSO 7.67 ＜ EBOwithCMAR 7.75 ＜ **`MC-ESO` 9.67（10 位）** ＜ AMALGAM-SO 10.33 … `MC-ESO-v3` 12.25（15 位）｜ 20D ARRDE 4.67 ＜ LSHADE-SPACMA 6.58 ＜ EBOwithCMAR 6.67 ＜ L-SRTDE 7.71 ＜ L-SHADE 7.92 ＜ ELSHADE-SPACMA / jSO 7.96 ＜ AMALGAM-SO 8.00 ＜ EA4eig 8.58 ＜ RCMAES 9.42 ＜ **`MC-ESO` 9.62（11 位）** ＜ IPOP-CMA-ES 9.79 … `MC-ESO-v3` 14.25（**最下位**）
- **関数別 SR@1e-10 の `MC-ESO` → `MC-ESO-v3`（動いた関数のみ、全 12 関数を照合）**: 10D は v3 が上 1 [G11 50→60]、下 2 [G02 55→50, G03 65→60]、同値 9（G01 100・G05 100 は両方、G04・G06-G10・G12 は両方 0%）｜ 20D は v3 が上 0、**下 2 [G01 100→70, G05 95→85]**、同値 10（G02-G04・G06-G12 が両方 0%）
- **`evals_succ_mean`（両方に成功のある関数だけの対応平均）**: 10D は 6 関数（G01/G02/G03/G05/G07/G11）で `MC-ESO` 8237.5 対 `MC-ESO-v3` 8648.8（v3 が +411.3）｜ 20D は 4 関数（G01/G03/G05/G11）で 22729.0 対 20390.0（**v3 が -2339.0 ＝ 成功時は速いが成功自体が少ない**。G11 36930→26810 が主因）
- **Wilcoxon（`best_f`、reference `MC-ESO`、α=0.05 両側）**: 対 `MC-ESO-v3` は **10D 4勝0敗**（G08 A12=0.79・G10 0.81 large、G12 0.69 medium、G07 0.57 small）、**20D 9勝0敗**（G07 0.86・G12 0.86・G01 0.75・G08 0.83・G11 0.83 large、G02 0.69・G04 0.70・G05 0.68・G09 0.71 medium）。比較手法に対しては 10D が BIPOP-aCMA-ES 8勝0敗・IMODE / IPOP-CMA-ES 7勝2敗 … ARRDE / LSHADE-SPACMA 2勝6敗、20D が BIPOP-aCMA-ES 9勝2敗・NL-SHADE-RSP 8勝1敗・IMODE 8勝1敗 … ARRDE 1勝8敗・LSHADE-SPACMA 2勝7敗
- **`MC-ESO` が比較手法に負けている関数（[3]）**: 10D は G02 55%（最良 RCMAES 80%）・G04 0%（EBOwithCMAR 20%）・G07 0%（ARRDE 10%）・G11 50%（LSHADE-SPACMA 100%）。18 手法すべて 0% は G06 / G08 / G09 / G10 / G12 ｜ 20D は G02 0%（BIPOP-aCMA-ES 70%）・G03 0%（RCMAES 45%）・G05 95%（EA4eig 100%）・G11 0%（BIPOP-aCMA-ES 5%）。18 手法すべて 0% は G04 / G06 / G07 / G08 / G09 / G10 / G12
- **記録**: `analysis/single/j61/m/aq_j61_{mc10,mc20}.txt`（`analyze_quick.py` 全文、[1][3][4][5]）と `arms_j61_{mc10,mc20}.txt`（関数別 SR@1e-10・平均誤差・`evals_succ_mean`・Wilcoxon の明細）。数値の本体は `runs/20261029_{193423_j61_mc10,200016_j61_mc20m}_quick`（`./run.sh publish` 済み、CSV.gz）。これで j61 (m) のジョブ 5 設定すべてが揃った
- **書かれたとおりに行かなかったこと**: (1) `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順どおり空 stub を `PYTHONPATH` に置き、`python3 -m pip`（3.11 側）で `--ignore-installed blinker` 付きに入れ直した（`minionpy==1.9.1` / `nevergrad==1.0.12` も必要）。(2) **20D は 1 本目が G10 の途中で OOM**（親プロセス 4.8 GB）。関数 shard（3 関数）に分けて取り直し、連結した。20D・18 手法を 1 プロセスで通すと容器のメモリに収まらない。(3) shard の連結は `scripts/single/merge_runs.py`（手法方向の結合用）ではなく summary / wilcoxon / stats の行連結で行った（findings.md「実行と並列」のとおり関数 shard は RNG 同一）。(4) `find analysis -type f` は **1446 で上限の約 400 を大きく超えている**（この run で +4。削除はしていない）

### 2026-10-29 j61 (m) 手法解説ページの成績の取り直し（BBOB 3 設定を完走）— 新しい既定 `MC-ESO` は 18 手法中 SR@1e-10 が 2D/5D/10D すべて 1 位（95.0 / 79.4 / 66.0%）、run 内 CEC 順位は 2D 4 位（8.04、首位 IMODE 7.40）・5D 1 位（7.10）・10D 1 位（6.23）。旧既定 `MC-ESO-v3` との差は 2D でほぼ無く（1 関数）、5D は新既定が SR で +1.1pt、10D は SR 同値で Wilcoxon 8勝0敗
- **ランした分**: ジョブの 5 設定のうち **BBOB の 3 つだけ**。`./run.sh quick --all [--dim 5|10] --max-evals {5000|12500|25000} --n-runs 20 --jobs 4 --methods <18 手法>`（instance 既定 1、shard 分割なし、1 本ずつ逐次）。`j61_md2` 17:39:13-17:48:52 = **9.7 分**、`j61_md5` 17:49:03-18:08:52 = **19.8 分**、`j61_md10` 18:09:05-18:47:30 = **38.4 分**、合計 wall **67.9 分**（setup 10 分を含めて約 78 分）。3 本とも `result.json` が `status=done`、**各 24/24 関数**（summary 432 行 = 24×18、wilcoxon 408 行 = 24×17、reference はすべて `MC-ESO`、(関数, 手法) の重複・欠落ゼロ）。**`--suite cec2022` の 2 設定（`j61_mc10` / `j61_mc20`）は 90 分に収まらないのでキューへ書き戻した**（j61 (m2)）
- **環境**: numpy 2.4.6 / cma 4.5.0 / scipy 1.17.1、python 3.11.17（`run.sh` の `PY` ＝ `/usr/local/bin/python3`）。CPU Intel Xeon @2.80GHz（family 6 / **model 85 / stepping 7**、4 CPUs）、avx512f あり、BLAS は openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1`。容器仕様は j59 (a5)・j59 (a20)・j60 (d2)・j58 と同じ
- **全体表 2D（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals_mean`）**: **`MC-ESO` 95.0 / 95.4 / 96.2 / 96.2、781（18 手法中 1 位）** ｜ `MC-ESO-v3` 95.2 / 95.6 / 96.2 / 96.2、773 ｜ IMODE 92.1 ｜ ARRDE 91.5 ｜ EBOwithCMAR / jSO / L-SHADE 91.2 ｜ ELSHADE-SPACMA 90.2 ｜ LSHADE-SPACMA 89.6 ｜ SPS-L-SHADE-EIG 89.2 ｜ NL-SHADE-RSP 87.5 ｜ L-SRTDE 85.8 ｜ IPOP-CMA-ES 82.9 ｜ AMALGAM-SO 82.5 ｜ EA4eig 79.8 ｜ BIPOP-CMA-ES 77.5 ｜ BIPOP-aCMA-ES 60.0 ｜ RCMAES 52.9
- **全体表 5D**: **`MC-ESO` 79.4 / 80.2 / 81.5 / 82.1、3849（1 位）** ｜ `MC-ESO-v3` 78.3 / 79.4 / 80.2 / 81.0、3927 ｜ ELSHADE-SPACMA 74.6 ｜ LSHADE-SPACMA 72.3 ｜ jSO 62.5 ｜ EBOwithCMAR 62.3 ｜ L-SRTDE 59.6 ｜ ARRDE 59.4 ｜ L-SHADE 58.5 ｜ AMALGAM-SO 57.7 ｜ IPOP-CMA-ES 57.3 ｜ BIPOP-CMA-ES 54.2 ｜ SPS-L-SHADE-EIG 52.5 ｜ RCMAES 52.1 ｜ BIPOP-aCMA-ES 47.9 ｜ EA4eig 45.2 ｜ NL-SHADE-RSP 43.8 ｜ IMODE 42.7
- **全体表 10D**: **`MC-ESO` 66.0 / 67.1 / 71.5 / 73.5、10508（1 位）** ｜ `MC-ESO-v3` 66.0 / 66.7 / 71.0 / 72.3、10449 ｜ LSHADE-SPACMA 53.1 ｜ AMALGAM-SO 52.9 ｜ IPOP-CMA-ES 49.8 ｜ RCMAES / BIPOP-CMA-ES 49.0 ｜ ELSHADE-SPACMA 41.7 ｜ BIPOP-aCMA-ES 41.5 ｜ L-SRTDE 36.9 ｜ EBOwithCMAR 34.8 ｜ ARRDE 34.6 ｜ jSO 34.4 ｜ L-SHADE 32.1 ｜ SPS-L-SHADE-EIG 29.2 ｜ NL-SHADE-RSP 20.2 ｜ IMODE 19.6 ｜ EA4eig 18.5
- **run 内の CEC 式の平均誤差の順位（18 手法、低いほど良い）**: 2D IMODE 7.40 ＜ EBOwithCMAR 7.62 ＜ `MC-ESO-v3` 7.73 ＜ **`MC-ESO` 8.04（4 位）** ＜ SPS-L-SHADE-EIG 8.25 ｜ 5D **`MC-ESO` 7.10（1 位）** ＜ EBOwithCMAR 7.17 ＜ `MC-ESO-v3` 7.25 ＜ LSHADE-SPACMA 7.56 ＜ ELSHADE-SPACMA 7.79 ｜ 10D **`MC-ESO` 6.23（1 位）** ＜ LSHADE-SPACMA 7.46 ＝ `MC-ESO-v3` 7.46 ＜ EBOwithCMAR 8.12 ＜ ELSHADE-SPACMA / IPOP-CMA-ES 8.31
- **関数別 SR@1e-10 の `MC-ESO` → `MC-ESO-v3`（動いた関数のみ）**: 2D v3 が上 1 [F19 95→100]、下 0、残り 23 関数は同値 ｜ 5D v3 が上 2 [F16 70→90, F22 95→100]、下 4 [F15 20→10, F18 75→65, F20 20→5, F23 60→45]、同値 18 ｜ 10D v3 が上 1 [F21 55→85]、下 5 [F07 100→95, F16 10→5, F17 35→30, F22 75→70, F23 10→0]、同値 18
- **新既定が比較手法に負けている関数（2D、`analyze_quick` [3]）**: F18 80%（最良 LSHADE-SPACMA 100%）・F19 95%（AMALGAM-SO 100%）・F20 95%（EA4eig 100%）・F24 15%（EBOwithCMAR 25%）。10D で全 18 手法が 0% なのは F18 / F19 / F20 / F24 を含む
- **`evals_succ_mean`（`MC-ESO` と `MC-ESO-v3` の両方で成功のある関数だけの対応平均）**: 2D 780.8 対 772.8（24 関数、v3 が -8.1）｜ 5D 3849.0 対 3926.5（22 関数、v3 が +77.5）｜ 10D 10508.0 対 10448.5（20 関数、v3 が -59.6）
- **Wilcoxon（`best_f`、reference `MC-ESO`、α=0.05 両側、新既定の 勝/敗）**: 2D は 16 手法すべてに勝ち越し（RCMAES 14勝0敗・BIPOP-aCMA-ES 14勝0敗・BIPOP-CMA-ES 11勝0敗 … IMODE 1勝1敗）、**`MC-ESO-v3` には 0勝0敗（有意差のある関数なし）** ｜ 5D は LSHADE-SPACMA にだけ 4勝6敗、他 15 手法に勝ち越し（EA4eig 17勝2敗が最大）、`MC-ESO-v3` には 1勝1敗 ｜ 10D は 16 手法すべてに勝ち越し（IMODE 20勝0敗・NL-SHADE-RSP 20勝2敗 … LSHADE-SPACMA 9勝5敗）、**`MC-ESO-v3` には 8勝0敗**（F15 A12=0.77・F19 0.84・F23 0.82 large、F20 0.65・F24 0.70 medium、F02 / F16 / F17 small）
- **2D の `MC-ESO` を `MC-ESO-v3` と比べたときの注意**: 2D では SR の全体平均で v3 が 0.2pt 上（F19 の 1 関数のみ）で、Wilcoxon の有意差はゼロ。**2D では新旧の既定を区別できない**
- **記録**: `analysis/single/j61/m/aq_j61_{md2,md5,md10}.txt`（`analyze_quick.py` 全文、[1][3][4][5]）と `arms_j61_{md2,md5,md10}.txt`（関数別 SR@1e-10・`evals_succ_mean`・Wilcoxon の明細）。数値の本体は `runs/20261029_{173913_j61_md2,174903_j61_md5,180905_j61_md10}_quick`（`./run.sh publish` 済み、CSV.gz）
- **書かれたとおりに行かなかったこと**: (1) `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順どおり空 stub を `PYTHONPATH` に置き、`python3 -m pip`（3.11 側）で `--ignore-installed blinker` 付きに入れ直した（容器の `pip` は 3.13 を指す。`minionpy==1.9.1` / `nevergrad==1.0.12` も必要）。(2) CEC2022 の 2 設定は時間が足りずキューへ書き戻した。(3) `find analysis -type f` は **1442 で上限の約 400 を大きく超えている**（この run で +6。削除はしていない）

### 2026-10-23 j60 (c20) CEC2022 20D を完走 — 3 腕すべてがパネル内 CEC 順位で基準 8.17 を下回る（最良 `v4_k50fR_r50` 5.92）が比較手法 5 つに届かず、20D では `recover_age=50` も効く（2D の bit 一致と違う）。Wilcoxon は 12 関数で基準が腕に勝った関数 0
### 2026-10-23 j60 (d2 / d2ho / c10) 2D の 2 設定と CEC2022 10D を完走 — 2D では `recover_age=50` が 1 セルも効かず（`v4_k50_r50` は既定と bit 一致、`v4_k50fR_r50` は `v4_k50fR` と bit 一致）、`fR` を含む 2 腕のパネル CEC 順位は 4.44（基準 4.52）／SR@1e-10 は調整 94.8%（基準 94.4%）・hold-out 94.4%（基準 94.6%）。CEC2022 10D はパネル順位で 3 腕すべてが基準 6.62 を下回る（`v4_k50fR_r50` 5.54 が最良）が、13 手法の run 内順位では基準 8.83・腕も 6.29〜7.92 で比較手法（ARRDE 4.00・LSHADE-SPACMA 4.54・L-SHADE 4.96）に届かない。**CEC2022 20D はキューへ書き戻した**
### 2026-10-23 j59 (a10 / a10ho) 10D の 2 設定を完走 — パネル CEC 順位は調整・hold-out ともに 6 腕中 5 腕が基準（4.62 / 4.75）を下回り（最良は `v4_k50fR` 4.21 / 4.46、どちらでも `v4_k50f_np` だけが基準より上）、SR@1e-10 は調整で基準 64.2% に並ぶのが `v4_k50_r50` `v4_k50fR_r50np` の 2 腕・上回る腕なし、hold-out では 4 腕が基準 63.3% を上回る（最良 `v4_k50fR_r50` `v4_k50fR_r50np` 63.8%）
### 2026-10-23 j59 (a5 / a5ho) 5D の 2 設定を完走 — 調整 instance ではパネル CEC 順位で基準 4.06 を下回る腕はゼロ（最良 `v4_k50_r50` / `v4_k50fR_r50` / `v4_k50fR_r50np` 4.12）、hold-out でも 6 腕すべてが基準 4.10 以上。SR@1e-10 は調整で `v4_k50f_np` 78.3% / `v4_k50f_r50` 77.9% / `v4_k50fR_r50` 76.9% が基準 76.7% 以上、hold-out では `v4_k50fR` 77.1% だけが基準 76.7% を上回る
### 2026-10-23 j59 (a20) 20D を完走 — パネル CEC 順位は 6 腕すべて基準 5.81 を下回る（`v4_k50f_r50` 4.56 が最良）、SR@1e-10 も 6 腕すべて基準 35.4% 以上（`v4_k50fR` / `v4_k50f_r50` 37.5%）。F21/F22 の低下は 6 腕すべてで残った

