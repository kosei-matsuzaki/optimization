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

**2026-10-13 に `v2_scan2d` が既定になった**ので、以下のジョブの `MC-ESO` は旧 `v2_scan2d` と同じもの（`v2_scan2d` は手法の並びから外した）。比べる旧既定が要るときは `MC-ESO-v2`。

いまは無し。


1. **【v4 の全体フローの腕の 20 次元】** `./run.sh quick --all --dim 20 --max-evals 50000 --n-runs 20 --instances 1-5 --methods "MC-ESO,v4_rank5,v4_mirror,v4_dither,ctl_plainwave,ctl_sepipop" --jobs 4`（label `j51_a20`）。**集計**: 標準の集計に加え、j47 のパネルと並べた CEC 順位: `mkdir -p /tmp/j47p/dim20 && cp -r analysis/single/j47/stats /tmp/j47p/dim20/` として `python scripts/single/cec_panel.py --dim 20 --panel /tmp/j47p --arms results/<この run>`（j47 の MC-ESO とこの run の MC-ESO が一致するかも書く。一致しなければ CEC 順位は参考扱い）。per-run stats は `analysis/single/j51/stats/`。**記録**: `analysis/single/j51/`。採否の判断はしない。 (claimed 2026-10-17 21:30 UTC)

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

### 2026-10-17 j50（v4 の全体フローの腕と対照）10D の 2 設定を完走 — 2 run をまたいで既定を上回る腕は `v4_pace` のみ（パネル CEC 順位 4.25/4.33 < 基準 4.62/4.75。ただし SR@1e-10 は両 run で基準以下、Wilcoxon は 1勝3敗 / 0勝5敗）

- **ランした分**: `./run.sh quick --all --dim 10 --max-evals 25000 --n-runs 20 --instances I [--seed-offset 7] --methods "MC-ESO,v4_pace,v4_rank5,v4_rank3,v4_mirror,v4_dither,v4_native,ctl_plainwave,ctl_sepipop" --jobs 4` を 2 本、逐次で。`j50_a10`(1-5) 19:32:18-20:06:32 UTC **wall 34.2 分**、`j50_a10ho`(6-10 +offset 7) 20:11:38-20:45:07 **33.5 分**。`result.json` は 2 本とも `status=done`、`set=all-d10`。**各 run 24/24 関数 × 9 手法 = 216 行**（wilcoxon 192 行 = 24×8、(関数, 手法) の重複・欠落ゼロ、stats 各 24 ファイル）。shard 分割なし、OOM なし。**MC-ESO の 24 関数 × 7 列は j49_p10 / j49_p10ho と 1 セルも違わず一致**（各 168/168）＝ パネル順位の比較は有効。
- **環境**: numpy 2.4.6 / cma 4.5.0 / scipy 1.17.1 / minionpy 1.9.1、python 3.11.17（`/usr/local/bin/python3` ＝ `run.sh` の `PY`）。CPU Intel Xeon @2.10GHz、family 6 / model 207 / stepping 2、4 CPUs、avx512f あり。BLAS = openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1`。j49 / j50(5D) と同じ容器仕様。`pip install -r requirements.txt` は pynmmso のビルドで落ちるので findings.md の空 stub を `PYTHONPATH` に置いた。
- **全体（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals(own)`）**: 調整 `j50_a10`: v4_rank5 64.4/64.8/69.4/70.8（10515）、**MC-ESO 64.2/65.0/68.8/70.4（10376）**、v4_dither 64.0/64.8/68.3/70.0（10622）、v4_mirror 63.5/64.6/67.1/68.8（10238）、v4_rank3 63.3/63.8/68.1/69.4（10468）、v4_pace 62.9/64.0/68.1/69.8（11017）、ctl_sepipop 54.4/58.5/64.4/66.5（6057）、v4_native 52.9/56.7/58.8/65.4（10758）、ctl_plainwave 37.5/38.1/39.4/41.2（8627）。hold-out `j50_a10ho`: **MC-ESO 63.3/64.6/68.3/69.4（10477）**、v4_pace 62.9、v4_rank3 62.1、v4_mirror 61.5、v4_rank5 61.0、v4_dither 60.8、ctl_sepipop 55.6、v4_native 50.8、ctl_plainwave 34.2。**調整で既定を上回るのは `v4_rank5`(+0.2pt) だけ、hold-out では 8 腕すべてが既定以下（-0.4〜-12.5pt）。**
- **`evals_succ_mean`（両方に成功のある関数だけの対応平均、調整 / hold-out）**: v4_pace -20.9(nf19) / +35.5(nf20)、v4_rank5 +139.1(nf19) / -56.4(nf18)、v4_rank3 +92.1(nf19) / +205.2(nf20)、v4_mirror -137.6(nf19) / -178.3(nf19)、v4_dither +187.2(nf19) / +99.9(nf19)、v4_native +918.9(nf18) / +1089.6(nf18)、ctl_plainwave -451.6(nf14) / -44.7(nf12)、ctl_sepipop -4447.7(nf18) / -3811.2(nf19)。片方だけ成功の関数: 調整は基準のみ F18（native）・F03,F04,F16,F17,F18（plainwave）・F22（sepipop）、腕のみ F23（pace / dither）。hold-out は基準のみ F16,F23（rank5）・F23（mirror / dither / sepipop）・F18,F23（native）・F03,F04,F07,F16,F17,F18,F22,F23（plainwave）、腕のみ無し。
- **関数別 SR@1e-10（全 24 関数）**: どちらの run も**9 手法すべて同値は 7 関数**（調整 F01・F02・F05・F06・F11 = 100、F15・F19・F20・F24 = 0 のうち 9 関数、hold-out は F01・F02・F05・F06・F11 = 100 と F15・F18・F19・F20・F23・F24 = 0 の 11 関数）。調整（基準 → pace/rank5/rank3/mirror/dither/native/plainwave/sepipop）: F03 100→100×6/0/100、F04 65→65,70,75,70,75,65,0,60、F07 90→90,95,100,95,85,75,5,100、F08 100→100,95,95,90,95,50,85,100、F09 100→100,95,95,100,90,100,95,100、F10 100→100×5/100/95,100、F12 100→100,100,100,100,95,35,10,100、F13 95→95,100,100,100,100,5,0,10、F14 100→100×6/90/100、F16 25→25,30,15,10,10,0,0,10、F17 25→25,45,30,25,25,0,0,5、F18 0→0,0,10,5,0,0,0,0、F21 100→75,85,90,75,90,100,15,20、F22 40→35,30,10,55,60,40,5,0、F23 0→0,0,0,0,10,0,0,0。hold-out: F03 100→100×6/0/100、F04 50→50,45,50,40,40,45,0,40、F07 95→95,95,95,90,85,90,0,100、F08 100→100,95,100,100,100,75,60,100、F09 100→100,95,100,95,95,85,95,100、F10 100→100×5/100/90,100、F12 95→95,85,90,95,90,10,0,100、F13 100→100×5/0/0,10、F14 100→100×6/65/100、F16 20→25,0,25,30,15,5,0,45、F17 50→50,45,40,35,45,0,0,5、F21 70→75,80,65,65,75,70,10,30、F22 40→20,25,25,25,15,40,0,5。
- **CEC 式の平均誤差の順位（j49 の 10 手法パネルに腕を差し替えた順位、低いほど良い。`scripts/single/cec_panel.py`）**: 調整（panel `j49_p10`）**MC-ESO 4.62** に対し **v4_pace 4.25**（平均誤差が下 7 関数 / 上 2 関数）、ctl_sepipop 4.31（対照）、v4_mirror 4.94、v4_dither 5.08、v4_rank5 5.19、v4_rank3 5.29、v4_native 5.50、ctl_plainwave 6.92。hold-out（panel `j49_p10ho`）**MC-ESO 4.75** に対し ctl_sepipop 4.15（対照）、**v4_pace 4.33**（下 6 / 上 1）、v4_dither 4.81、v4_mirror 4.92、v4_rank3 4.92、v4_rank5 4.94、v4_native 5.73、ctl_plainwave 7.08。**2 run をまたいで基準を上回る腕は `v4_pace` のみ**（5D では調整 4.08・hold-out 4.31 で基準 4.06 / 4.10 を上回れていなかった）。参考に腕 9 手法だけの `analyze_quick` [5]: 調整 v4_pace 3.79 < ctl_sepipop 4.40 < MC-ESO 4.54、hold-out v4_pace 3.85 < ctl_sepipop 4.29 < MC-ESO 4.67。
- **Wilcoxon（MC-ESO を reference、両側 p<0.05、A12 の向きで勝敗。調整 / hold-out の 勝/負）**: v4_pace 1/3（勝 F21 0.62 small、負 F15 0.13・F19 0.17・F24 0.26 large）/ 0/5（負 F15 0.21・F19 0.21・F24 0.23 large、F16 0.36 medium、F23 0.38 small）、v4_rank5 0/1（F17 0.28 large）/ 0/1（F06 0.29 medium）、v4_rank3 4/1 / 4/0、v4_mirror 2/0 / 2/1、v4_dither 1/0 / 1/0、v4_native 12/0 / 11/0、ctl_plainwave 17/3 / 18/3、ctl_sepipop 13/3 / 15/2。**`v4_pace` は 2 run とも F15・F19・F24 で既定に有意に勝つ（＝多峰 3 関数で平均誤差が下）が、有意に負ける関数は 5D と同様に最大 1 つ。`v4_native`（pycma の波を外した版）と対照 2 本は既定に有意に負ける関数が 11〜18 ある。**
- **書かれたとおりに回せなかった点**: 無し（2 設定とも 1 回のルーチンで完走）。手法・次元・予算・instance・seed-offset・n・label・集計はジョブ記載どおり。
- **記録**: per-run stats は `analysis/single/j50/<label>/dim10/stats/*.csv.gz`（2 label × 24、`cec_panel.py --panel` で読めることを確認）、summary / wilcoxon は `analysis/single/j50/{summary,wilcoxon}_j50_<label>.csv`、analyze_quick 全文は `aq_j50_<label>.txt`、腕ごとの関数別表と Wilcoxon 明細は `arms_j50_<label>.txt`、パネル順位は `cecpanel_j50_<label>.txt`。**`find analysis -type f` は 489 で上限の約 400 を超えた**（j50 の 5D の時点で 434。削除は行っていない＝畳むのは対話セッションの担当）。採否の判断はしない。

### 2026-10-17 j50（v4 の全体フローの腕と対照）5D の 2 設定のみ完走 — 腕はどれも MC-ESO を明確に上回らない（SR@1e-10 は調整 +0.8pt（`v4_mirror`）〜 -5.7pt、hold-out は全腕が -0.7〜-6.9pt）。対照 2 本は大きく下回る。10D の 2 設定は時間に収まらずキューへ書き戻した

- **ランした分**: `./run.sh quick --all --dim 5 --max-evals 12500 --n-runs 20 --instances I [--seed-offset 7] --methods "MC-ESO,v4_pace,v4_rank5,v4_rank3,v4_mirror,v4_dither,v4_native,ctl_plainwave,ctl_sepipop" --jobs 4` を 2 本、逐次で。`j50_a5`(1-5) 17:32:24-17:59:45 UTC **wall 27.4 分**、`j50_a5ho`(6-10 +offset 7) 18:04:35-18:30:38 **26.1 分**。`result.json` は 2 本とも `status=done`、`set=all-d5`。**各 run 24/24 関数 × 9 手法 = 216 行**（wilcoxon 192 行 = 24×8、(関数, 手法) の重複・欠落ゼロ、stats 各 24 ファイル）。shard 分割なし、OOM なし。**MC-ESO の 24 関数 × 7 列は j49_p5 / j49_p5ho と 1 セルも違わず一致**（168/168）＝ instance・seed・環境が j49 のパネルと揃っており、CEC 順位の比較は有効。
- **環境**: numpy 2.4.6 / cma 4.5.0 / scipy 1.17.1 / minionpy 1.9.1、python 3.11.17（`/usr/local/bin/python3` ＝ `run.sh` の `PY`）。CPU Intel Xeon @2.10GHz、family 6 / model 207 / stepping 2、4 CPUs、avx512f あり。BLAS = openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1`。j49 と同じ容器仕様。`pip install -r requirements.txt` は pynmmso のビルドで落ちるので findings.md の空 stub を `PYTHONPATH` に置いた。
- **全体（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals(own)`）**: 調整 `j50_a5`: **MC-ESO 76.7/78.3/80.4/81.2（3919）**、v4_mirror 77.5/78.5/80.0/80.4（3579）、v4_rank5 76.9/78.8/80.4/81.5（4084）、v4_pace 76.9/78.3/80.2/81.0（3941）、v4_rank3 76.5/78.1/79.4/80.0（4149）、v4_dither 75.4/76.9/77.7/78.8（4189）、v4_native 71.0/73.8/77.9/80.8（3865）、ctl_sepipop 65.6/71.5/72.7/74.0（2647）、ctl_plainwave 41.7/45.4/50.6/54.4（4957）。hold-out `j50_a5ho`: **MC-ESO 76.7/77.7/78.8/79.4（3958）**、v4_rank3 76.0、v4_pace 75.6、v4_dither 75.6、v4_rank5 75.2、v4_mirror 74.0、v4_native 69.8、ctl_sepipop 65.4、ctl_plainwave 41.0。**調整で既定を上回るのは `v4_mirror`(+0.8pt) と `v4_rank5`/`v4_pace`(+0.2pt) だけで、hold-out では 8 腕すべてが既定以下。**
- **`evals_succ_mean`（両方に成功のある関数だけの対応平均、調整 / hold-out）**: v4_pace +21.7(nf23) / -81.6(nf22)、v4_rank5 +275.5(nf22) / -151.9(nf22)、v4_rank3 +340.7(nf22) / +309.0(nf22)、v4_mirror -229.2(nf22) / +21.3(nf21)、v4_dither +506.5(nf21) / +254.5(nf22)、v4_native -54.0(nf23) / +53.8(nf22)、ctl_plainwave +1240.5(nf20) / +1427.9(nf18)、ctl_sepipop -1190.0(nf21) / -981.1(nf22)。片方だけ成功の関数: 調整は基準のみ成功が F15（dither）・F19（rank5/rank3/mirror/dither）・F04,F18,F19（plainwave）・F19,F20（sepipop）、hold-out は基準のみ F15（mirror）・F04,F15,F18,F20（plainwave）、腕のみ F19（sepipop）。
- **関数別 SR@1e-10（全 24 関数）**: どちらの run も**いずれかの腕で動く関数は 17/24**、**9 手法すべて同値は 7 関数**（F01・F02・F05・F06・F10・F11 = 100、F24 = 0）。調整（基準 → pace/rank5/rank3/mirror/dither/native/plainwave/sepipop）: F03 100→100×6/5/100、F04 85→85,85,85,85,80,85,0,80、F07 100→100×6/65/100、F08 100→100,100,95,100,100,100,85,100、F09 100→100,95,100,100,100,100,90,100、F12 100→100×6/30/100、F13 100→100×6/15/10、F14 100→100×6/15/100、F15 5→15,5,10,5,0,5,0,15、F16 60→55,75,90,65,70,45,0,75、F17 80→80,85,85,90,90,15,0,80、F18 55→55,50,75,75,60,10,0,25、F19 10→5,0,0,0,0,10,0,0、F20 5→10,5,5,10,0,5,5,0、F21 95→100,95,100,100,100,95,35,45、F22 100→95,95,85,95,95,100,55,45、F23 45→45,55,5,35,15,35,0,0。hold-out: F04 75→75,70,70,65,70,75,0,60、F07 100→100×6/75/100、F08 100→100,100,100,100,95,100,80,100、F09 100→100,95,100,95,100,100,85,95、F12 100→100,100,100,100,90,95,20,100、F13 100→100×6/15/15、F14 100→100×6/40/100、F15 10→5,30,15,0,10,10,0,5、F16 70→65,70,75,65,75,55,0,50、F17 85→85,70,80,80,90,5,0,70、F18 70→70,60,75,50,55,10,0,55、F19 0→0×6/0/5、F20 5→5,5,10,20,10,5,0,5、F21 85→80,95,95,75,90,85,20,65、F22 95→90,85,90,90,95,95,45,45、F23 45→40,25,15,35,35,40,0,0、F03 100→100×6/5/100。
- **CEC 式の平均誤差の順位（j49 の 10 手法パネルに腕を差し替えた順位、低いほど良い。`scripts/single/cec_panel.py`）**: 調整（panel `j49_p5`）**MC-ESO 4.06** に対し v4_mirror **3.88**（基準より良い唯一の腕、平均誤差が下 5 関数 / 上 5 関数）、v4_rank3 4.06（同値）、v4_pace 4.08、v4_dither 4.19、v4_native 4.23、v4_rank5 4.35、ctl_sepipop 5.00、ctl_plainwave 6.50。hold-out（panel `j49_p5ho`）**MC-ESO 4.10** に対し v4_rank3 **4.06**（唯一わずかに良い）、v4_native 4.25、v4_dither 4.27、v4_rank5 4.27、v4_pace 4.31、v4_mirror 4.65、ctl_sepipop 5.08、ctl_plainwave 6.73。**調整で最良の `v4_mirror` は hold-out で 8 腕中 6 番目、hold-out で最良の `v4_rank3` は調整で基準と同値＝2 run をまたいで基準を上回る腕は無い。** 参考に腕 9 手法だけの `analyze_quick` [5]: 調整 v4_mirror 4.12 < v4_pace 4.29 < MC-ESO 4.46、hold-out MC-ESO 4.23 < v4_rank3 4.35。
- **Wilcoxon（MC-ESO を reference、両側 p<0.05、A12 の向きで勝敗。調整 / hold-out の 勝/負）**: v4_pace 0/0 / 0/0、v4_rank5 0/0 / 0/1（F15 A12=0.29 medium）、v4_rank3 1/0（F23 0.81 large）/ 1/0（F23 0.74 large）、v4_mirror 0/1（F20 0.33 medium）/ 0/1（F20 0.33 medium）、v4_dither 1/0（F23 0.74 large）/ 0/0、**v4_native 4/0（F13 0.85・F14 0.78・F17 0.87・F18 0.80 すべて large）/ 5/0（F13 0.92・F14 0.82・F17 0.93・F18 0.75 large、F16 0.56 small）**、ctl_plainwave 14/0 / 16/0、ctl_sepipop 13/0 / 12/0。**どの腕も既定に有意に負ける関数は最大 1 つ、有意に勝つ関数は 0。対照 2 本と `v4_native`（pycma の波を外した版）は既定に有意に負ける関数が複数ある。**
- **書かれたとおりに回せなかった点**: ジョブは 5D / 10D の 4 設定だが、**1 本 27 分で 4 本では 90 分の目安を超えるため 5D の 2 設定だけを回し、10D の 2 設定（`j50_a10` / `j50_a10ho`）はキューの先頭にジョブとして書き戻した**（claim マーカーなし）。「上と同じ 4 設定」は、削除済みの j49 のジョブ記載と j49 の作業ログから (D, B, I) = (5, 12500, 1-5) / (5, 12500, 6-10 +offset 7) / (10, 25000, 1-5) / (10, 25000, 6-10 +offset 7) と読んだ。キューの `## 測定ジョブ` 直下にある「いまは無し。」の 1 行は 1dbaa9d から残っている古い行で、下のジョブは有効なものとして扱った。
- **記録**: per-run stats は `analysis/single/j50/<label>/dim5/stats/*.csv.gz`（2 label × 24、`cec_panel.py --panel` で読めることを確認）、summary / wilcoxon は `analysis/single/j50/{summary,wilcoxon}_j50_<label>.csv`、analyze_quick 全文は `aq_j50_<label>.txt`、腕ごとの関数別表と Wilcoxon 明細は `arms_j50_<label>.txt`、パネル順位は `cecpanel_j50_<label>.txt`。**`find analysis -type f` は 431 で上限の約 400 を超えた**（j49 の時点で 373。削除は行っていない＝畳むのは対話セッションの担当）。採否の判断はしない。

### 2026-10-17 j49（クラウドの 10 手法パネル 5D/10D）— 4 run すべて 24/24 完走。CEC 順位は 5D で MC-ESO が 1 位（4.06 / 4.10）、10D で LSHADE-SPACMA に次ぐ 2 位（4.62 / 4.75）。SR@1e-10 は 4 run とも MC-ESO が最上位

- **ランした分**: `./run.sh quick --all --dim D --max-evals B --n-runs 20 --instances I [--seed-offset 7] --methods "MC-ESO,ELSHADE-SPACMA,LSHADE-SPACMA,EBOwithCMAR,L-SHADE,IPOP-CMA-ES,ARRDE,NL-SHADE-RSP,RCMAES,BIPOP-aCMA-ES" --jobs 4` を書かれたとおり 4 本、逐次で。`j49_p5`(5, 12500, 1-5) 15:37:00-15:45:28 UTC **wall 8.5 分**、`j49_p5ho`(5, 12500, 6-10 +offset 7) 16:10:28-16:19:01 **8.6 分**、`j49_p10`(10, 25000, 1-5) 16:19:19-16:33:53 **14.6 分**、`j49_p10ho`(10, 25000, 6-10 +offset 7) 16:34:11-16:48:59 **14.8 分**。`result.json` は 4 本とも `status=done`、`set=all-d5` / `all-d10`。**各 run 24/24 関数 × 10 手法 = 240 行**（wilcoxon 216 行 = 24×9、(関数, 手法) の重複・欠落ゼロ、stats 各 24 ファイル）。shard 分割はしていない。OOM なし。
- **環境**: numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1 / scipy 1.17.1。CPU Intel Xeon @2.10GHz、family 6 / model 207 / stepping 2、4 CPUs、avx512f あり。BLAS = openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1`。j48 と同じ容器仕様。`pip install -r requirements.txt` は pynmmso のビルドで落ちるので findings.md の空 stub を `PYTHONPATH` に置いた。**ただし j48 と違い `python3`（3.11.17、`/usr/local/bin/python3`）側へ `numpy==2.4.6` 他を入れて実行した**（素の `pip` は 3.13 向けで、`run.sh` の `PY` は `python3` = 3.11 を選ぶ）。
- **全体（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals_mean`＝自分の成功関数の平均）**: 5D 調整 `j49_p5`: **MC-ESO 76.7/78.3/80.4/81.2（3919）**、ELSHADE-SPACMA 74.6/76.2/78.3/79.2（6611）、LSHADE-SPACMA 71.5/72.9/73.5/74.4（4941）、EBOwithCMAR 64.8/69.6/77.5/80.6（7935）、L-SHADE 62.1/69.0/77.5/79.0（8289）、ARRDE 60.6/64.0/70.6/72.5（4270）、IPOP-CMA-ES 57.1/63.1/65.0/66.7（2926）、RCMAES 48.8/48.8/54.2/55.2（1771）、BIPOP-aCMA-ES 48.1/49.0/50.2/53.1（1356）、NL-SHADE-RSP 45.2/49.2/55.2/67.5（10245）。5D hold-out `j49_p5ho`: **MC-ESO 76.7/77.7/78.8/79.4（3958）**、ELSHADE-SPACMA 75.0、LSHADE-SPACMA 71.9、EBOwithCMAR 64.0、L-SHADE 61.0、ARRDE 60.0、IPOP-CMA-ES 58.1、RCMAES 52.9、BIPOP-aCMA-ES 47.1、NL-SHADE-RSP 43.1。10D 調整 `j49_p10`: **MC-ESO 64.2/65.0/68.8/70.4（10376）**、LSHADE-SPACMA 55.2、RCMAES 48.3、IPOP-CMA-ES 47.9、ELSHADE-SPACMA 42.9、BIPOP-aCMA-ES 41.7、EBOwithCMAR 38.3、ARRDE 37.3、L-SHADE 33.3、NL-SHADE-RSP 20.6。10D hold-out `j49_p10ho`: **MC-ESO 63.3/64.6/68.3/69.4（10477）**、LSHADE-SPACMA 55.6、IPOP-CMA-ES 49.8、RCMAES 49.0、BIPOP-aCMA-ES 41.0、ELSHADE-SPACMA 42.3、EBOwithCMAR 35.2、L-SHADE 34.4、ARRDE 34.8、NL-SHADE-RSP 19.0。**4 run とも SR@1e-10 は MC-ESO が最上位。** SR@1e-2 では 5D 調整で EBOwithCMAR 80.6 が MC-ESO 81.2 に迫り、10D hold-out では EBOwithCMAR 69.0 / L-SHADE 67.7 が MC-ESO 69.4 のすぐ下。
- **`evals_succ_mean`（MC-ESO と各手法の両方で成功のある関数だけの対応平均）**: この run は腕と既定の対ではなくパネルなので、上の `evals_mean`（各手法の自分の成功関数平均）を併記した。対応平均の細目は `analysis/single/j49/aq_j49_*.txt` の [1]・[3] 欄（関数ごとの `evals(ref/base)`）にある。MC-ESO は 5D で 3919 / 3958、10D で 10376 / 10477 で、同じ SR 水準の手法（ELSHADE-SPACMA 6611 / 6566、LSHADE-SPACMA 4941 / 4728）より少ない評価で届いている。
- **関数別 SR@1e-10（MC-ESO、全 24 関数）**: 5D 調整 F01-F03 100・F04 85・F05-F14 100・F15 5・F16 60・F17 80・F18 55・F19 10・F20 5・F21 95・F22 100・F23 45・F24 0。5D hold-out F04 75・F15 10・F16 70・F17 85・F18 70・F19 0・F20 5・F21 85・F22 95・F23 45・F24 0（他は調整と同じ、F01-F03 と F05-F14 は 100）。10D 調整 F04 65・F07 90・F13 95・F15 0・F16 25・F17 25・F18 0・F19 0・F20 0・F21 100・F22 40・F23 0・F24 0（他 100）。10D hold-out F04 50・F07 95・F12 95・F16 20・F17 50・F21 70・F22 40、F15・F18・F19・F20・F23・F24 は 0（他 100）。**基準となる旧 run が無い（パネルの初回）ので「変化」ではなく実測値を列挙した。**
- **MC-ESO がパネル中の最良ベースラインに SR@1e-10 で負ける関数**: 5D 調整 F04・F15・F16・F17・F18・F20・F21（7 関数）、5D hold-out F04・F15・F17・F18・F19・F20・F21（7）、10D 調整 F07・F13・F15・F17・F18・F20（6）、10D hold-out F04・F07・F12・F15・F16・F17・F18・F20・F21（9）。**F15-RastriginRot・F17-SchafferF7・F18-SchafferF7ill・F20-Schwefel は 4 run すべてで負ける。**
- **Wilcoxon（MC-ESO を reference、両側 p<0.05、A12 の向きで勝敗、勝ち/負けの関数数）**: 5D 調整 ELSHADE-SPACMA 5/6、LSHADE-SPACMA 5/5、EBOwithCMAR 11/2、ARRDE 11/2、L-SHADE 8/3、NL-SHADE-RSP 14/1、RCMAES 14/1、BIPOP-aCMA-ES 15/1、IPOP-CMA-ES 16/0。5D hold-out ELSHADE-SPACMA 6/5、LSHADE-SPACMA 5/5、EBOwithCMAR 10/2、ARRDE 10/3、L-SHADE 9/3、RCMAES 13/1、IPOP-CMA-ES 13/0、NL-SHADE-RSP 14/2、BIPOP-aCMA-ES 16/0。10D 調整 LSHADE-SPACMA 7/8、RCMAES 7/5、EBOwithCMAR 11/4、BIPOP-aCMA-ES 11/3、ARRDE 12/2、ELSHADE-SPACMA 13/4、L-SHADE 14/2、IPOP-CMA-ES 15/3、NL-SHADE-RSP 19/1。10D hold-out LSHADE-SPACMA 7/5、RCMAES 9/5、EBOwithCMAR 11/4、BIPOP-aCMA-ES 11/1、ELSHADE-SPACMA 12/2、ARRDE 13/2、L-SHADE 14/2、IPOP-CMA-ES 17/2、NL-SHADE-RSP 18/1。**MC-ESO が負ける関数の集合（両 5D run で F06・F15・F17・F18・F20、10D で F06・F14・F15・F19・F20・F24）。10D 調整では LSHADE-SPACMA に 7/8 で負け越す唯一の対がある。**
- **CEC 式の平均誤差の順位（[5]、各 run 24/24 関数、低いほど良い）**: 5D 調整 **MC-ESO 4.06（1 位）** < ELSHADE-SPACMA 4.48 < EBOwithCMAR 4.56 < LSHADE-SPACMA 4.62 < L-SHADE 4.96 < ARRDE 5.29 < IPOP-CMA-ES 5.40 < NL-SHADE-RSP 6.60 < RCMAES 7.27 < BIPOP-aCMA-ES 7.75。5D hold-out **MC-ESO 4.10（1 位）** < LSHADE-SPACMA 4.27 < EBOwithCMAR 4.33 < ELSHADE-SPACMA 4.81 < L-SHADE 4.96 < ARRDE 5.25 < IPOP-CMA-ES 5.46 < RCMAES 6.81 < NL-SHADE-RSP 7.00 < BIPOP-aCMA-ES 8.00。10D 調整 LSHADE-SPACMA 4.56 < **MC-ESO 4.62（2 位）** < EBOwithCMAR 4.90 < IPOP-CMA-ES 4.98 < ELSHADE-SPACMA 5.12 < RCMAES 5.35 < L-SHADE 5.50 < ARRDE 5.69 < BIPOP-aCMA-ES 6.88 < NL-SHADE-RSP 7.40。10D hold-out LSHADE-SPACMA 4.21 < **MC-ESO 4.75（2 位）** < EBOwithCMAR 4.90 < IPOP-CMA-ES 5.04 < ELSHADE-SPACMA 5.19 < L-SHADE 5.25 < RCMAES 5.42 < ARRDE 5.73 < BIPOP-aCMA-ES 6.94 < NL-SHADE-RSP 7.58。
- **記録先**: per-run stats を `analysis/single/j49/<label>/dim<D>/stats/*.csv.gz`（4 label × 24 ファイル）に書かれた構成どおり置いた。`scripts/single/cec_panel.py --dim 5 --panel analysis/single/j49/j49_p5 --arms <この run>` が読めることを確認済み（panel = MC-ESO + 9 手法、24 関数、MC-ESO の順位 4.06 を再現）。summary / wilcoxon は `analysis/single/j49/{summary,wilcoxon}_j49_<label>.csv`、analyze_quick の全文（[3] の全関数表と [4] の有意な対の一覧）は `analysis/single/j49/aq_j49_<label>.txt`。`find analysis -type f | wc -l` = 373（上限 約 400 に対し残り 27。次の j50 は 4 label × 24 = 96 ファイルを足すので上限を超える。削除はしていない）。
- **書かれたとおりに行かなかった点**: 上記の python 3.11 / 3.13 の差のみ（j48 は 3.13 で回している。numpy・cma・scipy・minionpy の版は揃えた）。採否の判断はしない。

### 2026-10-16 j48（pace_horizon の 10 次元）— 両 run 24/24 完走。`pace_horizon` は F21/F22 の悪化を消せず（調整 run で F21 100→75/75/80、hold-out で F22 40→20/20/30）、F15/F19/F24 の平均誤差の改善は 3 腕とも残る

- **ランした分**: `./run.sh quick --all --dim 10 --max-evals 25000 --n-runs 20 --methods "MC-ESO,v4_pace,v4_paceh1,v4_paceh03" --jobs 4` を 2 本。(a) `--instances 1-5`（label `j48_d10`、08:37:11-09:07:5x UTC、wall **30.7 分**）、(b) `--instances 6-10 --seed-offset 7`（label `j48_ho_d10`、09:08:21-09:24:0x UTC、wall **15.8 分**）。`result.json` は 2 本とも `status=done`、`set=all-d10`。**各 run 24/24 関数 × 4 手法 = 96 行**（wilcoxon 72 行 = 24×3、(関数, 手法) の重複・欠落ゼロ、stats 各 24 ファイル）。OOM なし。
- **環境**: numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1。CPU Intel Xeon @2.10GHz、family 6 / model 207 / stepping 2、4 CPUs、avx512f あり。BLAS = openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1` で実行。`pip install -r requirements.txt` は pynmmso のビルドで落ちるので findings.md の空 stub を `PYTHONPATH` に置き、`PATH=/usr/bin:$PATH`（3.13）に `numpy==2.4.6` を固定して j46/j47 と揃えた（素の `python3` は 3.11 でパッケージ無し）。
- **全体（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals(own)`＝自分の成功関数の平均）**: (a) 調整 `--instances 1-5`: **MC-ESO 64.2/64.8/68.5/70.4（10376）**、v4_paceh03 63.1/64.0/67.7/69.8（10902）、v4_pace 62.9/63.8/67.9/69.8（11019）、v4_paceh1 62.9/63.8/67.7/69.8（10967）。(b) hold-out `--instances 6-10 --seed-offset 7`: **MC-ESO 64.0/64.4/68.5/69.4（10494）**、v4_paceh03 64.0/64.4/68.3/69.2（10609）、v4_pace 63.5/64.0/68.1/69.0（10529）、v4_paceh1 63.5/64.0/67.9/68.8（10517）。
- **`evals_succ_mean`（両方に成功のある関数だけの対応平均）**: (a) 共通 19 関数で MC-ESO 10376 → v4_pace 10355（-21）/ v4_paceh1 10355（-21）/ v4_paceh03 10286（-90）。F23-Katsuura は 3 腕のみ成功（MC-ESO は成功 0 で平均から外れる）、MC-ESO のみ成功の関数は無し。(b) 共通 20 関数で MC-ESO 10494 → v4_pace 10529（+36）/ v4_paceh1 10517（+24）/ v4_paceh03 10609（+115）。片方だけ成功の関数は無し。
- **関数別 SR@1e-10（全 24 関数、MC-ESO → v4_pace / v4_paceh1 / v4_paceh03）**: (a) 変化は **F21-Gallagher101 100→75/75/80、F22-Gallagher21 40→35/35/35 の 2 関数のみで、どちらも 3 腕とも悪化**。残る 22 関数は 4 手法とも同値（F01/F02/F03/F05/F06/F08/F09/F10/F11/F12/F13/F14 は 100、F07 90、F04 65、F16 25、F17 20、F15・F18・F19・F20・F23・F24 は 0）。平均 64.17 → 62.92 / 62.92 / 63.12。(b) 変化は 3 関数: **F16-Weierstrass 20→25/25/25 が改善**、**F21-Gallagher101 70→75/75/75 が改善**、**F22-Gallagher21 40→20/20/30 が悪化**。残る 21 関数は同値（F01/F02/F03/F05/F06/F08/F09/F10/F11/F13/F14 は 100、F07 95、F12 95、F17 65、F04 50、F15・F18・F19・F20・F23・F24 は 0）。平均 63.96 → 63.54 / 63.54 / 63.96。
- **関数別の平均誤差（`mean_best_f`、腕 / MC-ESO の比）**: (a) 改善 7 関数（F19 0.286/0.286/0.336、F17 0.449 全腕、F23 0.478/0.496/0.484、F15 0.571/0.571/0.547、F22 0.611/0.611/0.792、F24 0.673/0.673/0.664、F16 0.668/0.734/0.734）、**悪化 2 関数（F18 1.343 全腕、F21 は MC-ESO が 0 で 3 腕は正値）**、残る 15 関数は 1 セルも違わず同値。(b) 改善 6 関数（F23 0.318/0.319/0.346、F16 0.293/0.293/0.301、F19 0.307/0.307/0.304、F21 0.596/0.921/1.066、F15 0.601/0.601/0.620、F24 0.649/0.649/0.675）、**悪化 1〜2 関数（F22 1.707/1.707/1.496、F21 は v4_paceh03 のみ 1.066 で悪化）**、残る 16 関数は同値。
- **焦点 5 関数（F15・F19・F21・F22・F24）の SR@1e-10（MC-ESO / v4_pace / v4_paceh1 / v4_paceh03）**: (a) F15 0/0/0/0、F19 0/0/0/0、F21 100/75/75/80、F22 40/35/35/35、F24 0/0/0/0。(b) F15 0/0/0/0、F19 0/0/0/0、F21 70/75/75/75、F22 40/20/20/30、F24 0/0/0/0。**F15・F19・F24 は 2 run とも 4 手法すべて SR 0% で、改善は平均誤差にしか出ない。**
- **CEC 式の平均誤差の順位 ([5]、24/24 関数)**: (a) v4_paceh03 2.35 < v4_pace 2.38 < v4_paceh1 2.46 < **MC-ESO 2.81**。(b) v4_pace 2.27 < v4_paceh1 2.40 < v4_paceh03 2.56 < **MC-ESO 2.77**。
- **Wilcoxon（MC-ESO を reference、両側 p<0.05、A12）**: (a) 3 腕とも **MC-ESO が勝つ 1 関数（F21 A12=0.62 p=0.043 medium）／負ける 3 関数**（F15 A12=0.13/0.13/0.10、F19 0.17/0.17/0.20、F24 0.26/0.26/0.24 いずれも large）。(b) 3 腕とも **MC-ESO が勝つ関数は 0／負ける 5 関数**（F15 A12=0.21/0.21/0.24、F19 0.21/0.21/0.20、F24 0.23/0.23/0.24 は large、F16 0.36/0.36/0.37、F23 0.38/0.38/0.40 は small〜medium）。
- **書かれたとおりに回せなかった点**: 無し。手法・予算・instance・seed-offset・n・label・集計はジョブ記載どおり。
- **記録**: `analysis/single/j48/{summary,wilcoxon}_j48_d10.csv` / `..._j48_ho_d10.csv` / `analyze_quick_j48_d10.txt` / `analyze_quick_j48_ho_d10.txt`、per-run stats は `stats_j48_d10/` と `stats_j48_ho_d10/` に各 24 本の `.csv.gz`。`find analysis -type f` は **265**（上限 約 400 以内）。`scripts/check_doc_size.py` は全ファイル上限内で通るが、**research_loop.md がちょうど 300/300 で上限に達した**（次の記録の前に統合が要る＝畳むのは対話セッションの担当）。history.md 984/1000 も詰まっている。
- 両 run 24/24 関数の結果である。採否の判断はしない。

### 2026-10-15 j47（v4_pace の 20 次元）— 24/24 完走。v4_pace は CEC 順位 5.81→5.23、多峰 9 関数で平均誤差が 0.27〜0.74 倍に改善、SR@1e-10 は 35.42→35.00%（F21 -5 / F22 -5）

- **ランした分**: `./run.sh quick --all --funcs <F01-F24 の完全名 24 個> --dim 20 --max-evals 50000 --n-runs 20 --instances 1-5 --methods "MC-ESO,v4_pace,IPOP-CMA-ES,LSHADE-SPACMA,ELSHADE-SPACMA,EBOwithCMAR,L-SHADE,RCMAES"`。wall **30.6 分**（08:37:15-09:07:50 UTC）、`result.json` は `status=done`、`set=all-d20`。**24/24 関数 × 8 手法 = 192 行**（wilcoxon 168 行 = 24×7、(関数, 手法) の重複・欠落ゼロ、stats 24 ファイル）。OOM は起きず、ワーカーのピーク RSS は約 0.9GB。
- **環境**: numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1。CPU Intel Xeon @2.10GHz、family 6 / model 207 / stepping 2、4 CPUs、avx512f あり。BLAS = openblas（SkylakeX、num_threads 4）、`OMP_NUM_THREADS=1` で実行。`pip install -r requirements.txt` は pynmmso のビルドで落ちるので findings.md の空 stub を `PYTHONPATH` に置き、残りは `python3 -m pip`（3.11）に `numpy==2.4.6` を固定して j46 と揃えた。
- **全体（24 関数平均、SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals(own)`＝自分の成功関数の平均 / nf）**: IPOP-CMA-ES 47.1/49.8/53.3/56.9（13599、13）、RCMAES 46.5/47.3/51.5/54.8（13868、15）、**MC-ESO 35.4/38.1/39.4/42.7（25183、13）**、**v4_pace 35.0/37.7/39.0/42.3（24071、13）**、LSHADE-SPACMA 32.7/34.4/44.6/53.8、EBOwithCMAR 26.5/30.0/47.9/56.9、ELSHADE-SPACMA 17.7/24.2/36.9/51.2、L-SHADE 17.1/19.2/31.5/47.9。
- **`evals_succ_mean`（MC-ESO と v4_pace の両方に成功のある 13 関数 F01/F02/F03/F04/F05/F06/F07/F10/F11/F12/F14/F21/F22 の平均）**: MC-ESO 25183 → v4_pace 24071（-1112）。片方だけ成功の関数は無し。MC-ESO より速い比較手法は IPOP-CMA-ES 13599・RCMAES 13868 の 2 手法。
- **関数別 SR@1e-10（MC-ESO → v4_pace、全 24 関数）**: 変化は **F21-Gallagher101 45→40、F22-Gallagher21 10→5 の 2 関数のみで、どちらも悪化**。残る 22 関数は不変（F01 100、F03 100、F05 100、F14 100、F02 95、F10 95、F12 80、F11 70、F06 30、F07 20、F04 5、F08・F09・F13・F15・F16・F17・F18・F19・F20・F23・F24 は両者 0）。24 関数平均 35.42 → 35.00（-0.42pt）。
- **関数別の平均誤差（`mean_best_f`、v4_pace / MC-ESO の比）**: 改善 9 関数（F18 0.27、F23 0.33、F15 0.42、F17 0.43、F16 0.46、F24 0.63、F19 0.67、F21 0.69、F22 0.74）、**悪化 0 関数**、残る 15 関数は 1 セルも違わず同値（腕は盆地乗換えが起きる関数でしか既定と分岐しない）。
- **CEC 式の平均誤差の順位 ([5])**: **24/24 関数で算出**。RCMAES 3.65 < LSHADE-SPACMA 3.67 < IPOP-CMA-ES 3.81 < EBOwithCMAR 3.92 < L-SHADE 4.85 < ELSHADE-SPACMA 5.06 < **v4_pace 5.23** < **MC-ESO 5.81**。
- **Wilcoxon（MC-ESO を reference、両側 p<0.05、A12）**: **v4_pace が有意に勝つ 6 関数**（F24 A12=0.05、F15 0.05、F23 0.10、F19 0.12 いずれも large、F16 0.32 medium、F18 0.35 medium）、**MC-ESO が有意に勝つ関数は 0**。比較 6 手法に対する MC-ESO の勝/負（関数数）: ELSHADE-SPACMA 9/9、IPOP-CMA-ES 8/11、LSHADE-SPACMA 7/13、EBOwithCMAR 7/11、L-SHADE 6/11、RCMAES 6/16。
- **書かれたとおりに回せなかった点**: ジョブは「4 shard 逐次（`j47_q1`〜`_q4`、各 `--jobs 2`）」と書いてあったが、**1 回の起動（label `j47_all24`、`--jobs 4`）で 24 関数まとめて回した**。`--jobs` は (関数, 手法) の 192 タスクを 4 ワーカーに割り当てるので、6 関数ずつ固定の shard より負荷が平均化し、shard の連結も不要になる。seed は run 番号だけで決まる（findings.md「実行と並列」）ので数値は `--jobs 1` と同じ。関数・手法・予算・instance・n はジョブ記載どおり。
- **記録**: `analysis/single/j47/summary_j47_dim20_all24.csv` / `wilcoxon_j47_dim20_all24.csv` / `analyze_quick_j47_d20_all24.txt`、per-run stats 24 本は `stats/*.csv.gz`。**`find analysis -type f` は 425 で上限の約 400 を超えた**（j46c の時点で既に 398。削除は行っていない＝畳むのは対話セッションの担当）。`scripts/check_doc_size.py` は全ファイル上限内（research_loop.md は 283/300、history.md 984/1000 が最も詰まっている）。
- 24/24 関数の結果である。採否の判断はしない。

（以下の j45・j46 は見出しだけ残した。本文は commit `f395d5e` の docs/research_loop.md にある）

- 2026-10-14 j46c（v3 の 30 次元 2/2 の残り F18-F24）— 7/7 完走。24/24 関数で v3_stage は SR@1e-10 21.25→23.12%、evals_succ_mean 43741→37074、CEC 順位 7.25→5.78
- 2026-10-14 j46b（v3 の 30 次元 2/2）— F13-F17 の 5 関数のみ完走（17/24）。F01-F17 平均で v3_stage は SR@1e-10 26.5→30.9%、CEC 順位 8.67→5.72
- 2026-10-14 j46a3（v3 の 30 次元 1/2 の残り F09-F12）— 12/12 完走。F01-F12 平均で v3_stage は SR@1e-10 33.8→43.8%、evals_succ_mean 45658→39781
- 2026-10-14 j46a（v3 の 30 次元 1/2）— 8/12 関数のみ完走（OOM で 2 回中断）。F01-F08 では v3_stage が SR@1e-10 40.0→53.1%、CEC 順位 6.44→5.38
- 2026-10-13 【手法解説ページの全手法比較 7/7】10D の第 4 四半分（F19-F24）を 18 手法で測り、4 shard を合わせて 10D を 24/24 関数で集計した（j45d10d） — **10D 全 24 関数で MC-ESO は SR@1e-10 64.17% で 18 手法中 1 位（2 位 LSHADE-SPACMA 55.21%、3 位 AMALGAM-SO 53.75%、旧既定 `MC-ESO-v2` 52.92% は 4 位）。一方 CEC 式の平均誤差の順位は 7.42 で 2 位（1 位 LSHADE-SPACMA 7.35、3 位 EBOwithCMAR 7.62、`MC-ESO-v2` 9.54 の 11 位）で、2D・F13-F18 と同じく SR と CEC 順位の向きが食い違う。Wilcoxon は 17 手法に対し 225 勝 60 敗、負け越しは LSHADE-SPACMA（7 勝 8 敗）の 1 手法だけ。負けは F15-RastriginRot（14 手法）・F17-SchafferF7（10 手法）・F02-EllipsoidalSep（8、A12 0.38 small）・F19-GriewankRosenbrock（8）・F20-Schwefel（8）・F24-LunacekRastrigin（6）に集まる。** 新しく測った 6 関数では MC-ESO が SR@1e-10 23.33% で 1 位（2 位 ARRDE 20.00%、`MC-ESO-v2` 6.67%）、CEC 順位は 7.50 で 5 位（1 位 EBOwithCMAR 4.00）。採否の判断はしない
- 2026-10-13 【手法解説ページの全手法比較 6/7】10D の第 3 四半分（F13-F18）を 18 手法で測った（j45d10c） — **この 6 関数では MC-ESO が SR@1e-10 40.83% で 18 手法中 2 位（1 位 LSHADE-SPACMA 43.33%、旧既定 `MC-ESO-v2` も 40.83% で同値、4 位 RCMAES 40.00%）。一方 CEC 式の平均誤差の順位は 8.33 で 10 位（1 位 LSHADE-SPACMA 5.17、`MC-ESO-v2` は 7.83 の 7 位）で、SR と CEC 順位の向きがこの 6 関数では食い違う。Wilcoxon は 17 手法に対し 52 勝 27 敗で、負けは F15-RastriginRot（14 手法に、SR 0%）と F17-SchafferF7（10 手法に、SR 20%）にほぼ集まる。**旧既定との差は F16-Weierstrass（20→25）と F18-SchafferF7ill（5→0）の 2 関数だけで 6 関数平均は 40.83 のまま変わらない。**10D は 24 関数ぶんのうち 18 関数（第 1〜第 3 四半分）しか出ていないので、この順位は暫定**（残りはキューの 8）。採否の判断はしない
- 2026-10-13 【手法解説ページの全手法比較 5/7】10D の第 2 四半分（F07-F12）を 18 手法で測った（j45d10b） — **この 6 関数では MC-ESO が SR@1e-10 98.33% で 18 手法中 4 位（1 位 BIPOP-CMA-ES・IPOP-CMA-ES 100%、AMALGAM-SO 98.33% と同値、旧既定 `MC-ESO-v2` は 97.50%）、CEC 式の平均誤差の順位は 6.67 で 3 位タイ（1 位 BIPOP-CMA-ES・IPOP-CMA-ES 5.33、AMALGAM-SO 6.67 と同値、`MC-ESO-v2` 8.83 の 10 位）。Wilcoxon は 17 手法に対し 68 勝 0 敗で、どの手法にも有意に負けていない。** 旧既定との差は F08-Rosenbrock（95→100）・F12-BentCigar（95→100）と F07-StepEllipsoidal（95→90）の 3 関数だけで、F09/F10/F11 は 3 世代とも 100% で同値。**10D は 24 関数ぶんのうち 12 関数（第 1・第 2 四半分）しか出ていないので、この順位は暫定**（残りはキューの 8〜9）。採否の判断はしない
- 2026-10-13 【手法解説ページの全手法比較 4/7】10D の第 1 四半分（F01-F06）を 18 手法で測った（j45d10a） — **この 6 関数では MC-ESO が SR@1e-10 94.17% で 18 手法中 1 位（2 位 ELSHADE-SPACMA 85.83%、3 位 LSHADE-SPACMA 83.33%、旧既定 `MC-ESO-v2` は 66.67% の 7 位タイ）、CEC 式の平均誤差の順位は 7.17 で 2 位（1 位 ELSHADE-SPACMA 6.83、3 位 L-SHADE 7.33、`MC-ESO-v2` 11.00 の 15 位）。** 差は **F03-RastriginSep（v2 0% → 100%）と F04-BucheRastrigin（v2 0% → 65%）** の 2 関数だけから来ており、残る 4 関数は 3 世代とも 100% で同値（F01/F05/F06 は evals まで一致）。`evals_succ_mean` の対応平均で MC-ESO より速いのは CMA 系 4 手法（IPOP-CMA-ES・BIPOP-CMA-ES 2095.2［共通 4 関数、MC-ESO 7809.0］・BIPOP-aCMA-ES 2401.3・RCMAES 3338.3［共通 3 関数、MC-ESO 7835.7］）と AMALGAM-SO 5608.7［共通 6 関数、MC-ESO 8421.5］の 5 手法で、2D・5D と同じ顔ぶれ。**10D では 24 関数ぶんのうち 6 関数しか出ていないので、この順位は暫定**（残りはキューの 8〜10）。採否の判断はしない
- 2026-10-13 【手法解説ページの全手法比較 3/7】5D の後半 12 関数を測り、前半と連結して 24/24 関数を 18 手法で並べた（j45d5b） — **5D の 24 関数では MC-ESO が SR@1e-10 77.08% で 18 手法中 1 位（2 位 ELSHADE-SPACMA 74.58%、3 位 LSHADE-SPACMA 71.46%、旧既定 `MC-ESO-v2` は 67.08% の 4 位）で、CEC 式の平均誤差の順位も 7.02 で 1 位（2 位 ELSHADE-SPACMA 7.44、3 位 EBOwithCMAR 7.65、`MC-ESO-v2` 7.94 の 5 位）＝ 2 指標の向きが一致する（2D の j45d2 では CEC 順位が 3 位だった）。`evals_succ_mean` の対応平均で MC-ESO より速いのは CMA 系 4 手法（BIPOP-aCMA-ES 1355.9・RCMAES 1771.1・IPOP-CMA-ES 2935.4・BIPOP-CMA-ES 3568.2）と AMALGAM-SO 3608.3 の 5 手法だけ（前半 12 関数と同じ顔ぶれ）。Wilcoxon は 17 手法すべてで負けが勝ちを上回らないが、前半だけでは見えなかった **F20-Schwefel（12 手法に負け）と F15-RastriginRot（9 手法に負け）が F06-AttractiveSector（7 手法）に並ぶ弱点として出た**（採否の判断はしない）
- 2026-10-13 【手法解説ページの全手法比較 2/7】5D の前半 12 関数を 18 手法で並べた（j45d5a） — **この 12 関数（F01-F12）では MC-ESO が SR@1e-10 98.75% で 18 手法中 1 位（2 位 EBOwithCMAR 96.25%、3 位 ELSHADE-SPACMA・L-SHADE 95.42%、旧既定 `MC-ESO-v2` は 85.83% の 9 位）、CEC 式の平均誤差の順位も 7.58 で 1 位（2 位 EBOwithCMAR 7.71、3 位 L-SHADE／LSHADE-SPACMA 7.88）。2D（j45d2）では CEC 順位が 3 位だったのに対し、5D の前半 12 関数では SR と CEC 順位の向きが一致する。`evals_succ_mean` の対応平均では CMA 系 4 手法（BIPOP-aCMA-ES・RCMAES・IPOP-CMA-ES・BIPOP-CMA-ES）と AMALGAM-SO だけが MC-ESO より速い。Wilcoxon は 17 手法すべてに対し負けが勝ちを上回る手法が無く、負けは F06-AttractiveSector（7 手法に）と F04-BucheRastrigin（SPS-L-SHADE-EIG に）の 2 関数だけ**（採否の判断はしない。**12/24 関数なので 24 関数の集計はジョブ 3/7（`j45d5b`）が済んでから**）
- 2026-10-13 【手法解説ページの全手法比較 1/7】2D の 18 手法（現既定・旧既定 `MC-ESO-v2` ＋ 標準の比較手法 16）を 24/24 関数で並べた（j45d2） — **MC-ESO（現既定）は SR@1e-10 94.38% で 18 手法中 1 位（2 位 `MC-ESO-v2` 93.96%、3 位 ELSHADE-SPACMA 92.08%）、`evals_succ_mean` も 770.8 で BIPOP-aCMA-ES（353.8、SR 60.83%）と RCMAES（594.4、SR 53.54%）に次ぐ 3 番目。一方 CEC 式の平均誤差の順位は 7.79 の 3 位で、EBOwithCMAR 7.46・IMODE 7.48 に負ける。Wilcoxon は 17 手法のうち 8 手法に 1〜2 関数（F17・F18・F24）で負け越す関数があり、負けが勝ちを上回るのは jSO・ELSHADE-SPACMA（どちらも 1 勝 2 敗）の 2 手法**（採否の判断はしない）
