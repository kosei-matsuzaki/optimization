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

- **主指標**: BBOB-24 の SR@1e-10。補助に SR@1e-2 / 1e-4 / 1e-7、`evals_succ_mean`（比べる手法どうしで同じ関数集合を平均する）、Wilcoxon（MC-ESO を reference、α=0.05）＋ A12。
- **判定**: 2 次元 BBOB-24（n=20、5000 評価）。5 次元・10 次元（予算 `2500×D`）を併記し、改善の対象に含める。**2D の SR@1e-10 を下げる変更は採用しない。**
- **比較対象**: CMA-ES / IPOP-CMA-ES / BIPOP-CMA-ES / DE / L-SHADE。
- **現在地**: 2026-10-12 に既定を `v1_w4q3E` に変更（[mceso.md](mceso.md) の「高次元の終盤」）。正準の SR@1e-10 は 2D 95.00 / 5D 68.33 / 10D 52.71%（`analysis/single/j29/`、旧既定 `MC-ESO-v1` 93.33 / 59.17 / 42.08）。比較 5 手法との同じ run の並びは測定ジョブ。参考に修正後の比較手法の正準値（[findings.md](findings.md) の 0 節）は 5D L-SHADE 58.54 が最高、10D IPOP 49.58 が最高。
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

**対話セッションでやること（測定ルーチンは取らない）**: いまは無し。

---
1. **【v3 の 30 次元 1/2 の残り】BBOB 30D の F09-F12 で、v3 の候補 `v3_stage` と既定・比較手法 9 を並べる。** `./run.sh quick --all --funcs F09-RosenbrockRot,F10-EllipsoidalRot,F11-Discus,F12-BentCigar --dim 30 --max-evals 75000 --n-runs 20 --instances 1-5 --methods "MC-ESO,v3_stage,ELSHADE-SPACMA,LSHADE-SPACMA,EBOwithCMAR,L-SHADE,IPOP-CMA-ES,ARRDE,NL-SHADE-RSP,RCMAES,BIPOP-aCMA-ES" --jobs 1`（label `j46a3`）。 (claimed 2026-10-14 06:30 UTC)
   **`--jobs` は 1 にする**（2026-10-14 の測定で判明: 30D のワーカーは 1 本で 9.9GB まで育ち、クラウドのコンテナの memory cgroup 上限は 13.7GiB。`--jobs 3` は F05 の途中で、`--jobs 2` は F09 の途中で OOM kill された。キューにあった「1 ワーカー 2GB 前後」は 30D では過小）。
   **背景**: 2026-10-14 に F01-F08 は測り終えて `analysis/single/j46/` に入っている（下の作業ログ）。この項はその残り 4 関数。
   **集計**: F01-F08 の `analysis/single/j46/summary_j46a_dim30.csv` / `wilcoxon_j46a_dim30.csv` と連結して 12/12 を確かめ、標準の集計（[1]・[4]・[5]）を 11 手法で。加えて `v3_stage` と MC-ESO の関数別 SR@1e-10 の変化を全列挙。
   **記録**: `analysis/single/j46/`（ジョブ 2 と合わせて 24 関数）。
2. **【v3 の 30 次元 2/2】後半 12 関数。** 同じコマンドで `--funcs F13-SharpRidge,F14-DiffPowers,F15-RastriginRot,F16-Weierstrass,F17-SchafferF7,F18-SchafferF7ill,F19-GriewankRosenbrock,F20-Schwefel,F21-Gallagher101,F22-Gallagher21,F23-Katsuura,F24-LunacekRastrigin`（label `j46b`）。前半と連結して 24/24 を確かめ、【v3 の 30 次元 1/2】と同じ集計を 24 関数で。**記録**: `analysis/single/j46/`。

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

### 2026-10-14 j46a（v3 の 30 次元 1/2）— 8/12 関数のみ完走（OOM で 2 回中断）。F01-F08 では v3_stage が SR@1e-10 40.0→53.1%、CEC 順位 6.44→5.38

- **ランした分**: `./run.sh quick --all --dim 30 --max-evals 75000 --n-runs 20 --instances 1-5` 11 手法。**F01-F08 の 8/12 関数のみ**（88 行 = 8×11、重複なし）。F09-F12 は未測定でジョブに書き戻した。wall 76 分（04:54-06:10 UTC）。2 shard（`--jobs 3` で F01-F04、`--jobs 2` で F05-F08）を連結。
- **書かれたとおりに回せなかった点**: `--jobs 3` が 05:06 に `BrokenProcessPool` で落ちた。`dmesg` で memory cgroup の OOM kill を確認（ワーカー 1 本が anon-rss 5.3GB）。`--jobs 2` で F05-F08 は通ったが F09 の途中で再度 OOM（ワーカー 1 本が RSS 9.9GB、cgroup 上限 13.7GiB、cache は 127MB のみ）。**キューの「30D は 1 ワーカーが 2GB 前後」は 30D では過小**。残りは `--jobs 1` 指定で書き戻した。F05 は 1 回目の中断時に未完だったので捨てて 2 回目で測り直した。
- **環境**: numpy 2.4.6 / cma 4.5.0。CPU Intel Xeon @2.10GHz、family 6 / model 207 / stepping 2、4 CPUs、avx512f あり。BLAS = openblas（SkylakeX、num_threads 4、`libscipy_openblas64_`）。`pip install -r requirements.txt` は pynmmso のビルドで落ちるので findings.md の空 stub を PYTHONPATH に置いた。`pip` が 3.13 を指すので `python3 -m pip`（3.11）で入れ直した。
- **全体（8 関数平均、SR@1e-2/1e-4/1e-7/1e-10）**: MC-ESO 50.6/45.0/41.2/**40.0** ／ v3_stage 58.1/58.1/55.0/**53.1**。他: IPOP-CMA-ES 73.8、RCMAES 55.6、EBOwithCMAR 50.6、BIPOP-aCMA-ES 49.4、L-SHADE 47.5、ELSHADE-SPACMA 46.9、LSHADE-SPACMA 46.2、ARRDE 38.1、NL-SHADE-RSP 13.1（いずれも SR@1e-10）。
- **evals_succ_mean**（両方に成功のある関数だけ、F01/F02/F03/F05/F06 の 5 関数）: MC-ESO 34999 ／ v3_stage 27174（-7825）。
- **関数別 SR@1e-10 の変化（v3_stage − MC-ESO、変化のあった関数のみ）**: F02-EllipsoidalSep 40→30%（-10）、F06-AttractiveSector 0→100%（+100）、F07-StepEllipsoidal 0→15%（+15）。F01 100→100、F03 80→80、F04 0→0、F05 100→100、F08 0→0 は変化なし。
- **Wilcoxon（MC-ESO を reference、α=0.05、8 関数）**: v3_stage が有意に良い 4（F06 A12=0.00 large、F08 A12=0.00 large、F07 A12=0.09 large、F01 A12=0.39 small）／MC-ESO が有意に良い 1（F03-RastriginSep A12=0.84 large）／有意差なし 3。
- **[5] CEC 式の平均誤差の順位（8 関数）**: ELSHADE-SPACMA 5.00、L-SHADE 5.00、LSHADE-SPACMA 5.25、IPOP-CMA-ES 5.31、**v3_stage 5.38**、EBOwithCMAR 5.62、ARRDE 6.00、RCMAES 6.12、**MC-ESO 6.44**、BIPOP-aCMA-ES 7.50、NL-SHADE-RSP 8.38。
- **記録**: `analysis/single/j46/summary_j46a_dim30.csv` / `wilcoxon_j46a_dim30.csv`（8 関数分）。`analysis/` のファイル数 376。
- 8/12 関数の部分結果である。採否の判断はしない。

### 2026-10-13 【手法解説ページの全手法比較 7/7】10D の第 4 四半分（F19-F24）を 18 手法で測り、4 shard を合わせて 10D を 24/24 関数で集計した（j45d10d） — **10D 全 24 関数で MC-ESO は SR@1e-10 64.17% で 18 手法中 1 位（2 位 LSHADE-SPACMA 55.21%、3 位 AMALGAM-SO 53.75%、旧既定 `MC-ESO-v2` 52.92% は 4 位）。一方 CEC 式の平均誤差の順位は 7.42 で 2 位（1 位 LSHADE-SPACMA 7.35、3 位 EBOwithCMAR 7.62、`MC-ESO-v2` 9.54 の 11 位）で、2D・F13-F18 と同じく SR と CEC 順位の向きが食い違う。Wilcoxon は 17 手法に対し 225 勝 60 敗、負け越しは LSHADE-SPACMA（7 勝 8 敗）の 1 手法だけ。負けは F15-RastriginRot（14 手法）・F17-SchafferF7（10 手法）・F02-EllipsoidalSep（8、A12 0.38 small）・F19-GriewankRosenbrock（8）・F20-Schwefel（8）・F24-LunacekRastrigin（6）に集まる。** 新しく測った 6 関数では MC-ESO が SR@1e-10 23.33% で 1 位（2 位 ARRDE 20.00%、`MC-ESO-v2` 6.67%）、CEC 順位は 7.50 で 5 位（1 位 EBOwithCMAR 4.00）。採否の判断はしない

- 実行: ジョブ記載のコマンドをそのまま 1 回（`--all --funcs F19〜F24 --dim 10 --max-evals 25000 --n-runs 20 --instances 1-5`、18 手法、label `j45d10d`）に `--jobs 4` だけ足した（関数 × 手法の並列。seed は run 番号だけで決まるので結果は `--jobs 1` と同じ）。wall **8.1 分**（19:35:43→19:43:46）、`result.json` は `status=done`、`set=all-d10`。この shard は **6/6 関数 × 18 手法 = 108 行**（`wilcoxon.csv` 102 行＝6×17、(関数, 手法) の重複・欠落ゼロ、stats 6 ファイル）。4 shard を連結した 10D は **24/24 関数 × 18 手法 = 432 行**（`wilcoxon.csv` 408 行＝24×17、重複・欠落ゼロ、stats 24 ファイル）。numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1、CPU **Intel Xeon @ 2.10GHz 4 コア（family 6 / model 207 / stepping 2、avx512f あり）**＝ j45d2・j45d5a・j45d5b・j45d10a・j45d10b・j45d10c と同じ容器仕様。BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。
- 10D 24 関数の全体表（SR@1e-10 / 1e-7 / 1e-4 / 1e-2、`evals_mean`）: **MC-ESO 64.2 / 64.8 / 68.5 / 70.4（10375）**、LSHADE-SPACMA 55.2 / 56.7 / 63.5 / 65.4、AMALGAM-SO 53.8 / 57.7 / 60.4 / 64.2、`MC-ESO-v2` 52.9 / 54.6 / 55.6 / 58.5、RCMAES 48.3 / 48.3 / 53.5 / 55.6、IPOP-CMA-ES 48.1 / 51.7 / 57.7 / 60.0、BIPOP-CMA-ES 48.1 / 53.1 / 55.0 / 59.0、ELSHADE-SPACMA 42.9 / 52.9 / 64.6 / 70.2、BIPOP-aCMA-ES 41.7、EBOwithCMAR 38.3、ARRDE 37.3、L-SRTDE 37.1、jSO 35.4、L-SHADE 33.3、SPS-L-SHADE-EIG 29.6、IMODE 20.8、NL-SHADE-RSP 20.6、EA4eig 19.2。**SR@1e-2 では ELSHADE-SPACMA・EBOwithCMAR が MC-ESO と同じ 70.4% に並ぶ**ので、MC-ESO の差は深い水準で付いている。
- 24 関数での旧既定 `MC-ESO-v2` との SR@1e-10 の差（変化した 9 関数のみ、他 15 関数は同値）: **F03-RastriginSep 0→100・F04-BucheRastrigin 0→65・F21-Gallagher101 15→100・F22-Gallagher21 25→40・F08-Rosenbrock 95→100・F12-BentCigar 95→100・F16-Weierstrass 20→25** が改善、**F07-StepEllipsoidal 95→90・F18-SchafferF7ill 5→0** が悪化。平均は 52.92 → 64.17（+11.25pt）で、上げ幅の大半は F03・F21・F04 の 3 関数から来ている。新しく測った 6 関数だけで見ると差は F21（+85）と F22（+15）の 2 関数のみ。
- `evals_succ_mean` の対応平均（両方に成功のある関数だけ）で MC-ESO より速いのは **BIPOP-aCMA-ES 3439.5［共通 11 関数、MC-ESO 9474.9］・RCMAES 3857.8［共通 15、10004.4］・IPOP-CMA-ES 5546.9［共通 15、9982.5］・BIPOP-CMA-ES 6509.7［共通 15、9580.0］・AMALGAM-SO 7310.1［共通 17、9587.8］・`MC-ESO-v2` 9605.5［共通 16、9862.2］** の 6 つで、CMA 系 4 ＋ AMALGAM-SO という顔ぶれは 2D・5D・各 10D shard と同じ（旧既定が僅差で速いのは、新既定が F03・F04・F21 で成功した分だけ長く回っているため）。残る 11 手法は MC-ESO のほうが速い（最も遅いのは IMODE 22395.8［共通 6 関数］）。
- 書かれたとおりに行かなかった点: (1) ジョブのコマンドに `--jobs` の指定は無いが、**j45d2・j45d10a〜c の前例に合わせて `--jobs 4` で回した**。(2) 参照先の【全手法比較 1/7】はその記録で消えているので、2026-10-13 の 1/7 の作業ログ項目を参照先として使った。(3) 24 関数の集計は 4 shard の `summary.csv` / `wilcoxon.csv` を連結し、`stats_*.tar.gz` を 1 つの `stats/` に展開した作業コピーに対して `analyze_quick.py` を回した（[5] は `dim10/stats/` のディレクトリを読むため）。連結した `meta` 行の commit は shard a のもの（`7324dda`）が出る。(4) `detail_*.txt` の生成スクリプトは repo に無いので scratchpad に書いて使った（[1] の表・Wilcoxon の数は `analyze_quick.py` の出力と一致することを確認）。(5) この容器では PATH の `python3` が 3.11（numpy 無し）で、パッケージは `/usr/bin/python3`（3.13）側に入る。`PATH=/usr/bin:$PATH` で 3.13 を使い、**`numpy==2.4.6` に固定して j45 の他の shard と揃えた**（素では 2.5.3）。pynmmso は findings.md の手順どおり空 stub を `PYTHONPATH` に置いた。
- **`scripts/check_doc_size.py` が落ちる: `docs/research_loop.md` が上限 300 行を超過**（j45d10a 以降ずっと超過しており、この記録でさらに増えた）。統合は対話セッションの担当なのでこちらでは畳まない。
- 記録: `analysis/single/j45/dim10/` に 24 関数の `summary.csv`・`wilcoxon.csv`・`analyze_quick_d10_all24.txt`・`detail_d10_all24.txt`、この shard の `summary_f19-f24.csv`・`wilcoxon_f19-f24.csv`・`stats_f19-f24.tar.gz`・`analyze_quick_d10_f19-f24.txt`・`detail_d10_f19-f24.txt`、`analysis/single/j45/result_d10d.json`。`find analysis -type f | wc -l` = 374（上限 約 400）。

### 2026-10-13 【手法解説ページの全手法比較 6/7】10D の第 3 四半分（F13-F18）を 18 手法で測った（j45d10c） — **この 6 関数では MC-ESO が SR@1e-10 40.83% で 18 手法中 2 位（1 位 LSHADE-SPACMA 43.33%、旧既定 `MC-ESO-v2` も 40.83% で同値、4 位 RCMAES 40.00%）。一方 CEC 式の平均誤差の順位は 8.33 で 10 位（1 位 LSHADE-SPACMA 5.17、`MC-ESO-v2` は 7.83 の 7 位）で、SR と CEC 順位の向きがこの 6 関数では食い違う。Wilcoxon は 17 手法に対し 52 勝 27 敗で、負けは F15-RastriginRot（14 手法に、SR 0%）と F17-SchafferF7（10 手法に、SR 20%）にほぼ集まる。**旧既定との差は F16-Weierstrass（20→25）と F18-SchafferF7ill（5→0）の 2 関数だけで 6 関数平均は 40.83 のまま変わらない。**10D は 24 関数ぶんのうち 18 関数（第 1〜第 3 四半分）しか出ていないので、この順位は暫定**（残りはキューの 8）。採否の判断はしない

- 実行: ジョブ記載のコマンドをそのまま 1 回（`--all --funcs F13〜F18 --dim 10 --max-evals 25000 --n-runs 20 --instances 1-5`、18 手法、label `j45d10c`）に `--jobs 4` だけ足した（関数 × 手法の並列。seed は run 番号だけで決まるので結果は `--jobs 1` と同じ）。wall **7.6 分**（18:37:09→18:44:43）、`result.json` は `status=done`、`set=all-d10`。この shard は **6/6 関数 × 18 手法 = 108 行**（`wilcoxon.csv` 102 行＝6×17、(関数, 手法) の重複・欠落ゼロ、stats 6 ファイル）。**24/24 ではない**（ジョブ記載どおり 10D の第 3 四半分のみ。4 本そろってから 24 関数で集計する）。numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1、CPU **Intel Xeon @ 2.10GHz 4 コア（family 6 / model 207 / stepping 2、avx512f あり）**＝ j45d2・j45d5a・j45d5b・j45d10a・j45d10b と同じ容器仕様。BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。
- 全体表（6 関数平均、SR@1e-2 / 1e-4 / 1e-7 / **1e-10** ％、`evals(own)` ＝ 自分の成功関数の平均、nf ＝ 成功のある関数数）: LSHADE-SPACMA 67.50/64.17/46.67/**43.33**（12602.6、5）・**MC-ESO 65.83/58.33/43.33/40.83（13284.8、5）**・MC-ESO-v2 59.17/47.50/44.17/**40.83**（12503.6、5）・RCMAES 68.33/60.83/40.00/**40.00**（3881.4、5）・BIPOP-aCMA-ES 44.17/33.33/33.33/**33.33**（3653.5、2）・AMALGAM-SO 68.33/54.17/43.33/**29.17**（13601.0、6）・IPOP-CMA-ES 70.83/61.67/37.50/**23.33**（11580.8、5）。以下 SR@1e-10 のみ: BIPOP-CMA-ES・ELSHADE-SPACMA 18.33・L-SRTDE 17.50・L-SHADE・jSO 12.50・ARRDE 3.33・EA4eig 2.50・EBOwithCMAR 1.67・IMODE・NL-SHADE-RSP・SPS-L-SHADE-EIG 0.00。
- `evals_succ_mean`（両方で成功のある関数だけの対応平均。MC-ESO / 相手、nf）: 対 BIPOP-aCMA-ES 7856.5 / **3653.5**（2）、対 RCMAES 13284.8 / **3881.4**（5）、対 LSHADE-SPACMA 12463.2 / **11508.2**（4）、対 IPOP-CMA-ES 13284.8 / **11580.8**（5）、対 ARRDE 12463.2 / **11652.8**（4）、対 BIPOP-CMA-ES 13284.8 / **12223.2**（5）、対 MC-ESO-v2 13284.8 / **12503.6**（5）、対 AMALGAM-SO 13284.8 / **12520.6**（5）。MC-ESO の方が速いのは L-SRTDE 14065.0・ELSHADE-SPACMA 17081.5・EBOwithCMAR 17849.0・EA4eig 19406.0・jSO 19781.0・L-SHADE 20888.2・SPS-L-SHADE-EIG 21747.5・IMODE 22346.0・NL-SHADE-RSP 22425.0 の 9 手法に対してで、**この 6 関数では 8 手法に負けている**（2D では 2 手法だけだった）。
- 関数別 SR@1e-10（現既定 `MC-ESO` 対 旧既定 `MC-ESO-v2`、差のある関数のみ）: **F16-Weierstrass 20→25（+5）**、**F18-SchafferF7ill 5→0（-5）**。残る 4 関数は不変（F13-SharpRidge 100、F14-DiffPowers 100、F15-RastriginRot 0、F17-SchafferF7 20）で、**6 関数平均は 40.83 → 40.83（±0.00pt）**。
- 18 手法の関数別 SR@1e-10 で目立つ点: **F15-RastriginRot は MC-ESO 0% で、最高が LSHADE-SPACMA・EBOwithCMAR 10%**（18 手法すべて 0〜10% の難関数）、**F16-Weierstrass は MC-ESO 25% が 18 手法中 1 位**（2 位 MC-ESO-v2・RCMAES 20%、13 手法が 0%）、F13-SharpRidge は MC-ESO・MC-ESO-v2・RCMAES・BIPOP-aCMA-ES 100 に対し 7 手法が 0%、F17-SchafferF7 は MC-ESO 20 に対し LSHADE-SPACMA 45 が最高、F18-SchafferF7ill は MC-ESO 0 に対し LSHADE-SPACMA 10 が最高。F14-DiffPowers は 12 手法が 100%。
- Wilcoxon（ref=MC-ESO、p<0.05 両側、A12 で向き。勝ち/負け）: IMODE 6/0、NL-SHADE-RSP 6/0、SPS-L-SHADE-EIG 5/1、BIPOP-aCMA-ES 3/0、BIPOP-CMA-ES 3/1、EA4eig 3/1、EBOwithCMAR 3/1、AMALGAM-SO 3/2、ARRDE 3/2、ELSHADE-SPACMA 3/2、L-SHADE 3/2、L-SRTDE 3/2、jSO 3/2、IPOP-CMA-ES 2/2、**RCMAES 1/2**、**LSHADE-SPACMA 1/3**、**MC-ESO-v2 1/4**（合計 52 勝 27 敗）。負けが勝ちを上回るのは RCMAES・LSHADE-SPACMA・MC-ESO-v2 の 3 手法。負けの集まる関数は **F15-RastriginRot（14 手法に、A12=0.00〜0.15 large）・F17-SchafferF7（10 手法に、A12=0.22〜0.49）** の 2 つで、F13 に 1（MC-ESO-v2、A12=0.40 small）・F14 に 2（LSHADE-SPACMA・MC-ESO-v2、A12=0.23 large）。勝ちは F14-DiffPowers 15・F13-SharpRidge 13・F16-Weierstrass 13 に集まる。
- 参考 [5] CEC 式の平均誤差の順位（低いほど良い、6 関数、per-run 誤差 1e-8 未満は 0）: 1. LSHADE-SPACMA 5.17（4/6 で最良）、2. RCMAES 6.83、3. IPOP-CMA-ES 7.25、4. L-SRTDE 7.25、5. ELSHADE-SPACMA 7.42、6. L-SHADE 7.75、**7. MC-ESO-v2 7.83**、8. AMALGAM-SO 8.00、9. BIPOP-CMA-ES 8.25、**10. MC-ESO 8.33（2/6 で最良）**、11. jSO 8.75、以下 ARRDE 9.83・EBOwithCMAR 10.17・BIPOP-aCMA-ES 11.00・EA4eig 11.33・SPS-L-SHADE-EIG 12.67・NL-SHADE-RSP 15.83・IMODE 17.33。**SR@1e-10 では 2 位だが CEC 順位では 10 位**で、2D（SR 1 位 / CEC 3 位）より食い違いが大きい。
- 書かれたとおりに行かなかった点: (1) ジョブのコマンドに `--jobs` の指定は無いが、**j45d2・j45d10a・j45d10b の前例に合わせて `--jobs 4` で回した**。(2) 参照先の【全手法比較 1/7】はその記録で消えているので、2026-10-13 の 1/7 の作業ログ項目を参照先として使った。**第 1・第 2 四半分が同じ `dim10/` に入っているので、ファイル名に `_f13-f18` を付けて上書きを防いだ**（24 関数の集計は 7/7 の担当）。(3) `detail_*.txt` の生成スクリプトは repo に無いので scratchpad に書いて使った（[1] の表・Wilcoxon の数は `analyze_quick.py` の出力と一致することを確認）。(4) この容器では PATH の `python3` が 3.11（numpy 無し）で、パッケージは `/usr/bin/python3`（3.13）側に入る。`PATH=/usr/bin:$PATH` で 3.13 を使い、numpy は 2.5.4 が入ったので **`numpy==2.4.6` に固定して j45 の他の shard と揃えた**。pynmmso は findings.md の手順どおり空 stub を `PYTHONPATH` に置いた。
- **`scripts/check_doc_size.py` が落ちる: `docs/research_loop.md` が 322 / 300 行で上限超過**（この記録を足した結果）。統合は対話セッションの担当なのでこちらでは畳まない。
- 記録: `analysis/single/j45/dim10/summary_f13-f18.csv`・`wilcoxon_f13-f18.csv`・`stats_f13-f18.tar.gz`・`analyze_quick_d10_f13-f18.txt`・`detail_d10_f13-f18.txt`、`analysis/single/j45/result_d10c.json`。`find analysis -type f | wc -l` = 364。

### 2026-10-13 【手法解説ページの全手法比較 5/7】10D の第 2 四半分（F07-F12）を 18 手法で測った（j45d10b） — **この 6 関数では MC-ESO が SR@1e-10 98.33% で 18 手法中 4 位（1 位 BIPOP-CMA-ES・IPOP-CMA-ES 100%、AMALGAM-SO 98.33% と同値、旧既定 `MC-ESO-v2` は 97.50%）、CEC 式の平均誤差の順位は 6.67 で 3 位タイ（1 位 BIPOP-CMA-ES・IPOP-CMA-ES 5.33、AMALGAM-SO 6.67 と同値、`MC-ESO-v2` 8.83 の 10 位）。Wilcoxon は 17 手法に対し 68 勝 0 敗で、どの手法にも有意に負けていない。** 旧既定との差は F08-Rosenbrock（95→100）・F12-BentCigar（95→100）と F07-StepEllipsoidal（95→90）の 3 関数だけで、F09/F10/F11 は 3 世代とも 100% で同値。**10D は 24 関数ぶんのうち 12 関数（第 1・第 2 四半分）しか出ていないので、この順位は暫定**（残りはキューの 8〜9）。採否の判断はしない

- 実行: ジョブ記載のコマンドをそのまま 1 回（`--all --funcs F07〜F12 --dim 10 --max-evals 25000 --n-runs 20 --instances 1-5`、18 手法、label `j45d10b`）に `--jobs 4` だけ足した（関数 × 手法の並列。seed 固定なので結果は `--jobs 1` と同じ）。wall **7.7 分**（17:41:59→17:49:40）、`result.json` は `status=done`、ログの `set=all-d10`。この shard は **6/6 関数 × 18 手法 = 108 行**（`wilcoxon.csv` 102 行＝6×17、(関数, 手法) の重複・欠落ゼロ、stats 6 ファイル）。**24/24 ではない**（ジョブ記載どおり 10D の第 2 四半分のみ。4 本そろってから 24 関数で集計する）。numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1、CPU **Intel Xeon @ 2.10GHz 4 コア（family 6 / model 207 / stepping 2、avx512f あり）**＝ j45d2・j45d5a・j45d5b・j45d10a と同じ容器仕様。BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。
- 全体表（6 関数平均、SR@1e-2 / 1e-4 / 1e-7 / 1e-10、`evals(own)`）: BIPOP-CMA-ES 100/100/100/**100**（6293.2）・IPOP-CMA-ES 100/100/100/**100**（5847.0）・AMALGAM-SO 100/99.17/99.17/**98.33**（8547.7）・**MC-ESO 98.33/98.33/98.33/98.33（10213.5）**・MC-ESO-v2 98.33/98.33/97.50/**97.50**（10987.3）・RCMAES 95.83 の 4 水準同値（5015.7）・LSHADE-SPACMA 97.50/95.83/86.67/**85.83**・BIPOP-aCMA-ES 80.83 の 4 水準同値・L-SRTDE 91.67/80.83/77.50/**72.50**・ELSHADE-SPACMA 99.17/88.33/79.17/**57.50**。以下 SR@1e-10 のみ: ARRDE 56.67・jSO 53.33・EBOwithCMAR 52.50・SPS-L-SHADE-EIG 48.33・L-SHADE 45.00・IMODE 30.00・EA4eig 24.17・NL-SHADE-RSP 5.83。
- 関数別 SR@1e-10（MC-ESO）: F07 90・F08 100・F09 100・F10 100・F11 100・F12 100。旧既定 `MC-ESO-v2` からの変化は **F08 95→100（+5）・F12 95→100（+5）・F07 95→90（−5）** の 3 関数のみ。`analyze_quick.py` の [2] に within-run baseline として `MC-ESO-v2` を与えると、F07 の −5 により自動判定は REJECT と出る（この自動判定は同一手法の前後比較用の規則をそのまま世代間比較に当てたもの。採否の判断は対話セッションとユーザーの担当）。
- `evals_succ_mean` の対応平均（両方に成功のある関数だけ）で MC-ESO より速いのは **BIPOP-aCMA-ES 4547.6［共通 5 関数、MC-ESO 10832.6］・RCMAES 5015.7・IPOP-CMA-ES 5847.0・BIPOP-CMA-ES 6293.2・AMALGAM-SO 8547.7［いずれも共通 6 関数、MC-ESO 10213.5］** の 5 手法で、2D・5D・j45d10a と同じ顔ぶれ。残る 12 手法は MC-ESO のほうが速い（最も遅いのは IMODE 22806.4）。
- Wilcoxon（ref=MC-ESO、p_two<0.05、向きは A12）: **68 勝 0 敗**（MC-ESO が有意に負けた関数はゼロ）。6 勝 = IMODE・NL-SHADE-RSP、5 勝 = ARRDE・BIPOP-CMA-ES・EA4eig・EBOwithCMAR・ELSHADE-SPACMA・IPOP-CMA-ES・L-SHADE・L-SRTDE・SPS-L-SHADE-EIG・jSO、2 勝 = AMALGAM-SO（F11・F12）・LSHADE-SPACMA（F09・F12）、1 勝 = BIPOP-aCMA-ES（F07、A12=0.98）・MC-ESO-v2（F12、A12=0.71）、0 勝 0 敗 = RCMAES。
- 書かれたとおりに行かなかった点（測定値には影響しない環境側）: `pip install -r requirements.txt` は記録どおり `pynmmso` のビルドで落ちたので [findings.md](findings.md) の空 stub を `PYTHONPATH` に置いた。この容器では `python3` が 3.11（パッケージ無し）で pip が 3.13 を向いていたため、3.13 を指す shim を `PATH` の先頭に置いて `run.sh` に拾わせた。`ioh` を入れると numpy が 2.5.4 に上がるので、他の shard と同じ正準環境に合わせて **numpy を 2.4.6 に戻してから**測った。
- 記録先は `analysis/single/j45/dim10/`（`summary_f07-f12.csv`・`wilcoxon_f07-f12.csv`・`analyze_quick_d10_f07-f12.txt`・`detail_d10_f07-f12.txt`・`stats_f07-f12.tar.gz`）。`scripts/check_doc_size.py` は **`docs/research_loop.md` が上限 300 行を超過で OVER**（j45d10a の時点で 302 行、そこへこの追記）。統合は対話セッションの担当なので畳まずに commit した。`analysis` のファイル数は 358（上限 約 400）。

### 2026-10-13 【手法解説ページの全手法比較 4/7】10D の第 1 四半分（F01-F06）を 18 手法で測った（j45d10a） — **この 6 関数では MC-ESO が SR@1e-10 94.17% で 18 手法中 1 位（2 位 ELSHADE-SPACMA 85.83%、3 位 LSHADE-SPACMA 83.33%、旧既定 `MC-ESO-v2` は 66.67% の 7 位タイ）、CEC 式の平均誤差の順位は 7.17 で 2 位（1 位 ELSHADE-SPACMA 6.83、3 位 L-SHADE 7.33、`MC-ESO-v2` 11.00 の 15 位）。** 差は **F03-RastriginSep（v2 0% → 100%）と F04-BucheRastrigin（v2 0% → 65%）** の 2 関数だけから来ており、残る 4 関数は 3 世代とも 100% で同値（F01/F05/F06 は evals まで一致）。`evals_succ_mean` の対応平均で MC-ESO より速いのは CMA 系 4 手法（IPOP-CMA-ES・BIPOP-CMA-ES 2095.2［共通 4 関数、MC-ESO 7809.0］・BIPOP-aCMA-ES 2401.3・RCMAES 3338.3［共通 3 関数、MC-ESO 7835.7］）と AMALGAM-SO 5608.7［共通 6 関数、MC-ESO 8421.5］の 5 手法で、2D・5D と同じ顔ぶれ。**10D では 24 関数ぶんのうち 6 関数しか出ていないので、この順位は暫定**（残りはキューの 8〜10）。採否の判断はしない

- 実行: ジョブ記載のコマンドをそのまま 1 回（`--all --funcs F01〜F06 --dim 10 --max-evals 25000 --n-runs 20 --instances 1-5`、18 手法、label `j45d10a`）に `--jobs 4` だけ足した（関数 × 手法の並列。seed 固定なので結果は `--jobs 1` と同じ）。wall **7.9 分**（16:37:51→16:45:44）、`result.json` は `status=done`、ログの `set=all-d10`。この shard は **6/6 関数 × 18 手法 = 108 行**（`wilcoxon.csv` 102 行＝6×17、(関数, 手法) の重複・欠落ゼロ、stats 6 ファイル）。**24/24 ではない**（ジョブ記載どおり 10D の第 1 四半分のみ。4 本そろってから 24 関数で集計する）。numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1、CPU **Intel Xeon @ 2.10GHz 4 コア（family 6 / model 207 / stepping 2、avx512f あり）**＝ j45d2・j45d5a・j45d5b と同じ容器仕様。BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。
- 全体表（6 関数平均、SR@1e-2 / 1e-4 / 1e-7 / 1e-10、`evals(own)` は自分が成功した関数だけの平均）: MC-ESO 94.17/94.17/94.17/**94.17**（8421.5）, ELSHADE-SPACMA 97.50/97.50/95.83/85.83（16964.3）, LSHADE-SPACMA 84.17/84.17/84.17/83.33（13829.7）, EBOwithCMAR 95.83/95.83/91.67/82.50（16893.0）, AMALGAM-SO 73.33/73.33/73.33/72.50（5608.7）, ARRDE 69.17×3/69.17（9387.2）, BIPOP-CMA-ES・IPOP-CMA-ES・**MC-ESO-v2**・jSO が 66.67, L-SHADE 98.33/98.33/92.50/65.00, NL-SHADE-RSP 99.17/93.33/67.50/60.83, SPS-L-SHADE-EIG 100/100/71.67/53.33, BIPOP-aCMA-ES・IMODE・L-SRTDE・RCMAES が 50.00, EA4eig 85.00/80.00/65.83/39.17 の最下位。**浅い水準では MC-ESO は 1 位でない**（SR@1e-2 は SPS-L-SHADE-EIG 100 / NL-SHADE-RSP 99.17 / L-SHADE 98.33 が上）。
- 関数別 SR@1e-10（MC-ESO / 旧既定 `MC-ESO-v2`）: F01 100/100・F02 100/100・**F03-RastriginSep 100/0**・**F04-BucheRastrigin 65/0**・F05 100/100・F06 100/100。F04 は 18 手法で最高（2 位 NL-SHADE-RSP 65% と同値、3 位 EBOwithCMAR 45%）だが **MC-ESO 自身も 65% にとどまる 6 関数中唯一の未達関数**。F05-LinearSlope を 0% にする手法が 4 つ（BIPOP-aCMA-ES・L-SRTDE・NL-SHADE-RSP・RCMAES）、F06 を 0〜5% にする手法が 3 つ（EA4eig 5・IMODE 0・NL-SHADE-RSP 0）ある。
- Wilcoxon（`best_f`、ref=MC-ESO、p_two<0.05、向きは A12）: 17 手法の合計で **勝ち 55 / 負け 11**。負けが勝ちを上回るのは **LSHADE-SPACMA（1 勝 2 敗）だけ**（5D では 1 手法も無かった）。負けの中身は **F02-EllipsoidalSep が 8 手法（すべて A12=0.38, small）と F06-AttractiveSector が 3 手法（RCMAES A12=0.19 large・BIPOP-aCMA-ES 0.35・LSHADE-SPACMA 0.35 medium）に集中**。どちらも MC-ESO の SR@1e-10 は 100% の関数で、ゴール欄の注意書き（SR 100% の関数の `best_f` 差は主指標の何桁も下）が当てはまる。
- 環境のセットアップで 1 点つまずいた: 容器の `pip` は python3.13 を指すが `python3` は 3.11 なので、`pip install` 後も import が通らない。`python3 -m pip` で入れ直して解決した（numpy は 3.11 側で 2.4.6 ＝ 従来の正準値と同じ）。`pynmmso` はビルドできないので findings.md のとおり空 stub を `PYTHONPATH` に置いた。`scripts/analyze_quick.py` の [5] も `core` 経由で `pynmmso` を import するので、stub 無しでは [1]〜[4] だけ出て落ちる。
- `scripts/check_doc_size.py` が **`docs/research_loop.md` 302 / 300 行で OVER**（この作業ログの追記ぶん）。統合は対話セッションの担当なので畳まずに commit した。`analysis` のファイル数は 352（上限 約 400）。

### 2026-10-13 【手法解説ページの全手法比較 3/7】5D の後半 12 関数を測り、前半と連結して 24/24 関数を 18 手法で並べた（j45d5b） — **5D の 24 関数では MC-ESO が SR@1e-10 77.08% で 18 手法中 1 位（2 位 ELSHADE-SPACMA 74.58%、3 位 LSHADE-SPACMA 71.46%、旧既定 `MC-ESO-v2` は 67.08% の 4 位）で、CEC 式の平均誤差の順位も 7.02 で 1 位（2 位 ELSHADE-SPACMA 7.44、3 位 EBOwithCMAR 7.65、`MC-ESO-v2` 7.94 の 5 位）＝ 2 指標の向きが一致する（2D の j45d2 では CEC 順位が 3 位だった）。`evals_succ_mean` の対応平均で MC-ESO より速いのは CMA 系 4 手法（BIPOP-aCMA-ES 1355.9・RCMAES 1771.1・IPOP-CMA-ES 2935.4・BIPOP-CMA-ES 3568.2）と AMALGAM-SO 3608.3 の 5 手法だけ（前半 12 関数と同じ顔ぶれ）。Wilcoxon は 17 手法すべてで負けが勝ちを上回らないが、前半だけでは見えなかった **F20-Schwefel（12 手法に負け）と F15-RastriginRot（9 手法に負け）が F06-AttractiveSector（7 手法）に並ぶ弱点として出た**（採否の判断はしない）

- 実行: ジョブ記載のコマンドをそのまま 1 回（`--all --funcs F13〜F24 --dim 5 --max-evals 12500 --n-runs 20 --instances 1-5 --jobs 4`、18 手法、label `j45d5b`）。wall **8.4 分**（15:36:53→15:45:15）、`result.json` は `status=done`、ログの `set=all-d5`。この shard は **12/12 関数 × 18 手法 = 216 行**（`wilcoxon.csv` 204 行＝12×17、per-run stats 12 ファイル）、前半（`j45d5a`）と連結して **24/24 関数 × 18 手法 = 432 行**（`wilcoxon.csv` 408 行＝24×17、(関数, 手法) の重複・欠落ゼロ、stats 24 ファイル）。numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1、CPU **Intel Xeon @ 2.10GHz 4 コア（family 6 / model 207 / stepping 2、avx512f あり）**＝ j45d2・j45d5a と同じ容器仕様。BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。
- 全体（24 関数平均、SR@1e-2 / 1e-4 / 1e-7 / **1e-10** ％、`evals` ＝ 自分の成功関数の平均、nf ＝ 成功のある関数数）: **MC-ESO 81.25 / 80.62 / 78.54 / 77.08**（3929.6、23）、ELSHADE-SPACMA 79.17 / 78.33 / 76.25 / **74.58**（6610.6、21）、LSHADE-SPACMA 74.38 / 73.54 / 72.92 / **71.46**（4941.2、21）、**MC-ESO-v2 72.08 / 71.46 / 69.17 / 67.08**（4019.8、22）、jSO 75.62 / 75.21 / 71.04 / **66.25**（7393.2、20）、EBOwithCMAR 80.62 / 77.50 / 69.58 / **64.79**（7934.8、21）、L-SHADE 78.96 / 77.50 / 68.96 / **62.08**（8288.7、20）、L-SRTDE **61.67**、AMALGAM-SO **60.83**、ARRDE **60.62**、IPOP-CMA-ES **56.88**、BIPOP-CMA-ES **55.83**、SPS-L-SHADE-EIG **53.75**、RCMAES **48.75**、BIPOP-aCMA-ES **48.12**、EA4eig **47.50**、NL-SHADE-RSP **45.21**、IMODE **41.88**。全 18 手法の列は記録先。
- `evals_succ_mean`（両方で成功のある関数だけの対応平均。MC-ESO / 相手、nf）: 対 BIPOP-aCMA-ES 3603.7 / **1355.9**（18）、対 RCMAES 3729.8 / **1771.1**（17）、対 IPOP-CMA-ES 3800.6 / **2935.4**（20）、対 BIPOP-CMA-ES 3800.6 / **3568.2**（20）、対 AMALGAM-SO 3750.5 / **3608.3**（21）、対 MC-ESO-v2 3889.9 / 4019.8（22）、以下 ARRDE 4270.1・L-SRTDE 4534.8・LSHADE-SPACMA 4941.2・ELSHADE-SPACMA 6610.6・jSO 7393.2・EBOwithCMAR 7934.8・L-SHADE 8288.7・EA4eig 8572.4・SPS-L-SHADE-EIG 9295.0・NL-SHADE-RSP 10244.8・IMODE 10700.1 は MC-ESO より遅い。
- 関数別 SR@1e-10（現既定 `MC-ESO` 対 旧既定 `MC-ESO-v2`、24 関数のうち差のある 9 関数。全列挙は記録先）: **F04-BucheRastrigin 0→85（+85）**、**F03-RastriginSep 30→100（+70）**、**F22-Gallagher21 50→100（+50）**、**F21-Gallagher101 70→95（+25）**、**F23-Katsuura 25→45（+20）**、**F17-SchafferF7 75→80（+5）**、**F19-GriewankRosenbrock 5→10（+5）**、**F16-Weierstrass 75→60（-15）**、**F15-RastriginRot 10→5（-5）**。残る 15 関数は不変（24 関数平均 67.08→77.08、**+10.0pt**）。
- 18 手法の関数別 SR@1e-10 で目立つ点: **F23-Katsuura は MC-ESO 45 / MC-ESO-v2 25 / BIPOP-aCMA-ES 15 に対し他の 15 手法が 0%**、**F16-Weierstrass も MC-ESO 60 に対し IPOP-CMA-ES 85・MC-ESO-v2 75・BIPOP-CMA-ES 65 以外は 0〜40%**。逆に **F20-Schwefel は MC-ESO 5 に対し EBOwithCMAR 95・ELSHADE-SPACMA 85・NL-SHADE-RSP 90・jSO 75**、**F15-RastriginRot は MC-ESO 5 に対し L-SRTDE 45・ELSHADE-SPACMA 40・EA4eig 40**。**F24-LunacekRastrigin は 18 手法すべて 0%**、F01・F02・F05（RCMAES 除く）・F14 は上位手法すべて 100%。
- Wilcoxon（ref=MC-ESO、p<0.05 両側、A12 で向き。勝ち/負け）: IMODE 17/1、IPOP-CMA-ES 16/0、BIPOP-CMA-ES 16/0、BIPOP-aCMA-ES 15/1、EA4eig 15/2、RCMAES 14/1、NL-SHADE-RSP 14/1、ARRDE 11/2、SPS-L-SHADE-EIG 11/2、EBOwithCMAR 11/2、L-SRTDE 10/2、L-SHADE 9/3、AMALGAM-SO 8/2、jSO 8/3、LSHADE-SPACMA 5/4、**ELSHADE-SPACMA 5/5**、MC-ESO-v2 4/3。**負けが勝ちを上回る手法は無い**。負けの集まる関数は **F20-Schwefel（12 手法に、A12 0.04〜0.32）・F15-RastriginRot（9 手法に、A12 0.11〜0.28 large）・F06-AttractiveSector（7 手法に、A12 0.12〜0.29）**、ほかは F17-SchafferF7（2 手法）、F04・F16・F18・F19（各 1 手法）。全列挙は記録先の `analyze_quick_d5.txt` / `detail_d5.txt`。
- 参考 [5] CEC 式の平均誤差の順位（低いほど良い、24 関数、per-run 誤差 1e-8 未満は 0）: **1. MC-ESO 7.02（15/24 で最良）**、2. ELSHADE-SPACMA 7.44（15/24）、3. EBOwithCMAR 7.65（11/24）、4. LSHADE-SPACMA 7.75（13/24）、**5. MC-ESO-v2 7.94（13/24）**、6. L-SHADE 8.21、7. SPS-L-SHADE-EIG 8.38、8. ARRDE 8.83、9. jSO 9.04、10. IPOP-CMA-ES 9.85、11. BIPOP-CMA-ES 9.90、12. EA4eig 9.92、13. IMODE 10.00、14. AMALGAM-SO 10.31、15. L-SRTDE 10.75、16. NL-SHADE-RSP 11.52、17. RCMAES 12.88、18. BIPOP-aCMA-ES 13.62。**前半 12 関数（j45d5a）でも 24 関数でも CEC 順位 1 位＝ SR@1e-10 1 位**で、2D（CEC 3 位）とは向きが違う。
- 書かれたとおりに行かなかった点: (1) ジョブのコマンドに `--jobs` の指定は無いが、**j45d2・j45d5a の前例に合わせて `--jobs 4` で回した**（`--jobs>1` でも seed は run 番号だけで決まるので数値は変わらない）。(2) **この容器では `python3` が空の Python 3.11（`/usr/local/bin/python3`）を指し、パッケージは `/usr/bin/python3`（3.13）側に入る**ので、`PATH=/usr/bin:$PATH` で回した（`run.sh` の `PY` は `python3`）。環境は findings.md の手順（`--ignore-installed blinker`、pynmmso は空 stub を `PYTHONPATH`）で numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1 に固定（`minionpy` の導入で numpy が 2.5.4 に上がるので入れ直した）。(3) 24 関数の集計は、この shard の CSV と `j45d5a` の `summary_f01-f12.csv` / `wilcoxon_f01-f12.csv` ＋ 両方の stats を scratchpad で連結して `analyze_quick.py` に渡した（連結版を `summary.csv` / `wilcoxon.csv` として保存）。(4) `detail_*.txt` の生成スクリプトは repo に無いので scratchpad に書いて使った（[1] の表・Wilcoxon の数は `analyze_quick.py` の出力と一致することを確認）。
- 記録: `analysis/single/j45/dim5/summary.csv`・`wilcoxon.csv`（24 関数の連結版）、`summary_f13-f24.csv`・`wilcoxon_f13-f24.csv`・`stats_f13-f24.tar.gz`（この shard）、`analyze_quick_d5.txt`・`detail_d5.txt`、`analysis/single/j45/result_d5b.json`。`find analysis -type f | wc -l` = 346。

### 2026-10-13 【手法解説ページの全手法比較 2/7】5D の前半 12 関数を 18 手法で並べた（j45d5a） — **この 12 関数（F01-F12）では MC-ESO が SR@1e-10 98.75% で 18 手法中 1 位（2 位 EBOwithCMAR 96.25%、3 位 ELSHADE-SPACMA・L-SHADE 95.42%、旧既定 `MC-ESO-v2` は 85.83% の 9 位）、CEC 式の平均誤差の順位も 7.58 で 1 位（2 位 EBOwithCMAR 7.71、3 位 L-SHADE／LSHADE-SPACMA 7.88）。2D（j45d2）では CEC 順位が 3 位だったのに対し、5D の前半 12 関数では SR と CEC 順位の向きが一致する。`evals_succ_mean` の対応平均では CMA 系 4 手法（BIPOP-aCMA-ES・RCMAES・IPOP-CMA-ES・BIPOP-CMA-ES）と AMALGAM-SO だけが MC-ESO より速い。Wilcoxon は 17 手法すべてに対し負けが勝ちを上回る手法が無く、負けは F06-AttractiveSector（7 手法に）と F04-BucheRastrigin（SPS-L-SHADE-EIG に）の 2 関数だけ**（採否の判断はしない。**12/24 関数なので 24 関数の集計はジョブ 3/7（`j45d5b`）が済んでから**）

- 実行: ジョブ記載のコマンドをそのまま 1 回（`--all --funcs F01〜F12 --dim 5 --max-evals 12500 --n-runs 20 --instances 1-5 --jobs 4`、18 手法、label `j45d5a`）。wall **9.1 分**（14:33:13→14:42:18）、`result.json` は `status=done`、ログの `set=all-d5`。**12/12 関数 × 18 手法 = 216 行**（`wilcoxon.csv` 204 行＝12×17、(関数, 手法) の重複・欠落ゼロ、per-run stats 12 ファイル）。**24 関数のうち F13〜F24 は別ジョブ（3/7）で未測定**。numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1、CPU **Intel Xeon @ 2.10GHz 4 コア（family 6 / model 207 / stepping 2、avx512f あり）**＝ j45d2 と同じ容器仕様。BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。
- 全体（12 関数平均、SR@1e-2 / 1e-4 / 1e-7 / **1e-10** ％、`evals` ＝ 自分の成功関数の平均、nf ＝ 成功のある関数数）: **MC-ESO 98.75 / 98.75 / 98.75 / 98.75**（3502.8、12）、EBOwithCMAR 99.58 / 99.58 / 99.58 / **96.25**（7111.5、12）、ELSHADE-SPACMA 98.75 / 98.75 / 97.50 / **95.42**（6117.8、12）、L-SHADE 99.58 / 98.33 / 95.83 / **95.42**（7678.8、12）、LSHADE-SPACMA 94.17 ×2 / 94.17 / **92.50**（4628.8、12）、jSO 94.17 / 94.17 / 91.67 / **88.33**（6869.2、12）、AMALGAM-SO・ARRDE・**MC-ESO-v2** がいずれも **85.83**（2754.9 / 3802.1 / 3617.3）、BIPOP-CMA-ES **84.58**、IPOP-CMA-ES **83.75**、SPS-L-SHADE-EIG **79.58**、L-SRTDE **79.17**、BIPOP-aCMA-ES **75.00**、RCMAES **72.92**、IMODE **70.83**（SR@1e-2 は 100.00 で最高）、NL-SHADE-RSP **65.83**、EA4eig **64.17**。全 18 手法は記録先。
- `evals_succ_mean`（両方で成功のある関数だけの対応平均。MC-ESO / 相手、nf）: 対 BIPOP-aCMA-ES 3253.8 / **1295.7**（10）、対 RCMAES 3331.7 / **1328.7**（9）、対 IPOP-CMA-ES 3384.5 / **1863.2**（11）、対 BIPOP-CMA-ES 3384.5 / **2069.5**（11）、対 AMALGAM-SO 3502.8 / **2754.9**（12）、対 MC-ESO-v2 3384.5 / 3617.3（11）、対 ARRDE 3502.8 / 3802.1（12）、対 L-SRTDE 3384.5 / 4127.9（11）、以下 LSHADE-SPACMA 4628.8・ELSHADE-SPACMA 6117.8・jSO 6869.2・EBOwithCMAR 7111.5・L-SHADE 7678.8・EA4eig 8330.3・SPS-L-SHADE-EIG 8668.1・NL-SHADE-RSP 9914.9（11）・IMODE 10592.8。**MC-ESO より速いのは CMA 系 4 手法と AMALGAM-SO の 5 手法**（うち 4 手法は SR@1e-10 が 72〜85%）。
- 関数別 SR@1e-10（現既定 `MC-ESO` 対 旧既定 `MC-ESO-v2`、差のある 2 関数のみ。12 関数の全列挙は記録先）: **F03-RastriginSep 30→100（+70）**、**F04-BucheRastrigin 0→85（+85）**。残る 10 関数は 18 手法の多くと同じく不変（12 関数平均 85.83→98.75、**+12.92pt**）。悪化した関数は無い。
- 18 手法の関数別 SR@1e-10 で目立つ点: **F04-BucheRastrigin は MC-ESO 85 に対し 8 手法が 0〜10%**（SPS-L-SHADE-EIG と NL-SHADE-RSP だけが 100、L-SHADE 100、EBOwithCMAR 95、IMODE 95）、**F12-BentCigar は MC-ESO 100 に対し 11 手法が 60% 以下**（SPS-L-SHADE-EIG 0・NL-SHADE-RSP 0・IMODE 5・EA4eig 25・L-SRTDE 35・L-SHADE 45）、F03-RastriginSep は MC-ESO 100 に対し 7 手法が 30% 以下、F10・F11 は IMODE と NL-SHADE-RSP が 0。F01・F02 は 18 手法すべて 100%。
- Wilcoxon（ref=MC-ESO、p<0.05 両側、A12 で向き。勝ち/負け）: IPOP-CMA-ES 9/0、EA4eig 9/0、IMODE 8/0、BIPOP-CMA-ES 8/0、NL-SHADE-RSP 7/0、EBOwithCMAR 5/0、L-SRTDE 4/0、ARRDE 4/0、BIPOP-aCMA-ES 4/1、SPS-L-SHADE-EIG 4/1、AMALGAM-SO 3/1、jSO 3/1、RCMAES 3/1、LSHADE-SPACMA 2/1、MC-ESO-v2 2/0、**ELSHADE-SPACMA 1/1**、**L-SHADE 1/1**。**負けが勝ちを上回る手法は 1 つも無い**。負けは **F06-AttractiveSector**（LSHADE-SPACMA A12=0.12・ELSHADE-SPACMA 0.17・RCMAES 0.16・AMALGAM-SO 0.20・BIPOP-aCMA-ES 0.27・L-SHADE 0.29・jSO 0.13、large〜medium の 7 手法）と **F04-BucheRastrigin**（SPS-L-SHADE-EIG A12=0.35 medium）の 2 関数だけ。勝ちが最も多い関数は F12-BentCigar（10 手法に）と F04-BucheRastrigin（9 手法に）。全列挙は記録先の `analyze_quick_d5_f01-f12.txt` / `detail_d5_f01-f12.txt`。
- 参考 [5] CEC 式の平均誤差の順位（低いほど良い、12 関数、per-run 誤差 1e-8 未満は 0）: **1. MC-ESO 7.58（11/12 で最良）**、2. EBOwithCMAR 7.71（10/12）、3. L-SHADE 7.88（11/12）、4. LSHADE-SPACMA 7.88（11/12）、5. ELSHADE-SPACMA 8.00（10/12）、**6. MC-ESO-v2 8.58（10/12）**、7. jSO 8.67、8. ARRDE 9.00、9. BIPOP-CMA-ES 9.17、10. IPOP-CMA-ES 9.17、…16. RCMAES 11.79、17. NL-SHADE-RSP 12.58、18. BIPOP-aCMA-ES 12.79。**2D（j45d2）では MC-ESO は CEC 順位 3 位だった**ので、この 12 関数では SR@1e-10 の 1 位と CEC 順位の 1 位が一致している（24 関数で同じになるかは 3/7 が済むまで未確定）。
- 書かれたとおりに行かなかった点: (1) ジョブのコマンドに `--jobs` の指定は無いが、**j45d2 の前例に合わせて `--jobs 4` で回した**（`--jobs>1` でも seed は run 番号だけで決まるので数値は変わらない）。(2) ジョブは集計と記録を「【手法解説ページの全手法比較 1/7】と同じ」と参照しているが **1/7 はその記録で消えている**ので、2026-10-13 の 1/7 の作業ログ項目を参照先として使った。**後半（3/7）が同じ `dim5/` に入るので、ファイル名に `_f01-f12` を付けて上書きを防いだ**（24 関数の集計は 3/7 の担当）。(3) `detail_*.txt` の生成スクリプトは repo に無いので scratchpad に書いて使った（[1] の表・Wilcoxon の数は `analyze_quick.py` の出力と一致することを確認）。(4) 環境は findings.md の手順（`--ignore-installed blinker`、pynmmso は空 stub を `PYTHONPATH`）で numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1 に固定し j45d2 と揃えた。BLAS の確認のため `threadpoolctl` を追加で入れた（最適化には使われない）。
- 記録: `analysis/single/j45/dim5/summary_f01-f12.csv`・`wilcoxon_f01-f12.csv`・`stats_f01-f12.tar.gz`・`analyze_quick_d5_f01-f12.txt`・`detail_d5_f01-f12.txt`、`analysis/single/j45/result_d5a.json`。`find analysis -type f | wc -l` = 338。
### 2026-10-13 【手法解説ページの全手法比較 1/7】2D の 18 手法（現既定・旧既定 `MC-ESO-v2` ＋ 標準の比較手法 16）を 24/24 関数で並べた（j45d2） — **MC-ESO（現既定）は SR@1e-10 94.38% で 18 手法中 1 位（2 位 `MC-ESO-v2` 93.96%、3 位 ELSHADE-SPACMA 92.08%）、`evals_succ_mean` も 770.8 で BIPOP-aCMA-ES（353.8、SR 60.83%）と RCMAES（594.4、SR 53.54%）に次ぐ 3 番目。一方 CEC 式の平均誤差の順位は 7.79 の 3 位で、EBOwithCMAR 7.46・IMODE 7.48 に負ける。Wilcoxon は 17 手法のうち 8 手法に 1〜2 関数（F17・F18・F24）で負け越す関数があり、負けが勝ちを上回るのは jSO・ELSHADE-SPACMA（どちらも 1 勝 2 敗）の 2 手法**（採否の判断はしない）

- 実行: ジョブ記載のコマンドをそのまま 1 回（`--all --n-runs 20 --max-evals 5000 --instances 1-5 --jobs 4`、18 手法、label `j45d2`）。wall **8.1 分**（13:33:19→13:41:22）、`result.json` は `status=done`。**24/24 関数 × 18 手法 = 432 行**（`wilcoxon.csv` 408 行＝24×17、(関数, 手法) の重複・欠落ゼロ、per-run stats 24 ファイル）。numpy 2.4.6 / cma 4.5.0 / minionpy 1.9.1、CPU **Intel Xeon @ 2.10GHz 4 コア（family 6 / model 207 / stepping 2、avx512f あり）**＝ j35b・j43 と同じ容器仕様。BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。
- 全体（24 関数平均、SR@1e-2 / 1e-4 / 1e-7 / **1e-10** ％、`evals` ＝ 自分の成功関数の平均、nf ＝ 成功のある関数数）。SR@1e-10 の上位 10 ＋ MC-ESO 2 世代: **MC-ESO 95.83 / 95.83 / 95.21 / 94.38**（770.8、24）、**MC-ESO-v2 95.42 / 95.21 / 94.79 / 93.96**（853.7、24）、ELSHADE-SPACMA 94.38 / 93.75 / 93.12 / **92.08**（1650.1、24）、ARRDE 93.33 / 92.71 / 91.88 / **91.88**（1160.0、24）、EBOwithCMAR 96.88 / 95.62 / 93.96 / **91.04**（2081.9、24）、IMODE 96.67 / 95.62 / 92.71 / **91.04**（1946.2、24）、L-SHADE 92.71 / 92.50 / 91.67 / **91.04**（2148.1、24）、jSO 91.04 / 90.83 / 90.21 / **90.00**（1462.0、23）、LSHADE-SPACMA 92.92 / 91.67 / 90.21 / **89.17**（1271.0、24）、SPS-L-SHADE-EIG 93.12 / 91.25 / 88.75 / **86.67**（1993.4、23）。全 18 手法は記録先。
- `evals_succ_mean`（両方で成功のある関数だけの対応平均。MC-ESO / 相手、nf）: 対 BIPOP-aCMA-ES 735.6 / **353.8**（23）、対 RCMAES 733.5 / **594.4**（21）、対 AMALGAM-SO 735.6 / 792.9（23）、対 MC-ESO-v2 770.8 / 853.7（24）、対 IPOP-CMA-ES 735.6 / 866.7（23）、対 L-SRTDE 733.3 / 930.3（23）、対 BIPOP-CMA-ES 735.6 / 984.6（23）、対 ARRDE 770.8 / 1160.0（24）、以下 LSHADE-SPACMA 1271.0・jSO 1462.0・ELSHADE-SPACMA 1650.1・IMODE 1946.2・SPS-L-SHADE-EIG 1993.4・EBOwithCMAR 2081.9・L-SHADE 2148.1・EA4eig 2578.5（22）・NL-SHADE-RSP 2979.2（23）。**MC-ESO より速いのは BIPOP-aCMA-ES と RCMAES の 2 手法だけ**（どちらも SR@1e-10 は 53〜61%）。
- 関数別 SR@1e-10（現既定 `MC-ESO` 対 旧既定 `MC-ESO-v2`、差のある 6 関数のみ。24 関数の全列挙は記録先）: **F04-BucheRastrigin 90→100（+10）**、**F14-DiffPowers 95→100（+5）**、**F16-Weierstrass 95→100（+5）**、**F23-Katsuura 85→95（+10）**、**F13-SharpRidge 100→95（-5）**、**F18-SchafferF7ill 100→85（-15）**。残る 18 関数は不変（24 関数平均 93.96→94.38、**+0.42pt**）。
- 18 手法の関数別 SR@1e-10 で目立つ点: **F23-Katsuura は MC-ESO 95 / MC-ESO-v2 85 / BIPOP-aCMA-ES 60 に対し他の 15 手法が 0〜15%**、**F24-LunacekRastrigin は MC-ESO 10 に対し EBOwithCMAR 45・ARRDE 40・ELSHADE-SPACMA 35・IMODE 35**、F12-BentCigar は MC-ESO 90 に対し ELSHADE-SPACMA・RCMAES 100、F18-SchafferF7ill は MC-ESO 85 に対し 6 手法が 100。F01・F02・F06・F08〜F11 は 18 手法すべて 100%。
- Wilcoxon（ref=MC-ESO、p<0.05 両側、A12 で向き。勝ち/負け）: RCMAES 14/0、BIPOP-aCMA-ES 14/0、EA4eig 12/0、BIPOP-CMA-ES 10/0、IPOP-CMA-ES 9/0、NL-SHADE-RSP 6/1、L-SRTDE 5/1、AMALGAM-SO 5/0、LSHADE-SPACMA 3/1、SPS-L-SHADE-EIG 3/0、EBOwithCMAR 2/1、L-SHADE 2/0、IMODE 1/1、ARRDE 1/1、**jSO 1/2**、**ELSHADE-SPACMA 1/2**、MC-ESO-v2 0/0。負けの集まる関数は **F17-SchafferF7（5 手法に、A12=0.30 medium）・F18-SchafferF7ill（3 手法に、A12=0.20〜0.26 large）・F24-LunacekRastrigin（2 手法に、A12=0.21 large / 0.30 medium）** の 3 つだけ。勝ちは F23-Katsuura（13 手法に、A12 0.91〜1.00 large）が最も多い。全列挙は記録先の `analyze_quick_d2.txt` / `detail_d2.txt`。
- 参考 [5] CEC 式の平均誤差の順位（低いほど良い、24 関数、per-run 誤差 1e-8 未満は 0）: 1. EBOwithCMAR 7.46（19/24 で最良）、2. IMODE 7.48（19/24）、**3. MC-ESO 7.79（20/24）**、4. L-SHADE 7.92（20/24）、**5. MC-ESO-v2 7.98（18/24）**、6. ELSHADE-SPACMA 8.00（19/24）、7. ARRDE 8.71、8. SPS-L-SHADE-EIG 8.85、9. jSO 9.00、10. EA4eig 9.02。**SR@1e-10 では 1 位だが CEC 順位では 3 位**で、順位の向きが指標で食い違う（2026-10-13 のユーザー決定では CEC 順位を主に見る）。
- 書かれたとおりに行かなかった点: (1) ジョブのコマンドに `--jobs` の指定は無いが、**j35b・j43 の前例に合わせて `--jobs 4` で回した**（`--jobs>1` でも seed は run 番号だけで決まるので数値は変わらない）。(2) 集計欄は「MC-ESO 3 世代の行」を求めているが、**ジョブのコマンドに入っている MC-ESO 系は現既定と `MC-ESO-v2` の 2 つだけ**なので 2 世代で書いた。(3) ジョブは記録先が `methods_data.py` にそのまま読めると書いているが、`core.run_data.cec_mean_errors` は `dim2/stats/` の**ディレクトリ**を読むので、[5] を再計算するには `stats.tar.gz` を展開する必要がある。`result.json`（`methods_data.py` が要る）も `analysis/single/j45/` に写した。(4) 環境は findings.md の手順（`--ignore-installed blinker`、pynmmso は空 stub を `PYTHONPATH`）で、numpy 2.4.6 / cma 4.5.0 に固定して j35b・j43 と揃えた。 (5) **残るジョブ 2/7〜7/7 は集計と記録を「【手法解説ページの全手法比較 1/7】と同じ」と参照しているが、その 1/7 はこの記録で消えた**。参照先はこの作業ログの項目（`--jobs 4`・`analysis/single/j45/dim{N}/` に `summary.csv`・`wilcoxon.csv`・`stats.tar.gz`）になる。ジョブの文面は測定ルーチンが書き換えない。
- 記録: `analysis/single/j45/dim2/summary.csv`・`wilcoxon.csv`・`stats.tar.gz`・`analyze_quick_d2.txt`・`detail_d2.txt`、`analysis/single/j45/result.json`。`find analysis -type f | wc -l` = 332。
古い作業ログ（2026-10-06〜13 の改善案・全手法比較・新既定値の ablation・`v2_*` の候補・CEC2022 の 25000 / 200000 評価・一般化の検証・公式予算・20D）は [history.md](history.md) の該当日の節と [findings.md](findings.md) 7〜8 節に要約してある。全文は git の履歴（コミット `a16908a`）にある。
