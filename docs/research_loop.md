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

2. **【公式予算 3/3】10D の後半 12 関数。** 同じコマンドで `--funcs F13-SharpRidge,F14-DiffPowers,F15-RastriginRot,F16-Weierstrass,F17-SchafferF7,F18-SchafferF7ill,F19-GriewankRosenbrock,F20-Schwefel,F21-Gallagher101,F22-Gallagher21,F23-Katsuura,F24-LunacekRastrigin`（label `j41b`）。【公式予算 2/3】の結果と連結して 24/24 を確かめ、【公式予算 1/3】と同じ集計をする。**記録**: `analysis/single/j41/`。 (claimed 2026-10-13 04:30 UTC)
3. **【20 次元 1/2】20D を判定と同じ予算 2500·D で測る。** `./run.sh quick --all --dim 20 --max-evals 50000 --n-runs 20 --methods "MC-ESO,ELSHADE-SPACMA,LSHADE-SPACMA,EBOwithCMAR,IPOP-CMA-ES,L-SHADE"`（label `j42`）。90 分に収まらなければ規則どおり完走した関数までを記録し、残りを書き戻す。
    **背景**: BBOB の標準は 2〜40 次元で、論文は 20D を併記するのが普通。CEC の上位手法は 30〜100D を主戦場にしているので、低次元だけの比較は CMA-ES 系に有利な可能性がある。
    **集計**: 【公式予算 1/3】と同じ。**記録**: `analysis/single/j42/`。
4. **【20 次元 2/2】20D を公式予算 10000·D で、手法を 4 つに絞って測る。** `./run.sh quick --all --funcs FUNCS --dim 20 --max-evals 200000 --n-runs 20 --methods "MC-ESO,ELSHADE-SPACMA,LSHADE-SPACMA,IPOP-CMA-ES"` を 3 組に分けて回す（label `j43a` / `j43b` / `j43c`）。FUNCS は (a) `F01-Sphere,F02-EllipsoidalSep,F03-RastriginSep,F04-BucheRastrigin,F05-LinearSlope,F06-AttractiveSector,F07-StepEllipsoidal,F08-Rosenbrock`、(b) `F09-RosenbrockRot,F10-EllipsoidalRot,F11-Discus,F12-BentCigar,F13-SharpRidge,F14-DiffPowers,F15-RastriginRot,F16-Weierstrass`、(c) `F17-SchafferF7,F18-SchafferF7ill,F19-GriewankRosenbrock,F20-Schwefel,F21-Gallagher101,F22-Gallagher21,F23-Katsuura,F24-LunacekRastrigin`。1 回の起動で終わる分だけ回し、残りの組は書き戻す。
    **集計**: 24 関数がそろったら【公式予算 1/3】と同じ集計。**記録**: `analysis/single/j43/`。
5. **CEC2022（10D、hold-out）を公式予算 200000 評価で測る**（25000 評価の分は 2026-10-12 に測り終えた。`analysis/single/j35/` と作業ログ）。
   `./run.sh quick --suite cec2022 --all --n-runs 20 --max-evals 200000 --funcs <完全名> --methods "MC-ESO,EA4eig,LSHADE-SPACMA,L-SHADE,L-SRTDE,EBOwithCMAR,jSO"` を **1 関数ずつ 12 回**（G01-Zakharov / G02-Rosenbrock / G03-SchafferF7 / G04-Rastrigin / G05-Levy / G06-Hybrid1 / G07-Hybrid2 / G08-Hybrid3 / G09-Composition1 / G10-Composition2 / G11-Composition3 / G12-Composition4。label `j35b<関数番号>`）。1 回の起動で終わる分だけ回し、残りの関数を書き戻す。
   **目安**: 25000 評価では 12 関数 × 15 手法 × 20 run が `--jobs 4` で 57 分だった（2026-10-12）。手法を 8 つ（MC-ESO・最新の候補・25000 評価の CEC 順位の上位 6 手法）に絞ったので、1 関数あたりおよそ 20 分、**1 回の起動で 4 関数前後**。2026-10-13 に一般化の検証より後ろへ回した（対話セッション）。
   **背景**: 正準環境の 25000 評価では SR@1e-10 が MC-ESO 31.2%（15 手法中 1 位）、`v2_hlC` 30.8%（2 位）、L-SRTDE 30.4%。一方 CEC 流の平均誤差の順位では MC-ESO 9 位 / `v2_hlC` 11 位（ハイブリッド G06〜G08 と合成 G09・G10・G12 で DE 系に負ける）。公式予算で順位が入れ替わるかを確かめる。CEC2022 用にハイパーパラメータは調整しない（[experiments.md](experiments.md)）。
   **集計**: 標準の集計に加え、CEC の慣例どおり **平均誤差による関数ごとの順位の平均**（誤差 1e-8 未満は 0 とみなす）を 7 手法について書く。関数別に MC-ESO / 最良の比較手法の SR@1e-7 と中央値誤差を並べる。25000 評価の値（`analysis/single/j35/`）と並べる。
   **記録**: `analysis/single/j35/`（予算ごとに別ファイル。25000 評価は `*_e25k.*`）。
6. **【手法解説ページの全手法比較 1/7】2D。** `./run.sh quick --all --n-runs 20 --max-evals 5000 --instances 1-5 --methods "MC-ESO,MC-ESO-v2,MC-ESO-v1,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,PSO,DE,L-SHADE,SaVOA,NM-Restart,EA4eig,EA4eig-jSO-IDEbd,EA4eig-Simpl,LSHADE-SPACMA,AMALGAM-SO,AMALGAM-SO-DE,HSES,ICMAES-ILS,MOS,UMOEA-II,EBOwithCMAR,HMHH,HMHH-random,PS-CMA-ES,DEPSO,jSO,L-SRTDE,IMODE,ELSHADE-SPACMA,APGSK-IMODE,SPS-L-SHADE-EIG,CoBiDE,NG-Portfolio"`（label `j45d2`）。
   **背景**: 手法解説ページ（`/methods`）の成績表とヒートマップは 2026-10-12 の既定（現 `MC-ESO-v2`）で測った run のまま。現在の既定（2026-10-13〜）と比較手法 31 個・旧既定 2 世代を同じ run に並べて作り直す（ユーザー指示 2026-10-13）。2026-10-13 の方針に従い、run を instance 1〜5 に散らす（`--instances 1-5`、各 instance 4 run）。
   **集計**: 関数の組ごとの結果を連結して 24/24 を確かめ、標準の集計（[1] 全体表、[4] Wilcoxon、[5] CEC 式の順位）を 34 手法について書く。作業ログには全体表の上位 10 手法と MC-ESO 3 世代の行、CEC 式の順位の上位 10 と MC-ESO 3 世代だけを書けばよい（全表は記録先に）。
   **記録**: `analysis/single/j45/dim2/` に連結した `summary.csv`・`wilcoxon.csv` を置き、run ごとの `stats/` は **`dim2/stats.tar.gz` の 1 ファイルにまとめる**（ファイル数の上限のため）。`scripts/web/methods_data.py` がこの形をそのまま読む。
7. **【手法解説ページの全手法比較 2/7】5D の前半 12 関数。** `./run.sh quick --all --funcs F01-Sphere,F02-EllipsoidalSep,F03-RastriginSep,F04-BucheRastrigin,F05-LinearSlope,F06-AttractiveSector,F07-StepEllipsoidal,F08-Rosenbrock,F09-RosenbrockRot,F10-EllipsoidalRot,F11-Discus,F12-BentCigar --dim 5 --max-evals 12500 --n-runs 20 --instances 1-5 --methods "MC-ESO,MC-ESO-v2,MC-ESO-v1,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,PSO,DE,L-SHADE,SaVOA,NM-Restart,EA4eig,EA4eig-jSO-IDEbd,EA4eig-Simpl,LSHADE-SPACMA,AMALGAM-SO,AMALGAM-SO-DE,HSES,ICMAES-ILS,MOS,UMOEA-II,EBOwithCMAR,HMHH,HMHH-random,PS-CMA-ES,DEPSO,jSO,L-SRTDE,IMODE,ELSHADE-SPACMA,APGSK-IMODE,SPS-L-SHADE-EIG,CoBiDE,NG-Portfolio"`（label `j45d5a`）。集計と記録は【手法解説ページの全手法比較 1/7】と同じ（`dim5`。後半が済んでから 24 関数で集計）。
8. **【手法解説ページの全手法比較 3/7】5D の後半 12 関数。** `./run.sh quick --all --funcs F13-SharpRidge,F14-DiffPowers,F15-RastriginRot,F16-Weierstrass,F17-SchafferF7,F18-SchafferF7ill,F19-GriewankRosenbrock,F20-Schwefel,F21-Gallagher101,F22-Gallagher21,F23-Katsuura,F24-LunacekRastrigin --dim 5 --max-evals 12500 --n-runs 20 --instances 1-5 --methods "MC-ESO,MC-ESO-v2,MC-ESO-v1,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,PSO,DE,L-SHADE,SaVOA,NM-Restart,EA4eig,EA4eig-jSO-IDEbd,EA4eig-Simpl,LSHADE-SPACMA,AMALGAM-SO,AMALGAM-SO-DE,HSES,ICMAES-ILS,MOS,UMOEA-II,EBOwithCMAR,HMHH,HMHH-random,PS-CMA-ES,DEPSO,jSO,L-SRTDE,IMODE,ELSHADE-SPACMA,APGSK-IMODE,SPS-L-SHADE-EIG,CoBiDE,NG-Portfolio"`（label `j45d5b`）。集計と記録は【手法解説ページの全手法比較 1/7】と同じ（`dim5`。後半が済んでから 24 関数で集計）。
9. **【手法解説ページの全手法比較 4/7】10D の第 1 四半分（6 関数）。** `./run.sh quick --all --funcs F01-Sphere,F02-EllipsoidalSep,F03-RastriginSep,F04-BucheRastrigin,F05-LinearSlope,F06-AttractiveSector --dim 10 --max-evals 25000 --n-runs 20 --instances 1-5 --methods "MC-ESO,MC-ESO-v2,MC-ESO-v1,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,PSO,DE,L-SHADE,SaVOA,NM-Restart,EA4eig,EA4eig-jSO-IDEbd,EA4eig-Simpl,LSHADE-SPACMA,AMALGAM-SO,AMALGAM-SO-DE,HSES,ICMAES-ILS,MOS,UMOEA-II,EBOwithCMAR,HMHH,HMHH-random,PS-CMA-ES,DEPSO,jSO,L-SRTDE,IMODE,ELSHADE-SPACMA,APGSK-IMODE,SPS-L-SHADE-EIG,CoBiDE,NG-Portfolio"`（label `j45d10a`）。集計と記録は【手法解説ページの全手法比較 1/7】と同じ（`dim10`。4 つ済んでから 24 関数で集計）。
10. **【手法解説ページの全手法比較 5/7】10D の第 2 四半分（6 関数）。** `./run.sh quick --all --funcs F07-StepEllipsoidal,F08-Rosenbrock,F09-RosenbrockRot,F10-EllipsoidalRot,F11-Discus,F12-BentCigar --dim 10 --max-evals 25000 --n-runs 20 --instances 1-5 --methods "MC-ESO,MC-ESO-v2,MC-ESO-v1,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,PSO,DE,L-SHADE,SaVOA,NM-Restart,EA4eig,EA4eig-jSO-IDEbd,EA4eig-Simpl,LSHADE-SPACMA,AMALGAM-SO,AMALGAM-SO-DE,HSES,ICMAES-ILS,MOS,UMOEA-II,EBOwithCMAR,HMHH,HMHH-random,PS-CMA-ES,DEPSO,jSO,L-SRTDE,IMODE,ELSHADE-SPACMA,APGSK-IMODE,SPS-L-SHADE-EIG,CoBiDE,NG-Portfolio"`（label `j45d10b`）。集計と記録は【手法解説ページの全手法比較 1/7】と同じ（`dim10`。4 つ済んでから 24 関数で集計）。
11. **【手法解説ページの全手法比較 6/7】10D の第 3 四半分（6 関数）。** `./run.sh quick --all --funcs F13-SharpRidge,F14-DiffPowers,F15-RastriginRot,F16-Weierstrass,F17-SchafferF7,F18-SchafferF7ill --dim 10 --max-evals 25000 --n-runs 20 --instances 1-5 --methods "MC-ESO,MC-ESO-v2,MC-ESO-v1,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,PSO,DE,L-SHADE,SaVOA,NM-Restart,EA4eig,EA4eig-jSO-IDEbd,EA4eig-Simpl,LSHADE-SPACMA,AMALGAM-SO,AMALGAM-SO-DE,HSES,ICMAES-ILS,MOS,UMOEA-II,EBOwithCMAR,HMHH,HMHH-random,PS-CMA-ES,DEPSO,jSO,L-SRTDE,IMODE,ELSHADE-SPACMA,APGSK-IMODE,SPS-L-SHADE-EIG,CoBiDE,NG-Portfolio"`（label `j45d10c`）。集計と記録は【手法解説ページの全手法比較 1/7】と同じ（`dim10`。4 つ済んでから 24 関数で集計）。
12. **【手法解説ページの全手法比較 7/7】10D の第 4 四半分（6 関数）。** `./run.sh quick --all --funcs F19-GriewankRosenbrock,F20-Schwefel,F21-Gallagher101,F22-Gallagher21,F23-Katsuura,F24-LunacekRastrigin --dim 10 --max-evals 25000 --n-runs 20 --instances 1-5 --methods "MC-ESO,MC-ESO-v2,MC-ESO-v1,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,PSO,DE,L-SHADE,SaVOA,NM-Restart,EA4eig,EA4eig-jSO-IDEbd,EA4eig-Simpl,LSHADE-SPACMA,AMALGAM-SO,AMALGAM-SO-DE,HSES,ICMAES-ILS,MOS,UMOEA-II,EBOwithCMAR,HMHH,HMHH-random,PS-CMA-ES,DEPSO,jSO,L-SRTDE,IMODE,ELSHADE-SPACMA,APGSK-IMODE,SPS-L-SHADE-EIG,CoBiDE,NG-Portfolio"`（label `j45d10d`）。集計と記録は【手法解説ページの全手法比較 1/7】と同じ（`dim10`。4 つ済んでから 24 関数で集計）。

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

### 2026-10-13 【公式予算 2/3】10D の前半 12 関数を公式予算 100000 評価で（j41a、6 手法） — **予算を 2500·D から 10000·D（4 倍）にすると、F01〜F12 では MC-ESO の SR@1e-10 は 99.58% で 6 手法中 2 位。首位は L-SHADE 100%、次が MC-ESO、ELSHADE-SPACMA と EBOwithCMAR が 99.17%。同じ構成・同じ正準環境の 25000 評価（j36 の `v2_scan2d`、12 関数平均 99.58）からは ±0.00pt で、動いたのは F07 95→100 と F09 100→95 の 2 関数だけ（前半 12 関数は 4 手法が天井に張りつくため、判定は 24 関数がそろう【公式予算 3/3】の後）。CEC 式の平均誤差の順位は MC-ESO 4 位（3.46、首位は L-SHADE 3.21）。`evals_succ_mean` は対応平均で IPOP 以外のすべてに勝つ**（採否の判断はしない）

- 実行: ジョブ記載のコマンドをそのまま 1 回（`--all --funcs F01〜F12 --dim 10 --max-evals 100000 --n-runs 20 --methods "MC-ESO,ELSHADE-SPACMA,LSHADE-SPACMA,EBOwithCMAR,IPOP-CMA-ES,L-SHADE"`、label `j41a`、`--jobs 4`、shard 分割なし）。wall **41.6 分**（02:40:04→03:21:40）。**12/12 そろった**（`summary.csv` 72 行＝6 手法×12、`wilcoxon.csv` 60 行＝5 手法×12、(関数, 手法) の重複・欠落ゼロ、`result.json` は `status=done`）。numpy 2.4.6 / cma 4.5.0、CPU **Intel Xeon @ 2.80GHz 4 コア（family 6 / model 85 / stepping 7、avx512f あり）**、BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。j37〜j40 と同じ 2.80GHz のコンテナ。
- 全体（12 関数平均、SR@1e-2 / 1e-4 / 1e-7 / **1e-10** ％、`evals` ＝ 自分の成功関数の平均、nf ＝ 成功のある関数数）: L-SHADE 100.00 / 100.00 / 100.00 / **100.00**（33365.0、12）、**MC-ESO 99.58 / 99.58 / 99.58 / 99.58**（10361.8、12）、ELSHADE-SPACMA 99.17 / 99.17 / 99.17 / **99.17**（28613.0、12）、EBOwithCMAR 99.58 / 99.58 / 99.58 / **99.17**（37668.2、12）、LSHADE-SPACMA 91.25 / 91.25 / 91.25 / **91.25**（18266.7、11）、IPOP-CMA-ES 83.33 / 83.33 / 83.33 / **83.33**（4453.8、10）。
- `evals_succ_mean`（両方で成功のある関数だけの対応平均。MC-ESO / 相手、nf）: 対 EBOwithCMAR 10361.8 / 37668.2（12）、対 ELSHADE-SPACMA 10361.8 / 28613.0（12）、対 L-SHADE 10361.8 / 33365.0（12）、対 LSHADE-SPACMA 10410.6 / 18266.7（11）、対 IPOP-CMA-ES 10493.9 / **4453.8**（10）。**IPOP だけが MC-ESO より速い**（5D の j38〜j40・10D の j39 と同じ向き）。
- **予算を 4 倍にしたときの関数別 SR@1e-10（同じ構成・同じ正準環境、j36 の `v2_scan2d` 25000 評価 → 本回 100000 評価）**: **F07-StepEllipsoidal 95→100（+5）・F09-RosenbrockRot 100→95（−5）の 2 関数だけが動き**、残る 10 関数は 100% で不変。12 関数平均は 99.58 → 99.58（**±0.00pt**）。本回の MC-ESO は **F09 の 95% だけが 100% 未満**。比較手法側で 100% に届かないのは LSHADE-SPACMA（F04 0・F03 95）、IPOP（F03 0・F04 0）、ELSHADE-SPACMA（F04 90）、EBOwithCMAR（F04 95・F12 95）で、**F04-BucheRastrigin が比較手法の落ちどころ**（MC-ESO と L-SHADE だけが 100%）。
- Wilcoxon（ref=MC-ESO、p_two<0.05、A12 の向き。better ＝ MC-ESO が有意に良い関数数 / worse ＝ 有意に悪い関数数）: IPOP-CMA-ES **11/0**（F07 以外の全関数、A12 0.85〜1.00 の large）、LSHADE-SPACMA 1/1（better F04 A12=1.00 / worse F06 A12=0.00）、EBOwithCMAR 1/1（better F12 A12=0.93 / worse F06 A12=0.00）、L-SHADE 0/2（F04 A12=0.40 small・F06 A12=0.00 large）、ELSHADE-SPACMA 0/1（F06 A12=0.02 large）。**MC-ESO が負ける側は F06-AttractiveSector が 4 手法すべてに出る**（SR は両者 100% で、`best_f` の下位桁の差）。全関数が同値で p=nan になる組（天井に張りつく関数）は有意に数えていない。
- 参考 [5] CEC 式の平均誤差の順位（低いほど良い、12 関数）: L-SHADE 1 位（3.21、12/12 で最良）・EBOwithCMAR 2 位（3.33）・ELSHADE-SPACMA 3 位（3.42）・**MC-ESO 4 位（3.46）**・LSHADE-SPACMA 5 位（3.71）・IPOP-CMA-ES 6 位（3.88）。**SR@1e-10 で 2 位の MC-ESO がこの指標で 4 位**という向きは 5D の j40（3 位 → 4 位）と同じだが、12 関数すべてが天井近くなので差は 0.25 以内。
- 書かれたとおりに回せなかった点: **ジョブの指定どおり 24 関数の集計はしていない**（【公式予算 3/3】＝ `j41b` が未測定のため。連結用に本回の集計を `analysis/single/j41/` に置いた）。環境の注記は前回と同じ 2 点: `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順（空 stub を `PYTHONPATH`）に従い、`python3 -m pip install --ignore-installed blinker ...`（3.11）で正準の numpy 2.4.6 / cma 4.5.0 に固定した（`pip` は python3.13 を指す）。記録は `analysis/single/j41/{summary,wilcoxon}_d10_e100k_f01-12.csv`、`analyze_quick_d10_e100k_f01-12.txt`、`detail_d10_e100k_f01-12.txt`、生成スクリプト `agg_j41.py`。`analysis` のファイル数 294（上限 約 400）。`check_doc_size.py` は通る（research_loop.md は **211 / 300 行**。main での統合でログが history.md へ畳まれた分、前回の 293 行から減っている）。

### 2026-10-13 【公式予算 1/3】5D を CEC 公式予算 50000 評価で（j40d5、instance 1、6 手法） — **予算を 2500·D から 10000·D（4 倍）にすると MC-ESO の SR@1e-10 は 81.67% で 6 手法中 3 位。首位は ELSHADE-SPACMA 83.75%、次に EBOwithCMAR 82.71%、MC-ESO との差は −2.08pt。同じ構成・同じ正準環境の 12500 評価（j36d5 の `v2_scan2d` 78.33%）からは +3.34pt で、比較手法の伸びのほうが大きい（L-SHADE 60.00→81.46 ＝ +21.46pt、EBOwithCMAR 63.75→82.71 ＝ +18.96 ※ただし対照は Windows ローカル）。CEC 式の平均誤差の順位では MC-ESO 4 位（3.52、首位は EBOwithCMAR 2.90）。`evals_succ_mean` は対応平均で IPOP 以外のすべてに勝つ**（採否の判断はしない）

- 実行: ジョブ記載のコマンドをそのまま 1 回（`--all --dim 5 --max-evals 50000 --n-runs 20 --methods "MC-ESO,ELSHADE-SPACMA,LSHADE-SPACMA,EBOwithCMAR,IPOP-CMA-ES,L-SHADE"`、label `j40d5`、`--jobs 4`、shard 分割なし）。wall **44.9 分**（00:33:17→01:18:15）。**24/24 そろった**（`summary.csv` 144 行＝6 手法×24、`wilcoxon.csv` 120 行＝5 手法×24、(関数, 手法) の重複・欠落ゼロ、`result.json` は `status=done`）。numpy 2.4.6 / cma 4.5.0、CPU **Intel Xeon @ 2.80GHz 4 コア（family 6 / model 85 / stepping 7、avx512f あり）**、BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。j37〜j39 と同じ 2.80GHz のコンテナ。
- 全体（24 関数平均、SR@1e-2 / 1e-4 / 1e-7 / **1e-10** ％、`evals` ＝ 自分の成功関数の平均、nf ＝ 成功のある関数数）: ELSHADE-SPACMA 84.38 / 83.96 / 83.96 / **83.75**（12305.8、23）、EBOwithCMAR 86.46 / 85.00 / 83.75 / **82.71**（16329.8、22）、**MC-ESO 83.54 / 83.54 / 82.92 / 81.67**（8619.3、23）、L-SHADE 83.54 / 82.71 / 81.67 / **81.46**（16263.5、22）、LSHADE-SPACMA 75.42 / 74.38 / 74.38 / **74.38**（9071.7、22）、IPOP-CMA-ES 75.83 / 75.42 / 75.00 / **68.96**（8577.8、22）。
- `evals_succ_mean`（両方で成功のある関数だけの対応平均。MC-ESO / 相手、nf）: 対 ELSHADE-SPACMA 8619.3 / 12305.8（23）、対 LSHADE-SPACMA 8485.3 / 9071.7（22）、対 EBOwithCMAR 7418.7 / 16329.8（22）、対 L-SHADE 7418.7 / 16263.5（22）、対 IPOP-CMA-ES 8785.7 / **8577.8**（22）。**IPOP だけが MC-ESO より速い**（5D の j38・j39 と同じ向き）。
- **予算を 4 倍にしたときの関数別 SR@1e-10（同じ構成・同じ正準環境、j36d5 の `v2_scan2d` 12500 評価 → 本回 50000 評価。10 関数が動いた）**: 上昇 **F18 65→95（+30）・F21 75→100（+25）・F20 5→20（+15）・F23 45→60（+15）**・F08 95→100・F09 95→100・F19 0→5（各 +5）、低下 **F16 90→80（−10）・F15 10→5・F17 100→95（各 −5）**。F24 は 6 手法すべて 0%、F15 は MC-ESO 5% に対し EBOwithCMAR 95%・LSHADE-SPACMA 75%・IPOP 50%・L-SHADE 30% で最大の負け幅。
- **ジョブが並べるよう指定した対照（`runs/20261012_020743_allmethods2_d5_quick`、12500 評価）との関数別 SR@1e-10 の変化**。この対照は **Windows ローカルで測ったもので正準環境ではなく**（`result.json` の note）、その `MC-ESO` 行は **2026-10-13 以前の既定（v1_w4q3E）**なので、差には予算・既定変更・環境の 3 つが混ざる。MC-ESO 11 関数（F04 0→100・F03 20→100・F21 40→100・F22 70→100・F23 30→60・F18 75→95・F20 5→20・F08 90→100・F16 75→80・F17 90→95 が上昇、**F15 30→5（−25）**が低下）。比較手法も 9〜11 関数ずつ動き、EBOwithCMAR は F12・F13・F17・F18 が 0→100、L-SHADE は F17・F18 が 0→100・F12 5→100・F20 30→100、IPOP は F20 5→70・F18 50→100。全体の差は MC-ESO +13.75pt に対し L-SHADE +21.46・EBOwithCMAR +18.96・IPOP +12.08・ELSHADE-SPACMA +9.17・LSHADE-SPACMA +1.88。
- Wilcoxon（ref=MC-ESO、p_two<0.05、A12 の向き。better ＝ MC-ESO が有意に良い関数数 / worse ＝ 有意に悪い関数数）: IPOP-CMA-ES **10/2**、LSHADE-SPACMA 5/6、ELSHADE-SPACMA 3/6、L-SHADE 3/5、EBOwithCMAR 2/5。**MC-ESO が負ける側は F13・F15・F17・F18・F20 が 5 手法すべてに繰り返し出る**（A12 は 0.00〜0.25 の large）ほか F19 が 2 手法。勝つ側は F16・F21・F23・F24 が複数手法に出る。A12 つきの一覧は `detail_d5_e50k.txt` の [F]。
- 参考 [5] CEC 式の平均誤差の順位（低いほど良い）: EBOwithCMAR 1 位（2.90、24 関数中 20 で最良）・L-SHADE 2 位（3.46）・ELSHADE-SPACMA 3 位（3.50）・**MC-ESO 4 位（3.52）**・IPOP-CMA-ES 5 位（3.79）・LSHADE-SPACMA 6 位（3.83）。**SR@1e-10 で 3 位の MC-ESO がこの指標で 4 位**という向きは 5D の j38（5〜6 位）より縮む。
- 書かれたとおりに回せなかった点: **ジョブの集計欄は「7 手法」と書いているが `--methods` は 6 手法**（`v2_scan2d` が既定に入って並びから外れた分）なので 6 手法で集計した。環境の注記は前回と同じ 2 点: `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順（空 stub を `PYTHONPATH`）に従い、`python3 -m pip install --ignore-installed blinker ...`（3.11）で正準の numpy 2.4.6 / cma 4.5.0 に固定した。記録は `analysis/single/j40/{summary,wilcoxon}_d5_e50k.csv`、`analyze_quick_d5_e50k.txt`、`detail_d5_e50k.txt`、生成スクリプト `agg_j40.py`。`analysis` のファイル数 289（上限 約 400）。`check_doc_size.py` は通るが **research_loop.md が 293 / 300 行**で、次回のログで超える（統合は対話セッション）。

### 2026-10-13 【一般化の検証 3/3】調整外 instance の 10D（j39、I=2/3/4） — **MC-ESO の SR@1e-10 は instance 1 の 52.29% から 3 instance 平均 53.33%（**+1.04pt**。2D・5D と違い下がらない）、`v2_scan2d` は 66.04 → 64.10（−1.94pt）。比較手法はそろって上がり（LSHADE-SPACMA 53.75→57.01、IPOP 47.29→50.14、ELSHADE-SPACMA 41.25→42.29、EBOwithCMAR 35.42→37.36、L-SHADE 33.33→33.89）、最良ベースラインとの差は MC-ESO が −1.46pt → −3.68pt に開き、`v2_scan2d` は +12.29pt → +7.09pt に縮む。順位は instance 1 と i2/i3/i4 の 4 つすべてで `v2_scan2d` 1 位・LSHADE-SPACMA 2 位・MC-ESO 3 位・IPOP 4 位、L-SHADE は 4 つすべて 7 位**（採否の判断はしない）

- 実行: ジョブ記載のコマンドをそのまま 3 回（`--all --dim 10 --max-evals 25000 --n-runs 20 --instance {2,3,4} --seed-offset 7`、7 手法、`--jobs 4`、shard 分割なし、順番に）。wall は i2 26.2 分（22:41:01→23:07:12）／i3 24.7 分（23:12:46→23:37:28）／i4 23.5 分（23:37:36→00:01:03）＝ 計 **74.4 分**。**3 回すべて 24/24 そろった**（各 `summary.csv` 168 行＝7 手法×24、`wilcoxon.csv` 144 行＝6 手法×24、(関数, 手法) の重複・欠落ゼロ、`result.json` は 3 本とも `status=done`）。numpy 2.4.6 / cma 4.5.0、CPU **Intel Xeon @ 2.80GHz 4 コア（family 6 / model 85 / stepping 7、avx512f あり）**、BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。j37・j38 と同じ 2.80GHz のコンテナだが、**instance 1 の対照（`allmethods2_d10`・j36）は 2.10GHz のコンテナで測ったもの**。
- 全体（24 関数平均、SR@1e-2 / 1e-4 / 1e-7 / **1e-10** ％、`evals` ＝ 自分の成功関数の平均 evals、nf ＝ 成功のある関数数）。**i2**: MC-ESO 59.79 / 56.04 / 55.21 / **53.96**（9371.0、17）、`v2_scan2d` 68.75 / 66.46 / 63.33 / **62.71**（10177.1、19）、LSHADE-SPACMA **58.75**、IPOP **49.58**、ELSHADE-SPACMA **42.92**、EBOwithCMAR **36.67**、L-SHADE **33.96**。**i3**: MC-ESO 59.58 / 56.25 / 55.00 / **53.33**（9371.3、16）、`v2_scan2d` 71.25 / 70.00 / 66.25 / **65.21**（9936.7、18）、LSHADE-SPACMA **56.25**、IPOP **50.62**、ELSHADE-SPACMA **44.38**、EBOwithCMAR **34.58**、L-SHADE **34.38**。**i4**: MC-ESO 58.96 / 55.62 / 54.17 / **52.71**（9643.1、17）、`v2_scan2d` 69.79 / 68.54 / 66.04 / **64.38**（9728.7、18）、LSHADE-SPACMA **56.04**、IPOP **50.21**、EBOwithCMAR **40.83**、ELSHADE-SPACMA **39.58**、L-SHADE **33.33**。
- **3 instance 平均**: `v2_scan2d` 69.93 / 68.33 / 65.21 / **64.10**（evals 9947.5、nf 18.3）、LSHADE-SPACMA 66.25 / 64.31 / 57.85 / **57.01**（12730.6、17.3）、MC-ESO 59.44 / 55.97 / 54.79 / **53.33**（9461.8、16.7）、IPOP 60.90 / 58.75 / 53.89 / **50.14**（6491.5、16.3）、ELSHADE-SPACMA 69.03 / 62.99 / 52.01 / **42.29**（16035.5、13.7）、EBOwithCMAR 69.65 / 59.24 / 53.82 / **37.36**（17641.5、12.3）、L-SHADE 69.44 / 61.53 / 50.21 / **33.89**（16851.6、11.7）。`evals_succ_mean` は IPOP 6491.5 が最小、MC-ESO 9461.8 が次。両方で成功のある関数だけの対応平均（[1b]）でも IPOP だけが MC-ESO より速い（i2 6436.5 対 9041.1、i3 6562.7 対 9371.3、i4 5919.1 対 9560.0）。`v2_scan2d` は MC-ESO とほぼ同速（例 i4 9759.2 対 9560.0）。
- 関数別 SR@1e-10 の変化（instance 1 → i2/i3/i4 の平均）。**MC-ESO は 8 関数が動き**、上昇は F12 55→83.33（90/85/75、**+28.33pt**）・F21 10→28.33（+18.33）・F16 5→20.00（+15）・F17 30→38.33（+8.33）、低下は F22 35→10.00（5/10/15、**−25pt**）・F18 25→8.33（20/0/5、−16.67）・F09 100→98.33・F11 100→98.33。**`v2_scan2d` は 9 関数**: 低下は **F04 100→71.67（15/100/100、−28.33pt。instance 2 だけ 15%。5D の j38 でも F04 の i2 だけが落ちていた）**・F22 70→45.00（−25）・F13 100→93.33・F08 100→95.00・F21 85→80.00・F07 95→91.67、上昇は F16 5→18.33（+13.33）・F17 30→41.67（+11.67）・F23 0→1.67。比較手法側は **F21-Gallagher101 が instance 1 で全ベースライン異常に低かった**（L-SHADE 20→75.00 ＝ +55pt、EBOwithCMAR 45→91.67 ＝ +46.67、LSHADE-SPACMA 10→53.33 ＝ +43.33、ELSHADE-SPACMA 10→51.67 ＝ +41.67、IPOP 5→33.33 ＝ +28.33）のが最大で、5D と同じ向き。他に LSHADE-SPACMA の F20 15→36.67・F17 60→78.33 の上昇、L-SHADE の F14 80→66.67・F11 95→83.33 の低下。
- Wilcoxon（ref=MC-ESO、p_two<0.05、A12 の向き。better ＝ MC-ESO が有意に良い関数数 / worse ＝ 有意に悪い関数数）。**i2**: `v2_scan2d` 4/6、ELSHADE-SPACMA 10/7、LSHADE-SPACMA 3/9、EBOwithCMAR 9/8、IPOP 10/4、L-SHADE 12/7。**i3**: `v2_scan2d` 5/6、ELSHADE-SPACMA 12/6、LSHADE-SPACMA 8/9、EBOwithCMAR 10/8、IPOP 10/5、L-SHADE 12/7。**i4**: `v2_scan2d` 4/7、ELSHADE-SPACMA 11/5、LSHADE-SPACMA 5/9、EBOwithCMAR 8/9、IPOP 8/3、L-SHADE 12/5。**`v2_scan2d` が MC-ESO より有意に良い関数は 3 instance すべてで F03・F04・F12・F20・F21・F22（A12 は large 中心）**、MC-ESO が勝つ側は F13〜F15・F19 など回ごとに入れ替わる。LSHADE-SPACMA に負ける側も F03・F04・F15・F17・F18・F20・F21 が繰り返し出る。A12 の大きさつきの一覧は `detail_i2i3i4.txt` の [3]。
- 参考 [5] CEC 式の平均誤差の順位（低いほど良い）: i2 は IPOP 1 位（3.56）・LSHADE-SPACMA 2 位・`v2_scan2d` 4 位・**MC-ESO 6 位（4.42）**、i3 は LSHADE-SPACMA 1 位（3.54）・`v2_scan2d` 2 位・**MC-ESO 7 位（4.42、最下位）**、i4 は LSHADE-SPACMA 1 位（3.44）・`v2_scan2d` 4 位・**MC-ESO 7 位（4.52、最下位）**。**SR@1e-10 では 3 位の MC-ESO が、この指標では 6〜7 位**という 5D と同じ向きが 10D でより強く出る。
- 書かれたとおりに回せなかった点は無い（3 コマンドをそのまま、shard 分割もせずに回した）。環境の注記は前回と同じ 2 点: `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順（空 stub を `PYTHONPATH`）に従い、`python3 -m pip install --ignore-installed blinker ...`（3.11）で正準の numpy 2.4.6 / cma 4.5.0 に固定した（`pip` は python3.13 を指す）。記録は `analysis/single/j39/{summary,wilcoxon}_i{2,3,4}.csv`、`analyze_quick_i{2,3,4}.txt`、`detail_i2i3i4.txt`、生成スクリプト `agg_j39.py`。`analysis` のファイル数 381（上限 約 400）。

### 2026-10-12 【一般化の検証 2/3】調整外 instance の 5D（j38、I=2/3/4） — **MC-ESO の SR@1e-10 は instance 1 の 68.12% から 3 instance 平均 66.60%（−1.52pt）、`v2_scan2d` は 78.33 → 75.49（−2.84pt）。比較手法は逆に上がり（ELSHADE-SPACMA 74.58→76.39、LSHADE-SPACMA 72.50→73.06、EBOwithCMAR 63.75→65.69、L-SHADE 60.00→61.11、IPOP 56.88→57.57）、最良ベースラインとの差は MC-ESO が −6.46pt → −9.79pt に開き、`v2_scan2d` は +3.75pt → −0.90pt で首位を失う。順位は i3 で `v2_scan2d` 1 位、i2・i4 は ELSHADE-SPACMA 1 位・`v2_scan2d` 2 位。MC-ESO は i2・i3 で 4 位、i4 で 5 位**（採否の判断はしない）

- 実行: ジョブ記載のコマンドをそのまま 3 回（`--all --dim 5 --max-evals 12500 --n-runs 20 --instance {2,3,4} --seed-offset 7`、7 手法、`--jobs 4`、shard 分割なし、順番に）。wall は i2 11.7 分（21:33:57→21:45:40）／i3 12.2 分（22:02:33→22:14:46）／i4 11.9 分（22:14:54→22:26:50）＝ 計 **35.8 分**。**3 回すべて 24/24 そろった**（各 `summary.csv` 168 行＝7 手法×24、`wilcoxon.csv` 144 行＝6 手法×24、(関数, 手法) の重複・欠落ゼロ、`result.json` は 3 本とも `status=done`）。numpy 2.4.6 / cma 4.5.0、CPU **Intel Xeon @ 2.80GHz 4 コア（family 6 / model 85 / stepping 7、avx512f あり）**、BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。j37 と同じ 2.80GHz のコンテナだが、**instance 1 の対照（`allmethods2_d5`・j36）は 2.10GHz のコンテナで測ったもの**。
- 全体（24 関数平均、SR@1e-2 / 1e-4 / 1e-7 / **1e-10** ％、`evals` ＝ 自分の成功関数の平均 evals、nf ＝ 成功のある関数数）。**i2**: MC-ESO 70.00 / 69.79 / 68.33 / **66.88**（4430.9、22）、`v2_scan2d` 77.50 / 76.04 / 74.17 / **73.12**（4145.3、22）、ELSHADE-SPACMA **76.88**、LSHADE-SPACMA **72.50**、EBOwithCMAR **66.88**、L-SHADE **61.46**、IPOP **58.12**。**i3**: MC-ESO 71.04 / 69.79 / 68.54 / **67.29**（3958.6、22）、`v2_scan2d` 80.21 / 79.58 / 78.12 / **76.88**（3852.0、22）、ELSHADE-SPACMA **75.62**、LSHADE-SPACMA **73.12**、EBOwithCMAR **64.38**、L-SHADE **60.42**、IPOP **57.29**。**i4**: MC-ESO 71.46 / 70.62 / 68.33 / **65.62**（4202.9、23）、`v2_scan2d` 80.42 / 80.00 / 77.71 / **76.46**（3687.0、21）、ELSHADE-SPACMA **76.67**、LSHADE-SPACMA **73.54**、EBOwithCMAR **65.83**、L-SHADE **61.46**、IPOP **57.29**。
- **3 instance 平均**: ELSHADE-SPACMA 80.56 / 79.79 / 76.94 / **76.39**（evals 6487.5、nf 20.7）、`v2_scan2d` 79.38 / 78.54 / 76.67 / **75.49**（3894.8、21.7）、LSHADE-SPACMA 75.56 / 74.93 / 73.82 / **73.06**（4765.8、20.3）、MC-ESO 70.83 / 70.07 / 68.40 / **66.60**（4197.5、22.3）、EBOwithCMAR 81.74 / 79.44 / 70.00 / **65.69**（7377.1、17.0）、L-SHADE 78.19 / 76.32 / 66.94 / **61.11**（7874.7、17.3）、IPOP 65.83 / 64.72 / 62.99 / **57.57**（2990.9、19.0）。`evals_succ_mean` は 2D と違い MC-ESO は最小でない（IPOP 2990.9 が最小、`v2_scan2d` 3894.8 が次）。両方で成功のある関数だけの対応平均（[1b]）でも IPOP だけが MC-ESO より速い（i2 2817.5 対 3808.7、i3 2628.2 対 3605.8、i4 3114.2 対 3697.3）。
- 関数別 SR@1e-10 の変化（instance 1 → i2/i3/i4 の平均）。**MC-ESO は 14 関数が動き**、低下は F15 35→10.00（10/10/10、**−25pt**）・F16 85→70.00・F22 60→45.00・F23 45→30.00（各 −15pt）・F12 100→96.67・F17 85→81.67（各 −3.33）・F03 25→23.33・F19 5→3.33、上昇は F21 45→58.33（+13.33）・F20 5→15.00（+10）・F08 90→98.33（+8.33）・F09 95→100・F04 0→3.33・F18 60→63.33。**`v2_scan2d` は 11 関数**: 低下は **F04 100→70.00（10/100/100、−30pt。instance 2 だけ 10%）**・F17 100→73.33（−26.67）・F16 90→70.00（−20）・F22 100→85.00・F23 45→35.00（各 −10〜15）・F15 10→5.00、上昇は F21 75→96.67（+21.67）・F20 5→11.67・F09 95→100・F08 95→98.33・F18 65→66.67。比較手法側は **F21-Gallagher101 が instance 1 で異常に低かった**（ELSHADE-SPACMA 25→96.67 ＝ +71.67pt、LSHADE-SPACMA 30→93.33 ＝ +63.33pt、IPOP 10→38.33、EBOwithCMAR 80→96.67）ことと、F22 が 3 instance でそろって下がること（ELSHADE −16.67、LSHADE-SPACMA −26.67、IPOP −28.33、EBO −11.67、L-SHADE −10）が目立つ。**F21 の instance 1 は MC-ESO 系に有利な出方だった**。
- Wilcoxon（ref=MC-ESO、p_two<0.05、A12 の向き。better ＝ MC-ESO が有意に良い関数数 / worse ＝ 有意に悪い関数数）。**i2**: `v2_scan2d` 1/4、ELSHADE-SPACMA 5/7、LSHADE-SPACMA 3/8、EBOwithCMAR 13/6、IPOP **11/0**、L-SHADE 9/5。**i3**: `v2_scan2d` 0/4、ELSHADE-SPACMA 4/9、LSHADE-SPACMA 3/9、EBOwithCMAR 8/6、IPOP **10/0**、L-SHADE 7/4。**i4**: `v2_scan2d` 0/4、ELSHADE-SPACMA 3/8、LSHADE-SPACMA 2/8、EBOwithCMAR 8/6、IPOP **10/0**、L-SHADE 8/4。**`v2_scan2d` が MC-ESO より有意に良い 4 関数は 3 instance すべて同じ（F03・F04・F21・F22、A12 は large〜medium）。** ELSHADE-SPACMA / LSHADE-SPACMA に負ける側も F03・F04・F06・F15・F17・F20・F21・F22 が繰り返し出る。A12 の大きさつきの一覧は `detail_i2i3i4.txt` の [3]。
- 参考 [5] CEC 式の平均誤差の順位（低いほど良い）: i2 は EBOwithCMAR 1 位（3.54）・`v2_scan2d` 2 位・MC-ESO 5 位（4.04）、i3 は EBOwithCMAR 1 位（3.62）・ELSHADE-SPACMA 2 位・`v2_scan2d` 4 位・MC-ESO 6 位（4.29）、i4 は `v2_scan2d` 1 位（3.40）・EBOwithCMAR 2 位・MC-ESO 6 位（4.27）。IPOP は 3 instance とも最下位。**2D では MC-ESO が 1〜3 位だったこの指標でも、5D では 5〜6 位。**
- 書かれたとおりに回せなかった点は無い（3 コマンドをそのまま、shard 分割もせずに回した）。環境の注記は前回と同じ 2 点: `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順（空 stub を `PYTHONPATH`）に従い、`python3 -m pip install --ignore-installed blinker ...`（3.11）で正準の numpy 2.4.6 / cma 4.5.0 に固定した。記録は `analysis/single/j38/{summary,wilcoxon}_i{2,3,4}.csv`、`analyze_quick_i{2,3,4}.txt`、`detail_i2i3i4.txt`、生成スクリプト `agg_j38.py`。`analysis` のファイル数 370（上限 約 400）。

### 2026-10-12 【一般化の検証 1/3】調整外 instance の 2D（j37、I=2/3/4） — **MC-ESO の SR@1e-10 は instance 1 の 95.00% から 3 instance 平均 93.61%（−1.39pt）、`v2_scan2d` は 95.21 → 94.31（−0.90pt）。比較手法も同様に動き（EBOwithCMAR 92.92→90.14、IPOP 83.33→79.65、L-SHADE 91.46→90.07、ELSHADE-SPACMA 90.21→91.18、LSHADE-SPACMA 90.21→90.83）、最良ベースラインとの差は +2.08pt → +2.43pt で縮まらない。順位は i2 で MC-ESO 1 位、i3・i4 は `v2_scan2d` 1 位・MC-ESO 2 位、IPOP は 3 instance すべて 7 位**（採否の判断はしない）

- 実行: ジョブ記載のコマンドをそのまま 3 回（`--all --n-runs 20 --max-evals 5000 --instance {2,3,4} --seed-offset 7`、7 手法、`--jobs 4`、shard 分割なし、順番に）。wall は i2 10.0 分（19:42:45→19:52:43）／i3 9.8 分（20:03:59→20:13:47）／i4 9.8 分（20:25:47→20:35:35）＝ 計 **29.6 分**。**3 回すべて 24/24 そろった**（各 `summary.csv` 168 行＝7 手法×24、(関数, 手法) の重複・欠落ゼロ、`result.json` は 3 本とも `status=done`）。numpy 2.4.6 / cma 4.5.0、CPU **Intel Xeon @ 2.80GHz 4 コア（family 6 / model 85 / stepping 7、avx512f あり）**、BLAS は openblas / SkylakeX / num_threads=4（実行時は `OMP_NUM_THREADS=1`）。**j36・`allmethods2`（instance 1 の対照）は 2.10GHz のコンテナで測ったもので、本回は 2.80GHz の別コンテナ**。
- 全体（24 関数平均、SR@1e-2 / 1e-4 / 1e-7 / **1e-10** ％、`evals` ＝ 自分の成功関数の平均 evals、nf ＝ 成功のある関数数）。**i2**: MC-ESO 94.79 / 94.79 / 93.12 / **92.29**（905.4、24）、`v2_scan2d` 95.42 / 95.21 / 93.12 / **92.08**（881.0、24）、LSHADE-SPACMA **90.42**、ELSHADE-SPACMA **90.00**、L-SHADE **89.79**、EBOwithCMAR **88.12**、IPOP **82.29**。**i3**: MC-ESO 95.83 / 95.62 / 95.42 / **94.79**（776.0、24）、`v2_scan2d` 96.46 / 96.46 / 95.83 / **95.62**（796.1、24）、ELSHADE-SPACMA ＝ LSHADE-SPACMA ＝ EBOwithCMAR **92.29**、L-SHADE **90.42**、IPOP **78.75**。**i4**: MC-ESO 94.38 / 94.17 / 93.96 / **93.75**（786.0、24）、`v2_scan2d` 95.42 / 95.42 / 95.42 / **95.21**（823.5、24）、ELSHADE-SPACMA **91.25**、EBOwithCMAR ＝ L-SHADE **90.00**、LSHADE-SPACMA **89.79**、IPOP **77.92**。
- **3 instance 平均**: MC-ESO 95.00 / 94.86 / 94.17 / **93.61**（evals 822.5、nf 24.0）、`v2_scan2d` 95.76 / 95.69 / 94.79 / **94.31**（833.5、24.0）、ELSHADE-SPACMA 93.47 / 92.85 / 91.81 / **91.18**（1616.7、23.7）、LSHADE-SPACMA 93.40 / 92.64 / 91.67 / **90.83**（1259.6、24.0）、EBOwithCMAR 96.53 / 95.69 / 92.99 / **90.14**（2014.4、23.3）、L-SHADE 93.06 / 92.57 / 91.25 / **90.07**（2144.0、23.7）、IPOP 86.74 / 85.07 / 84.03 / **79.65**（943.1、22.7）。`evals_succ_mean` は MC-ESO が 7 手法で最小（IPOP 943.1 が次）。
- 関数別 SR@1e-10 の変化（instance 1 → i2/i3/i4 の平均。他 15 関数は 3 instance とも 100% で不変）。**MC-ESO**: F12 100→81.67（50/95/100）・F23 100→81.67（70/90/85）・F24 15→10.00（15/10/5）・F16 100→98.33・F19 100→98.33 が低下、F04 85→86.67（90/95/75）・F17 95→98.33・F18 95→98.33・F20 90→93.33 が上昇。**`v2_scan2d`**: F12 100→81.67・F23 95→86.67・F17 100→95.00・F20 95→93.33・F16 100→98.33・F19 100→98.33・F24 15→13.33 が低下、F18 80→96.67 が上昇。**F12-BentCigar の i2 の 50%（MC-ESO・`v2_scan2d` とも）が最大の落ち込み**で、i3 95 / i4 100 なので instance 2 固有。
- Wilcoxon（ref=MC-ESO、p_two<0.05、A12 の向き。better ＝ MC-ESO が有意に良い関数数 / worse ＝ 有意に悪い関数数）。**i2**: `v2_scan2d` 0/1（F23）、ELSHADE-SPACMA 3/3（better F19・F23・F24、worse F12・F17・F18）、LSHADE-SPACMA 1/1（F24 / F12）、EBOwithCMAR 1/0（F18）、IPOP 9/1（better F03・F04・F06・F13・F14・F17・F18・F19・F20、worse F12）、L-SHADE 3/0。**i3**: `v2_scan2d` 0/0、ELSHADE-SPACMA 1/1、LSHADE-SPACMA 1/1、EBOwithCMAR 3/1、IPOP **13/0**、L-SHADE 2/1。**i4**: `v2_scan2d` 0/1（F04）、ELSHADE-SPACMA 1/2、LSHADE-SPACMA 1/1、EBOwithCMAR 2/1、IPOP **13/0**、L-SHADE 1/1。A12 の大きさつきの一覧は `detail_i2i3i4.txt` の [3]。
- 参考 [5] CEC 式の平均誤差の順位（低いほど良い）: i2 は MC-ESO 1 位（3.54）・EBOwithCMAR 2 位・`v2_scan2d` 3 位、i3 は EBOwithCMAR 1 位（3.56）・ELSHADE-SPACMA 2 位・MC-ESO 3 位（3.83）・`v2_scan2d` 4 位、i4 は EBOwithCMAR 1 位（3.38）・`v2_scan2d` 2 位・MC-ESO 3 位（3.71）。IPOP は 3 instance とも最下位。
- 書かれたとおりに回せなかった点は無い（3 コマンドをそのまま、shard 分割もせずに回した）。環境の注記（前回と同じ 2 点）: `pip install -r requirements.txt` は `pynmmso` のビルドで落ちるので findings.md の手順（空 stub を `PYTHONPATH`）に従った。`pip` は python3.13 を指すので `python3 -m pip`（3.11）で入れ、正準の numpy 2.4.6 / cma 4.5.0 に固定した（`blinker` の debian 由来の RECORD 欠落で 1 度落ちたので `--ignore-installed blinker` で入れ直した）。記録は `analysis/single/j37/{summary,wilcoxon}_i{2,3,4}.csv`、`analyze_quick_i{2,3,4}.txt`、`detail_i2i3i4.txt`、生成スクリプト `agg_j37.py`。`analysis` のファイル数 359（上限 約 400）。

古い作業ログ（2026-10-06〜12 の改善案・全手法比較・新既定値の ablation・`v2_*` の候補の正準測定・CEC2022 の 25000 評価・main とのマージ衝突）は [history.md](history.md) の該当日の節と [findings.md](findings.md) 7 節に要約してある。全文は git の履歴（コミット `8634ad3`）にある。
