e86 — 問い 1: N16-CF3-5D / N18-CF3-10D で採用候補の合成腕は「被覆」を動かすか

背景（キューの問い 1 の本文どおり）
  status.md の算術で案 (A) の深さ側は閉じている。同一 run では PR@eps は eps に対して
  単調非増加なので PR@1e-1 はその run の PR@1e-5 の上限で、MC-ESO の上限は F14-F20 の
  7 関数すべてで公表最良を下回る。案 (A) が生き残る条件は「腕が被覆そのものを上げる」1 点だけ。
  よって測るのは判定水準ではなく PR@1e-1 / PRtrue@1e-1。

設計（数値を見る前に固定した。driver の docstring が事前登録）
- 腕 base: 出荷クラスそのもの（無効化した変種ではない）。
  腕 comp: 採用手順の (a)+(c)+(d) ＝ BBOB gate を通っており関数依存でない候補すべて。
      (a) rel_level=1e-5（c=1.0、その46 の角）＋ fis_floor=1e-12（その44 の採用形）
      (c) exhausted_sigma_tol=1.0（_sig10）、sigma_floor_ratio=1e-8（_fl08）
      (d) sol_trim_mode="rho"（soltrim_rho）
      (f) commit_place は事前登録では除外（その41・その42 で被覆律速専用と閉じており、
          BBOB gate を一度も通していないので「合成腕」の定義に入らない）。
- seed 0-11（optimiser seed = i*100、e73/e77/e85 と同一規約）、正規予算 4e5。core/ は不変。
- 採点は同一 run を 2 通り: 公式採点器（core.runner._niching_counts）と、
  報告集合の各点を最近傍の真の最適に帰属させた PRtrue（その76 の ρ アーティファクト対策。
  10D では盆地の 1e-1 等高面が ρ=0.01 より広く、公式 PR@1e-1 は同一盆地内の点を別 niche に数える）。
- base の新規 run は N16 のみ（N18 の base は e77 の保存 CSV を seed 番号で対にした ＝ 追加 run ゼロ）。
  両関数とも base の公式 PR を e85 の 6 手法表の MC-ESO 行と全 5 水準で照合してから読む（analyze.py が印字）。

事前登録した棄却条件
  comp の PRtrue@1e-1 が N16 で 0.677 を、N18 で 0.667 を超えない
  ＝ 案 (A) は 7 関数すべてで閉じる。

事後に足した腕（事前登録ではない。ログにもそう書いた）
  腕 compf: comp ＋ 採用手順 (f)（commit_sigma_mode="place", ratio=0.1）。
  理由: comp は深さ側と報告側のつまみだけで、再起動の「配置」を一切動かさない。
  配置を動かす出荷済みのつまみは commit_place だけで、それはその62・その63 で
  N09-Vincent3D の被覆天井を上げた実績がある。これを入れないと否定的結果は
  「深さ・報告のつまみは被覆を動かさない」（既知）にしかならず、
  入れて初めて「リポジトリにある全つまみで動かない」になる。

結果（数値は acceptance_topology.md のその86 節が正）
  1. 事前登録の棄却条件は両関数で成立。comp の PRtrue@1e-1 は N16 で 12 seed 全部 0.6667
     （base と 0/12/0）、N18 で 0.6389（base 0.6667 に対し 0/10/2 ＝ 2 seed で下がる）。
     どの seed でも公表最良（N16 0.677 / N18 0.667）を上回らない。
  2. これは未発火ではない: best_f は N18 で 11/12 seed 動く（p=0.001）。
     腕は軌跡を変えており、変えたうえで被覆が 1 ビットも動かない。
  3. compf（配置つまみ入り）も同じ。→ 本文参照。
  4. N16 は被覆律速: PR@1e-5 0.653 が PR@1e-1 0.667 のすぐ下 ＝ 入った盆地はほぼ全部
     1e-5 まで降りており、残差は 100% 被覆側。深さのつまみが効く余地が構造的に無い。

ファイル
  composite_cov.py : driver（事前登録の docstring つき）
  analyze.py       : 対応のある読み（Wilcoxon / w-t-l / A12、e77 と同じ統計）
  runs.csv.gz      : 全 run の 1 行 1 run 集計（func, arm, seed, best_f, n_rep, pr_*, prtrue_*）
  analysis_out.txt : analyze.py の全出力
