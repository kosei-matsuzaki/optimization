# analysis/

自動サイクルが書き出す測定結果。**テーマごとにサブディレクトリを切り、トップレベルには置かない。**

| ディレクトリ | テーマ | 状態 |
|---|---|---|
| `single/` | MC-ESO の単一解性能（現行テーマ）。測定ジョブは `j<N>/`、旧サイクルの問いは `e<NNN>/`、ローカル測定は `local_*/` | live |
| `hm/` | 多峰度の高い問題での peak ratio（多解路線、**一時停止中**）。残っているのは `e71/`（文献取得物 5 本）だけ | 一時停止。要約は [docs/archive/multisolution.md](../docs/archive/multisolution.md) |
| `audit/` | 受容集合と多様解の監査（2026-09-01〜09-02） | 終了。要約は [docs/archive/multisolution.md](../docs/archive/multisolution.md)（全文は git タグ `archive/multisolution-2026-09-29`） |

規約は [CLAUDE.md](../CLAUDE.md#ファイルを増やすときの規約) にある。要点:

- 行単位の生 CSV は `.csv.gz`。生のまま置いてよいのは数百行までの集計結果
- **数値は必ず `docs/` に書き出してから commit する。**ここは再解析用の控えであって結論の置き場ではない
- 路線を畳んだら、その生データも消す（git 履歴から取り出せる）

監査テーマの生データ（約 160 万行）は 2026-09-03 に削除済み。再現が要るなら
`scripts/audit/` のスクリプトで作り直せる。

**2026-10-06: `mmo2024/`（一時停止中の多解路線、354 ファイル）は作業ツリーから外した。** 中身はタグ `archive/multisolution-2026-09-29` と同一（外す前に差分ゼロを確認）。docs に残る `analysis/mmo2024/...` のパスはタグの中のパスとして読む。取り出すには `git checkout archive/multisolution-2026-09-29 -- analysis/mmo2024`。

**2026-10-16: `single/` の古いジョブ（`e177`〜`e183`、`j11`〜`j17`、`j24`・`j25`・`j28`、`j30`〜`j34`、`local_*` の 29 ディレクトリ、214 ファイル）を作業ツリーから外した。** 数値は `docs/findings.md`・`docs/history.md` に書いてあり、docs の該当パスには復元用の commit を添えた。まとめて取り出すには `git checkout f378fba -- analysis/single/<dir>`。

**2026-10-18: `single/` の `j36`〜`j43`・`j46`・`j48` の 10 ディレクトリ（152 ファイル）を作業ツリーから外した（517→365 ファイル）。** 数値は `docs/findings.md`・`docs/history.md`・`docs/research_loop.md` に書いてあり、docs の該当パスには復元用の commit を添えた。まとめて取り出すには `git checkout 95cf9b7 -- analysis/single/<dir>`。`j49/` の per-run stats は j52 のパネルとして、`j47/stats/` は j52 の 20D のパネルとして、`j50/` の per-run stats は v4_pace の再解析用に残してある。

**2026-10-30: `j49/` の `ctl_plainwave` の列は素の CMA の波ではない。** この腕は 2026-10-08 以前の既定で埋められて走り、波が一度も打たれていない（v0 の MC-ESO と同じ）。再解析に使わない。測り直しは j61（[docs/history.md](../docs/history.md) の 2026-10-17〜18 の訂正）。2026-10-30 から j49 / j47 / j57 のパネルの `MC-ESO` 列は旧既定（現 `MC-ESO-v3`）の値。
