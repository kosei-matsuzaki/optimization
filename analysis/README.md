# analysis/

自動サイクルが書き出す測定結果。**テーマごとにサブディレクトリを切り、トップレベルには置かない。**

| ディレクトリ | テーマ | 状態 |
|---|---|---|
| `hm/` | 多峰度の高い問題での peak ratio（現行） | live |
| `cal/` | 同上、hunt 数の較正 | live |
| `audit/` | 受容集合と多様解の監査（2026-09-01〜09-02） | 終了。要約は [docs/archive/multisolution.md](../docs/archive/multisolution.md)（全文は git タグ `archive/multisolution-2026-09-29`） |

規約は [CLAUDE.md](../CLAUDE.md#ファイルを増やすときの規約) にある。要点:

- 行単位の生 CSV は `.csv.gz`。生のまま置いてよいのは数百行までの集計結果
- **数値は必ず `docs/` に書き出してから commit する。**ここは再解析用の控えであって結論の置き場ではない
- 路線を畳んだら、その生データも消す（git 履歴から取り出せる）

監査テーマの生データ（約 160 万行）は 2026-09-03 に削除済み。再現が要るなら
`scripts/audit/` のスクリプトで作り直せる。

**2026-10-06: `mmo2024/`（一時停止中の多解路線、354 ファイル）は作業ツリーから外した。** 中身はタグ `archive/multisolution-2026-09-29` と同一（外す前に差分ゼロを確認）。docs に残る `analysis/mmo2024/...` のパスはタグの中のパスとして読む。取り出すには `git checkout archive/multisolution-2026-09-29 -- analysis/mmo2024`。
