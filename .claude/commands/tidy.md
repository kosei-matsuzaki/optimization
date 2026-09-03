---
description: リポジトリの整理（未整理のファイル・閉じた路線のデータ・ディレクトリ構成を点検する）
---

Launch the `curator` subagent to check the repository against the project's retention
rules and tidy what has accumulated.

$ARGUMENTS

Default to reporting and proposing rather than deleting. Apply changes when the user
asked for a tidy-up in their message, or when the change is a move rather than a
deletion. Anything that removes data the user has not agreed to lose comes back as a
proposal with the sizes, not as a completed action.

Relay the table it produces, and say explicitly what was left alone and why.
