---
description: docs とコードの一致を検証して直す（パラメータ・パス・指標・web の記述）
---

Launch the `docs-keeper` subagent to verify the documentation against the code.

$ARGUMENTS

If arguments name a document or an area, scope it to that. Otherwise check all of
`docs/` plus `README.md` and the results UI under `web/`.

Relay what it found and what it changed. Numbers whose source could not be traced are
the important part of the report — surface those rather than burying them under the
list of fixes.
