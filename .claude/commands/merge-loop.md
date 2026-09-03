---
description: research-loop を main にマージする（前に整合性を点検する）
---

Merge `research-loop` into `main`, after checking it is safe to.

Before merging:

1. `git fetch origin` and show how far apart the branches are — commits, files, and
   insertions split by directory. If the added lines are dominated by data files rather
   than code and documents, stop and say so: the project's rule is that raw result CSVs
   are gzipped and folded routes take their data with them, and merging a diff nobody
   can read defeats the point of merging often.
2. Confirm MC-ESO is untouched: `git diff origin/main...origin/research-loop -- core/optimizers/mceso.py`
   must be empty, or the change must be one the user has approved.
3. Say what the merge brings in — the research conclusions, not the file list.

Then merge and push `main`. Report the result and the new diff size.

Do not force, do not rebase `main`, and stop and ask if the merge is not a
fast-forward.
