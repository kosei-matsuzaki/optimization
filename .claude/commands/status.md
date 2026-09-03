---
description: 研究の現況を表示する（status.md の要点 ＋ 問いのキュー ＋ 直近サイクル）
---

Run `./run.sh loop --cycles 8` and report what it shows, in Japanese.

Lead with anything under 「ユーザーの判断を仰ぐこと」 in `docs/status.md` — that is
written for the user to act on and is the reason they asked. If it is empty, say so
in one line and move on.

Then give them: where the research stands against its goal, what the recent cycles
concluded, and whether the queue looks aimed at the goal. Keep it short; the full
documents are one command away and they can read them.

If the branch is behind, pull first. If `research-loop` is more than about a day
ahead of `main`, mention it — the project's rule is to merge roughly daily so the
diff stays reviewable.
