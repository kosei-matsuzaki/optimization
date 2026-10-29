"""Build web/static/methods_params.json — the parameter list on the /methods page.

The page is a translation, not a source: the meaning of each parameter comes
from the parameter table in docs/mceso.md, and the default value is read from
the code (`MultiChannelEpidemicOptimizer.__init__`). The script fails if a
constructor parameter is missing from the table or the table names one the
code no longer has, so the two cannot drift apart silently.

    python scripts/web/methods_params.py
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "web" / "static" / "methods_params.json"

# Mechanism groups, in page order: (label, predicate on the parameter name).
# v5 arms (2026-10-30, all off by default); kept in their own group.
V5 = ("pace_route", "pace_max", "herd_child", "keepair_slow", "chain_route")

GROUPS = [
    ("集団と選択", lambda n: n.startswith(("n_pop", "pop_")) or n in ("kill_fraction", "softmax_beta", "select_rank", "init_lhs")),
    ("伝染経路（接触・飛沫・空気）", lambda n: n.startswith(("air_", "h2h_", "droplet_")) or n in ("cc_air_ratio", "cc_h2h_ratio", "host_sigma_min_scale", "close_mirror")),
    ("チャネルルーター", lambda n: n.startswith(("channel_", "cond_", "align_", "close_", "route_", "router_"))),
    ("集団共分散の下限", lambda n: n.startswith(("empirical_cov", "cov_"))),
    ("学習共分散（3 次元以上）", lambda n: n.startswith("cc_")),
    ("歩幅 σ", lambda n: n.startswith(("sigma", "precision_", "hd_sigma")) or n == "log_slope_threshold"),
    ("スピルオーバー・乗換え・回復", lambda n: n not in V5 and n.startswith(("restart_", "basin_", "ir_", "spill_", "switch_", "pace_", "recover_", "chain_"))),
    ("予報の子（3 次元以上）", lambda n: n.startswith("forecast")),
    ("組換え波（3 次元以上）", lambda n: n.startswith(("wave_", "hd_wave")) or n == "handover_frac"),
    ("再集合の掃引", lambda n: n.startswith("reassort_")),
    ("変異株の競合・試し打ち（不採用）", lambda n: n.startswith(("strain_", "probe_"))),
    ("v5 の腕（実験中）", lambda n: n in V5),
    ("系統と逐次 niching", lambda n: n.startswith(("n_elite", "niche_", "exhausted_", "hunt_", "solution_"))),
]
STATUS = [("不採用", "rejected"), ("診断用", "diagnostic"), ("実験中", "experimental")]


def _cells(line: str) -> list[str]:
    """Split a markdown table row on unescaped pipes."""
    parts = re.split(r"(?<!\\)\|", line.strip())
    return [p.strip().replace("\\|", "|") for p in parts[1:-1]]


def _md_inline(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    return s


def main() -> None:
    from core.optimizers.mceso import MultiChannelEpidemicOptimizer as M
    sig = inspect.signature(M.__init__).parameters
    code = {k: p.default for k, p in sig.items() if k not in ("self", "benchmark", "seed")}

    doc = (ROOT / "docs" / "mceso.md").read_text(encoding="utf-8")
    table = doc[doc.index("## パラメータ一覧"):doc.index("## MC-ESO の新規性")]
    rows = []
    for line in table.splitlines():
        if not line.startswith("| `"):
            continue
        name_cell, default_cell, meaning = _cells(line)[:3]
        names = re.findall(r"`([A-Za-z_0-9]+)`", name_cell)
        status = next((tag for word, tag in STATUS if word in meaning[:40] or word in default_cell), "")
        rows.append({"names": names, "doc_default": _md_inline(default_cell),
                     "code_default": [repr(code[n]) if n in code else "?" for n in names],
                     "meaning": _md_inline(meaning), "status": status})

    documented = {n for r in rows for n in r["names"]}
    missing = sorted(set(code) - documented)
    stale = sorted(documented - set(code))
    if missing or stale:
        raise SystemExit(f"docs/mceso.md parameter table out of sync — missing {missing}, stale {stale}")

    groups = []
    for label, pred in GROUPS:
        g = [r for r in rows if pred(r["names"][0])]
        if g:
            groups.append({"label": label, "rows": g})
    placed = {id(r) for g in groups for r in g["rows"]}
    rest = [r for r in rows if id(r) not in placed]
    if rest:
        groups.append({"label": "その他", "rows": rest})
    # A row may match two groups by prefix; keep it in the first only.
    seen = set()
    for g in groups:
        g["rows"] = [r for r in g["rows"] if not (id(r) in seen or seen.add(id(r)))]

    OUT.write_text(json.dumps({"n_params": len(code), "groups": groups}, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(code)} parameters in {len(groups)} groups)")


if __name__ == "__main__":
    main()
