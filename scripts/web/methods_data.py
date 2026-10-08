"""Build web/static/methods_data.json — the numbers shown on the /methods page.

The page is a translation of quick results, not a source: every value it shows
comes from the summary.csv / wilcoxon.csv of the quick runs named here, with the
same aggregation as scripts/analyze_quick.py (mean over functions; evals =
mean of evals_succ_mean over functions with at least one success; Wilcoxon
win/loss = functions where the reference is significantly better / worse,
p_two < 0.05, direction by A12).

    python scripts/web/methods_data.py results/<d2 run> results/<d5 run> results/<d10 run>

Run directories may be local (results/, plain CSV) or shared (runs/, CSV.gz).
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "static" / "methods_data.json"
SR_KEYS = ("sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10")
REF = "MC-ESO"


def _pct(s: str) -> float:
    s = (s or "").strip().rstrip("%")
    try:
        return float(s) / 100.0
    except ValueError:
        return float("nan")


def _num(s: str) -> float:
    s = (s or "").strip()
    if s == "inf":
        return float("inf")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
    return sum(vals) / len(vals) if vals else None


def _rows(path: Path) -> list[dict]:
    gz = path.with_name(path.name + ".gz")
    f = (open(path, newline="") if path.exists()
         else io.TextIOWrapper(gzip.open(gz), newline=""))
    with f:
        return list(csv.DictReader(f))


def one_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    dim = int(meta["dim"])
    rows = _rows(run_dir / f"dim{dim}" / "summary.csv")
    wil = _rows(run_dir / f"dim{dim}" / "wilcoxon.csv")

    funcs = sorted({r["function"] for r in rows})
    methods: list[str] = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    better = {m: 0 for m in methods}
    worse = {m: 0 for m in methods}
    for w in wil:
        p, a12 = _num(w["p_value_two_sided"]), _num(w["a12"])
        if not math.isnan(p) and p < 0.05:
            if a12 > 0.5:
                better[w["method"]] += 1
            elif a12 < 0.5:
                worse[w["method"]] += 1

    overall = {}
    for m in methods:
        mr = [r for r in rows if r["method"] == m]
        overall[m] = {k: _mean([_pct(r[k]) for r in mr]) for k in SR_KEYS}
        overall[m]["evals"] = _mean([_num(r["evals_succ_mean"]) for r in mr])
        if m != REF:
            overall[m]["wins"], overall[m]["losses"] = better[m], worse[m]

    per_func = {fn: {r["method"]: _pct(r["sr_1e-10"]) for r in rows if r["function"] == fn}
                for fn in funcs}
    return {
        "run": run_dir.name, "dim": dim, "commit": meta.get("commit"),
        "n_runs": meta.get("n_runs"), "max_evals": meta.get("max_evals"),
        "created_at": meta.get("created_at"),
        "methods": methods, "funcs": funcs,
        "overall": overall, "sr10_by_func": per_func,
    }


def main() -> None:
    runs = [one_run(Path(p)) for p in sys.argv[1:]]
    if not runs:
        raise SystemExit(__doc__)
    data = {"reference": REF, "runs": sorted(runs, key=lambda r: r["dim"])}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({', '.join(r['run'] for r in data['runs'])})")


if __name__ == "__main__":
    main()
