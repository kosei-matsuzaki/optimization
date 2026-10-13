"""Merge two quick runs of the same suite / dimension / budget / seeds into one.

Used to add methods to a finished comparison without re-running it: the base
run keeps the methods named in --keep, the other run contributes --take. Both
runs must contain the reference (MC-ESO) with identical per-run results — the
script checks that, since the Wilcoxon rows of the added methods were computed
against the added run's reference.

    python scripts/single/merge_runs.py --dim 10 \\
        --base runs/<34-method run> --add results/<new-methods run> \\
        --keep MC-ESO,MC-ESO-v2,EBOwithCMAR,... --take ARRDE,RCMAES,... \\
        --out results/<YYYYMMDD_HHMMSS_label_quick> --note "..."

Writes plain CSVs (as a results/ run); `./run.sh publish` gzips them for runs/.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path

REF = "MC-ESO"


def _open(path: Path):
    if path.exists():
        return open(path, newline="", encoding="utf-8")
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        return io.TextIOWrapper(gzip.open(gz), newline="", encoding="utf-8")
    return None


def _rows(path: Path) -> tuple[list[str], list[dict]]:
    f = _open(path)
    if f is None:
        return [], []
    with f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _names(d: Path) -> list[str]:
    out = set()
    if d.is_dir():
        for p in d.iterdir():
            out.add(p.name[:-3] if p.name.endswith(".gz") else p.name)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--add", type=Path, required=True)
    ap.add_argument("--keep", required=True, help="comma list of methods taken from --base")
    ap.add_argument("--take", required=True, help="comma list of methods taken from --add")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    keep, take = a.keep.split(","), a.take.split(",")
    sub = f"dim{a.dim}"
    bdir, adir, odir = a.base / sub, a.add / sub, a.out / sub

    # The reference must agree run by run, or the added Wilcoxon rows are
    # against a different MC-ESO.
    bad = 0
    for name in _names(bdir / "stats"):
        _, rb = _rows(bdir / "stats" / name)
        _, ra = _rows(adir / "stats" / name)
        fb = {r["seed"]: r["best_f"] for r in rb if r["method"] == REF}
        fa = {r["seed"]: r["best_f"] for r in ra if r["method"] == REF}
        bad += sum(fb[s] != fa.get(s) for s in fb)
    if bad:
        raise SystemExit(f"{REF} differs between the runs in {bad} runs; not merging")

    for name in ("summary.csv", "wilcoxon.csv"):
        fb, rb = _rows(bdir / name)
        _, ra = _rows(adir / name)
        rows = [r for r in rb if r["method"] in keep] + [r for r in ra if r["method"] in take]
        order = {f: i for i, f in enumerate(dict.fromkeys(r["function"] for r in rb))}
        rows.sort(key=lambda r: order.get(r["function"], len(order)))
        _write(odir / name, fb, rows)
    for name in _names(bdir / "stats"):
        fb, rb = _rows(bdir / "stats" / name)
        _, ra = _rows(adir / "stats" / name)
        _write(odir / "stats" / name,
               fb, [r for r in rb if r["method"] in keep] + [r for r in ra if r["method"] in take])
    fm, rm = _rows(bdir / "mceso_runs.csv")
    if fm:
        _write(odir / "mceso_runs.csv", fm, [r for r in rm if r["method"] in keep])
    for name in _names(bdir / "curves"):
        def load(d: Path):
            p = d / "curves" / name
            if not p.exists():
                p = p.with_name(p.name + ".gz")
            if not p.exists():
                return None
            return json.loads(gzip.open(p).read() if p.suffix == ".gz" else p.read_text())
        cb, ca = load(bdir), load(adir)
        if cb is None:
            continue
        cb["methods"] = {m: v for m, v in cb["methods"].items() if m in keep}
        if ca is not None and ca.get("evals") == cb.get("evals"):
            cb["methods"].update({m: v for m, v in ca["methods"].items() if m in take})
        out = odir / "curves" / (name if name.endswith(".gz") else name + ".gz")
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out, "wt", encoding="utf-8") as f:
            json.dump(cb, f)

    meta = json.loads((a.base / "result.json").read_text(encoding="utf-8"))
    meta["methods"] = ",".join(keep + take)
    meta["merged_from"] = [a.base.name, a.add.name]
    meta["status"] = "done"
    if a.note:
        meta["note"] = a.note
    for k in ("published_at", "published_from", "source_dir"):
        meta.pop(k, None)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "result.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    env = a.base / "env.json"
    if env.exists():
        (a.out / "env.json").write_text(env.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"merged -> {a.out}  ({len(keep)} + {len(take)} methods)")


if __name__ == "__main__":
    main()
