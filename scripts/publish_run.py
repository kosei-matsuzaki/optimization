"""Copy a run's numbers from results/ (this PC) to runs/ (shared through git).

    python scripts/publish_run.py results/<run> [--name NEW_NAME] [--note TEXT] [--force]
    ./run.sh publish results/<run> ...                      # same thing

results/ is git-ignored, so each PC only sees its own runs. runs/ is tracked:
publish a run here, commit, push, and every PC that pulls sees it in the
results UI and can re-aggregate it with scripts/analyze_quick.py-style tools.

Only numbers travel. Figures stay on the PC that made them (they are hundreds
of MB per run and can be re-rendered). Every CSV is gzipped, following the
repository rule for row-level data; result.json stays plain and gains
``published_at`` / ``published_from`` / ``note``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def _gzip_copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    # mtime=0 keeps the archive byte-identical across publishes of the same file
    with open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                               filename="") as gz:
        gz.write(data)
    return dst.stat().st_size


def publish(src: Path, name: str | None, note: str | None, force: bool) -> Path:
    src = src.resolve()
    if not src.is_dir():
        raise SystemExit(f"not a directory: {src}")
    meta_path = src / "result.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("status") not in (None, "done"):
        raise SystemExit(f"{src.name}: status is {meta.get('status')!r}, not 'done' — "
                         "publish finished runs only")
    dims = sorted(d for d in src.iterdir() if d.is_dir() and d.name.startswith("dim"))
    if not dims or not any((d / "summary.csv").exists() for d in dims):
        raise SystemExit(f"{src.name}: no dim*/summary.csv to publish")

    dst = RUNS / (name or src.name)
    if dst.exists():
        if not force:
            raise SystemExit(f"{dst.relative_to(ROOT)} already exists (use --force to replace)")
        shutil.rmtree(dst)

    total = 0
    for d in dims:
        for csv_path in sorted(d.rglob("*.csv")):
            rel = csv_path.relative_to(src)
            total += _gzip_copy(csv_path, dst / rel.with_name(rel.name + ".gz"))
    meta.update({
        "published_at": datetime.now().isoformat(timespec="seconds"),
        "published_from": platform.node(),
        "source_dir": src.name,
    })
    if note:
        meta["note"] = note
    (dst / "result.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(f"published {src.name} -> {dst.relative_to(ROOT)}  ({total / 1024:.0f} KB of CSV.gz)")
    print("next: git add runs/ && git commit   (push when you want other PCs to see it)")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path, help="results/<run> directories")
    ap.add_argument("--name", help="name under runs/ (single run only; default: same name)")
    ap.add_argument("--note", help="free text stored in result.json")
    ap.add_argument("--force", action="store_true", help="replace an existing runs/<name>")
    args = ap.parse_args()
    if args.name and len(args.run_dirs) > 1:
        raise SystemExit("--name works with a single run")
    for d in args.run_dirs:
        publish(d, args.name, args.note, args.force)


if __name__ == "__main__":
    main()
