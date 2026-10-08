"""Merge quick-run shards into one run dir: <out>/dim<D>/{summary,wilcoxon}.csv

Usage: merge_shards.py <dim> <out_dir> <shard_dir> [<shard_dir> ...]
Concatenates with a single header, then reports coverage
(unique functions, (function, method) duplicates / gaps).
"""
import csv, os, sys

dim = int(sys.argv[1])
out = sys.argv[2]
shards = sys.argv[3:]
os.makedirs(os.path.join(out, f"dim{dim}"), exist_ok=True)

for fname, keycols in (("summary.csv", ("function", "method")),
                       ("wilcoxon.csv", ("function", "method"))):
    header, rows = None, []
    for sd in shards:
        p = os.path.join(sd, f"dim{dim}", fname)
        if not os.path.exists(p):
            print(f"MISSING {p}")
            continue
        with open(p, newline="") as f:
            r = csv.reader(f)
            h = next(r)
            if header is None:
                header = h
            elif h != header:
                sys.exit(f"header mismatch in {p}")
            rows.extend(list(r))
    dst = os.path.join(out, f"dim{dim}", fname)
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    idx = {c: header.index(c) for c in keycols}
    funcs = sorted({r[idx["function"]] for r in rows})
    methods = sorted({r[idx["method"]] for r in rows})
    keys = [(r[idx["function"]], r[idx["method"]]) for r in rows]
    dup = len(keys) - len(set(keys))
    gaps = [(fn, m) for fn in funcs for m in methods if (fn, m) not in set(keys)]
    print(f"{fname}: {len(rows)} rows, {len(funcs)} functions, {len(methods)} methods "
          f"({', '.join(methods)}), duplicates={dup}, gaps={len(gaps)}")
    if gaps:
        print("  gap keys:", gaps[:20])
    missing = [f"F{i:02d}" for i in range(1, 25)
               if not any(fn.startswith(f"F{i:02d}") for fn in funcs)]
    if missing:
        print("  functions absent from BBOB-24:", missing)
    else:
        print("  all 24 BBOB functions present (24/24)")
