#!/usr/bin/env python3
"""Entry 134 (queue 1) -- re-derive every number the log entry states, from the
files vendored in this repository.  No optimisation is run and nothing is
fetched: this reads `external/mmo2024/docs/` and `analysis/mmo2024/e92/refs/`.

    python3 analysis/mmo2024/e134/check_claims.py

(a) de Nobel+ 2024 (arXiv 2405.01226), the RR-CMA-ES paper: does it contain the
    statement the queue pre-registered ("comparisons among MMO methods, and
    against simple baselines, are rarely performed")?
(b) Is there a published PER-PROBLEM value for the GECCO'2024 competition, and
    is the one per-problem table that exists (N-DAM-CMA-ES, Table 1) the
    competition's PR?
"""
import gzip
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]


def read(p):
    p = ROOT / p
    if not p.exists():
        sys.exit(f"missing: {p}")
    return gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()


# ---------------------------------------------------------------- (a)
den = read("analysis/mmo2024/e92/refs/rrcmaes_denobel2024.txt.gz")
low = den.lower()
assert "arxiv:2405.01226" in low, "not the expected paper"
assert "avoiding redundant restarts" in low, "title phrase absent"

PROBE = ["baseline", "rarely", "seldom", "hardly ever", "not been compared",
         "lack of comparison", "no comparison", "little attention", "scarce",
         "underexplored", "under-explored", "few comparison"]
hits = {w: low.count(w) for w in PROBE}
print(f"(a) de Nobel+ 2024, {len(den)} chars extracted")
print("    probe word counts:", hits)
print(f"    PRE-REGISTERED CLAIM PRESENT: {any(hits.values())}  "
      f"-> refutation condition {'did NOT fire' if any(hits.values()) else 'FIRED'}")

flat = re.sub(r"\s+", " ", den)
for key in ("We compare a naive restart strategy",
            "Our proposed repelling-based restart method"):
    i = flat.find(key)
    assert i >= 0, f"quote not found: {key}"
    print(f"    quote OK: {flat[i:i + 96]}...")

# ---------------------------------------------------------------- (b)
deck = read("external/mmo2024/docs/gecco2024_results_deck.txt")
setup = read("external/mmo2024/docs/competition_setup_TR2024001.txt")

# the deck's only numeric block: 4 rows MPR, 4 rows mean-F1, 1 row Score,
# each "D RR-CMA-ES TRDE-LR Niching-CMA-ES"
rows = [[float(x) for x in m] for m in
        re.findall(r"^(?:\d+ )?(0\.\d+|\d\.\d+) (0\.\d+) (0\.\d+)\s*$", deck, re.M)]
assert len(rows) == 9, f"deck parse changed: {len(rows)} rows"
mpr, f1, overall = rows[:4], rows[4:8], rows[8]
dims = [2, 5, 10, 20]
names = ["RR-CMA-ES", "TRDE-LR", "N-DAM-CMA-ES"]
print(f"\n(b) GECCO'2024 deck: the only numeric table is {len(rows)} rows "
      f"(4 MPR + 4 mean-F1 + 1 overall), indexed by D, not by problem")
for j, n in enumerate(names):
    per_d = {d: round((mpr[i][j] + f1[i][j]) / 2, 4) for i, d in enumerate(dims)}
    print(f"    {n:14s} Score by D: {per_d}   overall {overall[j]}")

# how many distinct global minima a method must report to score the published MPR
K = {**{p: 20 for p in range(1, 9)}, **{p: 10 for p in range(9, 17)}}
assert re.search(r"1 High-Conditioned Elliptic -97\.8 20", setup), "setup table changed"
assert re.search(r"9 High-Conditioned Elliptic -97\.8 10", setup), "setup table changed"
one_each = sum(1 / K[p] for p in K) / len(K)
print(f"\n    K = 20 for PID 1-8, 10 for PID 9-16 (setup TR Table 1)")
print(f"    MPR of a method that reports exactly ONE global minimum per problem"
      f" = {one_each:.4f}")
print(f"    published N-DAM-CMA-ES MPR at D=10                              "
      f" = {mpr[2][2]:.4f}")

ndam = read("analysis/mmo2024/e92/refs/ndam_cmaes_2407.00939.txt.gz")
nflat = re.sub(r"\s+", " ", ndam)
assert "peak ratio (PR), defined as" in nflat, "N-DAM wording changed"
i = nflat.find("The average precision is")
blurb = nflat[i:i + 210]
self_score = float(re.search(r"overall score.*?is ([01]\.\d+)", blurb).group(1))
print(f"\n    N-DAM-CMA-ES paper's own aggregate: {blurb}")
print(f"    paper's self-reported Score {self_score} vs competition's "
      f"{overall[2]} -> {self_score / overall[2]:.1f}x")
print("\n    PER-PROBLEM PR/Score PUBLISHED FOR THE COMPETITION: no")
