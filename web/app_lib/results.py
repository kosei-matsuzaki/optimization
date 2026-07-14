"""Read-only data layer over the ``results/`` directory.

Pure helpers that list runs, scan media files, parse the per-dimension
``summary.csv`` / ``wilcoxon.csv`` tables, compute the Friedman ranking, and
read/write each run's ``result.json`` metadata. No Flask or threading here.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from .config import BASE_DIR, RESULTS_DIR


# ── run / dim / function listings ───────────────────────────────────────────

def list_results() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        (d.name for d in RESULTS_DIR.iterdir() if d.is_dir()),
        reverse=True,
    )


def list_dims(run_dir: Path) -> list[str]:
    return sorted(
        d.name for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("dim")
    )


def list_functions(run_dir: Path, dim: str) -> list[str]:
    dim_dir = run_dir / dim
    if not dim_dir.exists():
        return []

    funcs: set[str] = set()

    # Primary: {Func}_landscape.svg or {Func}_convergence.svg
    for p in dim_dir.glob("*_landscape.svg"):
        funcs.add(p.stem[: -len("_landscape")])
    for p in dim_dir.glob("*_convergence.svg"):
        funcs.add(p.stem[: -len("_convergence")])

    # Fallback: summary.csv (before landscape SVGs are written)
    if not funcs:
        summary_path = run_dir / dim / "summary.csv"
        if summary_path.exists():
            with open(summary_path, newline="") as f:
                for row in csv.DictReader(f):
                    if "function" in row:
                        funcs.add(row["function"])

    return sorted(funcs)


# Known visualization types for filename parsing (longest-first to avoid prefix clash)
_ANIM_TYPES = [
    "3dpopulation_failed", "3devals_failed",
    "population_failed", "evals_failed", "outbreak_dyn_failed",
    "3dpopulation", "3devals",
    "population", "evals", "outbreak_dyn", "runs",
]


def build_media_index(run_dir: Path, dim: str) -> dict:
    """Scan dim directory and return structured media file index."""
    dim_dir = run_dir / dim
    funcs = list_functions(run_dir, dim)
    if not funcs:
        return {"funcs": [], "methods": [], "types": [], "files": []}

    files: list[dict] = []
    methods_seen: set[str] = set()
    types_seen: set[str] = set()

    for func in funcs:
        # Function-level SVGs
        for type_, suffix in [("landscape", "_landscape.svg"), ("convergence", "_convergence.svg")]:
            if (dim_dir / f"{func}{suffix}").exists():
                files.append({"func": func, "method": None, "type": type_, "ext": "svg"})
                types_seen.add(type_)

        # Per-method files: {func}_{method}_{type}.{ext}
        for ext in ("webp", "gif", "svg"):
            for p in sorted(dim_dir.glob(f"{func}_*.{ext}")):
                stem = p.stem  # e.g. F01-Sphere_MC-ESO_evals
                rest = stem[len(func) + 1:]  # e.g. MC-ESO_evals
                if not rest or rest in ("landscape", "convergence"):
                    continue
                matched_type = next(
                    (t for t in _ANIM_TYPES if rest.endswith(f"_{t}") or rest == t),
                    None,
                )
                if not matched_type:
                    continue
                method_part = rest[: -(len(matched_type) + 1)] if rest != matched_type else None
                if not method_part:
                    continue
                # Prefer webp over gif (skip duplicates)
                if any(f["func"] == func and f["method"] == method_part
                       and f["type"] == matched_type for f in files):
                    continue
                methods_seen.add(method_part)
                types_seen.add(matched_type)
                files.append({"func": func, "method": method_part, "type": matched_type, "ext": ext})

    return {
        "funcs": sorted(funcs),
        "methods": sorted(methods_seen),
        "types": sorted(types_seen),
        "files": files,
    }


# ── CSV tables ──────────────────────────────────────────────────────────────

def read_summary(run_dir: Path, dim: str) -> list[dict]:
    path = run_dir / dim / "summary.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_wilcoxon(run_dir: Path, dim: str) -> list[dict]:
    """Read per-function Wilcoxon signed-rank rows (reference vs each method)."""
    path = run_dir / dim / "wilcoxon.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_stats(run_id: str, dim: str, func_name: str) -> dict:
    """Per-run/per-function raw stats CSV → {headers, rows}."""
    csv_path = RESULTS_DIR / run_id / dim / "stats" / f"{func_name}.csv"
    if not csv_path.exists():
        return {"headers": [], "rows": []}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return {"headers": headers, "rows": rows}


# ── overall ranking ─────────────────────────────────────────────────────────

def compute_overall_ranking(run_dir: Path, dim: str) -> dict:
    """Per-indicator Friedman ranking across all functions.

    Ranks methods within each function independently for three indicators:
      - "bf"    : mean_best_f across runs (lower is better)
      - "evals" : mean evals-to-target across *successful* runs only
                  (lower is better; inf if no run succeeds). Taken over
                  successful runs, where the spread is small and outliers are
                  unlikely, so the mean is used rather than the median.
      - "ecdf"  : ECDF AUC over BBOB targets (higher is better)
    For each indicator we also report Friedman χ²_F, p value, and Nemenyi
    critical difference at α=0.05 so the user can judge whether mean-rank
    gaps are statistically meaningful.
    """
    import numpy as np
    from scipy.stats import friedmanchisquare, rankdata, studentized_range

    rows = read_summary(run_dir, dim)
    if not rows:
        return {"methods": [], "categories": [], "funcs": [],
                "leaderboard": [], "func_ranks": {}, "func_scores": {},
                "func_categories": {}, "func_tags": {}, "all_tags": [],
                "friedman": {}, "scopes": [], "by_suite": {}}

    def parse_sr(s: str) -> float:
        if not s or s == "N/A":
            return 0.0
        return float(s.strip("%")) / 100.0

    def parse_float(s: str) -> float:
        try:
            return float(s)
        except (ValueError, TypeError):
            return float("inf")

    def parse_frac(s: str) -> float:
        # pr_* columns are stored as bare fractions (e.g. "0.28"), not "%".
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    funcs   = sorted(set(r["function"] for r in rows))
    methods = sorted(set(r["method"]   for r in rows))
    func_categories: dict[str, str] = {}
    data: dict[str, dict] = {}
    # Shape tags: prefer the CSV `tags` column (pipe-joined); for legacy runs
    # that predate it, fall back to the canonical map keyed by function name.
    from core.benchmarks import SHAPE_TAGS
    func_tags: dict[str, list[str]] = {}
    for row in rows:
        f, m = row["function"], row["method"]
        func_categories[f] = row.get("category", "unknown")
        raw_tags = (row.get("tags") or "").strip()
        func_tags[f] = (raw_tags.split("|") if raw_tags
                        else list(SHAPE_TAGS.get(f, [])))
        # Rank bf by the across-run MEAN best_f; fall back to median for runs
        # that lack mean_best_f.
        bf_raw = row.get("mean_best_f") or row.get("median_best_f") or "inf"
        # ECDF AUC: higher is better; missing → 0 (worst).
        try:
            ecdf_v = float(row.get("ecdf_auc", "0") or "0")
        except (TypeError, ValueError):
            ecdf_v = 0.0
        # Prefer success-only MEAN evals; fall back to the median, then ERT,
        # for older runs that predate the evals_succ_mean column.
        evals_raw = row.get("evals_succ_mean")
        if evals_raw is None or evals_raw == "":
            evals_raw = row.get("evals_succ_med")
        if evals_raw is None or evals_raw == "":
            evals_raw = row.get("ert", "inf")
        data.setdefault(f, {})[m] = {
            "sr":      parse_sr(row.get("sr_1e-4", "0%")),
            "sr_deep": parse_sr(row.get("sr_1e-10", "0%")),  # primary indicator (deepest)
            "pr":      parse_frac(row.get("pr_1e-4", "0")),   # multi-optima discovery rate
            "evals":   parse_float(evals_raw),
            "bf":      parse_float(bf_raw),
            "ecdf":    ecdf_v,
        }

    # bf / evals: lower-is-better. ecdf: higher-is-better — negated when ranking.
    INDICATORS = ("bf", "evals", "ecdf")
    HIGHER_BETTER = {"ecdf"}
    k = len(methods)

    def _build(fs: list[str]) -> dict:
        """Full overall-evaluation payload restricted to the function subset
        ``fs``. Ranking, means, category breakdown and Friedman stats are all
        computed within ``fs`` only, so mixing suites (BBOB / Custom / CEC2022)
        never contaminates a suite-scoped view. Ranks are per-function so
        recomputing them on a subset yields identical values."""
        fs = sorted(fs)
        # Per-function, per-indicator ranks via average-tie ranking.
        f_ranks: dict[str, dict[str, dict[str, float]]] = {ind: {} for ind in INDICATORS}
        r_matrix: dict[str, list[list[float]]] = {ind: [] for ind in INDICATORS}
        for func in fs:
            md = data.get(func, {})
            for ind in INDICATORS:
                missing = float("-inf") if ind in HIGHER_BETTER else float("inf")
                vals = np.array([md.get(m, {}).get(ind, missing) for m in methods],
                                dtype=float)
                if ind in HIGHER_BETTER:
                    vals = -vals
                ranks_arr = rankdata(vals, method="average")
                ranks_dict = {m: float(r) for m, r in zip(methods, ranks_arr)}
                f_ranks[ind][func] = ranks_dict
                r_matrix[ind].append([ranks_dict[m] for m in methods])

        cats = sorted({func_categories.get(f, "unknown") for f in fs})

        # Friedman χ²_F + Nemenyi CD per indicator (over the subset's blocks).
        fried: dict[str, dict] = {}
        n_blocks = len(fs)
        for ind in INDICATORS:
            cols = np.array(r_matrix[ind]) if r_matrix[ind] else np.empty((0, k))
            if n_blocks >= 2 and k >= 3:
                try:
                    chi2, pval = friedmanchisquare(*[cols[:, j] for j in range(k)])
                    chi2_f, p_f = float(chi2), float(pval)
                except ValueError:
                    chi2_f, p_f = float("nan"), float("nan")
            else:
                chi2_f, p_f = float("nan"), float("nan")
            if k >= 2 and n_blocks >= 1:
                try:
                    q_alpha = float(studentized_range.ppf(0.95, k, np.inf))
                    cd_f = q_alpha / np.sqrt(2.0) * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
                except Exception:
                    cd_f = float("nan")
            else:
                cd_f = float("nan")
            fried[ind] = {
                "chi2":     None if np.isnan(chi2_f) else round(chi2_f, 3),
                "p":        None if np.isnan(p_f)    else float(f"{p_f:.4g}"),
                "cd":       None if np.isnan(cd_f)   else round(cd_f, 3),
                "n_blocks": n_blocks,
                "k":        k,
            }

        lb = []
        for method in methods:
            sr_vals      = [data.get(f, {}).get(method, {}).get("sr", 0.0)      for f in fs]
            sr_deep_vals = [data.get(f, {}).get(method, {}).get("sr_deep", 0.0) for f in fs]
            pr_vals      = [data.get(f, {}).get(method, {}).get("pr", 0.0)      for f in fs]
            mean_sr      = float(np.mean(sr_vals))      if sr_vals      else 0.0
            mean_sr_deep = float(np.mean(sr_deep_vals)) if sr_deep_vals else 0.0
            mean_pr      = float(np.mean(pr_vals))      if pr_vals      else 0.0
            cat_sr: dict[str, float | None] = {}
            for cat in cats:
                cf = [f for f in fs if func_categories.get(f) == cat]
                cat_sr[cat] = (
                    float(np.mean([data.get(f, {}).get(method, {}).get("sr", 0.0) for f in cf]))
                    if cf else None
                )
            entry: dict = {
                "method":       method,
                "mean_sr":      round(mean_sr, 4),       # SR@1e-4 (auxiliary)
                "mean_sr_deep": round(mean_sr_deep, 4),  # SR@1e-10 (primary)
                "mean_pr":      round(mean_pr, 4),       # PR@1e-4 (multi-optima discovery rate)
                "category_sr":  {c: (round(v, 4) if v is not None else None)
                                 for c, v in cat_sr.items()},
            }
            for ind in INDICATORS:
                rvals = [f_ranks[ind].get(f, {}).get(method, float(k)) for f in fs]
                arr = np.array(rvals, dtype=float) if rvals else np.array([float(k)])
                entry[f"mean_rank_{ind}"] = round(float(np.mean(arr)), 2)
                entry[f"rank_std_{ind}"]  = round(float(np.std(arr)),  2)
                entry[f"n_best_{ind}"]    = int(np.sum(arr == 1.0))
                entry[f"n_worst_{ind}"]   = int(np.sum(arr == float(k)))
            lb.append(entry)
        # Primary sort: evals mean rank → SR@1e-10 → SR@1e-4.
        lb.sort(key=lambda x: (x["mean_rank_evals"], -x["mean_sr_deep"], -x["mean_sr"]))

        f_scores: dict[str, dict[str, dict[str, float]]] = {
            "sr_deep": {}, "sr": {}, "pr": {},
        }
        for f in fs:
            for skey in f_scores:
                f_scores[skey][f] = {
                    m: float(data.get(f, {}).get(m, {}).get(skey, 0.0)) for m in methods
                }
        sub_tags = {f: func_tags.get(f, []) for f in fs}
        return {
            "methods":         methods,
            "categories":      cats,
            "funcs":           fs,
            "func_categories": {f: func_categories.get(f, "unknown") for f in fs},
            "func_tags":       sub_tags,
            "all_tags":        sorted({t for ts in sub_tags.values() for t in ts}),
            "leaderboard":     lb,
            "func_ranks":      f_ranks,
            "func_scores":     f_scores,
            "friedman":        fried,
        }

    # Partition functions into benchmark suites by name prefix and build a full
    # payload per suite. F* = BBOB, C* = Custom, G* = CEC2022. An "all" scope is
    # added only when more than one suite is present (otherwise it is redundant).
    def _suite_of(name: str) -> str:
        return {"F": "bbob", "C": "custom", "G": "cec"}.get(name[:1], "other")

    SUITE_ORDER = ["bbob", "custom", "cec", "other"]
    suite_funcs: dict[str, list[str]] = {}
    for f in funcs:
        suite_funcs.setdefault(_suite_of(f), []).append(f)
    present = [s for s in SUITE_ORDER if s in suite_funcs]

    by_suite: dict[str, dict] = {s: _build(suite_funcs[s]) for s in present}
    if len(present) > 1:
        by_suite["all"] = _build(funcs)
    scopes = present + (["all"] if len(present) > 1 else [])

    # Backward compatibility: the top-level payload mirrors the default scope
    # (whole run when mixed, else the single suite) so any older consumer keeps
    # working. The frontend reads `by_suite` / `scopes` for the suite selector.
    default = by_suite.get("all") or by_suite[present[0]]
    return {**default, "scopes": scopes, "by_suite": by_suite}


# ── git + run metadata ──────────────────────────────────────────────────────

def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASE_DIR), text=True,
        ).strip()
    except Exception:
        return "nogit"


def write_result_meta(run_dir: Path, meta: dict) -> None:
    try:
        with open(run_dir / "result.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


def read_result_meta(run_dir: Path) -> dict:
    path = run_dir / "result.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback: derive from directory name
    name = run_dir.name
    parts = name.split("_")
    meta: dict = {"type": "quick" if "quick" in name else "workflow"}
    if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
        d, t = parts[0], parts[1]
        meta["created_at"] = f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}"
        if len(parts) >= 3:
            meta["commit"] = parts[2]
    return meta
