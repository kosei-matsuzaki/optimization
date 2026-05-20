"""Backend support package for the Flask results UI (web/app.py).

Split out of the original monolithic ``app.py`` into cohesive layers:

- :mod:`app_lib.config`  — paths and GitHub constants (also puts the project
  root on ``sys.path`` so ``core`` and ``quick_check`` stay importable).
- :mod:`app_lib.results` — read-only data layer over ``results/`` (listings,
  media index, summary/Wilcoxon CSVs, Friedman ranking, run metadata).
- :mod:`app_lib.jobs`    — background job layer (local quick runs and GitHub
  artifact downloads) plus their in-memory state.
"""
