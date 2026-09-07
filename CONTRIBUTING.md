# Contributing

1. Create a focused branch from the default branch.
2. Do not commit datasets, virtual environments, model checkpoints, personal
   documents, or generated results.
3. Run `python -m compileall src scripts app.py`, `pytest`, and `ruff check .`.
4. Explain dataset provenance and evaluation methodology for ML changes.
5. Never claim production accuracy from synthetic or ungrouped test data.
