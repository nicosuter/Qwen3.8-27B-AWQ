"""Marks this directory a regular package so it cannot be shadowed.

Without an `__init__.py` these tests are only a namespace package, and Python
prefers a regular package found anywhere on sys.path over a namespace one --
path order does not save you. `google_search_results`, pulled in by `bfcl-eval`
for BFCL's web-search category, ships a top-level `tests/` package, so on any
machine with the EvalScope dependencies installed `python -m unittest tests.X`
imported that one instead and reported our whole module missing.
"""
