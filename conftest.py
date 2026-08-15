"""Pytest root configuration.

Its only job is to exist at the repository root: pytest inserts a root
``conftest.py``'s directory at the front of ``sys.path``, which makes the
``experiments/`` package importable from the test suite. ``tda2s`` itself is
installed into the environment and needs no help; the experiment drivers are
not installed, and Phase 2's tests check invariants that live in
``experiments/phase2_imbalance_sweep.py``.
"""
