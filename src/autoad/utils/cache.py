"""Disk-backed memoization for expensive intermediates.

Wraps :class:`joblib.Memory` with a project-standard cache directory
under ``runs/_cache/``. Used to cache:

* TSFresh feature extraction (one-day amortized cost; recompute = seconds)
* Cluster assignments per ``(series_id, algo, C, seed, feature_version)``
* Per-model fits and score-time-series (so any future metric can be
  recomputed without retraining)
"""

from __future__ import annotations

from pathlib import Path

from joblib import Memory

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "runs" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

memory = Memory(location=str(CACHE_DIR), verbose=0)


def cached(func):
    """Decorator alias for ``memory.cache``."""
    return memory.cache(func)
