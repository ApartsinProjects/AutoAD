"""Anomaly detection model registry.

Importing this package triggers registration of all built-in detectors.
After import, :func:`autoad.models.base.list_families` returns every
registered family.
"""

from .base import BaseAD, enumerate_pool, get_model_class, list_families, register
from . import classical  # noqa: F401  (registration side-effects)

__all__ = [
    "BaseAD",
    "enumerate_pool",
    "get_model_class",
    "list_families",
    "register",
]
