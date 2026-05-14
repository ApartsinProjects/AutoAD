"""Base class and registry for anomaly detection models.

All AD models in AutoAD implement :class:`BaseAD`, which standardizes
fit/score I/O and hyperparameter grids. This makes the 60-candidate pool
addressable through a single interface for the LCO, synthetic, and
oracle pipelines.

Contract
--------
A subclass declares:

* ``name``: short identifier used as a key in result tables.
* ``family``: coarse grouping (e.g. ``"iforest"``, ``"lof"``, ``"deep"``).
* ``hyperparams``: dict of hyperparameters used by this instance.
* ``grid``: classmethod returning a list of hyperparameter dicts.

Method signatures:

* ``fit(X) -> self``: ``X`` shape ``(n_windows, window_len)`` for
  univariate or ``(n_windows, window_len, n_channels)`` for multivariate.
  Trained on normal data only.
* ``score(X) -> np.ndarray``: returns one anomaly score per window.
  Higher = more anomalous. Output is normalized to ``[0, 1]`` via empirical
  CDF on the training data so cross-fold scores are comparable.

Score normalization is enforced by :meth:`BaseAD.score`, not by subclasses.
Subclasses implement :meth:`_score_raw` and let the base class normalize.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


@dataclass
class ModelMeta:
    """Identity and provenance metadata attached to every fitted model."""

    name: str
    family: str
    hyperparams: dict[str, Any]
    train_n: int = 0
    train_shape: tuple[int, ...] = field(default_factory=tuple)


class BaseAD(ABC):
    """Abstract base class for anomaly detectors.

    Subclasses must set the class-level ``family`` attribute and implement
    :meth:`_fit_raw`, :meth:`_score_raw`, and :meth:`grid`.
    """

    family: ClassVar[str] = "abstract"

    def __init__(self, **hyperparams: Any) -> None:
        self.hyperparams: dict[str, Any] = dict(hyperparams)
        self._fitted: bool = False
        self._ecdf_anchor: np.ndarray | None = None
        self.meta = ModelMeta(
            name=self._derive_name(hyperparams),
            family=self.family,
            hyperparams=dict(hyperparams),
        )

    @classmethod
    def _derive_name(cls, hp: dict[str, Any]) -> str:
        if not hp:
            return cls.family
        # Sorted, short suffix: family_k1=v1_k2=v2
        parts = "_".join(f"{k}={v}" for k, v in sorted(hp.items()))
        return f"{cls.family}_{parts}"

    def fit(self, X: np.ndarray) -> "BaseAD":
        """Fit on normal data only.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_windows, window_len)`` (univariate) or
            ``(n_windows, window_len, n_channels)`` (multivariate).
        """
        X = self._validate(X)
        self._fit_raw(X)
        # Compute empirical CDF anchor on the training scores so that
        # downstream calls to ``score`` produce values in ``[0, 1]``.
        train_scores = self._score_raw(X)
        self._ecdf_anchor = np.sort(np.asarray(train_scores, dtype=np.float64))
        self.meta.train_n = X.shape[0]
        self.meta.train_shape = tuple(X.shape)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return per-window anomaly scores in ``[0, 1]`` (higher = more anomalous)."""
        if not self._fitted:
            raise RuntimeError(f"{self.meta.name}: must call fit() before score()")
        X = self._validate(X)
        raw = np.asarray(self._score_raw(X), dtype=np.float64)
        return self._ecdf_normalize(raw)

    @abstractmethod
    def _fit_raw(self, X: np.ndarray) -> None: ...

    @abstractmethod
    def _score_raw(self, X: np.ndarray) -> np.ndarray: ...

    @classmethod
    @abstractmethod
    def grid(cls) -> list[dict[str, Any]]:
        """Hyperparameter grid (list of dicts)."""
        ...

    def _ecdf_normalize(self, raw: np.ndarray) -> np.ndarray:
        """Map raw scores to [0, 1] via the empirical CDF of training scores."""
        assert self._ecdf_anchor is not None
        # searchsorted returns ranks; divide by N to get CDF values
        ranks = np.searchsorted(self._ecdf_anchor, raw, side="right")
        return np.clip(ranks / max(len(self._ecdf_anchor), 1), 0.0, 1.0)

    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim not in (2, 3):
            raise ValueError(f"X must be 1D, 2D, or 3D; got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        return X.astype(np.float32, copy=False)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.meta.name}>"


# ----------------------------------------------------------------------
# Registry: candidate models register themselves so the pipeline can
# enumerate the pool without hard-coding it.
# ----------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseAD]] = {}


def register(cls: type[BaseAD]) -> type[BaseAD]:
    """Class decorator that registers a model family by ``cls.family``."""
    if cls.family in _REGISTRY:
        raise ValueError(f"Family already registered: {cls.family}")
    _REGISTRY[cls.family] = cls
    return cls


def get_model_class(family: str) -> type[BaseAD]:
    if family not in _REGISTRY:
        raise KeyError(f"Unknown family {family!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[family]


def list_families() -> list[str]:
    return sorted(_REGISTRY)


def enumerate_pool() -> list[tuple[str, dict[str, Any]]]:
    """Return ``[(family, hyperparams), ...]`` for the entire registered pool."""
    out: list[tuple[str, dict[str, Any]]] = []
    for family in list_families():
        cls = _REGISTRY[family]
        for hp in cls.grid():
            out.append((family, hp))
    return out
