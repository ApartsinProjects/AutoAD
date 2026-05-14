"""Classical anomaly detectors wrapping scikit-learn and PyOD.

Reference set for Phase 0 smoke tests: IForest, LOF, OCSVM.
The full 60-candidate pool (Section 4 of the implementation plan) is
populated incrementally as subsequent phases come online.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .base import BaseAD, register


def _flatten(X: np.ndarray) -> np.ndarray:
    """Flatten (n, w) or (n, w, d) -> (n, w*d). Classical detectors expect 2D."""
    if X.ndim == 2:
        return X
    n = X.shape[0]
    return X.reshape(n, -1)


@register
class IForest(BaseAD):
    """Isolation Forest. Higher score = more anomalous."""

    family = "iforest"

    def _fit_raw(self, X: np.ndarray) -> None:
        self._model = IsolationForest(
            n_estimators=self.hyperparams.get("n_estimators", 100),
            contamination="auto",
            random_state=self.hyperparams.get("random_state", 42),
            n_jobs=1,
        )
        self._model.fit(_flatten(X))

    def _score_raw(self, X: np.ndarray) -> np.ndarray:
        # sklearn's score_samples returns higher = more normal; invert.
        return -self._model.score_samples(_flatten(X))

    @classmethod
    def grid(cls) -> list[dict[str, Any]]:
        return [{"n_estimators": n} for n in (50, 100, 200, 500)]


@register
class LOF(BaseAD):
    """Local Outlier Factor with novelty=True (predict on unseen data)."""

    family = "lof"

    def _fit_raw(self, X: np.ndarray) -> None:
        k = self.hyperparams.get("n_neighbors", 20)
        # Constrain k to be < n_samples to avoid sklearn errors on tiny inputs.
        k = min(k, max(2, X.shape[0] - 1))
        self._model = LocalOutlierFactor(
            n_neighbors=k,
            novelty=True,
            n_jobs=1,
        )
        self._model.fit(_flatten(X))

    def _score_raw(self, X: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal; invert so higher = more anomalous.
        return -self._model.decision_function(_flatten(X))

    @classmethod
    def grid(cls) -> list[dict[str, Any]]:
        return [{"n_neighbors": k} for k in (5, 10, 20, 50)]


@register
class OCSVM(BaseAD):
    """One-Class SVM (RBF kernel)."""

    family = "ocsvm"

    def _fit_raw(self, X: np.ndarray) -> None:
        self._model = OneClassSVM(
            nu=self.hyperparams.get("nu", 0.05),
            kernel=self.hyperparams.get("kernel", "rbf"),
            gamma="scale",
        )
        self._model.fit(_flatten(X))

    def _score_raw(self, X: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal; invert.
        return -self._model.decision_function(_flatten(X))

    @classmethod
    def grid(cls) -> list[dict[str, Any]]:
        return [{"nu": nu} for nu in (0.01, 0.05, 0.10, 0.20)]
