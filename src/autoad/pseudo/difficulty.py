"""Fold difficulty diagnostics for LCO fairness control.

For each leave-cluster-out fold, we compute a model-independent
"separability" score that measures how distinguishable the held-out
cluster is from the union of training clusters in the chosen feature
space. The simplest diagnostic (d4 in the implementation plan, Section
4.4) is the cross-validated AUC of a logistic regression on the
features. Folds with very high or very low d4 are excluded from the
main aggregation per the fairness protocol.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def logreg_separability(
    features_in: np.ndarray,
    features_out: np.ndarray,
    n_splits: int = 3,
    random_state: int = 42,
) -> float:
    """3-fold CV AUC of a LogisticRegression distinguishing in vs out features.

    ``features_in``: feature matrix of the training-cluster windows.
    ``features_out``: feature matrix of the held-out cluster windows.

    Returns a single float AUC in [0, 1]. Values near 0.5 mean the
    held-out cluster is indistinguishable from the rest (impossible
    pseudo-anomaly task); values near 1.0 mean trivially separable.
    """
    if features_in.shape[0] < n_splits or features_out.shape[0] < n_splits:
        return float("nan")
    X = np.vstack([features_in, features_out])
    y = np.concatenate(
        [np.zeros(features_in.shape[0], dtype=np.int8), np.ones(features_out.shape[0], dtype=np.int8)]
    )
    # Stratified to ensure each fold has both classes.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs: list[float] = []
    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=random_state)
        try:
            clf.fit(X[train_idx], y[train_idx])
            prob = clf.predict_proba(X[val_idx])[:, 1]
            if len(np.unique(y[val_idx])) < 2:
                continue
            aucs.append(float(roc_auc_score(y[val_idx], prob)))
        except Exception:
            continue
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def difficulty_bucket(
    auc: float,
    easy_threshold: float = 0.85,
    impossible_threshold: float = 0.55,
    trivial_threshold: float = 0.97,
) -> str:
    """Classify a fold by its separability AUC.

    Buckets used by aggregation:
    * ``trivial``  : AUC >= trivial_threshold (excluded from main aggregation)
    * ``easy``     : easy_threshold <= AUC < trivial_threshold
    * ``medium``   : 0.70 <= AUC < easy_threshold
    * ``hard``     : impossible_threshold <= AUC < 0.70
    * ``impossible``: AUC < impossible_threshold (excluded from main)
    """
    if not np.isfinite(auc):
        return "invalid"
    if auc >= trivial_threshold:
        return "trivial"
    if auc >= easy_threshold:
        return "easy"
    if auc >= 0.70:
        return "medium"
    if auc >= impossible_threshold:
        return "hard"
    return "impossible"


VALID_BUCKETS = ("easy", "medium", "hard")
