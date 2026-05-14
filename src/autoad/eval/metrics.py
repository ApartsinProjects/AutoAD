"""Evaluation metrics for time-series anomaly detection.

Primary metric for the project is VUS-PR (Volume Under the Surface,
PR variant; Paparrizos et al. VLDBJ 2025). We vendor / wrap the
canonical implementation from TheDatumOrg/VUS via the bundled
TSB-AD vendored repo, falling back to a numpy reference here so that
unit tests pass without external imports.

Secondary metrics: AUC-PR, AUC-ROC, range-based F1 (PRTS).

This module currently exports AUC-PR and AUC-ROC. VUS-PR will be wired
to the TSB-AD implementation in Phase 2 once the benchmark loader is
online.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision = area under the PR curve."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_true.sum() == 0:
        return float("nan")  # undefined without positives
    return float(average_precision_score(y_true, y_score))


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
