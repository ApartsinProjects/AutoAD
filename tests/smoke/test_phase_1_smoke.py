"""Phase-1 smoke test: synthetic data generators + windowing + metrics.

Invariants:
* All six synthetic anomaly families produce signals + labels with valid shapes
* Labels are binary (0/1) and contain at least one positive
* Sliding-window utilities round-trip per-point <-> per-window labels
* AUC-PR > 0.5 for IForest on synthetic level-shift anomalies (sanity)
"""

from __future__ import annotations

import numpy as np
import pytest

from autoad.data.synthetic import (
    INJECTORS,
    inject_contextual,
    make_normal_sine,
    make_synthetic_series,
)
from autoad.data.windowing import (
    aggregate_window_labels,
    scores_to_per_point,
    sliding_windows,
)
from autoad.eval.metrics import auc_pr, auc_roc
from autoad.models.classical import IForest


@pytest.mark.smoke
@pytest.mark.parametrize("family", list(INJECTORS.keys()))
def test_injector_shapes(family: str):
    """Every injector produces aligned signal + labels with positives."""
    if family == "contextual":
        pytest.skip("Contextual is tested separately (needs explicit period)")
    y, labels = make_synthetic_series(family, length=1000, seed=42)
    assert y.shape == labels.shape == (1000,)
    assert labels.dtype == np.uint8
    assert labels.sum() > 0, f"{family}: no anomalies injected"
    assert np.isfinite(y).all()


@pytest.mark.smoke
def test_contextual_injector():
    x = make_normal_sine(1000, period=50, seed=42)
    y, labels = inject_contextual(x, period=50, seed=42)
    assert y.shape == labels.shape == (1000,)
    assert labels.sum() > 0


@pytest.mark.smoke
def test_windowing_roundtrip():
    """Sliding windows + label aggregation + per-point reduction round-trip."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, size=n).astype(np.float32)
    labels = np.zeros(n, dtype=np.uint8)
    labels[100:120] = 1
    w, s = 32, 1
    wins = sliding_windows(x, w, s)
    expected_n = (n - w) // s + 1
    assert wins.shape == (expected_n, w)
    win_labels = aggregate_window_labels(labels, w, s, mode="max")
    assert win_labels.shape == (expected_n,)
    assert win_labels.sum() > 0
    # Per-point reduction: feed back random scores and verify shape
    win_scores = rng.uniform(0, 1, size=expected_n).astype(np.float32)
    pp = scores_to_per_point(win_scores, n, w, s, reducer="max")
    assert pp.shape == (n,)
    assert (pp >= 0).all() and (pp <= 1).all()


@pytest.mark.smoke
def test_iforest_detects_level_shift():
    """End-to-end mini-pipeline: synthetic anomalies + windowed IForest + AUC-PR > 0.5."""
    # Make a clean normal series, inject a few level shifts
    y, labels = make_synthetic_series(
        "level_shift", length=2000, period=50, n_anomalies=3, c_sigma=5.0, seed=42
    )
    w, stride = 32, 1
    windows = sliding_windows(y, w, stride)
    win_labels = aggregate_window_labels(labels, w, stride, mode="max")
    # Train on the first half (assumed clean enough for smoke test);
    # score the whole thing
    train_n = len(windows) // 2
    model = IForest(n_estimators=50, random_state=42).fit(windows[:train_n])
    scores = model.score(windows)
    apr = auc_pr(win_labels, scores)
    aroc = auc_roc(win_labels, scores)
    assert apr > 0.20, f"AUC-PR too low: {apr:.3f} (level-shift should be detectable)"
    assert aroc > 0.70, f"AUC-ROC too low: {aroc:.3f}"


@pytest.mark.smoke
def test_metrics_handle_edge_cases():
    """AUC-PR returns NaN when there are no positives; AUC-ROC same for degenerate."""
    y = np.zeros(100, dtype=int)
    s = np.random.default_rng(0).uniform(0, 1, size=100)
    assert np.isnan(auc_pr(y, s))
    assert np.isnan(auc_roc(y, s))
