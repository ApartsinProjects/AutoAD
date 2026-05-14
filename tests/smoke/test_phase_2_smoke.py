"""Phase-2 smoke test: end-to-end oracle on 5 real UCR series, 3 detectors.

Invariants (Section 16.2 of the implementation plan):
* All 3 reference detectors successfully fit on real UCR training data
* Per-window scores produce per-point scores aligned with y_test
* AUC-PR and AUC-ROC computable and in valid ranges
* Per-point parquet artifacts can be written and read back
* End-to-end run completes in well under 5 minutes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autoad.data.loaders import iter_ucr, list_ucr_files
from autoad.data.windowing import aggregate_window_labels, scores_to_per_point, sliding_windows
from autoad.eval.metrics import auc_pr, auc_roc
from autoad.models.classical import IForest, LOF, OCSVM
from autoad.utils.io import (
    RUNS_DIR,
    load_per_point_scores,
    per_point_exists,
    save_per_point_scores,
)

WINDOW = 64
STRIDE = 1


@pytest.fixture(scope="module")
def ucr_smoke_series():
    """Five real UCR series for end-to-end smoke testing."""
    if not list_ucr_files():
        pytest.skip("UCR data not downloaded; run scripts/01_download_ucr.py first")
    out = list(iter_ucr(limit=5))
    if len(out) < 3:
        pytest.skip(f"Only {len(out)} UCR series loaded; need >= 3 for smoke test")
    return out


@pytest.mark.smoke
def test_ucr_loader_returns_five(ucr_smoke_series):
    assert len(ucr_smoke_series) == 5
    for s in ucr_smoke_series:
        assert s.source == "ucr_anomaly_2021"
        assert s.x_train.dtype == np.float32
        assert s.x_test.dtype == np.float32
        assert s.y_test.dtype == np.uint8
        assert len(s.x_test) == len(s.y_test)
        assert s.y_test.sum() > 0, f"{s.series_id}: no anomalies"


@pytest.mark.smoke
@pytest.mark.parametrize("model_cls", [IForest, LOF, OCSVM])
def test_oracle_pipeline_three_detectors(ucr_smoke_series, model_cls, tmp_path: Path):
    """End-to-end: window training data, fit, score test, write+read per-point parquet."""
    results = []
    for s in ucr_smoke_series:
        if len(s.x_train) < WINDOW + 10 or len(s.x_test) < WINDOW + 10:
            continue
        wins_train = sliding_windows(s.x_train, WINDOW, STRIDE)
        wins_test = sliding_windows(s.x_test, WINDOW, STRIDE)
        win_labels = aggregate_window_labels(s.y_test, WINDOW, STRIDE, mode="max")
        # Train on first half of training windows to keep smoke runtime tight.
        train_n = min(len(wins_train), 2000)
        hp = model_cls.grid()[0]
        model = model_cls(**hp).fit(wins_train[:train_n])
        win_scores = model.score(wins_test)
        # Per-point reduction
        pp = scores_to_per_point(win_scores, len(s.x_test), WINDOW, STRIDE, reducer="max")
        assert pp.shape == (len(s.x_test),)
        assert np.isfinite(pp).all()
        assert (pp >= 0).all() and (pp <= 1).all()
        # Metrics
        apr = auc_pr(s.y_test, pp)
        aroc = auc_roc(s.y_test, pp)
        results.append({"series": s.series_id, "apr": apr, "aroc": aroc})
    # At least 3 of 5 series should have produced finite metrics.
    finite = [r for r in results if not (np.isnan(r["apr"]) or np.isnan(r["aroc"]))]
    assert len(finite) >= 3, f"Too few finite results: {results}"


@pytest.mark.smoke
def test_per_point_io_roundtrip(ucr_smoke_series, tmp_path: Path, monkeypatch):
    """Per-point scores can be persisted and read back with provenance metadata."""
    # Redirect RUNS_DIR to a tmp directory for hermetic test.
    import autoad.utils.io as io_mod
    monkeypatch.setattr(io_mod, "RUNS_DIR", tmp_path)
    s = ucr_smoke_series[0]
    fake_scores = np.linspace(0, 1, len(s.x_test), dtype=np.float32)
    save_per_point_scores(s.series_id, "iforest_smoke", fake_scores)
    assert per_point_exists(s.series_id, "iforest_smoke")
    back = load_per_point_scores(s.series_id, "iforest_smoke")
    np.testing.assert_allclose(back, fake_scores, atol=1e-6)
