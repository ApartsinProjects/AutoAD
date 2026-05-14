"""Phase-0 smoke test: BaseAD interface + 3 reference detectors round-trip.

Invariants asserted (Section 16.2 of the implementation plan):
* BaseAD.fit -> score round-trip works for IForest, LOF, OCSVM
* Scores are in [0, 1] (post-ECDF normalization)
* Models register themselves into the global registry
* Fixed seed 42 yields deterministic outputs
* End-to-end runs in well under 5 minutes on a laptop
"""

from __future__ import annotations

import numpy as np
import pytest

from autoad.models import enumerate_pool, get_model_class, list_families


@pytest.mark.smoke
def test_registry_has_three_reference_families():
    families = set(list_families())
    assert {"iforest", "lof", "ocsvm"}.issubset(families), (
        f"Phase-0 reference families missing: got {sorted(families)}"
    )


@pytest.mark.smoke
def test_enumerate_pool_phase0_size():
    pool = enumerate_pool()
    # Phase 0: 3 families with grids of (4, 4, 4) variants = 12 candidates.
    family_counts: dict[str, int] = {}
    for family, _hp in pool:
        family_counts[family] = family_counts.get(family, 0) + 1
    assert family_counts["iforest"] == 4
    assert family_counts["lof"] == 4
    assert family_counts["ocsvm"] == 4


@pytest.mark.smoke
@pytest.mark.parametrize("family", ["iforest", "lof", "ocsvm"])
def test_fit_score_roundtrip(family: str):
    """Each reference detector fits and scores without error; scores in [0, 1]."""
    rng = np.random.default_rng(42)
    n_train, n_test, w = 200, 100, 32
    X_train = rng.normal(0, 1, size=(n_train, w)).astype(np.float32)
    X_test = rng.normal(0, 1, size=(n_test, w)).astype(np.float32)
    cls = get_model_class(family)
    hp = cls.grid()[0]
    model = cls(**hp).fit(X_train)
    scores = model.score(X_test)
    assert scores.shape == (n_test,), f"{family}: scores shape mismatch"
    assert scores.dtype == np.float64 or scores.dtype == np.float32
    assert (scores >= 0).all() and (scores <= 1).all(), (
        f"{family}: scores must be in [0, 1] after ECDF normalization, "
        f"got min={scores.min()}, max={scores.max()}"
    )
    assert np.isfinite(scores).all(), f"{family}: scores contain NaN/inf"


@pytest.mark.smoke
@pytest.mark.parametrize("family", ["iforest", "lof", "ocsvm"])
def test_anomalies_score_higher_than_normals(family: str):
    """Sanity: on a trivially separable problem, anomalies should score higher."""
    rng = np.random.default_rng(42)
    n, w = 200, 32
    X_train = rng.normal(0, 1, size=(n, w)).astype(np.float32)
    X_normal = rng.normal(0, 1, size=(50, w)).astype(np.float32)
    X_anom = rng.normal(0, 1, size=(50, w)).astype(np.float32) + 5.0  # shifted
    cls = get_model_class(family)
    hp = cls.grid()[0]
    model = cls(**hp).fit(X_train)
    s_normal = model.score(X_normal).mean()
    s_anom = model.score(X_anom).mean()
    assert s_anom > s_normal, (
        f"{family}: shifted samples should score higher; "
        f"got normal={s_normal:.3f}, anomalous={s_anom:.3f}"
    )


@pytest.mark.smoke
def test_determinism_with_fixed_seed():
    """IForest with the same random_state produces identical scores."""
    from autoad.models.classical import IForest
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(100, 32)).astype(np.float32)
    s1 = IForest(n_estimators=50, random_state=42).fit(X).score(X)
    s2 = IForest(n_estimators=50, random_state=42).fit(X).score(X)
    np.testing.assert_allclose(s1, s2, err_msg="IForest with fixed seed must be deterministic")
